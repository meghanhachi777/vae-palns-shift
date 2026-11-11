"""Parallel ALNS with periodic synchronization.

``W`` workers each run ALNS on their own trajectory for ``iters_per_epoch``
iterations, following one global simulated-annealing schedule.  After every
epoch the master
  1. updates the global best,
  2. (elite sharing) restarts the worst ``restart_frac`` of workers from the
     global best solution,
  3. (weight sync) replaces every worker's operator weights by the average
     of all workers' weights.
Setting ``sync=False`` gives independent multi-start ALNS, kept as an ablation.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .alns import ALNSParams, initial_temperature, run_alns
from .instance import Instance
from .operators import DESTROY_OPS, REPAIR_OPS
from .solution import Solution, clarke_wright


@dataclass
class PALNSConfig:
    n_workers: int = 4
    epochs: int = 10
    iters_per_epoch: int = 300
    sync: bool = True
    restart_frac: float = 0.5
    destroy: tuple = tuple(DESTROY_OPS)
    repair: tuple = tuple(REPAIR_OPS)
    use_processes: bool = True
    params: ALNSParams = field(default_factory=ALNSParams)
    # time-controlled mode: CPU seconds per worker over the whole run (split equally over
    # epochs). Each worker-epoch stops when its CPU budget is used, including the set-up
    # of the (learned) operators; iters_per_epoch is then only a safety cap.
    cpu_per_worker: float | None = None

    @property
    def iters_per_worker(self) -> int:
        return self.epochs * self.iters_per_epoch

    @property
    def total_iterations(self) -> int:
        return self.n_workers * self.iters_per_worker


def build_operators(inst: Instance, destroy, repair, guidance=None):
    d = {k: DESTROY_OPS[k] for k in destroy if k in DESTROY_OPS}
    r = {k: REPAIR_OPS[k] for k in repair}
    if guidance is not None:
        d.update(guidance.destroy_operators(inst))
    return d, r


def _epoch_worker(payload):
    c0 = time.process_time()
    (inst, cur_routes, best_routes, dw, rw, seed, it_off, iters, total, t0,
     destroy, repair, guidance, params) = payload[:14]
    tc = payload[14] if len(payload) > 14 else None   # (budget, offset, total) or None
    rng = np.random.default_rng(seed)
    d_ops, r_ops = build_operators(inst, destroy, repair, guidance)
    cur = Solution(inst, cur_routes)
    best = Solution(inst, best_routes)
    res = run_alns(inst, cur, d_ops, r_ops, iters, rng, params=params,
                   d_weights=dw, r_weights=rw, it_offset=it_off,
                   total_iterations=total, t0=t0, best=best,
                   **({} if tc is None else {"cpu_budget": tc[0], "cpu_t0": c0,
                                             "cpu_offset": tc[1], "cpu_total": tc[2]}))
    return {"iterations": res.iterations, "cpu_total": time.process_time() - c0,"best": res.best.routes, "best_cost": res.best.cost,
            "cur": res.current.routes, "cur_cost": res.current.cost,
            "dw": res.d_weights, "rw": res.r_weights, "history": res.history,
            "d_stats": res.d_stats, "r_stats": res.r_stats, "seconds": res.seconds,
            "cpu_seconds": res.cpu_seconds}


def _avg_weights(ws):
    """Average of normalized weights over workers (union of operator keys)."""
    ws = [w for w in ws if w]
    if not ws:
        return None
    keys = list(dict.fromkeys(k for w in ws for k in w))
    out = {}
    for k in keys:
        vals = [w[k] / sum(w.values()) for w in ws if k in w]
        out[k] = float(np.mean(vals)) * len(keys)
    return out


def _merge_stats(acc, new):
    for k, v in new.items():
        a = acc.setdefault(k, {kk: 0 for kk in v})
        for kk, vv in v.items():
            a[kk] += vv


def run_palns(inst: Instance, cfg: PALNSConfig, seed: int = 0, guidance=None,
              start: Solution | None = None, executor=None, verbose: bool = False,
              race_epochs: int = 0) -> dict:
    """Parallel ALNS.

    ``race_epochs > 0`` (requires ``guidance``) runs the online shift-aware
    race: for the first ``race_epochs`` epochs even workers use the learned
    operators and odd workers do not, each group synchronizing only
    internally; the group with the better best cost then wins, and all
    workers continue with the winner's operator set, restarting the losing
    group from the winner's best solution and weights.
    """
    t_wall = time.perf_counter()
    start = start or clarke_wright(inst)
    t0 = initial_temperature(start.cost, cfg.params)
    W = cfg.n_workers
    racing = guidance is not None and race_epochs > 0 and W >= 2
    guided = [(w % 2 == 0) if racing else (guidance is not None) for w in range(W)]
    groups = [[w for w in range(W) if guided[w]], [w for w in range(W) if not guided[w]]]
    groups = [g for g in groups if g] if racing else [list(range(W))]
    cur = [start.routes for _ in range(W)]
    wbest = [start.routes for _ in range(W)]
    dws = [None] * W
    rws = [None] * W
    gbest, gbest_cost = start.routes, start.cost
    history = [(0, gbest_cost, 0.0)]
    d_stats, r_stats = {}, {}
    worker_seconds = 0.0
    race_info = {}
    timed = cfg.cpu_per_worker is not None
    ep_budget = cfg.cpu_per_worker / cfg.epochs if timed else None
    w_iters = [0] * W
    own_pool = None
    if executor is None and cfg.use_processes and W > 1:
        own_pool = ProcessPoolExecutor(max_workers=W)
        executor = own_pool
    try:
        for ep in range(cfg.epochs):
            payloads = [(inst, cur[w], wbest[w], dws[w], rws[w],
                         int(seed) * 100_003 + w * 1009 + ep,
                         w_iters[w] if timed else ep * cfg.iters_per_epoch,
                         cfg.iters_per_epoch, cfg.iters_per_worker, t0,
                         tuple(cfg.destroy), tuple(cfg.repair),
                         guidance if guided[w] else None, cfg.params)
                        + (((ep_budget, ep * ep_budget, cfg.cpu_per_worker),) if timed else ())
                        for w in range(W)]
            if executor is not None and W > 1:
                outs = list(executor.map(_epoch_worker, payloads))
            else:
                outs = [_epoch_worker(p) for p in payloads]
            for w, o in enumerate(outs):
                cur[w], wbest[w] = o["cur"], o["best"]
                dws[w], rws[w] = o["dw"], o["rw"]
                _merge_stats(d_stats, o["d_stats"])
                _merge_stats(r_stats, o["r_stats"])
                worker_seconds += o["cpu_total"] if timed else o["cpu_seconds"]
                w_iters[w] += o["iterations"]
                if o["best_cost"] < gbest_cost - 1e-9:
                    gbest, gbest_cost = o["best"], o["best_cost"]
            if cfg.sync and W > 1:
                for g in groups:
                    gb = min(g, key=lambda w: outs[w]["best_cost"])
                    order = sorted(g, key=lambda w: -outs[w]["cur_cost"])   # worst first
                    for w in order[: int(round(cfg.restart_frac * len(g)))]:
                        cur[w] = outs[gb]["best"]
                    dw_avg, rw_avg = _avg_weights([dws[w] for w in g]), _avg_weights([rws[w] for w in g])
                    for w in g:
                        dws[w], rws[w] = dict(dw_avg), dict(rw_avg)
            if racing and ep + 1 == race_epochs:
                gcost = [min(outs[w]["best_cost"] for w in g) for g in groups]
                win = int(np.argmin(gcost)) if len(groups) == 2 and gcost[0] < gcost[1] - 1e-9 else len(groups) - 1
                win_guided = guided[groups[win][0]]
                wb = min(groups[win], key=lambda w: outs[w]["best_cost"])
                for w in range(W):
                    if w not in groups[win]:
                        cur[w], wbest[w] = outs[wb]["best"], outs[wb]["best"]
                        dws[w], rws[w] = dict(dws[wb]), dict(rws[wb])
                    guided[w] = win_guided
                race_info = {"race_guided_best": gcost[0], "race_unguided_best": gcost[-1],
                             "race_winner": "guided" if win_guided else "unguided"}
                groups = [list(range(W))]
                racing = False
            history.append(((ep + 1) * cfg.iters_per_epoch * W, gbest_cost,
                            time.perf_counter() - t_wall))
            if verbose:
                print(f"  epoch {ep + 1}/{cfg.epochs}: best {gbest_cost:.1f}")
    finally:
        if own_pool is not None:
            own_pool.shutdown()
    best = Solution(inst, gbest)
    return {"best": best, "cost": best.cost, "history": history,
            "d_weights": dws[0], "r_weights": rws[0], "d_stats": d_stats,
            "r_stats": r_stats, "seconds": time.perf_counter() - t_wall,
            "worker_seconds": worker_seconds, "race": race_info,
            "total_iterations": sum(w_iters) if timed else cfg.total_iterations}
