"""Adaptive Large Neighborhood Search (Ropke & Pisinger, 2006) for CVRP.

* roulette-wheel operator selection with segment-based adaptive weights
  (scores sigma1/sigma2/sigma3, reaction factor r, segment length L)
* simulated-annealing acceptance with a global geometric schedule, so that
  a run split into epochs (parallel PALNS) follows the same schedule as one
  long run
* destroy and repair weights are adapted independently
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .instance import Instance
from .solution import Solution


@dataclass
class ALNSParams:
    sigma1: float = 33.0       # new global best
    sigma2: float = 9.0        # improved current, not seen before
    sigma3: float = 13.0       # accepted worse, not seen before
    reaction: float = 0.1
    segment: int = 100
    q_min_frac: float = 0.05
    q_max_frac: float = 0.30
    q_min_abs: int = 2
    q_max_abs: int = 60
    t_start_worse: float = 0.05   # a 5% worse solution accepted w.p. 0.5 at start
    t_end_ratio: float = 1e-3     # final temperature = T0 * ratio
    min_weight: float = 0.05


@dataclass
class OpStats:
    used: int = 0
    best: int = 0
    improved: int = 0
    accepted: int = 0
    time: float = 0.0


@dataclass
class RunResult:
    best: Solution
    current: Solution
    d_weights: dict
    r_weights: dict
    history: list = field(default_factory=list)       # (global_iter, best_cost)
    d_stats: dict = field(default_factory=dict)
    r_stats: dict = field(default_factory=dict)
    iterations: int = 0
    seconds: float = 0.0
    cpu_seconds: float = 0.0


def initial_temperature(cost0: float, p: ALNSParams) -> float:
    return -p.t_start_worse * cost0 / math.log(0.5)


def temperature(t0: float, it: int, total: int, p: ALNSParams) -> float:
    frac = min(1.0, it / max(1, total))
    return t0 * (p.t_end_ratio ** frac)


def _roulette(names, weights, rng):
    w = np.array([weights[n] for n in names], float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def run_alns(inst: Instance, start: Solution, destroy_ops: dict, repair_ops: dict,
             iterations: int, rng: np.random.Generator, params: ALNSParams | None = None,
             d_weights: dict | None = None, r_weights: dict | None = None,
             it_offset: int = 0, total_iterations: int | None = None,
             t0: float | None = None, best: Solution | None = None,
             history_every: int = 50, cpu_budget: float | None = None,
             cpu_t0: float | None = None, cpu_offset: float = 0.0,
             cpu_total: float | None = None) -> RunResult:
    """Run ``iterations`` ALNS iterations starting from ``start``.

    ``it_offset``/``total_iterations``/``t0`` let PALNS run the global
    schedule in epochs. Returns best, final current solution and weights.

    Time-controlled mode (``cpu_budget`` set): iterate until this process has
    used ``cpu_budget`` CPU seconds since ``cpu_t0`` (default: the call), with
    ``iterations`` as a safety cap. The annealing schedule then follows CPU
    time: the fraction elapsed is ``(cpu_offset + used) / cpu_total``.
    """
    p = params or ALNSParams()
    total = total_iterations or iterations
    t0 = t0 if t0 is not None else initial_temperature(start.cost, p)
    dn, rn = list(destroy_ops), list(repair_ops)
    dw = dict(d_weights) if d_weights else {k: 1.0 for k in dn}
    rw = dict(r_weights) if r_weights else {k: 1.0 for k in rn}
    for k in dn:
        dw.setdefault(k, 1.0)
    for k in rn:
        rw.setdefault(k, 1.0)
    d_seg = {k: [0.0, 0] for k in dn}
    r_seg = {k: [0.0, 0] for k in rn}
    d_stats = {k: OpStats() for k in dn}
    r_stats = {k: OpStats() for k in rn}

    cur = start.copy()
    best = (best if best is not None and best.cost <= cur.cost else cur).copy()
    seen = {cur.key()}
    hist = [(it_offset, best.cost)]
    n = inst.n
    qmin = max(p.q_min_abs, int(p.q_min_frac * n))
    qmax = max(qmin, min(p.q_max_abs, int(p.q_max_frac * n)))
    qmin, qmax = min(qmin, n), min(qmax, n)
    t_start = time.perf_counter()
    c_start = time.process_time() if cpu_t0 is None else cpu_t0
    done_iters = 0

    for local in range(iterations):
        it = it_offset + local
        if cpu_budget is not None:
            used = time.process_time() - c_start
            if used >= cpu_budget:
                break
            frac = min(1.0, (cpu_offset + used) / cpu_total)
            T = t0 * (p.t_end_ratio ** frac)
        else:
            T = temperature(t0, it, total, p)
        done_iters += 1
        dname, rname = _roulette(dn, dw, rng), _roulette(rn, rw, rng)
        q = int(rng.integers(qmin, qmax + 1))

        cand = cur.copy()
        t1 = time.perf_counter()
        removed = destroy_ops[dname](inst, cand, q, rng)
        t2 = time.perf_counter()
        repair_ops[rname](inst, cand, removed, rng)
        t3 = time.perf_counter()
        d_stats[dname].time += t2 - t1
        r_stats[rname].time += t3 - t2
        d_stats[dname].used += 1
        r_stats[rname].used += 1

        score = 0.0
        key = cand.key()
        new = key not in seen
        if cand.cost < best.cost - 1e-9:
            best = cand.copy()
            cur = cand
            score = p.sigma1
            d_stats[dname].best += 1; r_stats[rname].best += 1
        elif cand.cost < cur.cost - 1e-9:
            cur = cand
            score = p.sigma2 if new else 0.0
            d_stats[dname].improved += 1; r_stats[rname].improved += 1
        elif T > 0 and rng.random() < math.exp(-(cand.cost - cur.cost) / T):
            cur = cand
            score = p.sigma3 if new else 0.0
            d_stats[dname].accepted += 1; r_stats[rname].accepted += 1
        if new:
            seen.add(key)
        d_seg[dname][0] += score; d_seg[dname][1] += 1
        r_seg[rname][0] += score; r_seg[rname][1] += 1

        if (local + 1) % p.segment == 0:
            for seg, w in ((d_seg, dw), (r_seg, rw)):
                for k, (s, c) in seg.items():
                    if c > 0:
                        w[k] = max(p.min_weight, (1 - p.reaction) * w[k] + p.reaction * s / c)
                    seg[k] = [0.0, 0]
        if (local + 1) % history_every == 0:
            hist.append((it + 1, best.cost))

    hist.append((it_offset + done_iters, best.cost))
    return RunResult(best=best, current=cur, d_weights=dw, r_weights=rw, history=hist,
                     d_stats={k: vars(v) for k, v in d_stats.items()},
                     r_stats={k: vars(v) for k, v in r_stats.items()},
                     iterations=done_iters, seconds=time.perf_counter() - t_start,
                     cpu_seconds=time.process_time() - c_start)
