"""Run ALNS / PALNS variants on benchmark or generated instances.

Methods (all receive the SAME total number of destroy/repair iterations):
  ALNS          one trajectory, W*E*I iterations
  PALNS-indep   W independent workers, no synchronization (multi-start control)
  PALNS         W workers, elite sharing + operator-weight synchronization
  VAE-PALNS     PALNS + VAE-guided destroy operators (model chosen by alias)
  RACE          shift-aware online race: guided vs unguided worker groups for
                --race-epochs epochs, then all workers follow the winner

Instances: file paths (.vrp CVRPLIB or Solomon .txt) or generator specs
  gen:n=100,depot=C,customers=R,demand=1-100,route_size=8,seed=11

Example:
  python scripts/run_benchmarks.py --instances data/cvrplib/X-n101-k25.vrp \
      --methods ALNS,PALNS-indep,PALNS,VAE-PALNS:mixed --vae mixed=models/vae_mixed.npz \
      --seeds 5 --out results/bench.csv
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vaepalns import check_feasible, generate_uchoa, load_any  # noqa: E402
from vaepalns.alns import ALNSParams  # noqa: E402
from vaepalns.guidance import VAEGuidance  # noqa: E402
from vaepalns.palns import PALNSConfig, run_palns  # noqa: E402
from vaepalns.vae import VAE  # noqa: E402


def parse_instance(spec: str):
    if spec.startswith("gen:"):
        kv = dict(t.split("=") for t in spec[4:].split(","))
        args = {"n": int(kv["n"]), "depot": kv.get("depot", "R"),
                "customers": kv.get("customers", "R"), "demand": kv.get("demand", "1-100"),
                "route_size": float(kv["route_size"]) if "route_size" in kv else None,
                "seed": int(kv.get("seed", 0))}
        return generate_uchoa(**args)
    inst = load_any(spec)
    if not spec.lower().endswith(".vrp"):
        inst.name = os.path.splitext(os.path.basename(spec))[0]
    return inst


def load_refs():
    p = os.path.join(ROOT, "data", "reference_costs.csv")
    if not os.path.exists(p):
        return {}
    r = pd.read_csv(p)
    best = r[["ref_cost", "bks_literature"]].min(axis=1)
    return dict(zip(r.instance, best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--methods", default="ALNS,PALNS-indep,PALNS,VAE-PALNS:mixed")
    ap.add_argument("--vae", nargs="*", default=[], help="alias=path.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--iters-per-epoch", type=int, default=150)
    ap.add_argument("--race-epochs", type=int, default=3)
    ap.add_argument("--vae-p", type=float, default=3.0)
    ap.add_argument("--t-start", type=float, default=None, help="SA start (fraction worse)")
    ap.add_argument("--q-max", type=float, default=None, help="max removal fraction")
    ap.add_argument("--no-processes", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/bench.csv")
    a = ap.parse_args()

    specs = []
    for s in a.instances:
        specs += sorted(glob.glob(s)) if any(ch in s for ch in "*?") else [s]
    models = {}
    for kv in a.vae:
        alias, path = kv.split("=", 1)
        models[alias] = VAE.load(path)
    refs = load_refs()
    methods = a.methods.split(",")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out)
        done = set(zip(old.instance, old.method, old.seed))
    ex = None if a.no_processes else ProcessPoolExecutor(max_workers=a.workers)
    try:
        for spec in specs:
            inst = parse_instance(spec)
            ref = refs.get(inst.name, np.nan)
            ood = {al: VAEGuidance(m).ood_percentile(inst) for al, m in models.items()}
            score = {al: VAEGuidance(m).instance_score(inst) for al, m in models.items()}
            for seed in range(a.seed_offset, a.seed_offset + a.seeds):
                for meth in methods:
                    if (inst.name, meth, seed) in done:
                        continue
                    base, _, alias = meth.partition(":")
                    W = a.workers
                    prm = ALNSParams()
                    if a.t_start is not None:
                        prm.t_start_worse = a.t_start
                    if a.q_max is not None:
                        prm.q_max_frac = a.q_max
                    if base == "ALNS":
                        cfg = PALNSConfig(n_workers=1, epochs=a.epochs,
                                          iters_per_epoch=a.iters_per_epoch * W, params=prm)
                    else:
                        cfg = PALNSConfig(n_workers=W, epochs=a.epochs,
                                          iters_per_epoch=a.iters_per_epoch,
                                          sync=(base != "PALNS-indep"),
                                          use_processes=not a.no_processes, params=prm)
                    guid = VAEGuidance(models[alias], p=a.vae_p) if base in ("VAE-PALNS", "RACE") else None
                    out = run_palns(inst, cfg, seed=seed, guidance=guid, executor=ex,
                                    race_epochs=a.race_epochs if base == "RACE" else 0)
                    ok, msg = check_feasible(inst, out["best"])
                    used = {k: v["used"] for k, v in out["d_stats"].items()}
                    best_by = {k: v["best"] for k, v in out["d_stats"].items()}
                    tot = max(1, sum(used.values()))
                    vae_share = sum(v for k, v in used.items() if k.startswith("vae")) / tot
                    row = {"tag": a.tag, "instance": inst.name, "n": inst.n,
                           "layout": inst.meta.get("customers", inst.meta.get("format")),
                           "method": meth, "seed": seed, "cost": out["cost"],
                           "ref_cost": ref, "gap_pct": 100 * (out["cost"] / ref - 1) if ref == ref else np.nan,
                           "routes": out["best"].n_routes, "feasible": ok,
                           "wall_s": out["seconds"], "cpu_s": out["worker_seconds"],
                           "iterations": out["total_iterations"], "vae_destroy_share": vae_share,
                           "ood_pct": ood.get(alias, np.nan),
                           "shift_score": score.get(alias, np.nan),
                           "race_winner": out.get("race", {}).get("race_winner", ""),
                           "destroy_used": json.dumps(used), "destroy_best": json.dumps(best_by),
                           "final_d_weights": json.dumps({k: round(v, 3) for k, v in out["d_weights"].items()}),
                           "history": json.dumps([(int(i), float(c)) for i, c, _ in out["history"]])}
                    pd.DataFrame([row]).to_csv(a.out, mode="a", index=False,
                                               header=not os.path.exists(a.out))
                    print(f"{inst.name:28s} {meth:22s} seed {seed}  cost {out['cost']:10.2f}  "
                          f"gap {row['gap_pct']:6.2f}%  cpu {row['cpu_s']:6.1f}s  "
                          f"vae {vae_share:.2f}  feasible={ok}", flush=True)
    finally:
        if ex is not None:
            ex.shutdown()


if __name__ == "__main__":
    main()
