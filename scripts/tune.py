"""Small tuning pass on generated tuning instances (disjoint seeds from all test sets).

Stage 1: SA start temperature x max removal fraction (unguided ALNS).
Stage 2: VAE removal bias p (guided ALNS, best stage-1 setting).
Selection criterion: mean % deviation from the best cost found for each
instance across all configurations. Budget: 4000 iterations per run.

    python scripts/tune.py --out results/tuning
"""
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vaepalns import DESTROY_OPS, REPAIR_OPS, clarke_wright, generate_uchoa, run_alns  # noqa: E402
from vaepalns.alns import ALNSParams  # noqa: E402
from vaepalns.guidance import VAEGuidance  # noqa: E402
from vaepalns.vae import VAE  # noqa: E402

TUNING_SPECS = [
    dict(n=100, depot="C", customers="R", demand="1-100", route_size=8, seed=5001),
    dict(n=100, depot="R", customers="RC", demand="SL", route_size=10, seed=5002),
    dict(n=100, depot="C", customers="C", demand="Q", route_size=6, seed=5003),
    dict(n=120, depot="E", customers="R", demand="5-10", route_size=12, seed=5004),
    dict(n=80, depot="R", customers="RC", demand="1-10", route_size=5, seed=5005),
    dict(n=100, depot="C", customers="R", demand="50-100", route_size=14, seed=5006),
]


def one(args):
    spec, seed, t_start, qmax, vae_path, p = args
    inst = generate_uchoa(**spec)
    d_ops = dict(DESTROY_OPS)
    if vae_path:
        d_ops.update(VAEGuidance(VAE.load(vae_path), p=p).destroy_operators(inst))
    prm = ALNSParams(t_start_worse=t_start, q_max_frac=qmax)
    res = run_alns(inst, clarke_wright(inst), d_ops, REPAIR_OPS, 4000,
                   np.random.default_rng(seed), params=prm)
    return {"instance": inst.name, "seed": seed, "t_start": t_start, "qmax": qmax,
            "p": p if vae_path else np.nan, "cost": res.best.cost}


def summarize(df, keys):
    best = df.groupby("instance").cost.transform("min")
    df = df.assign(dev=100 * (df.cost / best - 1))
    return df.groupby(keys).dev.agg(["mean", "std", "median"]).sort_values("mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="models/vae_uniform.npz")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/tuning")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    grid = list(itertools.product([0.05, 0.01, 0.003], [0.2, 0.3, 0.4]))
    jobs = [(s, sd, t, q, None, None) for s in TUNING_SPECS for sd in range(a.seeds) for t, q in grid]
    with ProcessPoolExecutor(a.jobs) as ex:
        r1 = pd.DataFrame(list(ex.map(one, jobs)))
    s1 = summarize(r1, ["t_start", "qmax"])
    print("Stage 1 (SA start x max removal):\n", s1.round(3))
    t_best, q_best = s1.index[0]

    jobs = [(s, sd, t_best, q_best, a.vae, p) for s in TUNING_SPECS for sd in range(a.seeds)
            for p in (1.5, 3.0, 6.0)]
    with ProcessPoolExecutor(a.jobs) as ex:
        r2 = pd.DataFrame(list(ex.map(one, jobs)))
    s2 = summarize(r2, ["p"])
    print("Stage 2 (VAE removal bias p):\n", s2.round(3))
    p_best = float(s2.index[0])

    r1.to_csv(os.path.join(a.out, "stage1.csv"), index=False)
    r2.to_csv(os.path.join(a.out, "stage2.csv"), index=False)
    chosen = {"t_start_worse": float(t_best), "q_max_frac": float(q_best), "vae_p": p_best}
    with open(os.path.join(a.out, "chosen.json"), "w") as fh:
        json.dump(chosen, fh, indent=1)
    with open(os.path.join(a.out, "tuning.md"), "w") as fh:
        fh.write("# Tuning (6 generated tuning instances x %d seeds, 4000 iterations)\n\n" % a.seeds)
        fh.write("## Stage 1: SA start temperature x max removal fraction\n\n")
        fh.write(s1.round(3).to_markdown() + "\n\n## Stage 2: VAE removal bias p\n\n")
        fh.write(s2.round(3).to_markdown() + f"\n\nChosen: `{json.dumps(chosen)}`\n")
    print("chosen", chosen)


if __name__ == "__main__":
    main()
