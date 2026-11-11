"""Equal-CPU sensitivity check (default) or same-budget baseline re-run (--rerun)
for the configuration-controlled design (RQ2).

--rerun: unguided PALNS with the SAME iteration budget (4,000) but independent
seeds (seed offset 100), to measure pure stochastic run-to-run variability.

At equal iterations the guided method uses more CPU (VAE inference). Here the
unguided PALNS of every instance receives extra iterations, scaled by that
instance's measured CPU ratio guided/unguided in the equal-iteration runs
(same configuration, mean over seeds), so that both methods use about the same
CPU time. Results are compared with the existing guided runs (same seeds).

    python scripts/run_equal_cpu.py --out results/bench_equal_cpu.csv
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_benchmarks import parse_instance  # noqa: E402
from vaepalns import check_feasible  # noqa: E402
from vaepalns.alns import ALNSParams  # noqa: E402
from vaepalns.palns import PALNSConfig, run_palns  # noqa: E402

R = os.path.join(ROOT, "results")
DEM = ["1-100", "1-10", "Q", "SL", "5-10", "U", "50-100", "1-100"]
ID_SPECS = [f"gen:n=100,depot=C,customers=R,route_size=8,demand={DEM[k]},seed={9000 + k}" for k in range(8)]
X_FILES = [os.path.join(ROOT, "data", "cvrplib", f + ".vrp") for f in
           ["X-n101-k25", "X-n110-k13", "X-n120-k6", "X-n129-k18", "X-n148-k46", "X-n157-k13",
            "X-n176-k26", "X-n200-k36"]]


def cells():
    tc = pd.read_csv(os.path.join(R, "bench_tuning_control.csv"))
    sT = pd.read_csv(os.path.join(R, "bench_shift.csv"))
    mn = pd.read_csv(os.path.join(R, "bench_main.csv"))
    return {
        "ID_CD": (tc[tc.tag == "ID_untuned"], "VAE-PALNS:uniform", ID_SPECS, 4, ALNSParams(t_start_worse=0.05, q_max_frac=0.3)),
        "ID_CT": (sT[sT.tag == "ID"], "VAE-PALNS:uniform", ID_SPECS, 4, ALNSParams(t_start_worse=0.01, q_max_frac=0.2)),
        "X_CD": (mn[mn.instance.str.startswith("X")], "VAE-PALNS:mixed", X_FILES, 5, ALNSParams(t_start_worse=0.05, q_max_frac=0.3)),
        "X_CT": (tc[tc.tag == "X_tuned"], "VAE-PALNS:mixed", X_FILES, 5, ALNSParams(t_start_worse=0.01, q_max_frac=0.2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "bench_equal_cpu.csv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rerun", action="store_true", help="same budget, independent seeds")
    ap.add_argument("--seed-offset", type=int, default=100)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out)
        done = set(zip(old.cell, old.instance, old.seed))
    ex = ProcessPoolExecutor(max_workers=a.workers)
    try:
        for cell, (df, g, specs, seeds, prm) in cells().items():
            cpu = df[df.method.isin([g, "PALNS"])].pivot_table(index="instance", columns="method",
                                                               values="cpu_s", aggfunc="mean")
            for spec in specs:
                inst = parse_instance(spec)
                ratio = float(cpu.loc[inst.name, g] / cpu.loc[inst.name, "PALNS"])
                ipe = 100 if a.rerun else int(round(100 * ratio))
                off = a.seed_offset if a.rerun else 0
                for seed in range(off, off + seeds):
                    if (cell, inst.name, seed) in done:
                        continue
                    cfg = PALNSConfig(n_workers=a.workers, epochs=10, iters_per_epoch=ipe, params=prm)
                    out = run_palns(inst, cfg, seed=seed, executor=ex)
                    ok, _ = check_feasible(inst, out["best"])
                    row = {"cell": cell, "instance": inst.name, "seed": seed, "method": "PALNS-rerun" if a.rerun else "PALNS-eqCPU",
                           "cost": out["cost"], "cpu_s": out["worker_seconds"], "iterations": out["total_iterations"],
                           "cpu_ratio_target": ratio, "feasible": ok}
                    pd.DataFrame([row]).to_csv(a.out, mode="a", index=False, header=not os.path.exists(a.out))
                    print(f"{cell} {inst.name:28s} seed {seed} ipe {ipe} cost {out['cost']:.1f} "
                          f"cpu {out['worker_seconds']:.1f} feasible={ok}", flush=True)
    finally:
        ex.shutdown()


if __name__ == "__main__":
    main()
