"""Strictly time-controlled comparison for the configuration-controlled design (RQ2).

Both guided and unguided PALNS run under the SAME CPU-time limit; the number of
iterations is whatever each method completes within it. Each of the 4 workers
gets an equal CPU budget per epoch, measured with the worker process's own CPU
clock and including the set-up of the (learned) destroy operators, so VAE
inference counts against the guided method. The annealing schedule follows CPU
time. Runs are executed one at a time, so both methods share the machine in the
same way.

Budget: for every instance and configuration, the CPU time that unguided PALNS
needs for 4,000 iterations on this machine (calibration run with seed 999),
and four times that budget.

    python scripts/run_time_controlled.py --out results/bench_time_controlled.csv
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_benchmarks import parse_instance  # noqa: E402
from run_equal_cpu import ID_SPECS, X_FILES  # noqa: E402
from vaepalns import check_feasible  # noqa: E402
from vaepalns.alns import ALNSParams  # noqa: E402
from vaepalns.guidance import VAEGuidance  # noqa: E402
from vaepalns.palns import PALNSConfig, run_palns  # noqa: E402
from vaepalns.vae import VAE  # noqa: E402

CD = dict(t_start_worse=0.05, q_max_frac=0.3)
CT = dict(t_start_worse=0.01, q_max_frac=0.2)
CELLS = {  # cell: (instances, VAE model, p, configuration, seeds) -- as in the equal-iteration design
    "ID_CD": (ID_SPECS, "vae_uniform", 3.0, CD, 4),
    "ID_CT": (ID_SPECS, "vae_uniform", 6.0, CT, 4),
    "X_CD": (X_FILES, "vae_mixed", 3.0, CD, 5),
    "X_CT": (X_FILES, "vae_mixed", 6.0, CT, 5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench_time_controlled.csv"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--multipliers", default="1,4")
    a = ap.parse_args()
    mults = [float(m) for m in a.multipliers.split(",")]
    done, calib = set(), {}
    if os.path.exists(a.out):
        old = pd.read_csv(a.out)
        done = set(zip(old.cell, old.instance, old.budget_mult, old.method, old.seed))
        calib = {(r.cell, r.instance): r.budget_base for r in old.itertuples()}
    ex = ProcessPoolExecutor(max_workers=a.workers)
    try:
        for mult in mults:
            for cell, (specs, model, p, conf, seeds) in CELLS.items():
                guid = VAEGuidance(VAE.load(os.path.join(ROOT, "models", model + ".npz")), p=p)
                for spec in specs:
                    inst = parse_instance(spec)
                    prm = ALNSParams(**conf)
                    if (cell, inst.name) not in calib:
                        cal = run_palns(inst, PALNSConfig(n_workers=a.workers, epochs=10, iters_per_epoch=100,
                                                          params=prm), seed=999, executor=ex)
                        calib[(cell, inst.name)] = cal["worker_seconds"]
                    base = calib[(cell, inst.name)]
                    for seed in range(seeds):
                        for meth, g in (("PALNS", None), ("VAE-PALNS", guid)):
                            if (cell, inst.name, mult, meth, seed) in done:
                                continue
                            cfg = PALNSConfig(n_workers=a.workers, epochs=10, iters_per_epoch=10 ** 7,
                                              params=prm, cpu_per_worker=mult * base / a.workers)
                            out = run_palns(inst, cfg, seed=seed, guidance=g, executor=ex)
                            ok, _ = check_feasible(inst, out["best"])
                            row = {"cell": cell, "instance": inst.name, "budget_mult": mult,
                                   "budget_base": base, "budget_cpu": mult * base, "method": meth,
                                   "seed": seed, "cost": out["cost"], "cpu_s": out["worker_seconds"],
                                   "wall_s": out["seconds"], "iterations": out["total_iterations"],
                                   "feasible": ok}
                            pd.DataFrame([row]).to_csv(a.out, mode="a", index=False,
                                                       header=not os.path.exists(a.out))
                            print(f"x{mult:g} {cell} {inst.name:28s} {meth:10s} seed {seed} cost {out['cost']:.1f} "
                                  f"cpu {out['worker_seconds']:.1f}/{mult * base:.1f} it {out['total_iterations']} "
                                  f"feasible={ok}", flush=True)
    finally:
        ex.shutdown()


if __name__ == "__main__":
    main()
