"""Does VAE reconstruction error predict improvement potential?

For solution snapshots taken at three stages of an unguided ALNS run
(Clarke-Wright start, after 1,000 and after 4,000 iterations; configuration
C_D) we measure, for every customer i:

  recon     VAE reconstruction error of i's context (the score the learned
            operators use)
  detour    d(prev,i) + d(i,next) - d(prev,next) (the score of worst removal)
  gain1     relocation gain: remove i alone, reinsert at its best feasible
            position; cost(S) - cost(S')  (>= 0)
  gain10    neighbourhood gain: remove i and its 9 nearest visited customers,
            repair with greedy and with regret-2 insertion, keep the better;
            cost(S) - cost(S'')  (may be negative)
  improves  gain10 > 0

and report Spearman correlations of recon and detour with the gains, and the
mean gain of the top-10% customers ranked by each score.

    python scripts/recon_vs_gain.py --out results/recon_vs_gain
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_benchmarks import parse_instance  # noqa: E402
from vaepalns import DESTROY_OPS, REPAIR_OPS, clarke_wright, run_alns  # noqa: E402
from vaepalns.alns import ALNSParams  # noqa: E402
from vaepalns.features import customer_features  # noqa: E402
from vaepalns.operators import positions, remove_customers  # noqa: E402
from vaepalns.vae import VAE  # noqa: E402

DEM = ["1-100", "1-10", "Q", "SL", "5-10", "U", "50-100", "1-100"]
SETS = {
    "Generated ID": ([f"gen:n=100,depot=C,customers=R,route_size=8,demand={DEM[k]},seed={9000 + k}"
                      for k in range(8)], "models/vae_uniform.npz"),
    "Uchoa X": ([os.path.join(ROOT, "data", "cvrplib", f + ".vrp") for f in
                 ["X-n101-k25", "X-n110-k13", "X-n120-k6", "X-n129-k18", "X-n148-k46", "X-n157-k13",
                  "X-n176-k26", "X-n200-k36"]], "models/vae_mixed.npz"),
}
STAGES = [0, 1000, 4000]


def measure(args):
    iset, spec, vae_path, seed = args
    inst = parse_instance(spec)
    model = VAE.load(os.path.join(ROOT, vae_path))
    prm = ALNSParams(t_start_worse=0.05, q_max_frac=0.3)
    rng = np.random.default_rng(seed)
    rows = []
    for stage in STAGES:
        s0 = clarke_wright(inst)
        sol = s0 if stage == 0 else run_alns(inst, s0, DESTROY_OPS, REPAIR_OPS, stage,
                                             np.random.default_rng(seed), params=prm).current
        cust, F = customer_features(inst, sol)
        recon = model.recon_error(F)
        pv, c, nx, _ = positions(sol)
        det = dict(zip(c, inst.dist[pv, c] + inst.dist[c, nx] - inst.dist[pv, nx]))
        visited = np.asarray(c)
        for i, e in zip(cust, recon):
            s1 = sol.copy()
            rem = remove_customers(inst, s1, [i])
            REPAIR_OPS["greedy"](inst, s1, rem, rng)
            g1 = sol.cost - s1.cost
            others = visited[visited != i]
            near = others[np.argsort(inst.dist[i, others])[:9]]
            best = np.inf
            for rep in ("greedy", "regret2"):
                s2 = sol.copy()
                rem = remove_customers(inst, s2, [i, *near.tolist()])
                REPAIR_OPS[rep](inst, s2, rem, rng)
                best = min(best, s2.cost)
            rows.append({"set": iset, "instance": inst.name, "seed": seed, "stage": stage, "customer": int(i),
                         "recon": float(e), "detour": float(det[i]), "gain1": float(g1),
                         "gain10": float(sol.cost - best), "rel_gain10": float(100 * (sol.cost - best) / sol.cost)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "recon_vs_gain"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(iset, spec, vp, sd) for iset, (specs, vp) in SETS.items() for spec in specs for sd in range(a.seeds)]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = [r for out in ex.map(measure, jobs) for r in out]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "customers.csv"), index=False)
    summ = []
    for (iset, stage), d in df.groupby(["set", "stage"]):
        per = []
        for _, s in d.groupby(["instance", "seed"]):
            k = max(1, int(round(0.1 * len(s))))
            per.append({
                "rho_recon_g1": stats.spearmanr(s.recon, s.gain1).statistic,
                "rho_detour_g1": stats.spearmanr(s.detour, s.gain1).statistic,
                "rho_recon_g10": stats.spearmanr(s.recon, s.gain10).statistic,
                "rho_detour_g10": stats.spearmanr(s.detour, s.gain10).statistic,
                "top_recon": s.nlargest(k, "recon").rel_gain10.mean(),
                "top_detour": s.nlargest(k, "detour").rel_gain10.mean(),
                "all": s.rel_gain10.mean(),
                "p_improve": (s.gain10 > 1e-9).mean()})
        p = pd.DataFrame(per)
        summ.append({"set": iset, "stage": stage, "snapshots": len(p), **p.mean().to_dict(),
                     "p_rho_recon_g10": stats.wilcoxon(p.rho_recon_g10).pvalue,
                     "p_recon_vs_detour_g10": stats.wilcoxon(p.rho_recon_g10, p.rho_detour_g10).pvalue})
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
