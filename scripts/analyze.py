"""Summarize benchmark CSVs: tables, significance tests, figures.

    python scripts/analyze.py results/bench_main.csv --baseline PALNS --out results/summary_main
Statistics
  * Friedman test across methods (blocks = instances, values = mean over seeds)
  * paired Wilcoxon signed-rank tests of each method vs the baseline on
    (instance, seed) pairs, Holm-corrected
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--baseline", default="PALNS")
    ap.add_argument("--group", default=None, help="extra grouping column, e.g. layout")
    ap.add_argument("--out", default="results/summary")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(c) for c in a.csv], ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    assert df.feasible.all(), "infeasible solutions present!"
    methods = list(dict.fromkeys(df.method))
    # relative deviation from the best cost found by any method on that instance
    best = df.groupby("instance").cost.transform("min")
    df["dev_best_pct"] = 100 * (df.cost / best - 1)
    metric = "gap_pct" if df.gap_pct.notna().all() else "dev_best_pct"

    lines = [f"# Summary ({', '.join(os.path.basename(c) for c in a.csv)})", "",
             f"Metric: `{metric}` (%; gap to reference = HGS/BKS, or deviation from best found)",
             f"Instances: {df.instance.nunique()}, seeds per instance: {df.seed.nunique()}", ""]

    groups = [("all", df)]
    if a.group:
        groups += [(f"{a.group}={g}", d) for g, d in df.groupby(a.group)]
    for gname, d in groups:
        tab = d.groupby("method").agg(mean_gap=(metric, "mean"), sd_gap=(metric, "std"),
                                      median_gap=(metric, "median"), cpu_s=("cpu_s", "mean"),
                                      wall_s=("wall_s", "mean"), routes=("routes", "mean"),
                                      vae_share=("vae_destroy_share", "mean")).reindex(methods)
        lines += [f"## {gname}", "", tab.round(3).to_markdown(), ""]
        piv = d.groupby(["instance", "method"])[metric].mean().unstack()[methods].dropna()
        if piv.shape[0] >= 3 and piv.shape[1] >= 3:
            fr = stats.friedmanchisquare(*[piv[m] for m in methods])
            lines += [f"Friedman chi2={fr.statistic:.3f}, p={fr.pvalue:.4g} "
                      f"(blocks={piv.shape[0]})", ""]
        if a.baseline in methods:
            rows, ps = [], []
            for m in methods:
                if m == a.baseline:
                    continue
                pr = d.pivot_table(index=["instance", "seed"], columns="method",
                                   values="cost")[[a.baseline, m]].dropna()
                diff = pr[m] - pr[a.baseline]
                if (diff != 0).sum() == 0:
                    p = 1.0
                else:
                    p = stats.wilcoxon(pr[m], pr[a.baseline], zero_method="wilcox").pvalue
                rows.append({"method": m, "vs": a.baseline, "pairs": len(pr),
                             "wins": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
                             "losses": int((diff > 0).sum()),
                             "mean_rel_diff_pct": float(100 * (diff / pr[a.baseline]).mean()),
                             "p_raw": p})
                ps.append(p)
            if rows:
                t = pd.DataFrame(rows)
                t["p_holm"] = holm(t.p_raw)
                lines += ["Paired Wilcoxon vs baseline (negative diff = better than baseline):", "",
                          t.round(4).to_markdown(index=False), ""]

    with open(os.path.join(a.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))

    # figures
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [df[df.method == m][metric].values for m in methods]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_title("Solution quality by method (all runs)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "gap_boxplot.png"), dpi=200)
    plt.close()

    inst0 = df.groupby("instance").n.first().sort_values().index[-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in methods:
        hs = [json.loads(h) for h in df[(df.instance == inst0) & (df.method == m)].history]
        grid = np.linspace(0, max(h[-1][0] for h in hs), 50)
        ys = [np.interp(grid, [p[0] for p in h], [p[1] for p in h]) for h in hs]
        ax.plot(grid, np.mean(ys, 0), label=m)
    ax.set_xlabel("total iterations")
    ax.set_ylabel("best cost (mean over seeds)")
    ax.set_title(f"Convergence on {inst0}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "convergence.png"), dpi=200)
    plt.close()
    df.to_csv(os.path.join(a.out, "all_runs.csv"), index=False)


if __name__ == "__main__":
    main()
