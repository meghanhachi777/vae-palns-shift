"""Pilot figure: gain of VAE guidance over PALNS per layout, and the VAE shift score.

    python scripts/plot_shift.py results/bench_pilot.csv --out results/summary_pilot/shift.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--baseline", default="PALNS")
    ap.add_argument("--out", default="results/summary_pilot/shift.png")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    order = ["R", "RC", "C"]
    piv = d.pivot_table(index=["instance", "seed", "layout"], columns="method", values="cost").reset_index()
    guided = [m for m in d.method.unique() if m != a.baseline]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.8 / len(guided)
    for k, m in enumerate(guided):
        rel = 100 * (piv[m] / piv[a.baseline] - 1)
        g = rel.groupby(piv.layout)
        mean = g.mean().reindex(order)
        se = (g.std() / np.sqrt(g.count())).reindex(order)
        x = np.arange(len(order)) + (k - (len(guided) - 1) / 2) * w
        axes[0].bar(x, mean, w, yerr=1.96 * se, capsize=3, label=m)
        ood = d[d.method == m].groupby("layout").ood_pct.mean().reindex(order)
        axes[1].plot(order, ood, marker="o", label=m)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(["uniform (R)", "mixed (RC)", "clustered (C)"])
    axes[0].set_ylabel(f"cost vs {a.baseline} (%)  (<0 = guidance helps)")
    axes[0].set_title("Effect of VAE guidance by customer layout (95% CI)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("instance shift score (percentile of training)")
    axes[1].set_title("VAE out-of-distribution score")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
