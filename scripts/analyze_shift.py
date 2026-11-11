"""Analyze the distribution-shift study and evaluate shift-aware policies.

Policies per (instance, seed), built from the PALNS / VAE-PALNS / RACE runs:
  always    VAE-PALNS
  never     PALNS
  prior     conformal gate on the VAE instance score: ID -> VAE-PALNS, OOD -> PALNS
  race      RACE run
  combined  ID -> VAE-PALNS, OOD -> RACE
  oracle    per instance, whichever of PALNS / VAE-PALNS has the lower mean
            cost over seeds (unrealizable reference)

Reported per condition: mean cost change vs PALNS (negative = better),
worst instance (max mean change; robustness), regret vs oracle, gate flag
rate; paired Wilcoxon tests vs PALNS and vs always-guided (Holm-corrected).

    python scripts/analyze_shift.py results/bench_shift.csv --gate models/gate_uniform.json
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re  # noqa: E402

from vaepalns.shift import ConformalGate, MondrianGate  # noqa: E402


def demand_class(name):
    m = re.match(r"^G-n\d+-([CER])(RC|R|C)-(.+)-r\d+-s\d+$", name)
    return m.group(3) if m else None

ORDER = ["ID", "RC", "C", "E-depot", "long-routes", "n200"]
POLICIES = ["always", "prior", "race", "combined", "oracle"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def wilcox(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(a, b).pvalue)


def build(df, gate):
    meth = {"PALNS": "never", "VAE-PALNS:uniform": "always", "RACE:uniform": "race"}
    df = df[df.method.isin(meth)].copy()
    df["pol"] = df.method.map(meth)
    w = df.pivot_table(index=["tag", "instance", "seed"], columns="pol", values="cost").reset_index()
    info = df.groupby("instance").agg(score=("shift_score", "first")).reset_index()
    w = w.merge(info, on="instance")
    if isinstance(gate, MondrianGate):
        w["id_flag"] = [gate.in_distribution(sc, demand_class(i)) for sc, i in zip(w.score, w.instance)]
    else:
        w["id_flag"] = w.score.apply(gate.in_distribution)
    w["prior"] = np.where(w.id_flag, w["always"], w["never"])
    w["combined"] = np.where(w.id_flag, w["always"], w["race"])
    better = (w.groupby("instance")[["never", "always"]].mean()
              .assign(pick=lambda t: np.where(t.always < t.never, "always", "never"))["pick"])
    w["oracle"] = [r[better[r.instance]] for _, r in w.iterrows()]
    race_win = df[df.pol == "race"].set_index(["instance", "seed"]).race_winner
    w["race_winner"] = [race_win.get((r.instance, r.seed), "") for _, r in w.iterrows()]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--gate", default="models/gate_uniform.json")
    ap.add_argument("--mondrian", action="store_true", help="gate file is a MondrianGate")
    ap.add_argument("--out", default="results/summary_shift")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv)
    assert df.feasible.all(), "infeasible solutions present"
    gate = MondrianGate.load(a.gate) if a.mondrian else ConformalGate.load(a.gate)
    w = build(df, gate)
    for p in POLICIES:
        w[f"d_{p}"] = 100 * (w[p] / w["never"] - 1)
    conds = [c for c in ORDER if c in set(w.tag)]

    lines = ["# Distribution-shift study: shift-aware policies", "",
             f"Runs: {len(df)} (all feasible). Instances per condition: "
             f"{w.groupby('tag').instance.nunique().to_dict()}; seeds: {w.seed.nunique()}.",
             (f"Mondrian conformal gate (per demand type), alpha="
              f"{next(iter(gate.gates.values())).alpha}." if a.mondrian else
              f"Conformal gate: alpha={gate.alpha}, threshold={gate.threshold:.3f}."), "",
             "Values = mean % cost change vs PALNS (negative = better). "
             "`worst` = largest per-instance mean change (robustness).", ""]
    rows, prow = [], []
    for c in conds + ["ALL"]:
        d = w if c == "ALL" else w[w.tag == c]
        r = {"condition": c, "flagged_OOD": round(1 - d.id_flag.mean(), 2),
             "race_guided_wins": round((d.race_winner == "guided").mean(), 2)}
        for p in POLICIES:
            r[p] = d[f"d_{p}"].mean()
            r[f"worst_{p}"] = d.groupby("instance")[f"d_{p}"].mean().max()
        rows.append(r)
        ps = {}
        for p in ["always", "prior", "race", "combined"]:
            ps[f"{p}_vs_PALNS"] = wilcox(d[p], d["never"])
        for p in ["prior", "race", "combined"]:
            ps[f"{p}_vs_always"] = wilcox(d[p], d["always"])
        adj = holm(list(ps.values()))
        prow.append({"condition": c, **{k: round(v, 4) for k, v in zip(ps, adj)}})
    tab = pd.DataFrame(rows).set_index("condition")
    main_cols = ["flagged_OOD", "race_guided_wins"] + POLICIES
    lines += ["## Mean cost change vs PALNS (%)", "", tab[main_cols].round(3).to_markdown(), "",
              "## Worst-instance cost change vs PALNS (%)", "",
              tab[[f"worst_{p}" for p in POLICIES]].round(3).to_markdown(), "",
              "## Holm-corrected Wilcoxon p-values (paired over instance x seed)", "",
              pd.DataFrame(prow).set_index("condition").to_markdown(), ""]
    regret = {p: (w[p] / w["oracle"] - 1).mean() * 100 for p in ["never", "always", "prior", "race", "combined"]}
    lines += ["## Mean regret vs per-instance oracle (%)", "",
              pd.Series(regret).round(3).to_frame("regret_%").to_markdown(), ""]
    txt = "\n".join(lines)
    open(os.path.join(a.out, "summary.md"), "w").write(txt)
    w.to_csv(os.path.join(a.out, "policy_table.csv"), index=False)
    print(txt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(conds))
    wd = 0.8 / len(POLICIES)
    for k, p in enumerate(POLICIES):
        m = [w[w.tag == c][f"d_{p}"].mean() for c in conds]
        se = [w[w.tag == c][f"d_{p}"].std() / np.sqrt(len(w[w.tag == c])) for c in conds]
        axes[0].bar(x + (k - 2) * wd, m, wd, yerr=1.96 * np.array(se), capsize=2, label=p)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conds)
    axes[0].set_ylabel("cost change vs PALNS (%)  (<0 better)")
    axes[0].set_title("Policies by shift condition (mean, 95% CI)")
    axes[0].legend(fontsize=8, ncol=3)
    sc = [w[w.tag == c].groupby("instance").score.first().values for c in conds]
    axes[1].boxplot(sc)
    axes[1].set_xticks(range(1, len(conds) + 1))
    axes[1].set_xticklabels(conds)
    if not a.mondrian:
        axes[1].axhline(gate.threshold, color="r", ls="--", label=f"conformal threshold (alpha={gate.alpha})")
    axes[1].set_ylabel("VAE instance shift score")
    axes[1].set_title("Shift score by condition")
    if not a.mondrian:
        axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "shift_policies.png"), dpi=200)


if __name__ == "__main__":
    main()
