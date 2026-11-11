"""Independent re-computation of key manuscript numbers from the raw CSVs
(different code path from make_assets.py: merges instead of pivots, explicit
loops). Prints every check and exits non-zero on any mismatch.

    cd paper && python3 verify_numbers.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(HERE), "results")
N = json.load(open(os.path.join(HERE, "numbers.json")))
bad = 0


def rel_change(df, a, b):
    A = df[df.method == a][["instance", "seed", "cost"]]
    B = df[df.method == b][["instance", "seed", "cost"]]
    m = A.merge(B, on=["instance", "seed"], suffixes=("_a", "_b"))
    ch = [100.0 * (ra / rb - 1.0) for ra, rb in zip(m.cost_a, m.cost_b)]
    p = stats.wilcoxon(m.cost_a, m.cost_b).pvalue if (m.cost_a != m.cost_b).any() else 1.0
    return float(np.mean(ch)), float(p), len(m)


def check(key, value, fmt):
    global bad
    got = fmt.format(value)
    ok = got == N[key] or (N[key] == "<0.001" and value < 0.001)
    bad += 0 if ok else 1
    print(f"{'OK ' if ok else 'BAD'} {key:18s} paper={N[key]:>8s} recomputed={got:>8s}")


main = pd.read_csv(os.path.join(R, "bench_main.csv"))
X = main[main.instance.str.startswith("X")]
v, p, n = rel_change(X, "PALNS-indep", "PALNS"); check("XindepDiff", v, "{:+.2f}")
v, p, n = rel_change(X, "VAE-PALNS:mixed", "PALNS"); check("XvaeDiff", v, "{:+.2f}"); check("XvaeP", p, "{:.2f}")
tc = pd.read_csv(os.path.join(R, "bench_tuning_control.csv"))
v, p, n = rel_change(tc[tc.tag == "ID_untuned"], "VAE-PALNS:uniform", "PALNS")
check("IDuntunedEffect", v, "{:+.2f}"); check("IDuntunedP", p, "{:.3f}")
v, p, n = rel_change(tc[tc.tag == "X_tuned"], "VAE-PALNS:mixed", "PALNS")
check("XtunedEffect", v, "{:+.2f}"); check("XtunedP", p, "{:.3f}")
sT = pd.read_csv(os.path.join(R, "bench_shift.csv"))
v, p, n = rel_change(sT[sT.tag == "ID"], "VAE-PALNS:uniform", "PALNS")
check("IDtunedEffect", v, "{:+.2f}"); check("IDtunedP", p, "{:.3f}")
# interaction (independent: merge-based, explicit loop over instances)
def eff_series(df, g):
    A = df[df.method == g][["instance", "seed", "cost"]].merge(
        df[df.method == "PALNS"][["instance", "seed", "cost"]], on=["instance", "seed"], suffixes=("_g", "_u"))
    A["d"] = 100.0 * (A.cost_g / A.cost_u - 1.0)
    return A[["instance", "seed", "d"]]
for k, dD, dT in [("ID", eff_series(tc[tc.tag == "ID_untuned"], "VAE-PALNS:uniform"),
                   eff_series(sT[sT.tag == "ID"], "VAE-PALNS:uniform")),
                  ("X", eff_series(X, "VAE-PALNS:mixed"),
                   eff_series(tc[tc.tag == "X_tuned"], "VAE-PALNS:mixed"))]:
    m = dD.merge(dT, on=["instance", "seed"], suffixes=("_D", "_T"))
    m["i"] = m.d_T - m.d_D
    check(k + "Int", m.i.mean(), "{:+.2f}")
    per = m.groupby("instance").i.mean().values
    import itertools
    obs = abs(per.mean())
    cnt = sum(1 for sg in itertools.product([1, -1], repeat=len(per)) if abs((per * np.array(sg)).mean()) >= obs - 1e-12)
    check(k + "IntPflip", cnt / 2 ** len(per), "{:.3f}")
    check(k + "IntPseed", stats.wilcoxon(m.i).pvalue, "{:.3f}")
# same-budget A/A re-run of the unguided baseline
rer = pd.read_csv(os.path.join(R, "bench_baseline_rerun.csv")); rer["seed"] -= 100
origs = {"ID_CD": tc[(tc.tag == "ID_untuned") & (tc.method == "PALNS")], "ID_CT": sT[(sT.tag == "ID") & (sT.method == "PALNS")],
         "X_CD": X[X.method == "PALNS"], "X_CT": tc[(tc.tag == "X_tuned") & (tc.method == "PALNS")]}
diffs = []
for c, o in origs.items():
    m = rer[rer.cell == c].merge(o[["instance", "seed", "cost"]], on=["instance", "seed"], suffixes=("_r", "_o"))
    diffs.append(float(np.mean(100 * (m.cost_r / m.cost_o - 1))))
check("sameMin", min(diffs), "{:+.2f}"); check("sameMax", max(diffs), "{:+.2f}")
sD = pd.read_csv(os.path.join(R, "bench_shift_default.csv"))
for c, k in [("ID", "ID"), ("RC", "RC"), ("C", "C"), ("E-depot", "Edep"), ("long-routes", "Long"), ("n200", "Big")]:
    v, p, n = rel_change(sD[sD.tag == c], "VAE-PALNS:uniform", "PALNS")
    check("shiftAlways" + k, v, "{:+.2f}")
    v, p, n = rel_change(sD[sD.tag == c], "RACE:uniform", "PALNS")
    check("shiftRace" + k, v, "{:+.2f}")
    rr = sD[(sD.tag == c) & (sD.method == "RACE:uniform")]
    check("raceGuided" + k, 100 * (rr.race_winner == "guided").mean(), "{:.0f}")
# oracle over all shift conditions (C_D)
pal = sD[sD.method == "PALNS"].set_index(["instance", "seed"]).cost
vae = sD[sD.method == "VAE-PALNS:uniform"].set_index(["instance", "seed"]).cost.reindex(pal.index)
orc = []
for inst in pal.index.get_level_values(0).unique():
    a, b = vae.loc[inst], pal.loc[inst]
    pick = a if a.mean() < b.mean() else b
    orc += list(100 * (pick / b - 1))
check("shiftOracle", float(np.mean(orc)), "{:+.2f}")
if os.path.exists(os.path.join(R, "bench_road.csv")):
    road = pd.read_csv(os.path.join(R, "bench_road.csv"))
    assert road.feasible.all()
    for m, k in [("VAE-PALNS:uniform", "roadVAEuni"), ("VAE-PALNS:mixed", "roadVAEmix"), ("RACE:uniform", "roadRace")]:
        v, p, n = rel_change(road, m, "PALNS")
        check(k, v, "{:+.2f}"); check(k + "P", p, "{:.2f}")
    check("roadRuns", len(road), "{:d}")
for f in ["bench_main.csv", "bench_pilot.csv", "bench_shift.csv", "bench_shift_default.csv",
          "bench_tuning_control.csv", "bench_road.csv", "bench_ablation_p3.csv", "bench_ablation_p1.5.csv"]:
    d = pd.read_csv(os.path.join(R, f))
    assert d.feasible.all(), f
    print(f"OK  all {len(d):4d} runs feasible in {f}")
print("MISMATCHES:", bad)
sys.exit(1 if bad else 0)
