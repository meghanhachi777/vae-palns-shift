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

# instance as the unit of analysis (independent: per-instance means, explicit enumeration of sign patterns)
import itertools


def inst_flip(df, a, b):
    A = df[df.method == a][["instance", "seed", "cost"]]
    B = df[df.method == b][["instance", "seed", "cost"]]
    m = A.merge(B, on=["instance", "seed"])
    m["ch"] = 100.0 * (m.cost_x / m.cost_y - 1.0)
    per = m.groupby("instance").ch.mean().tolist()
    obs = abs(sum(per) / len(per))
    cnt = sum(abs(sum(s * x for s, x in zip(sg, per)) / len(per)) >= obs - 1e-12
              for sg in itertools.product([1, -1], repeat=len(per)))
    return cnt / 2 ** len(per), float(stats.wilcoxon(per).pvalue)


for k, df, a, b in [("unitIndep", X, "PALNS-indep", "PALNS"), ("unitXvae", X, "VAE-PALNS:mixed", "PALNS"),
                    ("unitIDD", tc[tc.tag == "ID_untuned"], "VAE-PALNS:uniform", "PALNS"),
                    ("unitIDT", sT[sT.tag == "ID"], "VAE-PALNS:uniform", "PALNS"),
                    ("unitXT", tc[tc.tag == "X_tuned"], "VAE-PALNS:mixed", "PALNS")]:
    pf, pw = inst_flip(df, a, b)
    check(k + "Pinst", pf, "{:.3f}"); check(k + "Pwil", pw, "{:.3f}")

# power analysis (independent: explicit sums of squares; t quantiles from scipy)
def comps_ss(a_cost, b_cost):
    m = a_cost.merge(b_cost, on=["instance", "seed"])
    m["d"] = 100.0 * (m.cost_x / m.cost_y - 1.0)
    I = m.instance.nunique(); n = len(m); S = n / I
    grand = m.d.mean()
    ssb = sum(len(g) * (g.d.mean() - grand) ** 2 for _, g in m.groupby("instance"))
    ssw = sum(((g.d - g.d.mean()) ** 2).sum() for _, g in m.groupby("instance"))
    msb, msw = ssb / (I - 1), ssw / (n - I)
    return max(0.0, (msb - msw) / S) ** 0.5, msw ** 0.5


def pick(df, meth):
    return df[df.method == meth][["instance", "seed", "cost"]]


gcomps = [comps_ss(pick(tc[tc.tag == "ID_untuned"], "VAE-PALNS:uniform"), pick(tc[tc.tag == "ID_untuned"], "PALNS")),
          comps_ss(pick(sT[sT.tag == "ID"], "VAE-PALNS:uniform"), pick(sT[sT.tag == "ID"], "PALNS")),
          comps_ss(pick(X, "VAE-PALNS:mixed"), pick(X, "PALNS")),
          comps_ss(pick(tc[tc.tag == "X_tuned"], "VAE-PALNS:mixed"), pick(tc[tc.tag == "X_tuned"], "PALNS"))]
si_m = float(np.median([c[0] for c in gcomps])); ss_m = float(np.median([c[1] for c in gcomps]))
check("pwSiMed", si_m, "{:.2f}"); check("pwSsMed", ss_m, "{:.2f}")


def mde_i(si, ss, I, S):
    return (stats.t.ppf(0.975, I - 1) + stats.t.ppf(0.8, I - 1)) * (si ** 2 / I + ss ** 2 / (I * S)) ** 0.5


def need(si, ss, S, target):
    I = 3
    while mde_i(si, ss, I, S) > target:
        I += 1
    return I


check("pwNtwoSfive", need(si_m, ss_m, 5, 0.2), "{:d}")
check("pwNoneSfive", need(si_m, ss_m, 5, 0.1), "{:d}")
check("pwNtwoSone", need(si_m, ss_m, 1, 0.2), "{:d}")
check("pwMdeEightInfMed", mde_i(si_m, ss_m, 8, 10 ** 6), "{:.2f}")

# literature audit counts
lit = pd.read_csv(os.path.join(os.path.dirname(R), "data", "literature_audit.csv"), dtype=str)
check("litN", len(lit), "{:d}")
check("litTest", int((lit.stat_test == "Y").sum()), "{:d}")
check("litRuns", int((lit.repeated_runs != "N").sum()), "{:d}")
check("litMult", int((lit.multiplicity == "Y").sum()), "{:d}")
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
# stronger base search: longer budget and larger instances (independent: merge-based)
if os.path.exists(os.path.join(R, "bench_strength.csv")):
    S = pd.read_csv(os.path.join(R, "bench_strength.csv"))
    if os.path.exists(os.path.join(R, "bench_strength_large16k.csv")):
        S = pd.concat([S, pd.read_csv(os.path.join(R, "bench_strength_large16k.csv"))])
    assert S.feasible.all()
    for tag, key in [("X_16k", "Xlong"), ("large_4k", "Lshort"), ("large_16k", "Llong")]:
        d = S[S.tag == tag]
        if d.empty or f"str{key}D" not in N:
            continue
        v, p, n = rel_change(d, "VAE-PALNS:mixed", "PALNS")
        check(f"str{key}D", v, "{:+.2f}"); check(f"str{key}P", p, "{:.2f}")
        g = d[d.method == "PALNS"]
        bks = {"X-n303-k21": 21736, "X-n344-k43": 42050, "X-n401-k29": 66154, "X-n502-k39": 69226,
               "X-n101-k25": 27591, "X-n110-k13": 14971, "X-n120-k6": 13332, "X-n129-k18": 28940,
               "X-n148-k46": 43448, "X-n157-k13": 16876, "X-n176-k26": 47812, "X-n200-k36": 58578}
        check(f"str{key}gapP", float(np.mean([100 * (c / bks[i] - 1) for i, c in zip(g.instance, g.cost)])), "{:.2f}")

# ---- strictly time-controlled comparison (independent recomputation)
ftc = os.path.join(R, "bench_time_controlled.csv")
if os.path.exists(ftc) and "tcRuns" in N:
    T = pd.read_csv(ftc)
    assert T.feasible.all()
    check("tcRuns", len(T), "{:d}")
    check("tcCpuDev", float((T.cpu_s / T.budget_cpu - 1).abs().max() * 100), "{:.1f}")
    for mult, w in [(1.0, "One"), (4.0, "Four")]:
        effs = {}
        for cell in ["ID_CD", "ID_CT", "X_CD", "X_CT"]:
            d = T[(T.budget_mult == mult) & (T.cell == cell)]
            g = d[d.method == "VAE-PALNS"].set_index(["instance", "seed"]).cost
            u = d[d.method == "PALNS"].set_index(["instance", "seed"]).cost
            e = (g / u.loc[g.index] - 1) * 100
            effs[cell] = e
            check(f"tc{cell.replace('_','')}{w}D", float(e.mean()), "{:+.2f}")
            it = d[d.method == "VAE-PALNS"].iterations.sum() / d[d.method == "PALNS"].iterations.sum() - 1
            check(f"tc{cell.replace('_','')}{w}It", float(it * 100), "{:+.1f}")
        for s_, a_, b_ in [("ID", "ID_CD", "ID_CT"), ("X", "X_CD", "X_CT")]:
            I = (effs[b_] - effs[a_]).groupby(level=0).mean()
            check(f"tc{s_}{w}Int", float(I.mean()), "{:+.2f}")
            n = len(I); vals = I.values; obs = abs(vals.mean()); cnt = 0
            for m in range(2 ** n):
                sg = np.array([1 if (m >> k) & 1 else -1 for k in range(n)])
                cnt += abs((sg * vals).mean()) >= obs - 1e-12
            check(f"tc{s_}{w}IntP", cnt / 2 ** n, "{:.3f}")
for f in ["bench_main.csv", "bench_pilot.csv", "bench_shift.csv", "bench_shift_default.csv",
          "bench_tuning_control.csv", "bench_road.csv", "bench_ablation_p3.csv", "bench_ablation_p1.5.csv"]:
    d = pd.read_csv(os.path.join(R, f))
    assert d.feasible.all(), f
    print(f"OK  all {len(d):4d} runs feasible in {f}")
print("MISMATCHES:", bad)
sys.exit(1 if bad else 0)
