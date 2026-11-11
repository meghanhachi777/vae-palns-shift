"""Generate all numbers (numbers.tex), tables (tables/*.tex) and figures
(figures/*.pdf) of the manuscript directly from the result CSVs, so that no
number in the paper is typed by hand.

    cd paper && python3 make_assets.py
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
from vaepalns.shift import ConformalGate, MondrianGate  # noqa: E402

os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
NUM = {}


def num(key, val, fmt="{:.2f}"):
    NUM[key] = fmt.format(val) if not isinstance(val, str) else val
    return NUM[key]


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def signed(x, d=2):
    return f"{x:+.{d}f}".replace("+", "+").replace("-", "$-$") if False else f"{x:+.{d}f}"


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired(df, a, b, keys=("instance", "seed")):
    """Return (mean % change of a vs b, wins, ties, losses, wilcoxon p) on paired runs."""
    pv = df.pivot_table(index=list(keys), columns="method", values="cost")[[a, b]].dropna()
    d = 100 * (pv[a] / pv[b] - 1)
    diff = pv[a] - pv[b]
    p = 1.0 if (diff != 0).sum() == 0 else stats.wilcoxon(pv[a], pv[b]).pvalue
    return d.mean(), int((diff < 0).sum()), int((diff == 0).sum()), int((diff > 0).sum()), float(p), d


def tex_table(df, path, col_fmt=None):
    s = df.to_latex(index=False, escape=False, column_format=col_fmt)
    s = s.replace("\\toprule", "\\toprule").replace("NaN", "--")
    open(path, "w").write(s)


# ------------------------------------------------------------------ RQ1 main
main = pd.read_csv(os.path.join(R, "bench_main.csv"))
assert main.feasible.all()
X = main[main.instance.str.startswith("X")]
order = ["ALNS", "PALNS-indep", "PALNS", "VAE-PALNS:mixed"]
label = {"ALNS": "ALNS (sequential)", "PALNS-indep": "PALNS-indep (no sync.)",
         "PALNS": "PALNS (synchronized)", "VAE-PALNS:mixed": "VAE-PALNS (mixed VAE)"}
rows, ps = [], []
for m in order:
    g = X[X.method == m]
    r = {"Method": label[m], "Mean gap (\\%)": f"{g.gap_pct.mean():.2f}",
         "Median gap (\\%)": f"{g.gap_pct.median():.2f}", "CPU s": f"{g.cpu_s.mean():.1f}"}
    if m != "PALNS":
        mean, w, t, l, p, _ = paired(X, m, "PALNS")
        r.update({"$\\Delta$ (\\%)": f"{mean:+.2f}", "W/T/L": f"{w}/{t}/{l}", "$p$": pfmt(p), "_p": p})
        ps.append(p)
    else:
        r.update({"$\\Delta$ (\\%)": "--", "W/T/L": "--", "$p$": "--", "_p": np.nan})
    rows.append(r)
adj = iter(holm(ps))
for r in rows:
    r["Holm $p$"] = "--" if np.isnan(r["_p"]) else pfmt(next(adj))
tm = pd.DataFrame(rows).drop(columns="_p")
tex_table(tm, os.path.join(HERE, "tables", "main.tex"), "lrrrrrrr")
piv = X.groupby(["instance", "method"]).gap_pct.mean().unstack()[order]
fr = stats.friedmanchisquare(*[piv[m] for m in order])
num("XFriedmanChi", fr.statistic)
num("XFriedmanP", fr.pvalue, "{:.3f}")
m_ind = paired(X, "PALNS-indep", "PALNS")
m_vae = paired(X, "VAE-PALNS:mixed", "PALNS")
hp = holm([paired(X, m, "PALNS")[4] for m in ["ALNS", "PALNS-indep", "VAE-PALNS:mixed"]])
num("XindepDiff", m_ind[0], "{:+.2f}")
num("XindepP", hp[1], "{:.4f}") if hp[1] >= 0.0001 else num("XindepP", "<0.001")
num("XvaeDiff", m_vae[0], "{:+.2f}")
num("XvaeP", m_vae[4], "{:.2f}")
num("XvaeHolm", hp[2], "{:.2f}")
num("XvaeWins", m_vae[1], "{:d}")
num("XvaeLosses", m_vae[3], "{:d}")

# --------------------------------------------------- RQ2 configuration
shift_T = pd.read_csv(os.path.join(R, "bench_shift.csv"))          # small-grid tuned configuration C_T
shift = pd.read_csv(os.path.join(R, "bench_shift_default.csv"))    # literature-default configuration C_D
tc = pd.read_csv(os.path.join(R, "bench_tuning_control.csv"))
assert shift.feasible.all() and shift_T.feasible.all() and tc.feasible.all()
CD, CT = "$C_D$ (default)", "$C_T$ (small-grid tuned)"
cells = {
    ("Generated ID", CD): tc[tc.tag == "ID_untuned"].replace({"method": {"VAE-PALNS:uniform": "G"}}),
    ("Generated ID", CT): shift_T[(shift_T.tag == "ID") & shift_T.method.isin(["PALNS", "VAE-PALNS:uniform"])]
    .replace({"method": {"VAE-PALNS:uniform": "G"}}),
    ("Uchoa X", CD): main[main.instance.str.startswith("X") & main.method.isin(["PALNS", "VAE-PALNS:mixed"])]
    .replace({"method": {"VAE-PALNS:mixed": "G"}}),
    ("Uchoa X", CT): tc[tc.tag == "X_tuned"].replace({"method": {"VAE-PALNS:mixed": "G"}}),
}
rows, eff, devs = [], {}, {}
for iset in ["Generated ID", "Uchoa X"]:
    allc = pd.concat([cells[(iset, c)].assign(conf=c) for c in (CD, CT)])
    best = allc.groupby("instance").cost.transform("min")
    allc["dev"] = 100 * (allc.cost / best - 1)
    for c in (CD, CT):
        d = cells[(iset, c)]
        mean, w_, t_, l_, p, dd = paired(d, "G", "PALNS")
        eff[(iset, c)] = dd
        a_ = allc[allc.conf == c]
        devs[(iset, c, "PALNS")] = a_[a_.method == "PALNS"].dev.mean()
        devs[(iset, c, "G")] = a_[a_.method == "G"].dev.mean()
        rows.append({"Instances": iset, "Configuration": c,
                     "Unguided": f"{devs[(iset, c, 'PALNS')]:.2f}", "Guided": f"{devs[(iset, c, 'G')]:.2f}",
                     "$\\Delta$ (\\%)": f"{mean:+.2f}", "W/T/L": f"{w_}/{t_}/{l_}", "$p$": pfmt(p)})
tex_table(pd.DataFrame(rows), os.path.join(HERE, "tables", "tuning.tex"), "llrrrrr")
for iset, k in [("Generated ID", "ID"), ("Uchoa X", "X")]:
    num(k + "untunedEffect", eff[(iset, CD)].mean(), "{:+.2f}")
    num(k + "tunedEffect", eff[(iset, CT)].mean(), "{:+.2f}")
    num(k + "untunedP", paired(cells[(iset, CD)], "G", "PALNS")[4], "{:.3f}")
    num(k + "tunedP", paired(cells[(iset, CT)], "G", "PALNS")[4], "{:.3f}")
    num(k + "devPD", devs[(iset, CD, "PALNS")])
    num(k + "devPT", devs[(iset, CT, "PALNS")])
    num(k + "devGD", devs[(iset, CD, "G")])
    num(k + "devGT", devs[(iset, CT, "G")])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for k, iset in enumerate(["Generated ID", "Uchoa X"]):
    ms = [eff[(iset, c)].mean() for c in (CD, CT)]
    cis = [1.96 * eff[(iset, c)].std() / np.sqrt(len(eff[(iset, c)])) for c in (CD, CT)]
    ax.errorbar(np.array([0, 1]) + (k - 0.5) * 0.06, ms, yerr=cis, marker="o", capsize=3, label=iset)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["default configuration $C_D$", "small-grid tuned $C_T$"])
ax.set_ylabel("guided vs unguided cost (%)\n(<0: guidance helps)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figures", "tuning_interaction.pdf"))
plt.close()

# ----------------------------------------------- RQ2 interaction analysis
import itertools  # noqa: E402


def interaction(dD, dT, n_boot=10000, seed=0):
    """Configuration x guidance interaction per (instance, seed):
    Delta_int = (guided_T/unguided_T - 1) - (guided_D/unguided_D - 1), in %.
    Primary unit = instance (mean over seeds): exact sign-flip permutation test;
    95% CI by cluster bootstrap over instances. Secondary: Wilcoxon on seed-matched pairs."""
    j = dD.index.intersection(dT.index)
    I = dT[j] - dD[j]
    inst = I.groupby(level=0).mean()
    rng_ = np.random.default_rng(seed)
    groups = [I.xs(i, level=0).values for i in inst.index]
    bs = [np.mean(np.concatenate([groups[g] for g in rng_.integers(0, len(groups), len(groups))]))
          for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    v = inst.values
    obs = abs(v.mean())
    flips = [abs((v * np.array(sg)).mean()) >= obs - 1e-12 for sg in itertools.product([1, -1], repeat=len(v))]
    p_flip = float(np.mean(flips))
    p_seed = float(stats.wilcoxon(I.values).pvalue)
    return {"mean": I.mean(), "lo": lo, "hi": hi, "p_flip": p_flip, "p_seed": p_seed,
            "n_inst": len(v), "pos": int((v > 0).sum()), "pairs": len(I)}


inter = {iset: interaction(eff[(iset, CD)], eff[(iset, CT)]) for iset in ["Generated ID", "Uchoa X"]}
hflip = holm([inter[k]["p_flip"] for k in inter])
rows = []
for (iset, r), ph in zip(inter.items(), hflip):
    rows.append({"Instances": iset, "Instances$\\times$seeds": f"{r['n_inst']}$\\times${r['pairs'] // r['n_inst']}",
                 "$\\Delta_{\\mathrm{int}}$ (\\%)": f"{r['mean']:+.2f}",
                 "95\\% CI": f"[{r['lo']:+.2f}, {r['hi']:+.2f}]",
                 "Inst.\\ $>0$": f"{r['pos']}/{r['n_inst']}",
                 "$p$ sign-flip": f"{r['p_flip']:.3f}", "Holm": f"{ph:.3f}",
                 "$p$ seed-paired": f"{r['p_seed']:.3f}"})
tex_table(pd.DataFrame(rows), os.path.join(HERE, "tables", "interaction.tex"), "lrrrrrrr")
for iset, k in [("Generated ID", "ID"), ("Uchoa X", "X")]:
    r = inter[iset]
    num(k + "Int", r["mean"], "{:+.2f}")
    num(k + "IntLo", r["lo"], "{:+.2f}")
    num(k + "IntHi", r["hi"], "{:+.2f}")
    num(k + "IntPflip", r["p_flip"], "{:.3f}")
    num(k + "IntPseed", r["p_seed"], "{:.3f}")
    num(k + "IntPos", r["pos"], "{:d}")
num("IDIntHolm", hflip[0], "{:.3f}")
num("XIntHolm", hflip[1], "{:.3f}")

# ------------------------------------------- RQ2 equal-CPU sensitivity
eq = pd.read_csv(os.path.join(R, "bench_equal_cpu.csv"))
assert eq.feasible.all()
cellmap = {"ID_CD": ("Generated ID", CD), "ID_CT": ("Generated ID", CT), "X_CD": ("Uchoa X", CD), "X_CT": ("Uchoa X", CT)}
rows, eqeff = [], {}
for cell, (iset, conf) in cellmap.items():
    G = cells[(iset, conf)]
    G = G[G.method == "G"][["instance", "seed", "cost"]]
    E = eq[eq.cell == cell][["instance", "seed", "cost", "cpu_ratio_target"]]
    m = G.merge(E, on=["instance", "seed"], suffixes=("_g", "_e"))
    d = 100 * (m.cost_g / m.cost_e - 1)
    eqeff[(iset, conf)] = pd.Series(d.values, index=pd.MultiIndex.from_frame(m[["instance", "seed"]]))
    p = stats.wilcoxon(m.cost_g, m.cost_e).pvalue if (m.cost_g != m.cost_e).any() else 1.0
    rows.append({"Instances": iset, "Configuration": conf,
                 "Extra iter.\\ (\\%)": f"{100 * (m.cpu_ratio_target.mean() - 1):.0f}",
                 "$\\Delta$ equal iter.\\ (\\%)": f"{eff[(iset, conf)].mean():+.2f}",
                 "$\\Delta$ equal CPU (\\%)": f"{d.mean():+.2f}",
                 "W/T/L": f"{int((d < 0).sum())}/{int((d == 0).sum())}/{int((d > 0).sum())}", "$p$": pfmt(p)})
    key = ("ID" if iset.startswith("Gen") else "X") + ("D" if conf == CD else "T")
    num("eq" + key, d.mean(), "{:+.2f}")
    num("eq" + key + "P", p, "{:.3f}")
    num("eqExtra" + key, 100 * (m.cpu_ratio_target.mean() - 1), "{:.0f}")
tex_table(pd.DataFrame(rows), os.path.join(HERE, "tables", "equal_cpu.tex"), "llrrrrr")
ieq = interaction(eqeff[("Generated ID", CD)], eqeff[("Generated ID", CT)])
num("eqIDInt", ieq["mean"], "{:+.2f}")
num("eqIDIntLo", ieq["lo"], "{:+.2f}")
num("eqIDIntHi", ieq["hi"], "{:+.2f}")
num("eqIDIntPflip", ieq["p_flip"], "{:.3f}")
num("eqExtraMin", min(100 * (eq.groupby("cell").cpu_ratio_target.mean() - 1)), "{:.0f}")
num("eqExtraMax", max(100 * (eq.groupby("cell").cpu_ratio_target.mean() - 1)), "{:.0f}")

# baseline re-run variability: unguided PALNS with ~10% more iterations vs original unguided PALNS
rr_rows = []
for cell, (iset, conf) in cellmap.items():
    O = cells[(iset, conf)]
    O = O[O.method == "PALNS"][["instance", "seed", "cost"]]
    E = eq[eq.cell == cell][["instance", "seed", "cost"]]
    m = O.merge(E, on=["instance", "seed"], suffixes=("_o", "_e"))
    d = 100 * (m.cost_e / m.cost_o - 1)
    rr_rows.append((cell, d.mean(), stats.wilcoxon(m.cost_e, m.cost_o).pvalue,
                    O.groupby("instance").cost.apply(lambda x: 100 * x.std() / x.mean()).mean()))
num("rerunMin", min(r[1] for r in rr_rows), "{:+.2f}")
num("rerunMax", max(r[1] for r in rr_rows), "{:+.2f}")
num("rerunPmin", min(r[2] for r in rr_rows), "{:.3f}")
num("seedSDmin", min(r[3] for r in rr_rows), "{:.2f}")
num("seedSDmax", max(r[3] for r in rr_rows), "{:.2f}")
num("rerunIDT", [r[1] for r in rr_rows if r[0] == "ID_CT"][0], "{:+.2f}")

# same-budget re-run of the unguided baseline with independent seeds (A/A comparison)
rer = pd.read_csv(os.path.join(R, "bench_baseline_rerun.csv"))
assert rer.feasible.all()
rer = rer.assign(seed=rer.seed - 100)
aa, aa_rows = {}, []
for cell, (iset, conf) in cellmap.items():
    O = cells[(iset, conf)]
    O = O[O.method == "PALNS"][["instance", "seed", "cost"]]
    E = rer[rer.cell == cell][["instance", "seed", "cost"]]
    m = O.merge(E, on=["instance", "seed"], suffixes=("_o", "_r"))
    d = 100 * (m.cost_r / m.cost_o - 1)
    aa[(iset, conf)] = pd.Series(d.values, index=pd.MultiIndex.from_frame(m[["instance", "seed"]]))
    p = stats.wilcoxon(m.cost_r, m.cost_o).pvalue if (m.cost_r != m.cost_o).any() else 1.0
    inst_diff = (m.groupby("instance").cost_r.mean() / m.groupby("instance").cost_o.mean() - 1) * 100
    aa_rows.append((cell, d.mean(), p, inst_diff.abs().max()))
num("sameMin", min(r[1] for r in aa_rows), "{:+.2f}")
num("sameMax", max(r[1] for r in aa_rows), "{:+.2f}")
num("samePmin", min(r[2] for r in aa_rows), "{:.3f}")
num("sameInstMax", max(r[3] for r in aa_rows), "{:.2f}")
aai = interaction(aa[("Generated ID", CD)], aa[("Generated ID", CT)])
num("aaInt", aai["mean"], "{:+.2f}")
num("aaIntLo", aai["lo"], "{:+.2f}")
num("aaIntHi", aai["hi"], "{:+.2f}")
num("aaIntP", aai["p_flip"], "{:.3f}")
num("sameRuns", len(rer), "{:d}")

# ----------------------------------- reconstruction error vs improvement
rv = pd.read_csv(os.path.join(R, "recon_vs_gain", "summary.csv"))
rows = []
for _, r in rv.iterrows():
    rows.append({"Set": "Gen.\\ ID" if r["set"].startswith("Gen") else "X", "Iter.": f"{int(r['stage'])}",
                 "$\\rho_1$ rec": f"{r['rho_recon_g1']:+.2f}", "$\\rho_1$ det": f"{r['rho_detour_g1']:+.2f}",
                 "$\\rho_{10}$ rec": f"{r['rho_recon_g10']:+.2f}", "$\\rho_{10}$ det": f"{r['rho_detour_g10']:+.2f}",
                 "Top rec": f"{r['top_recon']:.2f}", "Top det": f"{r['top_detour']:.2f}",
                 "All": f"{r['all']:.2f}", "P(impr.)": f"{r['p_improve']:.2f}"})
tex_table(pd.DataFrame(rows), os.path.join(HERE, "tables", "recon.tex"), "llrrrrrrrr")
cust = pd.read_csv(os.path.join(R, "recon_vs_gain", "customers.csv"))
num("rvSnapshots", cust.groupby(["set", "instance", "seed", "stage"]).ngroups, "{:d}")
num("rvCustomers", len(cust), "{:,}".replace(",", "{,}") if False else "{:d}")
num("rvRhoReconMin", rv.rho_recon_g10.min(), "{:+.2f}")
num("rvRhoReconMax", rv.rho_recon_g10.max(), "{:+.2f}")
num("rvRhoDetourMin", rv.rho_detour_g10.min(), "{:+.2f}")
num("rvRhoDetourMax", rv.rho_detour_g10.max(), "{:+.2f}")
late = rv[rv.stage == rv.stage.max()]
for _, r in late.iterrows():
    k = "ID" if r["set"].startswith("Gen") else "X"
    num("rvLateRecon" + k, r["rho_recon_g10"], "{:+.2f}")
    num("rvLateDetour" + k, r["rho_detour_g10"], "{:+.2f}")
    num("rvLateImprove" + k, 100 * r["p_improve"], "{:.0f}")
    num("rvLatePdiff" + k, r["p_recon_vs_detour_g10"], "{:.3f}")

# --------------------------------------------------------------- RQ3 shift
pooled = ConformalGate.load(os.path.join(ROOT, "models", "gate_uniform.json"))
mond = MondrianGate.load(os.path.join(ROOT, "models", "gate_uniform_mondrian.json"))


def dclass(name):
    m = re.match(r"^G-n\d+-([CER])(RC|R|C)-(.+)-r\d+-s\d+$", name)
    return m.group(3) if m else "1-100"


def policy_table(sh):
    S = sh[sh.method.isin(["PALNS", "VAE-PALNS:uniform", "RACE:uniform"])].copy()
    S["pol"] = S.method.map({"PALNS": "never", "VAE-PALNS:uniform": "always", "RACE:uniform": "race"})
    w = S.pivot_table(index=["tag", "instance", "seed"], columns="pol", values="cost").reset_index()
    sc = S.groupby("instance").shift_score.first()
    w["score"] = w.instance.map(sc)
    w["id_pooled"] = w.score.apply(pooled.in_distribution)
    w["id_mond"] = [mond.in_distribution(s_, dclass(i)) for s_, i in zip(w.score, w.instance)]
    w["prior"] = np.where(w.id_mond, w.always, w.never)
    w["combined"] = np.where(w.id_mond, w.always, w.race)
    better = w.groupby("instance")[["never", "always"]].mean()
    pickd = dict(zip(better.index, np.where(better.always < better.never, "always", "never")))
    w["oracle"] = [r[pickd[r.instance]] for _, r in w.iterrows()]
    for p in ["always", "prior", "race", "combined", "oracle"]:
        w[f"d_{p}"] = 100 * (w[p] / w.never - 1)
    rw = S[S.pol == "race"].set_index(["instance", "seed"]).race_winner
    w["race_guided"] = [rw.get((r.instance, r.seed), "") == "guided" for _, r in w.iterrows()]
    return w


w = policy_table(shift)
wT = policy_table(shift_T)
pols = ["always", "prior", "race", "combined", "oracle"]
conds = ["ID", "RC", "C", "E-depot", "long-routes", "n200"]
rows, minp = [], 1.0
for c in conds + ["All"]:
    d = w if c == "All" else w[w.tag == c]
    dT = wT if c == "All" else wT[wT.tag == c]
    inst = d.groupby("instance")[["id_pooled", "id_mond"]].first()
    r = {"Condition": c, "Flag$_{P}$": f"{1 - inst.id_pooled.mean():.2f}",
         "Flag$_{M}$": f"{1 - inst.id_mond.mean():.2f}", "Race$\\to$G": f"{d.race_guided.mean():.2f}"}
    for p in pols:
        r[p] = f"{d[f'd_{p}'].mean():+.2f}"
    plist = []
    for p in ["always", "prior", "race", "combined"]:
        plist.append(1.0 if ((d[p] - d.never) != 0).sum() == 0 else stats.wilcoxon(d[p], d.never).pvalue)
    for p in ["prior", "race", "combined"]:
        plist.append(1.0 if ((d[p] - d.always) != 0).sum() == 0 else stats.wilcoxon(d[p], d.always).pvalue)
    mp = holm(plist).min()
    minp = min(minp, mp)
    r["$p_{\\mathrm{Holm}}$"] = f"{mp:.2f}"
    r["always ($C_T$)"] = f"{dT.d_always.mean():+.2f}"
    rows.append(r)
tex_table(pd.DataFrame(rows), os.path.join(HERE, "tables", "shift.tex"), "lrrrrrrrrrr")
num("shiftMinP", np.floor(minp * 100) / 100, "{:.2f}")
num("shiftOracle", w.d_oracle.mean(), "{:+.2f}")
num("shiftAlwaysAll", w.d_always.mean(), "{:+.2f}")
num("shiftPriorAll", w.d_prior.mean(), "{:+.2f}")
num("shiftRaceAll", w.d_race.mean(), "{:+.2f}")
num("shiftCombinedAll", w.d_combined.mean(), "{:+.2f}")
for c, k in [("ID", "ID"), ("RC", "RC"), ("C", "C"), ("E-depot", "Edep"), ("long-routes", "Long"), ("n200", "Big")]:
    num("shiftAlways" + k, w[w.tag == c].d_always.mean(), "{:+.2f}")
    num("shiftRace" + k, w[w.tag == c].d_race.mean(), "{:+.2f}")
    num("raceGuided" + k, 100 * w[w.tag == c].race_guided.mean(), "{:.0f}")
    num("shiftAlwaysT" + k, wT[wT.tag == c].d_always.mean(), "{:+.2f}")
num("raceGuidedOther", 100 * w[w.tag != "ID"].race_guided.mean(), "{:.0f}")
idw = w[w.tag == "ID"]
num("shiftIDrawP", stats.wilcoxon(idw.always, idw.never).pvalue, "{:.3f}")
for c, k in [("ID", "flagIDm"), ("C", "flagCm"), ("long-routes", "flagLongm"), ("E-depot", "flagEdepm"),
             ("n200", "flagBigm"), ("RC", "flagRCm")]:
    inst = w[w.tag == c].groupby("instance").id_mond.first()
    num(k, 100 * (1 - inst.mean()), "{:.0f}")
reg = {p: 100 * (w[p] / w.oracle - 1).mean() for p in ["never", "always", "prior", "race", "combined"]}
for p in reg:
    num("regret" + p.capitalize(), reg[p], "{:.2f}")
idp = shift_T[(shift_T.tag == "ID") & (shift_T.method == "PALNS")].set_index(["instance", "seed"]).cost
for f, k in [("bench_ablation_p3.csv", "ablPthree"), ("bench_ablation_p1.5.csv", "ablPonefive")]:
    a_ = pd.read_csv(os.path.join(R, f)).set_index(["instance", "seed"]).cost.reindex(idp.index)
    num(k, (100 * (a_ / idp - 1)).mean(), "{:+.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
x = np.arange(len(conds))
wd = 0.8 / len(pols)
for k, p in enumerate(pols):
    m = [w[w.tag == c][f"d_{p}"].mean() for c in conds]
    se = [w[w.tag == c][f"d_{p}"].std() / np.sqrt((w.tag == c).sum()) for c in conds]
    axes[0].bar(x + (k - 2) * wd, m, wd, yerr=1.96 * np.array(se), capsize=2, label=p)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(conds, fontsize=8)
axes[0].set_ylabel("cost vs unguided PALNS (%)")
axes[0].legend(fontsize=7, ncol=3, frameon=False)
axes[1].boxplot([w[w.tag == c].groupby("instance").score.first().values for c in conds])
axes[1].set_xticks(range(1, len(conds) + 1))
axes[1].set_xticklabels(conds, fontsize=8)
axes[1].axhline(pooled.threshold, color="r", ls="--", lw=1, label="pooled conformal threshold")
axes[1].set_ylabel("VAE instance shift score")
axes[1].legend(fontsize=7, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "figures", "shift_policies.pdf"))
plt.close()

# ---------------------------------------------------------------- RQ4 road
road = pd.read_csv(os.path.join(R, "bench_road.csv"))
assert road.feasible.all()
desc = pd.read_csv(os.path.join(ROOT, "data", "roads", "instances", "descriptors.csv"))
zones = ["Lima-core", "Lima-north", "Lima-south", "Lima-east", "Milan", "Oslo", "Hiroshima"]
meths = {"VAE-PALNS:uniform": "VAE uniform", "VAE-PALNS:mixed": "VAE mixed", "RACE:uniform": "Race"}
rows = []
road_sc = road[road.method == "VAE-PALNS:uniform"].groupby("instance").shift_score.first()
for z in zones + ["All"]:
    d = road if z == "All" else road[road.tag == z]
    dz = desc if z == "All" else desc[desc.zone == z]
    r = {"Zone": z, "Circuity": f"{dz.circuity.mean():.2f}",
         "Junct./km$^2$": f"{dz.junction_density.mean():.0f}",
         "Dead-end": f"{dz.dead_end_share.mean():.3f}", "CV$_{nn}$": f"{dz.cv_nn.mean():.2f}"}
    for m, lab in meths.items():
        mean, wn, t, l, p, _ = paired(d, m, "PALNS")
        r[lab] = f"{mean:+.2f}" + ("$^{*}$" if p < 0.05 else "")
    insts = d.instance.unique()
    r["Flag"] = f"{np.mean([not mond.in_distribution(road_sc[i], '1-100') for i in insts]):.2f}"
    rows.append(r)
tex_table(pd.DataFrame(rows), os.path.join(HERE, "tables", "road.tex"), "lrrrrrrrr")
for m, k in [("VAE-PALNS:uniform", "roadVAEuni"), ("VAE-PALNS:mixed", "roadVAEmix"), ("RACE:uniform", "roadRace")]:
    mean, wn, t, l, p, _ = paired(road, m, "PALNS")
    num(k, mean, "{:+.2f}")
    num(k + "P", p, "{:.2f}")
num("roadInstances", road.instance.nunique(), "{:d}")
num("roadSeeds", road.seed.nunique(), "{:d}")
num("roadRuns", len(road), "{:d}")
num("roadFlag", 100 * np.mean([not mond.in_distribution(road_sc[i], "1-100") for i in road.instance.unique()]), "{:.0f}")
zeff = {z: paired(road[road.tag == z], "VAE-PALNS:uniform", "PALNS")[0] for z in zones}
zmin = min(zeff, key=zeff.get)
num("roadBestZone", zmin)
num("roadBestZoneEff", zeff[zmin], "{:+.2f}")
num("roadBestZoneP", paired(road[road.tag == zmin], "VAE-PALNS:uniform", "PALNS")[4], "{:.2f}")
_, _, _, _, _, dd = paired(road, "VAE-PALNS:uniform", "PALNS")
per_inst = dd.groupby(level=0).mean()
dj = desc.set_index("instance").reindex(per_inst.index)
rho, p = stats.spearmanr(per_inst.values, dj.dead_end_share.values)
num("roadRhoDead", rho, "{:+.2f}")
num("roadRhoDeadP", p, "{:.2f}")
rho, p = stats.spearmanr(per_inst.values, dj.circuity.values)
num("roadRhoCirc", rho, "{:+.2f}")
num("roadRhoCircP", p, "{:.2f}")
# Figure 3: two stacked panels at full text width; zone names on a horizontal axis-free y scale so labels never collide
okabe = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#000000"]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 9})
fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.4), gridspec_kw={"height_ratios": [1.15, 1]})
ax = axes[0]
y = np.arange(len(zones))[::-1]
bh = 0.26
mcol = {"VAE uniform": okabe[0], "VAE mixed": okabe[1], "Race": okabe[2]}
for k, (m, lab) in enumerate(meths.items()):
    vals, cis = [], []
    for z in zones:
        _, _, _, _, _, d_ = paired(road[road.tag == z], m, "PALNS")
        vals.append(d_.mean())
        cis.append(1.96 * d_.std() / np.sqrt(len(d_)))
    ax.barh(y + (1 - k) * bh, vals, bh, xerr=cis, capsize=2, color=mcol[lab], label=lab,
            error_kw={"elinewidth": 0.8, "ecolor": "0.3"})
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(zones)
ax.set_xlabel("cost change vs unguided PALNS (%)  (negative = guidance helps)")
ax.set_title("(a) Effect by zone (mean and 95% CI over instance-seed pairs)", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
ax.grid(axis="x", color="0.9", lw=0.6)
ax.set_axisbelow(True)
ax = axes[1]
dz_ = dj.groupby("zone").dead_end_share.mean().reindex(zones).sort_values()
rng_j = np.random.default_rng(3)
for k, z in enumerate(dz_.index):
    v = per_inst[dj.zone == z].values
    ax.scatter(k + rng_j.uniform(-0.12, 0.12, len(v)), v, s=24, color=okabe[0], alpha=0.85,
               edgecolor="white", linewidth=0.4, label="instance" if k == 0 else None, zorder=3)
    ax.hlines(v.mean(), k - 0.28, k + 0.28, color="k", lw=1.6, label="zone mean" if k == 0 else None, zorder=4)
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(range(len(dz_)))
ax.set_xticklabels([f"{z}\n{v:.3f}" for z, v in dz_.items()])
ax.set_xlabel("zone, ordered by dead-end share of its street network (value below name)")
ax.set_ylabel("guidance effect, uniform VAE (%)")
ax.set_title("(b) Per-instance effect of uniform-VAE guidance by zone", loc="left")
ax.legend(loc="lower left", frameon=False)
ax.grid(axis="y", color="0.9", lw=0.6)
ax.set_axisbelow(True)
plt.tight_layout(h_pad=1.6)
plt.savefig(os.path.join(HERE, "figures", "road.pdf"))
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

# ------------------------------------- strength of the base search (RQ1 robustness)
# X instances at 4,000 (main benchmark) and 16,000 iterations; larger X instances (303-502 customers)
# at 4,000 and 16,000 iterations. PALNS vs VAE-PALNS (mixed VAE), configuration C_D.
pX = X[X.method == "PALNS"].groupby("instance").gap_pct.mean()
num("gapXminP", pX.min()); num("gapXmaxP", pX.max())
strength_parts = []
fS, fL = os.path.join(R, "bench_strength.csv"), os.path.join(R, "bench_strength_large16k.csv")
if os.path.exists(fS):
    S = pd.read_csv(fS)
    if os.path.exists(fL):
        S = pd.concat([S, pd.read_csv(fL)])
    assert S.feasible.all()
    strength_parts = [S]
    cells_strength = [("Xshort", "Uchoa X, 101--201", "4{,}000", X[X.method.isin(["PALNS", "VAE-PALNS:mixed"])]),
             ("Xlong", "Uchoa X, 101--201", "16{,}000", S[S.tag == "X_16k"]),
             ("Lshort", "Uchoa X, 303--502", "4{,}000", S[S.tag == "large_4k"]),
             ("Llong", "Uchoa X, 303--502", "16{,}000", S[S.tag == "large_16k"])]
    rows, ps = [], []
    for key, inst, bud, d in cells_strength:
        if d.empty:
            continue
        gP = d[d.method == "PALNS"].gap_pct.mean()
        gV = d[d.method == "VAE-PALNS:mixed"].gap_pct.mean()
        mean, ws, t, l, p, _ = paired(d, "VAE-PALNS:mixed", "PALNS")
        nI, nS = d.instance.nunique(), d.seed.nunique()
        num(f"str{key}gapP", gP); num(f"str{key}gapV", gV)
        num(f"str{key}D", mean, "{:+.2f}"); num(f"str{key}P", p, "{:.2f}")
        num(f"str{key}W", ws, "{:d}"); num(f"str{key}L", l, "{:d}")
        ps.append(p)
        rows.append({"Instances": inst, "Iter.": bud, "Runs": f"{nI}$\\times${nS}",
                     "PALNS (\\%)": f"{gP:.2f}", "Guided (\\%)": f"{gV:.2f}",
                     "$\\Delta$ (\\%)": f"{mean:+.2f}", "W/T/L": f"{ws}/{t}/{l}", "$p$": f"{p:.3f}"})
    for r, a in zip(rows, holm(ps)):
        r["Holm $p$"] = f"{a:.3f}"
    num("strHolmMin", min(holm(ps)), "{:.2f}")
    num("strXshortHolm", holm(ps)[0], "{:.2f}")
    tex_table(pd.DataFrame(rows), os.path.join(HERE, "tables", "strength.tex"), "llrrrrrrr")
    if "strLlonggapP" in NUM:
        pL = S[(S.tag == "large_16k") & (S.method == "PALNS")].groupby("instance").gap_pct.mean()
        num("gapLminP", pL.min()); num("gapLmaxP", pL.max())
        num("strRuns", len(S), "{:d}")

# ------------------------------------------------------------------ totals

# ------------------------------------------- unit-of-analysis sensitivity
def inst_level(df, a, b, n_mc=200000, seed=1):
    """Compare a vs b with the instance as the unit: mean over seeds of the paired % change per instance;
    exact (n<=20) or Monte Carlo sign-flip test and Wilcoxon signed-rank test over instance means.
    Also returns the run-level (instance-seed) Wilcoxon p used elsewhere in the paper."""
    pv = df.pivot_table(index=["instance", "seed"], columns="method", values="cost")[[a, b]].dropna()
    d = 100 * (pv[a] / pv[b] - 1)
    p_pair = 1.0 if (pv[a] != pv[b]).sum() == 0 else float(stats.wilcoxon(pv[a], pv[b]).pvalue)
    v = d.groupby(level=0).mean().values
    obs = abs(v.mean())
    if len(v) <= 20:
        sg = np.array(list(itertools.product([1, -1], repeat=len(v))))
    else:
        sg = np.random.default_rng(seed).choice([1, -1], size=(n_mc, len(v)))
    p_flip = float(np.mean(np.abs((sg * v).mean(axis=1)) >= obs - 1e-12))
    p_wil = 1.0 if np.all(v == 0) else float(stats.wilcoxon(v).pvalue)
    return {"mean": d.mean(), "n": len(v), "pairs": len(d), "p_pair": p_pair, "p_flip": p_flip, "p_wil": p_wil}


def mk(series_frame):
    return series_frame


unit_rows, unit_changed, unit_p = [], [], {}
XG = X.replace({"method": {"VAE-PALNS:mixed": "G"}})
comps = [("RQ1", "Uchoa X, $C_D$", "ALNS", X, "ALNS", "PALNS"),
         ("RQ1", "Uchoa X, $C_D$", "PALNS-indep", X, "PALNS-indep", "PALNS"),
         ("RQ1", "Uchoa X, $C_D$", "Guided", X, "VAE-PALNS:mixed", "PALNS")]
for (iset, conf), nm in [(("Generated ID", CD), "Generated, $C_D$"), (("Generated ID", CT), "Generated, $C_T$"),
                         (("Uchoa X", CT), "Uchoa X, $C_T$")]:
    comps.append(("RQ2", nm, "Guided", cells[(iset, conf)], "G", "PALNS"))
for cell, (iset, conf) in cellmap.items():
    G_ = cells[(iset, conf)]
    G_ = G_[G_.method == "G"][["instance", "seed", "cost"]].assign(method="G")
    E_ = eq[eq.cell == cell][["instance", "seed", "cost"]].assign(method="E")
    nm = ("Generated" if iset.startswith("Gen") else "Uchoa X") + (", $C_D$" if conf == CD else ", $C_T$")
    comps.append(("RQ2 CPU", nm, "Guided (vs PALNS$^{+}$)", pd.concat([G_, E_]), "G", "E"))
for c in conds[1:]:  # ID under C_D is the same comparison as RQ2 "Generated, C_D"
    d_ = w[w.tag == c].melt(id_vars=["instance", "seed"], value_vars=["always", "never"],
                            var_name="method", value_name="cost")
    comps.append(("RQ3", f"{c}, $C_D$", "Guided", d_, "always", "never"))
for m, lab in [("VAE-PALNS:uniform", "Guided (uniform)"), ("VAE-PALNS:mixed", "Guided (mixed)"),
               ("RACE:uniform", "Race")]:
    comps.append(("RQ4", "Road, $C_D$", lab, road, m, "PALNS"))
for key, inst_, bud, d_ in cells_strength:
    if not d_.empty and key != "Xshort":  # Xshort repeats RQ1 "Guided vs PALNS"
        comps.append(("Strength", f"X {inst_.split(', ')[1]}, {bud}", "Guided", d_, "VAE-PALNS:mixed", "PALNS"))
for rq, setn, comp, df_, a, b in comps:
    r_ = inst_level(df_, a, b)
    unit_p[(rq, setn, comp)] = r_
    sig_pair, sig_inst = r_["p_pair"] < 0.05, r_["p_flip"] < 0.05
    if sig_pair != sig_inst:
        unit_changed.append(f"{rq} {setn} {comp}")
    unit_rows.append({"RQ": rq, "Instances": setn, "Comparison": comp, "Inst.": f"{r_['n']}",
                      "Runs": f"{r_['pairs']}", "$\\Delta$ (\\%)": f"{r_['mean']:+.2f}",
                      "$p_{\\mathrm{runs}}$": pfmt(r_["p_pair"]), "$p_{\\mathrm{flip}}$": pfmt(r_["p_flip"]),
                      "$p_{\\mathrm{W}}$": pfmt(r_["p_wil"])})
tex_table(pd.DataFrame(unit_rows), os.path.join(HERE, "tables", "unit.tex"), "lllrrrrrr")
num("unitComparisons", len(unit_rows), "{:d}")
for k_, key_ in [("unitIndep", ("RQ1", "Uchoa X, $C_D$", "PALNS-indep")),
                 ("unitXvae", ("RQ1", "Uchoa X, $C_D$", "Guided")),
                 ("unitIDD", ("RQ2", "Generated, $C_D$", "Guided")),
                 ("unitIDT", ("RQ2", "Generated, $C_T$", "Guided")),
                 ("unitXT", ("RQ2", "Uchoa X, $C_T$", "Guided"))]:
    num(k_ + "Pruns", unit_p[key_]["p_pair"], "{:.3f}")
    num(k_ + "Pinst", unit_p[key_]["p_flip"], "{:.3f}")
    num(k_ + "Pwil", unit_p[key_]["p_wil"], "{:.3f}")
num("unitChanged", len(unit_changed), "{:d}")
print("UNIT CHANGED:", unit_changed)


# ------------------------------------------- power and minimum detectable effect
# Variance components of the paired percentage difference d_is = 100 (cost_a / cost_b - 1) for instance i and
# seed s, from a balanced one-way random-effects model d_is = mu + a_i + e_is: sigma_inst^2 = Var(a_i), the
# heterogeneity of the effect across instances; sigma_seed^2 = Var(e_is), the run-to-run noise of a paired
# difference. Minimum detectable effect (two-sided alpha = 0.05, power 0.8, t approximation):
#   instance as the unit (generalizes to instances):  MDE = (t_{.975,I-1} + t_{.8,I-1}) sqrt(s_i^2/I + s_s^2/(I S))
#   runs as the unit (fixed instances):               MDE = (t_{.975,IS-1} + t_{.8,IS-1}) sqrt(s_s^2/(I S))
def var_components(d):
    g = d.groupby(level=0)
    S = g.size().median()
    msw = g.var(ddof=1).mean()
    msb = S * g.mean().var(ddof=1)
    return float(np.sqrt(max(0.0, (msb - msw) / S))), float(np.sqrt(msw)), int(g.ngroups), int(S)


def mde(si, ss, I, S, unit="inst", alpha=0.05, power=0.8):
    if unit == "inst":
        se, df_ = np.sqrt(si ** 2 / I + ss ** 2 / (I * S)), I - 1
    else:
        se, df_ = np.sqrt(ss ** 2 / (I * S)), I * S - 1
    return float((stats.t.ppf(1 - alpha / 2, df_) + stats.t.ppf(power, df_)) * se)


def n_needed(si, ss, S, target, unit="inst"):
    for I in range(3, 5001):
        if mde(si, ss, I, S, unit) <= target:
            return I
    return np.nan


pw_rows, comps_pw = [], {}
for cell, (iset, conf) in cellmap.items():
    G_ = cells[(iset, conf)]
    pv = G_.pivot_table(index=["instance", "seed"], columns="method", values="cost")
    comps_pw[f"Guided, {iset.replace('Generated ID', 'generated')}, " + ("$C_D$" if conf == CD else "$C_T$")] = \
        100 * (pv["G"] / pv["PALNS"] - 1)
pv = road.pivot_table(index=["instance", "seed"], columns="method", values="cost")
comps_pw["Guided (uniform), road, $C_D$"] = 100 * (pv["VAE-PALNS:uniform"] / pv["PALNS"] - 1)
for cell, (iset, conf) in cellmap.items():
    comps_pw[f"A/A rerun, {iset.replace('Generated ID', 'generated')}, " + ("$C_D$" if conf == CD else "$C_T$")] = aa[(iset, conf)]
vc = {}
for lab, d in comps_pw.items():
    si, ss, I, S = var_components(d)
    vc[lab] = (si, ss)
    pw_rows.append({"Comparison": lab.replace("Uchoa X", "X"), "$I\\times S$": f"{I}$\\times${S}",
                    "$\\sigma_{\\mathrm{inst}}$": f"{si:.2f}", "$\\sigma_{\\mathrm{seed}}$": f"{ss:.2f}",
                    "MDE inst.": f"{mde(si, ss, I, S):.2f}", "MDE runs": f"{mde(si, ss, I, S, 'runs'):.2f}",
                    "$I$ for 0.2": f"{n_needed(si, ss, 5, 0.2)}", "$I$ for 0.1": f"{n_needed(si, ss, 5, 0.1)}"})
tex_table(pd.DataFrame(pw_rows), os.path.join(HERE, "tables", "power.tex"), "lrrrrrrr")
guided_keys = [k for k in vc if k.startswith("Guided") and "road" not in k]
si_med = float(np.median([vc[k][0] for k in guided_keys]))
ss_med = float(np.median([vc[k][1] for k in guided_keys]))
num("pwSiMed", si_med); num("pwSsMed", ss_med)
num("pwSiMin", min(vc[k][0] for k in guided_keys)); num("pwSiMax", max(vc[k][0] for k in guided_keys))
num("pwSsMin", min(vc[k][1] for k in comps_pw)); num("pwSsMax", max(vc[k][1] for k in comps_pw))
aa_keys = [k for k in vc if k.startswith("A/A")]
num("pwAaSiMax", max(vc[k][0] for k in aa_keys))
mde_inst_paper = [mde(vc[k][0], vc[k][1], 8, 5 if "X" in k else 4) for k in guided_keys]
mde_runs_paper = [mde(vc[k][0], vc[k][1], 8, 5 if "X" in k else 4, "runs") for k in guided_keys]
num("pwMdeInstMin", min(mde_inst_paper)); num("pwMdeInstMax", max(mde_inst_paper))
num("pwMdeRunsMin", min(mde_runs_paper)); num("pwMdeRunsMax", max(mde_runs_paper))
for S_ in (1, 5, 20):
    for t_, lab in ((0.3, "three"), (0.2, "two"), (0.1, "one")):
        num(f"pwN{lab}S{['one', 'five', 'twenty'][[1, 5, 20].index(S_)]}", n_needed(si_med, ss_med, S_, t_), "{:d}")
num("pwMdeEightFiveMed", mde(si_med, ss_med, 8, 5))
num("pwMdeEightInfMed", mde(si_med, ss_med, 8, 10 ** 6))
# with instance heterogeneity, more seeds cannot push the MDE below this floor for a fixed number of instances
num("pwMdeThirtyTwoFiveMed", mde(si_med, ss_med, 32, 5))
num("pwMdeThirtyTwoOneMed", mde(si_med, ss_med, 32, 1))
num("pwMdeSixtyFourFiveMed", mde(si_med, ss_med, 64, 5))
print("POWER:", {k: tuple(round(x, 3) for x in v) for k, v in vc.items()}, si_med, ss_med,
      [n_needed(si_med, ss_med, S_, t_) for S_ in (1, 5, 20) for t_ in (0.3, 0.2, 0.1)])


# ------------------------------------------- literature audit (data/literature_audit.csv)
lit = pd.read_csv(os.path.join(ROOT, "data", "literature_audit.csv"), dtype=str)
sym = {"Y": "$\\bullet$", "P": "$\\circ$", "N": "--"}
eff = {"T": "time", "I": "iter.", "U": "--"}
la_rows = []
for r in lit.itertuples():
    la_rows.append({"Study": f"\\citet{{{r.key}}}", "Problem": r.problem, "Effort": eff[r.effort],
                    "Tuning": sym[r.baseline_tuning], "Config.": sym[r.config_sensitivity],
                    "Runs": r.repeated_runs if r.repeated_runs != "N" else "--",
                    "Test": sym[r.stat_test], "Mult.": sym[r.multiplicity], "OOD": sym[r.ood],
                    "Ablation": sym[r.ablation]})
la_rows.append({"Study": "This study", "Problem": "CVRP", "Effort": "iter., CPU", "Tuning": sym["Y"],
                "Config.": sym["Y"], "Runs": "3--5", "Test": sym["Y"], "Mult.": sym["Y"], "OOD": sym["Y"],
                "Ablation": sym["Y"]})
tex_table(pd.DataFrame(la_rows), os.path.join(HERE, "tables", "audit.tex"), "llcccccccc")
num("litN", len(lit), "{:d}")
num("litTime", (lit.effort == "T").sum(), "{:d}")
num("litIter", (lit.effort == "I").sum(), "{:d}")
num("litUnmatched", (lit.effort == "U").sum(), "{:d}")
num("litTuneY", (lit.baseline_tuning == "Y").sum(), "{:d}")
num("litTuneN", (lit.baseline_tuning == "N").sum(), "{:d}")
num("litConfY", (lit.config_sensitivity == "Y").sum(), "{:d}")
num("litConfP", (lit.config_sensitivity == "P").sum(), "{:d}")
num("litRuns", (lit.repeated_runs != "N").sum(), "{:d}")
num("litTest", (lit.stat_test == "Y").sum(), "{:d}")
num("litMult", (lit.multiplicity == "Y").sum(), "{:d}")
num("litOodY", (lit.ood == "Y").sum(), "{:d}")
num("litOodP", (lit.ood == "P").sum(), "{:d}")
num("litAblY", (lit.ablation == "Y").sum(), "{:d}")
num("litEffortRunsTest", ((lit.effort != "U") & (lit.repeated_runs != "N") & (lit.stat_test == "Y")).sum(), "{:d}")


# ------------------------------------------- RQ2 strictly time-controlled comparison
ftc = os.path.join(R, "bench_time_controlled.csv")
tc_runs = 0
if os.path.exists(ftc):
    tcr = pd.read_csv(ftc)
    assert tcr.feasible.all()
    tcr = tcr[tcr.groupby(["cell", "instance", "budget_mult", "seed"]).method.transform("nunique") == 2]
    tc_runs = len(tcr)
    tc_rows = []
    tc_p = []
    cell_lab = {"ID_CD": ("Generated", CD), "ID_CT": ("Generated", CT), "X_CD": ("Uchoa X", CD), "X_CT": ("Uchoa X", CT)}
    for mult in sorted(tcr.budget_mult.unique()):
        effs = {}
        for cell, (iset, conf) in cell_lab.items():
            d_ = tcr[(tcr.budget_mult == mult) & (tcr.cell == cell)]
            if d_.empty:
                continue
            pv = d_.pivot_table(index=["instance", "seed"], columns="method", values="cost")
            effs[cell] = 100 * (pv["VAE-PALNS"] / pv["PALNS"] - 1)
            it = d_.pivot_table(index=["instance", "seed"], columns="method", values="iterations")
            r_ = inst_level(d_.replace({"method": {"VAE-PALNS": "G"}}), "G", "PALNS")
            k_ = f"tc{cell.replace('_', '')}{'One' if mult == 1 else 'Four'}"
            num(k_ + "D", effs[cell].mean(), "{:+.2f}")
            num(k_ + "P", r_["p_pair"], "{:.3f}")
            tc_p.append(r_["p_pair"])
            num(k_ + "Pinst", r_["p_flip"], "{:.3f}")
            num(k_ + "It", 100 * (it["VAE-PALNS"].mean() / it["PALNS"].mean() - 1), "{:+.1f}")
        for iset, cD, cT in [("Generated", "ID_CD", "ID_CT"), ("Uchoa X", "X_CD", "X_CT")]:
            if cD not in effs or cT not in effs:
                continue
            ri = interaction(effs[cD], effs[cT])
            k_ = f"tc{'ID' if iset == 'Generated' else 'X'}{'One' if mult == 1 else 'Four'}Int"
            num(k_, ri["mean"], "{:+.2f}"); num(k_ + "Lo", ri["lo"], "{:+.2f}"); num(k_ + "Hi", ri["hi"], "{:+.2f}")
            num(k_ + "P", ri["p_flip"], "{:.3f}")
            itD = tcr[(tcr.budget_mult == mult) & (tcr.cell == cD)]
            tc_rows.append({"Budget": f"{mult:g}$\\times$", "Instances": iset,
                            "Iter.\\ guided (\\%)": "",
                            "$\\Delta$ $C_D$ (\\%)": f"{effs[cD].mean():+.2f}",
                            "$\\Delta$ $C_T$ (\\%)": f"{effs[cT].mean():+.2f}",
                            "$\\Delta_{\\mathrm{int}}$ (\\%)": f"{ri['mean']:+.2f}",
                            "95\\% CI": f"[{ri['lo']:+.2f}, {ri['hi']:+.2f}]",
                            "$p$ sign-flip": f"{ri['p_flip']:.3f}"})
            # iteration share of the guided method relative to unguided PALNS under the same CPU limit
            itc = []
            for c in (cD, cT):
                d_ = tcr[(tcr.budget_mult == mult) & (tcr.cell == c)]
                itc.append(d_[d_.method == "VAE-PALNS"].iterations.mean() / d_[d_.method == "PALNS"].iterations.mean() - 1)
            tc_rows[-1]["Iter.\\ guided (\\%)"] = f"{100 * itc[0]:+.1f} / {100 * itc[1]:+.1f}"
    tex_table(pd.DataFrame(tc_rows), os.path.join(HERE, "tables", "timecontrol.tex"), "llrrrrrr")
    num("tcRuns", tc_runs, "{:d}")
    num("tcHolmMin", min(holm(tc_p)), "{:.2f}")
    tc_eff = [100 * (g.set_index(["instance", "seed"]).query("method == 'VAE-PALNS'").cost
                     / g.set_index(["instance", "seed"]).query("method == 'PALNS'").cost - 1).mean()
              for _, g in tcr.groupby(["cell", "budget_mult"])]
    num("tcDMin", min(tc_eff), "{:+.2f}"); num("tcDMax", max(tc_eff), "{:+.2f}")
    tc_it = [100 * (1 - g[g.method == "VAE-PALNS"].iterations.mean() / g[g.method == "PALNS"].iterations.mean())
             for _, g in tcr.groupby(["cell", "budget_mult"])]
    num("tcFewerMin", min(tc_it), "{:.1f}"); num("tcFewerMax", max(tc_it), "{:.1f}")   # guided effect, 8 cells (4 cells x 2 budgets)
    num("tcCpuOne", tcr[tcr.budget_mult == 1].budget_cpu.mean(), "{:.1f}")
    num("tcItMin", 100 * min(
        (g[g.method == "VAE-PALNS"].iterations.mean() / g[g.method == "PALNS"].iterations.mean() - 1)
        for _, g in tcr.groupby(["cell", "budget_mult"])), "{:+.1f}")
    num("tcItMax", 100 * max(
        (g[g.method == "VAE-PALNS"].iterations.mean() / g[g.method == "PALNS"].iterations.mean() - 1)
        for _, g in tcr.groupby(["cell", "budget_mult"])), "{:+.1f}")
    num("tcCpuDev", 100 * (tcr.cpu_s / tcr.budget_cpu - 1).abs().max(), "{:.1f}")

tot = tc_runs + len(main) + len(shift_T) + sum(len(s) for s in strength_parts) \
    + len(shift[shift.tag != "ID"]) + len(shift[(shift.tag == "ID") & (shift.method == "RACE:uniform")]) \
    + len(tc) + len(road) + len(eq) + len(rer) \
    + len(pd.read_csv(os.path.join(R, "bench_ablation_p3.csv"))) \
    + len(pd.read_csv(os.path.join(R, "bench_ablation_p1.5.csv")))
num("totalRuns", f"{tot:,}".replace(",", "{,}"))
with open(os.path.join(HERE, "numbers.tex"), "w") as fh:
    for k, v in sorted(NUM.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(NUM, open(os.path.join(HERE, "numbers.json"), "w"), indent=1)
print(json.dumps(NUM, indent=1))
