# vae-palns-shift

**Does learned guidance in adaptive large neighborhood search survive configuration changes and distribution shift?**
Code, data and complete run logs for a controlled study of VAE-guided parallel ALNS (PALNS) on the
capacitated vehicle routing problem (CVRP). The study covers Uchoa X benchmarks, controlled generated
shifts, and real OpenStreetMap road networks of Lima (core and sprawled periphery), Milan, Oslo and Hiroshima.

Pure Python + NumPy, with no deep-learning framework needed: the VAE is implemented in NumPy with gradients
verified by finite differences. Every solution reported is independently re-checked for feasibility.

## Contents

| | |
|---|---|
| `vaepalns/` | the library: instances and loaders, ALNS, synchronized PALNS, operators, VAE, VAE guidance, shift-aware control (conformal and Mondrian gates, online race) |
| `scripts/` | training, tuning, gate calibration, instance building, experiment runners, analysis |
| `tests/` | 13 correctness tests (feasibility of every operator pair, known optimal cost, VAE gradient check, conformal guarantee, explicit road matrices, …) |
| `data/` | benchmark instances, reference costs, OSM road extracts and the 42 road instances (see `data/LICENSES.md`) |
| `models/` | the trained VAEs (`vae_uniform.npz`, `vae_mixed.npz`) and calibrated gates |
| `results/` | raw CSV logs of all 3,088 runs reported in the paper, plus summaries |
| `paper/` | `make_assets.py` regenerates every number, table and figure of the paper from `results/`; `verify_numbers.py` re-computes them independently |

## Install

```bash
python3 -m pip install -r requirements.txt     # numpy, scipy, pandas, matplotlib, tabulate, pytest (+ geopandas, shapely, pyarrow for road instances)
python3 -m pytest -q tests
```

## Reproduce

All experiments use equal budgets of 4,000 destroy–repair iterations (4 workers × 10 epochs × 100), except the stronger-search check, which also uses 16,000 iterations (4 × 10 × 400).
Scripts resume where they stopped. Times are on 2 CPU cores.

| Step | Command | Output |
|---|---|---|
| Train VAEs (~10–15 min each) | `python3 scripts/train_vae.py --layouts R,C,RC --out models/vae_mixed.npz`<br>`python3 scripts/train_vae.py --layouts R --depots C,R --out models/vae_uniform.npz` | `models/` |
| Calibrate gates | `python3 scripts/calibrate_gate.py --out models/gate_uniform.json`<br>`python3 scripts/calibrate_gate.py --mondrian --m 30 --out models/gate_uniform_mondrian.json` | `models/` |
| RQ1 main benchmark (~1.5 h) | `bash scripts/run_main.sh` | `results/bench_main.csv` |
| Small-grid tuning (C_T) | `python3 scripts/tune.py` | `results/tuning/` |
| RQ2 configuration control | `bash scripts/run_tuning_control.sh` | `results/bench_tuning_control.csv` |
| RQ3 shift study, C_D (~2 h) | `bash scripts/run_shift_study_default.sh` | `results/bench_shift_default.csv` |
| RQ3 shift study, C_T (~2 h) | `bash scripts/run_shift_study.sh` | `results/bench_shift.csv` |
| RQ4 road instances | `python3 scripts/make_road_instances.py` (needs the district file, see `data/LICENSES.md`) | `data/roads/instances/` |
| RQ4 road study (~2 h) | `bash scripts/run_road_study.sh` | `results/bench_road.csv` |
| RQ2 approximately-equal-CPU sensitivity (~1 h) | `python3 scripts/run_equal_cpu.py` | `results/bench_equal_cpu.csv` |
| RQ2 same-budget baseline re-run, independent seeds (~1 h) | `python3 scripts/run_equal_cpu.py --rerun --out results/bench_baseline_rerun.csv` | `results/bench_baseline_rerun.csv` |
| RQ2 identical CPU-time limits, 1× and 4× budget (~5 h) | `python3 scripts/run_time_controlled.py` | `results/bench_time_controlled.csv` |
| RQ1 stronger-search check: 16,000 iterations and X instances with 303–502 customers (~3.5 h) | `bash scripts/run_strength_check.sh` | `results/bench_strength.csv`, `results/bench_strength_large16k.csv` |
| Reconstruction error vs improvement | `python3 scripts/recon_vs_gain.py` | `results/recon_vs_gain/` |
| Paper numbers, tables and figures | `cd paper && python3 make_assets.py && python3 verify_numbers.py` | `paper/numbers.tex`, `tables/`, `figures/` |

Single runs:
```bash
python3 scripts/run_benchmarks.py --instances data/cvrplib/X-n101-k25.vrp \
  --methods PALNS,VAE-PALNS:mixed,RACE:mixed --vae mixed=models/vae_mixed.npz --seeds 3 --out results/my_runs.csv
```

## Key results (from the paper)

| Question | Result |
|---|---|
| RQ1: synchronization | Independent multi-start is +0.68% worse than synchronized PALNS at equal iterations (p < 0.001); synchronized PALNS does not significantly outperform sequential ALNS at equal iterations; wall-clock speed-up not evaluated |
| RQ1: guidance on X | −0.15% (not significant) |
| RQ1: stronger base search | Mean PALNS gap to the best-known solution falls from 1.34% (4,000 iterations) to 0.51% (16,000) on X-n101–201, and is 1.22% at 16,000 iterations on X-n303–502. The guidance effect stays within ±0.15 points (−0.15, +0.05, −0.03, +0.03); none significant |
| Unit of analysis | With the instance as the unit (exact sign-flip test over per-instance means, `paper/tables/unit.tex`), 2 of 21 principal comparisons change at the 5% level: the in-distribution benefit under C_D (p = 0.043 over runs → 0.164 over instances) and the benefit on X under C_T (0.018 → 0.062). Synchronization (+0.68%) and the harm of guidance under C_T (+0.25%) stay significant (p = 0.023) |
| RQ2: configuration dependence | At equal iterations on generated instances: −0.23% (C_D) vs +0.25% (C_T); interaction +0.48 points, 95% CI [+0.17, +0.83], sign-flip p = 0.016 (an A/A comparison of the baseline gives a pseudo-interaction of +0.21, p = 0.34). At approximately equal CPU the evidence is insufficient: +0.14 [−0.10, +0.47], p = 0.51. Under identical CPU-time limits (both methods stop at the same per-worker CPU budget, VAE overhead included; guided completes 3.8–8.8% fewer iterations) the interaction is −0.09 [−0.37, +0.21] (p = 0.61) at the base budget and −0.05 [−0.30, +0.19] (p = 0.77) at 4×; guidance effects lie in −0.11% to +0.14% in all eight cells (none significant after Holm). Two independent sets of unguided runs at the same budget differ by −0.17% to +0.17%, the same order as the guidance effects |
| Power | Paired-difference noise across seeds σ_seed ≈ 0.66% dominates instance heterogeneity σ_inst ≈ 0.09%. With 8 instances the MDE (α = 0.05, power 0.8) is ≈ 0.4% with the instance as the unit; detecting 0.2% with 5 seeds needs 21 instances, 0.1% needs 78 (`paper/tables/power.tex`) |
| Literature audit | 20 learned-LNS/ALNS studies coded for evaluation controls (`data/literature_audit.csv`): none varied the base-search configuration or corrected for multiple comparisons; 4 used a statistical test |
| Guidance signal | VAE reconstruction error does not predict the gain of 10-customer removal-and-repair moves (Spearman ρ −0.08 to +0.01); only ~1% of such moves improve late-stage solutions |
| RQ3: distribution shift | No policy difference survives Holm correction. The in-distribution benefit (C_D) vanishes under layout, route-length and size shift and persists under depot shift; the Mondrian gate flags 0% ID, 38% clustered, 62% long-route instances; retrospective per-instance selection between the two methods would gain −0.14% |
| RQ4: real roads | −0.04% (uniform VAE), −0.01% (mixed VAE), not significant, unrelated to sprawl descriptors |

## Method notes

* **CVRP**: unlimited homogeneous fleet, total distance; Solomon/Gehring–Homberger files are used as a CVRP projection (time windows ignored); road instances use undirected shortest-path distances in metres.
* **ALNS** (Ropke & Pisinger 2006): 5 destroy (random, worst, Shaw, string, route) and 4 repair (greedy, greedy+noise, regret-2, regret-3) operators, segment-based adaptive weights, simulated annealing.
* **PALNS**: after each epoch, elite sharing (worst half restart from the global best) and averaging of operator weights; `PALNS-indep` is the unsynchronized control.
* **VAE guidance**: 12 scale-free features of each customer's context in elite solutions; two learned destroy operators (`vae`, `vae_shaw`) compete with the classical ones through the adaptive layer.
* **Shift-aware control**: split-conformal and Mondrian (per demand type) gates on the VAE instance score; online race between guided and unguided worker groups; retrospective per-instance selection benchmark (choosing the better of the two standalone methods per instance).

## Citation

This repository accompanies the manuscript:

> Marino, C. A. *Evaluating learned guidance in adaptive large neighborhood search
under configuration changes and distribution shift: a controlled
study on vehicle routing* Manuscript submitted to the International Transactions in Operational Research.

Machine-readable metadata are in `CITATION.cff`. Road data © OpenStreetMap contributors (ODbL).

Author: Carlos Antonio Marino (Pontificia Universidad Católica del Perú; CENTRUM Católica Graduate Business School), cmarino@pucp.pe, ORCID [0000-0002-6393-6346](https://orcid.org/0000-0002-6393-6346).

## Licence

Code: MIT (`LICENSE`). Data: see `data/LICENSES.md`.
