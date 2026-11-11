#!/usr/bin/env bash
# Main benchmark: ALNS vs PALNS-indep vs PALNS vs VAE-PALNS (equal total iterations = 4000).
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-5}
python3 scripts/run_benchmarks.py --instances \
  data/user_files/25-cust.txt data/user_files/C201.txt data/user_files/R201.txt \
  data/user_files/RC101.txt data/user_files/C1_6_1.TXT data/user_files/100-cust.txt \
  data/cvrplib/X-n101-k25.vrp data/cvrplib/X-n110-k13.vrp data/cvrplib/X-n120-k6.vrp \
  data/cvrplib/X-n129-k18.vrp data/cvrplib/X-n148-k46.vrp data/cvrplib/X-n157-k13.vrp \
  data/cvrplib/X-n176-k26.vrp data/cvrplib/X-n200-k36.vrp \
  --methods ALNS,PALNS-indep,PALNS,VAE-PALNS:mixed --vae mixed=models/vae_mixed.npz \
  --seeds "$SEEDS" --epochs 10 --iters-per-epoch 100 --tag main --out results/bench_main.csv
python3 scripts/analyze.py results/bench_main.csv --baseline PALNS --out results/summary_main
