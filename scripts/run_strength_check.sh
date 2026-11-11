#!/usr/bin/env bash
# Robustness of RQ1 to a stronger base search (ITOR revision):
#  (a) 4x longer budget (16,000 iterations) on the eight X instances of the main benchmark;
#  (b) four larger X instances (303-502 customers) at the standard 4,000 and the longer 16,000 iterations.
# Configuration C_D (literature default), guidance = mixed VAE, same seeds as the main benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/bench_strength.csv
X="data/cvrplib/X-n101-k25.vrp data/cvrplib/X-n110-k13.vrp data/cvrplib/X-n120-k6.vrp data/cvrplib/X-n129-k18.vrp \
   data/cvrplib/X-n148-k46.vrp data/cvrplib/X-n157-k13.vrp data/cvrplib/X-n176-k26.vrp data/cvrplib/X-n200-k36.vrp"
L="data/cvrplib_large/X-n303-k21.vrp data/cvrplib_large/X-n344-k43.vrp data/cvrplib_large/X-n401-k29.vrp data/cvrplib_large/X-n502-k39.vrp"
M="PALNS,VAE-PALNS:mixed"
V="--vae mixed=models/vae_mixed.npz"
python3 scripts/run_benchmarks.py --instances $L --methods $M $V --seeds 5 --epochs 10 --iters-per-epoch 100 --tag large_4k --out $OUT
python3 scripts/run_benchmarks.py --instances $X --methods $M $V --seeds 5 --epochs 10 --iters-per-epoch 400 --tag X_16k --out $OUT
python3 scripts/run_benchmarks.py --instances $L --methods $M $V --seeds 3 --epochs 10 --iters-per-epoch 400 --tag large_16k --out results/bench_strength_large16k.csv
