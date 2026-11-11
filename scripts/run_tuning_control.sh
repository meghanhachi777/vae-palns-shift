#!/usr/bin/env bash
# Tuning-controlled 2x2 design: {untuned, tuned} base search x {unguided, guided}.
#  * ID shift-study instances: tuned arm = results/bench_shift.csv (tag ID); untuned arm run here.
#  * X instances: untuned arm = results/bench_main.csv; tuned arm run here.
set -euo pipefail
cd "$(dirname "$0")/.."
DEM=(1-100 1-10 Q SL 5-10 U 50-100 1-100); INST=()
for k in 0 1 2 3 4 5 6 7; do INST+=("gen:n=100,depot=C,customers=R,route_size=8,demand=${DEM[$k]},seed=$((9000+k))"); done
python3 scripts/run_benchmarks.py --instances "${INST[@]}" --methods PALNS,VAE-PALNS:uniform \
  --vae uniform=models/vae_uniform.npz --seeds 4 --epochs 10 --iters-per-epoch 100 \
  --t-start 0.05 --q-max 0.3 --vae-p 3 --tag ID_untuned --out results/bench_tuning_control.csv
python3 scripts/run_benchmarks.py --instances data/cvrplib/X-n101-k25.vrp data/cvrplib/X-n110-k13.vrp \
  data/cvrplib/X-n120-k6.vrp data/cvrplib/X-n129-k18.vrp data/cvrplib/X-n148-k46.vrp \
  data/cvrplib/X-n157-k13.vrp data/cvrplib/X-n176-k26.vrp data/cvrplib/X-n200-k36.vrp \
  --methods PALNS,VAE-PALNS:mixed --vae mixed=models/vae_mixed.npz --seeds 5 --epochs 10 \
  --iters-per-epoch 100 --t-start 0.01 --q-max 0.2 --vae-p 6 --tag X_tuned \
  --out results/bench_tuning_control.csv
