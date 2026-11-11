#!/usr/bin/env bash
# Distribution-shift pilot.
# Guidance VAE trained on UNIFORM customer layouts only (models/vae_uniform.npz)
# is tested on uniform (in-distribution), random-clustered and clustered
# (out-of-distribution) instances; the mixed-layout model is a comparison arm.
# Question: does learned guidance stop helping (or hurt) as layouts shift?
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-5}
INST=()
DEMANDS=(1-100 1-10 Q SL)
for layout in R RC C; do
  for k in 0 1 2 3; do
    INST+=("gen:n=100,depot=C,customers=${layout},demand=${DEMANDS[$k]},route_size=8,seed=$((900+k))")
  done
done
python3 scripts/run_benchmarks.py --instances "${INST[@]}" \
  --methods PALNS,VAE-PALNS:uniform,VAE-PALNS:mixed \
  --vae uniform=models/vae_uniform.npz mixed=models/vae_mixed.npz \
  --seeds "$SEEDS" --epochs 10 --iters-per-epoch 100 --tag pilot \
  --out results/bench_pilot.csv
python3 scripts/analyze.py results/bench_pilot.csv --baseline PALNS --group layout \
  --out results/summary_pilot
python3 scripts/plot_shift.py results/bench_pilot.csv --out results/summary_pilot/shift.png
