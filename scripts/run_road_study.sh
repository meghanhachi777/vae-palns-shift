#!/usr/bin/env bash
# Real road-network study: 7 zones (4 Lima, Milan, Oslo, Hiroshima) x 6 instances x SEEDS.
# Literature-default search settings (t_start 5%, q_max 0.3, p 3); uniform-trained VAE (shift arm), mixed-trained VAE, online race.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-3}
for z in Lima-core Lima-north Lima-south Lima-east Milan Oslo Hiroshima; do
  python3 scripts/run_benchmarks.py --instances "data/roads/instances/OSM-${z}-n100-*.vrp" \
    --methods PALNS,VAE-PALNS:uniform,VAE-PALNS:mixed,RACE:uniform \
    --vae uniform=models/vae_uniform.npz mixed=models/vae_mixed.npz \
    --seeds "$SEEDS" --epochs 10 --iters-per-epoch 100 --race-epochs 3 \
    --t-start 0.05 --q-max 0.3 --vae-p 3 --tag "$z" --out results/bench_road.csv
done
