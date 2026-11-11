#!/usr/bin/env bash
# Full distribution-shift study (uniform-trained VAE).
# Conditions (100 customers unless stated, centre depot, route size 8):
#   ID           uniform customers                       (training distribution)
#   RC           half clustered / half uniform customers (layout shift, moderate)
#   C            clustered customers                     (layout shift, strong)
#   E-depot      uniform customers, depot in the corner  (depot shift)
#   long-routes  uniform customers, ~30 customers/route  (route-length shift; training 4-16)
#   n200         uniform customers, 200 customers       (size shift; training 50-150)
# Runs: PALNS, VAE-PALNS, RACE (online race). The a-priori gate and the combined
# policy are evaluated post hoc from these runs (scripts/analyze_shift.py).
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-4}
NPER=${NPER:-8}
TUNED=$(python3 -c "import json;d=json.load(open('results/tuning/chosen.json'));print(d['t_start_worse'],d['q_max_frac'],d['vae_p'])")
read T_START Q_MAX VAE_P <<< "$TUNED"
DEMANDS=(1-100 1-10 Q SL 5-10 U 50-100 1-100)
declare -A COND=(
  [ID]="n=100,depot=C,customers=R,route_size=8"
  [RC]="n=100,depot=C,customers=RC,route_size=8"
  [C]="n=100,depot=C,customers=C,route_size=8"
  [E-depot]="n=100,depot=E,customers=R,route_size=8"
  [long-routes]="n=100,depot=C,customers=R,route_size=30"
  [n200]="n=200,depot=C,customers=R,route_size=8"
)
ORDER=(ID RC C E-depot long-routes n200)
for c in "${ORDER[@]}"; do
  INST=()
  for ((k=0; k<NPER; k++)); do
    INST+=("gen:${COND[$c]},demand=${DEMANDS[$k]},seed=$((9000+k))")
  done
  python3 scripts/run_benchmarks.py --instances "${INST[@]}" \
    --methods PALNS,VAE-PALNS:uniform,RACE:uniform --vae uniform=models/vae_uniform.npz \
    --seeds "$SEEDS" --epochs 10 --iters-per-epoch 100 --race-epochs 3 \
    --t-start "$T_START" --q-max "$Q_MAX" --vae-p "$VAE_P" --tag "$c" \
    --out results/bench_shift.csv
done
python3 scripts/analyze_shift.py results/bench_shift.csv --gate models/gate_uniform.json \
  --out results/summary_shift
