"""Calibrate the conformal shift gate on held-out in-distribution instances.

The calibration instances are drawn from the VAE's training distribution
(stored in the model file) with seeds disjoint from training and testing.

    python scripts/calibrate_gate.py --vae models/vae_uniform.npz --alpha 0.1 \
        --out models/gate_uniform.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vaepalns import generate_uchoa  # noqa: E402
from vaepalns.guidance import VAEGuidance  # noqa: E402
from vaepalns.shift import ConformalGate, MondrianGate  # noqa: E402
from vaepalns.vae import VAE  # noqa: E402

DEMANDS = ["U", "1-10", "5-10", "1-100", "50-100", "Q", "SL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="models/vae_uniform.npz")
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="models/gate_uniform.json")
    ap.add_argument("--mondrian", action="store_true",
                    help="one threshold per demand type (m instances per type)")
    a = ap.parse_args()
    model = VAE.load(a.vae)
    ex = model.extra
    layouts, depots = ex["layouts"], ex["depots"]
    lo, hi = ex["n_range"]
    rng = np.random.default_rng(a.seed)
    g = VAEGuidance(model)
    if a.mondrian:
        by = {}
        for dm in DEMANDS:
            by[dm] = []
            for i in range(a.m):
                spec = dict(n=int(rng.integers(lo, hi + 1)), depot=str(rng.choice(depots)),
                            customers=str(rng.choice(layouts)), demand=dm,
                            route_size=float(rng.uniform(4, 16)),
                            seed=8_000_000 + a.seed * 10_000 + DEMANDS.index(dm) * 1000 + i)
                by[dm].append(g.instance_score(generate_uchoa(**spec)))
        gate = MondrianGate.calibrate(by, a.alpha)
        gate.save(a.out)
        for dm, gg in gate.gates.items():
            print(f"  {dm:7s} threshold {gg.threshold:.3f}")
        return
    scores = []
    for i in range(a.m):
        spec = dict(n=int(rng.integers(lo, hi + 1)), depot=str(rng.choice(depots)),
                    customers=str(rng.choice(layouts)), demand=str(rng.choice(DEMANDS)),
                    route_size=float(rng.uniform(4, 16)), seed=7_000_000 + a.seed * 1000 + i)
        scores.append(g.instance_score(generate_uchoa(**spec)))
    gate = ConformalGate.calibrate(scores, a.alpha)
    gate.save(a.out)
    print(f"calibrated on {a.m} ID instances: threshold={gate.threshold:.3f} "
          f"(alpha={a.alpha}); score range {min(scores):.2f}-{max(scores):.2f}")


if __name__ == "__main__":
    main()
