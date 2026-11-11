"""Train the guidance VAE on elite solutions of generated training instances.

Example (mixed layouts):
    python scripts/train_vae.py --layouts R,C,RC --n-instances 60 --out models/vae_mixed.npz
Uniform-only model for the distribution-shift pilot:
    python scripts/train_vae.py --layouts R --n-instances 60 --out models/vae_uniform.npz
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vaepalns import DESTROY_OPS, REPAIR_OPS, clarke_wright, generate_uchoa, run_alns  # noqa: E402
from vaepalns.features import FEATURE_NAMES, customer_features  # noqa: E402
from vaepalns.guidance import VAEGuidance  # noqa: E402
from vaepalns.vae import VAE  # noqa: E402

DEMANDS = ["U", "1-10", "5-10", "1-100", "50-100", "Q", "SL"]


def make_spec(i, layouts, depots, n_lo, n_hi, seed):
    r = np.random.default_rng(seed * 7919 + i)
    return dict(n=int(r.integers(n_lo, n_hi + 1)), depot=str(r.choice(depots)),
                customers=str(r.choice(layouts)), demand=str(r.choice(DEMANDS)),
                route_size=float(r.uniform(4, 16)), seed=int(seed * 100_000 + i))


def solve_and_extract(args):
    spec, iters, keep_every = args
    inst = generate_uchoa(**spec)
    rng = np.random.default_rng(spec["seed"])
    s0 = clarke_wright(inst)
    res = run_alns(inst, s0, DESTROY_OPS, REPAIR_OPS, iters, rng)
    _, F = customer_features(inst, res.best)
    return {"name": inst.name, "n": inst.n, "cw": s0.cost, "best": res.best.cost,
            "F": F, "spec": spec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", default="R,C,RC")
    ap.add_argument("--depots", default="R,C,E")
    ap.add_argument("--n-instances", type=int, default=60)
    ap.add_argument("--n-min", type=int, default=50)
    ap.add_argument("--n-max", type=int, default=150)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--latent", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="models/vae_mixed.npz")
    a = ap.parse_args()

    layouts, depots = a.layouts.split(","), a.depots.split(",")
    specs = [make_spec(i, layouts, depots, a.n_min, a.n_max, a.seed) for i in range(a.n_instances)]
    t = time.time()
    print(f"Solving {len(specs)} training instances (layouts={layouts}) with ALNS x{a.iters} ...")
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        outs = list(ex.map(solve_and_extract, [(s, a.iters, 0) for s in specs]))
    print(f"  done in {time.time() - t:.0f}s; mean improvement over CW "
          f"{np.mean([1 - o['best'] / o['cw'] for o in outs]) * 100:.1f}%")

    X = np.vstack([o["F"] for o in outs])
    print(f"Training VAE on {X.shape[0]} customer contexts x {X.shape[1]} features")
    m = VAE(X.shape[1], d_lat=a.latent, hidden=(64, 32), beta=a.beta, seed=a.seed)
    m.fit(X, epochs=a.epochs, batch=256, lr=1e-3, kl_warmup=10, seed=a.seed, verbose=True)

    # calibration: instance-level shift scores of the training instances
    m.extra = {}
    g = VAEGuidance(m)
    scores = [g.instance_score(generate_uchoa(**o["spec"])) for o in outs]
    extra = {"train_instance_scores": scores, "layouts": layouts, "depots": depots,
             "n_instances": len(outs), "n_range": [a.n_min, a.n_max], "alns_iters": a.iters,
             "feature_names": FEATURE_NAMES, "train_specs": [o["spec"] for o in outs]}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    m.save(a.out, extra=extra)
    with open(a.out.replace(".npz", "_log.json"), "w") as fh:
        json.dump({"train_log": m.train_log, **{k: v for k, v in extra.items()
                                                 if k != "train_specs"}}, fh, indent=1)
    print(f"Saved {a.out}  (final train {m.train_log[-1]['train']:.4f}, "
          f"val {m.train_log[-1]['val']:.4f}); total {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
