"""VAE guidance for ALNS and the data pipeline to train it.

Training:  solve many training instances with (unguided) PALNS, take the
elite solutions, extract per-customer context features, fit the VAE.

Search:    two learned destroy operators join the adaptive operator pool
  * ``vae``       removes customers whose context the VAE reconstructs worst
                  (most atypical for good solutions), randomized by y^p
  * ``vae_shaw``  picks a seed customer by VAE score, then removes customers
                  related to it (distance + demand), i.e. guided Shaw removal
The ALNS adaptive weights decide how often learned operators are used, so
guidance competes with the classical operators instead of replacing them.

Shift detection: ``instance_score`` = mean negative ELBO of the customer
contexts of the Clarke-Wright solution of an instance; compared against the
scores of the training instances it gives an out-of-distribution signal.
"""
from __future__ import annotations

import numpy as np

from .features import customer_features
from .operators import _randomized_pick, remove_customers
from .solution import clarke_wright
from .vae import VAE


class VAEGuidance:
    def __init__(self, model: VAE, p: float = 3.0, ops=("vae", "vae_shaw")):
        self.model, self.p, self.ops = model, p, tuple(ops)

    def scores(self, inst, sol):
        cust, F = customer_features(inst, sol)
        return cust, self.model.recon_error(F)

    def destroy_operators(self, inst):
        model_scores, p = self.scores, self.p

        def vae_removal(inst_, sol, q, rng):
            cust, s = model_scores(inst_, sol)
            order = list(cust[np.argsort(-s)])
            out = [order.pop(_randomized_pick(len(order), p, rng))
                   for _ in range(min(q, len(order)))]
            return remove_customers(inst_, sol, out)

        def vae_shaw_removal(inst_, sol, q, rng):
            cust, s = model_scores(inst_, sol)
            order = cust[np.argsort(-s)]
            seed = int(order[_randomized_pick(len(order), p, rng)])
            remaining = [c for c in cust if c != seed]
            removed = [seed]
            while len(removed) < q and remaining:
                ref = removed[int(rng.integers(len(removed)))]
                ra = np.asarray(remaining)
                rel = (inst_.dist[ref, ra] / inst_.dmax
                       + 0.5 * np.abs(inst_.demand[ref] - inst_.demand[ra]) / inst_.capacity)
                o = np.argsort(rel)
                removed.append(remaining.pop(int(o[_randomized_pick(len(o), 6.0, rng)])))
            return remove_customers(inst_, sol, removed)

        all_ops = {"vae": vae_removal, "vae_shaw": vae_shaw_removal}
        return {k: all_ops[k] for k in self.ops}

    # ---------------------------------------------------------- shift score
    def instance_score(self, inst) -> float:
        sol = clarke_wright(inst)
        _, F = customer_features(inst, sol)
        return float(np.mean(self.model.neg_elbo(F)))

    def ood_percentile(self, inst) -> float:
        """Percentile of this instance's score among training-instance scores
        (stored in model.extra['train_instance_scores'] by the trainer)."""
        ref = np.asarray(getattr(self.model, "extra", {}).get("train_instance_scores", []))
        if ref.size == 0:
            return float("nan")
        return float((ref < self.instance_score(inst)).mean() * 100.0)
