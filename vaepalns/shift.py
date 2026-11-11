"""Shift-aware control of learned guidance.

A-priori gate (split-conformal): the VAE instance score s(I) is computed on a
calibration set of m in-distribution instances that were NOT used to train
the VAE.  The threshold is the ceil((m+1)(1-alpha))-th smallest calibration
score.  For a new instance that is exchangeable with the calibration set,
P(score > threshold) <= alpha, i.e. the false-fallback rate on
in-distribution instances is at most alpha (Vovk et al. 2005; Angelopoulos &
Bates 2023).

Policies (applied per instance):
  always   -> VAE-PALNS
  never    -> PALNS
  prior    -> VAE-PALNS if score <= threshold else PALNS
  race     -> online race (guided vs unguided workers, see palns.run_palns)
  combined -> VAE-PALNS if score <= threshold else race
"""
from __future__ import annotations

import json
import math

import numpy as np


class ConformalGate:
    def __init__(self, threshold: float, alpha: float, cal_scores=None):
        self.threshold, self.alpha = float(threshold), float(alpha)
        self.cal_scores = list(cal_scores or [])

    @classmethod
    def calibrate(cls, scores, alpha: float = 0.1) -> "ConformalGate":
        s = np.sort(np.asarray(scores, float))
        m = len(s)
        k = math.ceil((m + 1) * (1 - alpha))
        thr = np.inf if k > m else s[k - 1]
        return cls(thr, alpha, s.tolist())

    def in_distribution(self, score: float) -> bool:
        return bool(score <= self.threshold)

    def p_value(self, score: float) -> float:
        """Conformal p-value: (1 + #{cal >= score}) / (m + 1)."""
        s = np.asarray(self.cal_scores)
        return float((1 + (s >= score).sum()) / (len(s) + 1))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"threshold": self.threshold, "alpha": self.alpha,
                       "cal_scores": self.cal_scores}, fh, indent=1)

    @classmethod
    def load(cls, path) -> "ConformalGate":
        d = json.load(open(path))
        return cls(d["threshold"], d["alpha"], d["cal_scores"])


class MondrianGate:
    """Class-conditional (Mondrian) conformal gate: one threshold per instance
    class (here: demand type), so that the false-fallback guarantee holds
    within every class and heterogeneous classes do not dominate the
    threshold (Vovk et al. 2005, ch. 4)."""

    def __init__(self, gates: dict):
        self.gates = dict(gates)

    @classmethod
    def calibrate(cls, scores_by_class: dict, alpha: float = 0.1) -> "MondrianGate":
        return cls({k: ConformalGate.calibrate(v, alpha) for k, v in scores_by_class.items()})

    def in_distribution(self, score: float, cls_key) -> bool:
        g = self.gates.get(cls_key)
        return True if g is None else g.in_distribution(score)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({k: {"threshold": g.threshold, "alpha": g.alpha, "cal_scores": g.cal_scores}
                       for k, g in self.gates.items()}, fh, indent=1)

    @classmethod
    def load(cls, path) -> "MondrianGate":
        d = json.load(open(path))
        return cls({k: ConformalGate(v["threshold"], v["alpha"], v["cal_scores"]) for k, v in d.items()})
