"""A small Gaussian VAE in pure NumPy (no TensorFlow / PyTorch dependency).

Encoder  x -> [h1 -> h2] -> (mu, logvar);  z = mu + exp(logvar/2) * eps
Decoder  z -> [g1 -> g2] -> x_hat
Loss     0.5*||x - x_hat||^2 + beta * KL(q(z|x) || N(0, I))   (per sample)

Gradients are derived by hand and verified by a finite-difference test
(tests/test_core.py).  Inputs are standardized with a scaler stored inside
the model, so ``recon_error`` / ``neg_elbo`` take raw features.
"""
from __future__ import annotations

import json

import numpy as np

LEAK = 0.01


def _act(x):
    return np.where(x > 0, x, LEAK * x)


def _dact(x):
    return np.where(x > 0, 1.0, LEAK)


class VAE:
    def __init__(self, d_in: int, d_lat: int = 4, hidden=(64, 32), beta: float = 1.0,
                 seed: int = 0):
        self.d_in, self.d_lat, self.hidden, self.beta = d_in, d_lat, tuple(hidden), beta
        rng = np.random.default_rng(seed)
        h1, h2 = self.hidden

        def lin(a, b):
            return rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)

        P = {}
        P["W1"], P["b1"] = lin(d_in, h1)
        P["W2"], P["b2"] = lin(h1, h2)
        P["Wm"], P["bm"] = lin(h2, d_lat)
        P["Wv"], P["bv"] = lin(h2, d_lat)
        P["Wv"] *= 0.1
        P["V1"], P["c1"] = lin(d_lat, h2)
        P["V2"], P["c2"] = lin(h2, h1)
        P["V3"], P["c3"] = lin(h1, d_in)
        self.P = P
        self.mean = np.zeros(d_in)
        self.std = np.ones(d_in)
        self.train_log = []

    # ------------------------------------------------------------------ core
    def _encode(self, x):
        P = self.P
        a1 = x @ P["W1"] + P["b1"]; h1 = _act(a1)
        a2 = h1 @ P["W2"] + P["b2"]; h2 = _act(a2)
        mu = h2 @ P["Wm"] + P["bm"]
        lv = np.clip(h2 @ P["Wv"] + P["bv"], -8.0, 8.0)
        return (a1, h1, a2, h2), mu, lv

    def _decode(self, z):
        P = self.P
        b1 = z @ P["V1"] + P["c1"]; g1 = _act(b1)
        b2 = g1 @ P["V2"] + P["c2"]; g2 = _act(b2)
        xh = g2 @ P["V3"] + P["c3"]
        return (b1, g1, b2, g2), xh

    def loss_and_grads(self, x, eps):
        """Mean loss over the batch and gradients for every parameter."""
        P, B, beta = self.P, x.shape[0], self.beta
        (a1, h1, a2, h2), mu, lv = self._encode(x)
        sd = np.exp(0.5 * lv)
        z = mu + sd * eps
        (b1, g1, b2, g2), xh = self._decode(z)
        rec = 0.5 * ((xh - x) ** 2).sum(1)
        kl = -0.5 * (1 + lv - mu ** 2 - np.exp(lv)).sum(1)
        loss = float((rec + beta * kl).mean())

        G = {}
        dxh = (xh - x) / B
        G["V3"] = g2.T @ dxh; G["c3"] = dxh.sum(0)
        db2 = (dxh @ P["V3"].T) * _dact(b2)
        G["V2"] = g1.T @ db2; G["c2"] = db2.sum(0)
        db1 = (db2 @ P["V2"].T) * _dact(b1)
        G["V1"] = z.T @ db1; G["c1"] = db1.sum(0)
        dz = db1 @ P["V1"].T
        dmu = dz + beta * mu / B
        lv_raw = h2 @ P["Wv"] + P["bv"]
        mask = ((lv_raw > -8.0) & (lv_raw < 8.0)).astype(float)
        dlv = (dz * eps * 0.5 * sd + beta * 0.5 * (np.exp(lv) - 1.0) / B) * mask
        G["Wm"] = h2.T @ dmu; G["bm"] = dmu.sum(0)
        G["Wv"] = h2.T @ dlv; G["bv"] = dlv.sum(0)
        dh2 = dmu @ P["Wm"].T + dlv @ P["Wv"].T
        da2 = dh2 * _dact(a2)
        G["W2"] = h1.T @ da2; G["b2"] = da2.sum(0)
        da1 = (da2 @ P["W2"].T) * _dact(a1)
        G["W1"] = x.T @ da1; G["b1"] = da1.sum(0)
        return loss, G

    # -------------------------------------------------------------- training
    def fit(self, X, epochs: int = 60, batch: int = 256, lr: float = 1e-3,
            kl_warmup: int = 10, val_frac: float = 0.1, seed: int = 0, verbose=False):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, float)
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
        Z = (X - self.mean) / self.std
        idx = rng.permutation(len(Z))
        nv = max(1, int(val_frac * len(Z)))
        Zv, Zt = Z[idx[:nv]], Z[idx[nv:]]
        m = {k: np.zeros_like(v) for k, v in self.P.items()}
        v = {k: np.zeros_like(v) for k, v in self.P.items()}
        b1, b2, t = 0.9, 0.999, 0
        beta_target = self.beta
        for ep in range(epochs):
            self.beta = beta_target * min(1.0, (ep + 1) / max(1, kl_warmup))
            perm = rng.permutation(len(Zt))
            tl = []
            for s in range(0, len(Zt), batch):
                xb = Zt[perm[s:s + batch]]
                eps = rng.standard_normal((len(xb), self.d_lat))
                loss, G = self.loss_and_grads(xb, eps)
                tl.append(loss)
                t += 1
                for k in self.P:
                    g = np.clip(G[k], -5, 5)
                    m[k] = b1 * m[k] + (1 - b1) * g
                    v[k] = b2 * v[k] + (1 - b2) * g * g
                    mh = m[k] / (1 - b1 ** t)
                    vh = v[k] / (1 - b2 ** t)
                    self.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            self.beta = beta_target
            vl, _ = self.loss_and_grads(Zv, rng.standard_normal((len(Zv), self.d_lat)))
            self.train_log.append({"epoch": ep + 1, "train": float(np.mean(tl)), "val": vl})
            if verbose and (ep + 1) % 10 == 0:
                print(f"    VAE epoch {ep + 1}: train {np.mean(tl):.4f} val {vl:.4f}")
        return self

    # ------------------------------------------------------------- inference
    def _std(self, X):
        return (np.asarray(X, float) - self.mean) / self.std

    def encode(self, X):
        _, mu, lv = self._encode(self._std(X))
        return mu, lv

    def recon_error(self, X):
        """Per-sample squared reconstruction error using z = mu."""
        Z = self._std(X)
        _, mu, _ = self._encode(Z)
        _, xh = self._decode(mu)
        return ((xh - Z) ** 2).sum(1)

    def neg_elbo(self, X):
        """Per-sample negative ELBO (decoding the mean; analytic KL)."""
        Z = self._std(X)
        _, mu, lv = self._encode(Z)
        _, xh = self._decode(mu)
        rec = 0.5 * ((xh - Z) ** 2).sum(1)
        kl = -0.5 * (1 + lv - mu ** 2 - np.exp(lv)).sum(1)
        return rec + self.beta * kl

    # ------------------------------------------------------------ save/load
    def save(self, path: str, extra: dict | None = None):
        meta = {"d_in": self.d_in, "d_lat": self.d_lat, "hidden": list(self.hidden),
                "beta": self.beta, "train_log": self.train_log, "extra": extra or {}}
        np.savez(path, mean=self.mean, std=self.std, meta=json.dumps(meta),
                 **{f"P_{k}": v for k, v in self.P.items()})

    @classmethod
    def load(cls, path: str) -> "VAE":
        f = np.load(path, allow_pickle=False)
        meta = json.loads(str(f["meta"]))
        m = cls(meta["d_in"], meta["d_lat"], meta["hidden"], meta["beta"])
        m.mean, m.std = f["mean"], f["std"]
        for k in m.P:
            m.P[k] = f[f"P_{k}"]
        m.train_log = meta.get("train_log", [])
        m.extra = meta.get("extra", {})
        return m
