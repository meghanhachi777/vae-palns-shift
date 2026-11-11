"""Correctness tests. Run:  python -m pytest -q tests   (or python tests/test_core.py)"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vaepalns import (DESTROY_OPS, REPAIR_OPS, check_feasible, clarke_wright,  # noqa: E402
                      generate_uchoa, load_solomon, load_vrplib, run_alns)
from vaepalns.features import customer_features  # noqa: E402
from vaepalns.guidance import VAEGuidance  # noqa: E402
from vaepalns.palns import PALNSConfig, run_palns  # noqa: E402
from vaepalns.solution import Solution  # noqa: E402
from vaepalns.vae import VAE  # noqa: E402

DATA = os.path.join(ROOT, "data")


def test_loaders():
    x = load_vrplib(os.path.join(DATA, "cvrplib", "X-n101-k25.vrp"))
    assert x.n == 100 and x.capacity == 206 and x.round_dist
    assert x.demand[0] == 0 and x.demand.sum() > 0
    s = load_solomon(os.path.join(DATA, "user_files", "25-cust.txt"))
    assert s.n == 25 and s.capacity == 200
    h = load_solomon(os.path.join(DATA, "user_files", "data00.txt"))   # no header
    assert h.n == 100 and h.capacity == 200
    r = load_solomon(os.path.join(DATA, "user_files", "R201.txt"))
    assert r.capacity == 1000
    g = load_solomon(os.path.join(DATA, "user_files", "C1_6_1.TXT"))
    assert g.n == 60


def test_known_solution_cost():
    """Cost of the optimal X-n101-k25 routes (from HGS) must equal BKS 27591."""
    inst = load_vrplib(os.path.join(DATA, "cvrplib", "X-n101-k25.vrp"))
    sol_file = os.path.join(DATA, "cvrplib", "X-n101-k25.sol")
    if not os.path.exists(sol_file):
        return
    routes = []
    for ln in open(sol_file):
        if ln.startswith("Route"):
            routes.append([int(t) for t in ln.split(":")[1].split()])
    sol = Solution(inst, routes)
    assert check_feasible(inst, sol)[0]
    assert abs(sol.cost - 27591) < 1e-6


def test_clarke_wright_feasible():
    for f in ["X-n101-k25.vrp", "X-n148-k46.vrp"]:
        inst = load_vrplib(os.path.join(DATA, "cvrplib", f))
        sol = clarke_wright(inst)
        ok, msg = check_feasible(inst, sol)
        assert ok, msg


def test_every_operator_pair_keeps_feasibility():
    inst = generate_uchoa(60, "R", "RC", "1-100", route_size=6, seed=3)
    rng = np.random.default_rng(0)
    base = clarke_wright(inst)
    for dn, dop in DESTROY_OPS.items():
        for rn, rop in REPAIR_OPS.items():
            for _ in range(5):
                s = base.copy()
                q = int(rng.integers(2, 20))
                removed = dop(inst, s, q, rng)
                assert len(removed) >= 1
                rop(inst, s, removed, rng)
                ok, msg = check_feasible(inst, s)
                assert ok, f"{dn}+{rn}: {msg}"


def test_alns_improves_and_is_feasible():
    inst = load_vrplib(os.path.join(DATA, "cvrplib", "X-n101-k25.vrp"))
    s0 = clarke_wright(inst)
    res = run_alns(inst, s0, DESTROY_OPS, REPAIR_OPS, 300, np.random.default_rng(1))
    assert res.best.cost <= s0.cost
    assert check_feasible(inst, res.best)[0]
    assert check_feasible(inst, res.current)[0]


def test_vae_gradients_finite_difference():
    rng = np.random.default_rng(0)
    m = VAE(5, d_lat=3, hidden=(7, 6), beta=0.7, seed=1)
    x = rng.standard_normal((4, 5))
    eps = rng.standard_normal((4, 3))
    _, G = m.loss_and_grads(x, eps)
    h = 1e-6
    for k, P in m.P.items():
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + h; lp, _ = m.loss_and_grads(x, eps)
            P[idx] = old - h; lm, _ = m.loss_and_grads(x, eps)
            P[idx] = old
            num = (lp - lm) / (2 * h)
            assert abs(num - G[k][idx]) < 1e-5 * max(1.0, abs(num)), (k, idx, num, G[k][idx])


def test_vae_learns_and_ranks_outliers():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 2)) @ np.array([[1.0, 0.9], [0.0, 0.3]])
    X = np.column_stack([X, X[:, 0] * 0.5])
    m = VAE(3, d_lat=2, hidden=(16, 8), beta=0.5).fit(X, epochs=40, batch=128)
    assert m.train_log[-1]["train"] < m.train_log[0]["train"]
    inlier = m.recon_error(X[:50]).mean()
    outlier = m.recon_error(np.array([[3.0, -3.0, 4.0]]))[0]
    assert outlier > inlier


def test_features_and_guided_ops():
    inst = generate_uchoa(50, "C", "R", "5-10", route_size=8, seed=1)
    sol = clarke_wright(inst)
    cust, F = customer_features(inst, sol)
    assert F.shape == (inst.n, 12) and np.isfinite(F).all()
    assert sorted(cust.tolist()) == list(range(1, inst.n + 1))
    m = VAE(12, d_lat=4).fit(F, epochs=3, batch=32)
    g = VAEGuidance(m)
    rng = np.random.default_rng(2)
    for name, op in g.destroy_operators(inst).items():
        s = sol.copy()
        rem = op(inst, s, 10, rng)
        assert len(rem) == 10
        REPAIR_OPS["regret2"](inst, s, rem, rng)
        assert check_feasible(inst, s)[0], name
    assert np.isfinite(g.instance_score(inst))


def test_palns_sync_serial():
    inst = generate_uchoa(40, "R", "R", "1-10", route_size=6, seed=5)
    cfg = PALNSConfig(n_workers=3, epochs=3, iters_per_epoch=40, use_processes=False)
    out = run_palns(inst, cfg, seed=1)
    assert check_feasible(inst, out["best"])[0]
    assert out["cost"] <= clarke_wright(inst).cost


def test_generator_variants():
    for dep in "CER":
        for cu in ("R", "C", "RC"):
            for dm in ("U", "1-10", "50-100", "Q", "SL"):
                inst = generate_uchoa(30, dep, cu, dm, seed=7)
                assert inst.n == 30 and (inst.demand[1:] <= inst.capacity).all()


def test_race_and_union_weights():
    inst = generate_uchoa(40, "R", "R", "1-10", route_size=6, seed=11)
    sol = clarke_wright(inst)
    _, F = customer_features(inst, sol)
    g = VAEGuidance(VAE(12, d_lat=3).fit(np.vstack([F] * 5), epochs=3, batch=32))
    cfg = PALNSConfig(n_workers=4, epochs=4, iters_per_epoch=30, use_processes=False)
    out = run_palns(inst, cfg, seed=2, guidance=g, race_epochs=2)
    assert out["race"]["race_winner"] in ("guided", "unguided")
    assert check_feasible(inst, out["best"])[0]


def test_conformal_gate():
    from vaepalns.shift import ConformalGate
    rng = np.random.default_rng(0)
    cal = rng.normal(size=200)
    gate = ConformalGate.calibrate(cal, alpha=0.1)
    test = rng.normal(size=5000)
    false_flag = np.mean([not gate.in_distribution(x) for x in test])
    assert false_flag <= 0.14
    assert not gate.in_distribution(10.0) and gate.p_value(10.0) < 0.01


def test_explicit_matrix_instance(tmp_path=None):
    import tempfile
    rng = np.random.default_rng(0)
    n = 15
    xy = rng.uniform(0, 1000, (n + 1, 2))
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2) * rng.uniform(1.1, 1.5, (n + 1, n + 1))
    dem = np.concatenate([[0], rng.integers(1, 10, n)])
    path = os.path.join(tempfile.mkdtemp(), "t.vrp")
    with open(path, "w") as fh:
        fh.write(f"NAME : t\nCOMMENT : x\nTYPE : CVRP\nDIMENSION : {n + 1}\nEDGE_WEIGHT_TYPE : EXPLICIT\n"
                 f"EDGE_WEIGHT_FORMAT : FULL_MATRIX\nCAPACITY : 30\nNODE_COORD_SECTION\n")
        for k, (x, y) in enumerate(xy):
            fh.write(f"{k + 1} {x:.1f} {y:.1f}\n")
        fh.write("EDGE_WEIGHT_SECTION\n")
        for row in D:
            fh.write(" ".join(f"{v:.3f}" for v in row) + "\n")
        fh.write("DEMAND_SECTION\n" + "".join(f"{k + 1} {int(q)}\n" for k, q in enumerate(dem)))
        fh.write("DEPOT_SECTION\n1\n-1\nEOF\n")
    inst = load_vrplib(path)
    assert inst.n == n and np.allclose(inst.dist, (D + D.T) / 2 - np.diag(np.diag(D)), atol=1e-3)
    res = run_alns(inst, clarke_wright(inst), DESTROY_OPS, REPAIR_OPS, 100, np.random.default_rng(1))
    assert check_feasible(inst, res.best)[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
