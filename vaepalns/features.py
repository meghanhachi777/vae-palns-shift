"""Scale-free per-customer context features of a CVRP solution.

Each visited customer is described by how it sits in its route: detour it
causes, how near its route neighbours are (in absolute terms and as a
neighbour rank), its position relative to the depot and to its own and other
routes, and load information.  Features are normalized by instance-level
scales, so a VAE trained on many instances can be applied to new ones.  In
high-quality solutions these contexts follow a characteristic distribution;
customers whose context is atypical (high reconstruction error) are likely
misplaced and are good removal candidates.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance
from .operators import positions
from .solution import Solution

FEATURE_NAMES = [
    "log_d_prev", "log_d_next", "log_detour", "rank_prev", "rank_next",
    "depot_dist", "demand_frac", "route_load_frac", "route_len_rel",
    "log_d_own_centroid", "own_vs_other_centroid", "angle_dev",
]


def customer_features(inst: Instance, sol: Solution):
    """Return (customers, F) with F of shape (n_visited, 12)."""
    d = inst.dist
    pv, c, nx, rid = positions(sol)
    nn, ds = inst.nn_scale, inst.depot_scale
    logn = np.log1p(inst.n)
    detour = np.maximum(0.0, d[pv, c] + d[c, nx] - d[pv, nx])

    xy = inst.coords
    R = len(sol.routes)
    cent = np.zeros((R, 2))
    np.add.at(cent, rid, xy[c])
    sizes = np.bincount(rid, minlength=R).astype(float)
    cent /= sizes[:, None]
    dc = np.linalg.norm(xy[c][:, None, :] - cent[None, :, :], axis=2)   # (m, R)
    own = dc[np.arange(len(c)), rid]
    if R > 1:
        other = np.where(np.arange(R)[None, :] == rid[:, None], np.inf, dc).min(axis=1)
        ratio = own / (own + other + 1e-9)
    else:
        ratio = np.zeros(len(c))

    dep = xy[0]
    ang_c = np.arctan2(xy[c, 1] - dep[1], xy[c, 0] - dep[0])
    ang_r = np.arctan2(cent[rid, 1] - dep[1], cent[rid, 0] - dep[0])
    dang = np.abs((ang_c - ang_r + np.pi) % (2 * np.pi) - np.pi) / np.pi

    loads = np.asarray(sol.loads)[rid] / inst.capacity
    rlen = sizes[rid] / sizes.mean()

    F = np.column_stack([
        np.log1p(d[pv, c] / nn),
        np.log1p(d[c, nx] / nn),
        np.log1p(detour / nn),
        np.log1p(inst.rank[c, pv]) / logn,
        np.log1p(inst.rank[c, nx]) / logn,
        d[0, c] / ds,
        inst.demand[c] / inst.capacity,
        loads,
        rlen,
        np.log1p(own / nn),
        ratio,
        dang,
    ])
    return c, F
