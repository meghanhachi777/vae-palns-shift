"""Destroy and repair operators for CVRP-ALNS.

Every destroy operator has signature ``op(inst, sol, q, rng) -> removed`` and
removes (about) ``q`` customers from ``sol`` in place.  Every repair operator
has signature ``op(inst, sol, removed, rng)`` and re-inserts all removed
customers in place, respecting vehicle capacity (a new route is opened when
no feasible position exists).  Each operator is a genuinely different
procedure (no aliases).
"""
from __future__ import annotations

import numpy as np

from .instance import Instance
from .solution import Solution


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def remove_customers(inst: Instance, sol: Solution, rem) -> list:
    rem = set(int(c) for c in rem)
    touched = []
    for k, r in enumerate(sol.routes):
        if any(c in rem for c in r):
            sol.routes[k] = [c for c in r if c not in rem]
            touched.append(k)
    sol.refresh(inst, touched)
    return list(rem)


def positions(sol: Solution):
    """Arrays (prev, cur, next, route_index) over all visited customers."""
    prev, cur, nxt, rid = [], [], [], []
    for k, r in enumerate(sol.routes):
        prev += [0] + r[:-1]
        cur += r
        nxt += r[1:] + [0]
        rid += [k] * len(r)
    return (np.asarray(prev, int), np.asarray(cur, int),
            np.asarray(nxt, int), np.asarray(rid, int))


def _randomized_pick(n: int, p: float, rng) -> int:
    """Index into a sorted list, biased to the front (Ropke & Pisinger 2006)."""
    return min(n - 1, int(n * rng.random() ** p))


# ----------------------------------------------------------------------------
# destroy operators
# ----------------------------------------------------------------------------
def random_removal(inst, sol, q, rng):
    cust = np.concatenate([np.asarray(r) for r in sol.routes])
    q = min(q, len(cust))
    return remove_customers(inst, sol, rng.choice(cust, size=q, replace=False))


def worst_removal(inst, sol, q, rng, p: float = 3.0):
    """Remove customers with the largest detour cost, randomized by y^p."""
    pv, c, nx, _ = positions(sol)
    d = inst.dist
    sav = d[pv, c] + d[c, nx] - d[pv, nx]
    order = list(c[np.argsort(-sav)])
    out = []
    for _ in range(min(q, len(order))):
        out.append(order.pop(_randomized_pick(len(order), p, rng)))
    return remove_customers(inst, sol, out)


def shaw_removal(inst, sol, q, rng, p: float = 6.0, w_dist: float = 1.0, w_dem: float = 0.5):
    """Related removal (Shaw 1998): distance + demand similarity."""
    _, c, _, _ = positions(sol)
    remaining = list(c)
    seed = remaining.pop(int(rng.integers(len(remaining))))
    removed = [seed]
    dmax, Q = inst.dmax, inst.capacity
    while len(removed) < q and remaining:
        ref = removed[int(rng.integers(len(removed)))]
        rem_arr = np.asarray(remaining)
        rel = (w_dist * inst.dist[ref, rem_arr] / dmax
               + w_dem * np.abs(inst.demand[ref] - inst.demand[rem_arr]) / Q)
        order = np.argsort(rel)
        j = order[_randomized_pick(len(order), p, rng)]
        removed.append(remaining.pop(int(j)))
    return remove_customers(inst, sol, removed)


def string_removal(inst, sol, q, rng, l_max: int = 10):
    """Adjacent string removal (simplified SISR, Christiaens & Vanden Berghe 2020):
    remove contiguous strings from routes close to a random seed customer."""
    route_of = {}
    for k, r in enumerate(sol.routes):
        for i, cst in enumerate(r):
            route_of[cst] = (k, i)
    seed = int(rng.integers(1, inst.n + 1))
    removed, used_routes = [], set()
    for cst in [seed, *inst.neighbors[seed]]:
        if len(removed) >= q:
            break
        if cst == 0 or cst not in route_of:
            continue
        k, i = route_of[cst]
        if k in used_routes:
            continue
        r = sol.routes[k]
        L = int(rng.integers(1, min(l_max, len(r), q - len(removed)) + 1))
        start = int(rng.integers(max(0, i - L + 1), min(i, len(r) - L) + 1))
        removed += r[start:start + L]
        used_routes.add(k)
    return remove_customers(inst, sol, removed)


def route_removal(inst, sol, q, rng):
    """Remove whole (preferably short) routes -- helps reduce fleet size."""
    sizes = np.array([len(r) for r in sol.routes], float)
    w = 1.0 / sizes
    order = rng.choice(len(sizes), size=len(sizes), replace=False, p=w / w.sum())
    removed = []
    for k in order:
        if removed and len(removed) + sizes[k] > 1.5 * q:
            continue
        removed += sol.routes[k]
        if len(removed) >= q:
            break
    return remove_customers(inst, sol, removed)


# ----------------------------------------------------------------------------
# repair operators (vectorized insertion-cost matrices)
# ----------------------------------------------------------------------------
def _edges(sol: Solution):
    us, vs, rid = [], [], []
    for k, r in enumerate(sol.routes):
        path = [0] + r + [0]
        us += path[:-1]
        vs += path[1:]
        rid += [k] * (len(path) - 1)
    return np.asarray(us, int), np.asarray(vs, int), np.asarray(rid, int)


def _insert_all(inst, sol, removed, rng, k_regret: int = 1, noise: float = 0.0):
    d, dem, Q = inst.dist, inst.demand, inst.capacity
    us, vs, rid = _edges(sol)
    loads = np.asarray(sol.loads, float)
    rem = np.asarray(removed, int)
    amp = noise * inst.dmax
    touched = set()
    BIG = 1e12
    while rem.size:
        if us.size:
            delta = d[np.ix_(us, rem)] + d[np.ix_(vs, rem)] - d[us, vs][:, None]
            feas = loads[rid][:, None] + dem[rem][None, :] <= Q + 1e-9
            if amp > 0:
                delta = np.maximum(0.0, delta + rng.uniform(-amp, amp, delta.shape))
            delta = np.where(feas, delta, np.inf)
        else:
            delta = np.full((0, rem.size), np.inf)
        newc = 2.0 * d[0, rem]
        if amp > 0:
            newc = np.maximum(0.0, newc + rng.uniform(-amp, amp, newc.shape))
        col_min = delta.min(axis=0) if us.size else np.full(rem.size, np.inf)
        if k_regret <= 1:
            j = int(np.argmin(np.minimum(col_min, newc)))
        else:
            if us.size:
                starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
                rm = np.minimum.reduceat(delta, starts, axis=0)
                opts = np.vstack([rm, newc[None, :]])
            else:
                opts = newc[None, :]
            s = np.sort(np.where(np.isinf(opts), BIG, opts), axis=0)
            kk = min(k_regret, s.shape[0])
            regret = (s[1:kk] - s[0]).sum(axis=0) if kk > 1 else np.zeros(rem.size)
            if kk < k_regret:                      # fewer options than k -> urgent
                regret = regret + (k_regret - kk) * BIG
            j = int(np.lexsort((s[0], -regret))[0])
        c = int(rem[j])
        if us.size and col_min[j] < newc[j] and np.isfinite(col_min[j]):
            e = int(np.argmin(delta[:, j]))
            k = int(rid[e])
            start = int(np.searchsorted(rid, k))
            sol.routes[k].insert(e - start, c)
            loads[k] += dem[c]
            old_v = vs[e]
            vs[e] = c
            us = np.insert(us, e + 1, c)
            vs = np.insert(vs, e + 1, old_v)
            rid = np.insert(rid, e + 1, k)
            touched.add(k)
        else:
            k = len(sol.routes)
            sol.routes.append([c])
            sol.costs.append(0.0)
            sol.loads.append(0.0)
            loads = np.append(loads, dem[c])
            us = np.append(us, [0, c])
            vs = np.append(vs, [c, 0])
            rid = np.append(rid, [k, k])
            touched.add(k)
        rem = np.delete(rem, j)
    sol.refresh(inst, sorted(touched))


def greedy_repair(inst, sol, removed, rng):
    _insert_all(inst, sol, removed, rng, k_regret=1, noise=0.0)


def greedy_noise_repair(inst, sol, removed, rng):
    _insert_all(inst, sol, removed, rng, k_regret=1, noise=0.025)


def regret2_repair(inst, sol, removed, rng):
    _insert_all(inst, sol, removed, rng, k_regret=2, noise=0.0)


def regret3_repair(inst, sol, removed, rng):
    _insert_all(inst, sol, removed, rng, k_regret=3, noise=0.0)


DESTROY_OPS = {
    "random": random_removal,
    "worst": worst_removal,
    "shaw": shaw_removal,
    "string": string_removal,
    "route": route_removal,
}

REPAIR_OPS = {
    "greedy": greedy_repair,
    "greedy_noise": greedy_noise_repair,
    "regret2": regret2_repair,
    "regret3": regret3_repair,
}
