"""Solution representation, cost evaluation, feasibility check, construction."""
from __future__ import annotations

import numpy as np

from .instance import Instance


class Solution:
    """A CVRP solution: list of routes; each route is a list of customer ids
    (depot 0 implicit at both ends). Route costs and loads are cached."""

    __slots__ = ("routes", "costs", "loads", "cost")

    def __init__(self, inst: Instance, routes):
        self.routes = [list(r) for r in routes if len(r) > 0]
        self.costs = [route_cost(inst, r) for r in self.routes]
        self.loads = [float(inst.demand[r].sum()) for r in self.routes]
        self.cost = float(sum(self.costs))

    def copy(self) -> "Solution":
        s = Solution.__new__(Solution)
        s.routes = [r[:] for r in self.routes]
        s.costs = self.costs[:]
        s.loads = self.loads[:]
        s.cost = self.cost
        return s

    def refresh(self, inst: Instance, which=None):
        """Recompute cached values (all routes, or route indices in ``which``)."""
        idx = range(len(self.routes)) if which is None else which
        for k in idx:
            self.costs[k] = route_cost(inst, self.routes[k])
            self.loads[k] = float(inst.demand[self.routes[k]].sum())
        self.drop_empty()
        self.cost = float(sum(self.costs))

    def drop_empty(self):
        keep = [k for k, r in enumerate(self.routes) if len(r) > 0]
        if len(keep) != len(self.routes):
            self.routes = [self.routes[k] for k in keep]
            self.costs = [self.costs[k] for k in keep]
            self.loads = [self.loads[k] for k in keep]

    def key(self) -> int:
        """Order-independent hash used to detect revisited solutions."""
        canon = []
        for r in self.routes:
            rr = r if r[0] <= r[-1] else r[::-1]
            canon.append(tuple(rr))
        return hash(tuple(sorted(canon)))

    @property
    def n_routes(self) -> int:
        return len(self.routes)


def route_cost(inst: Instance, r) -> float:
    if len(r) == 0:
        return 0.0
    d = inst.dist
    path = np.fromiter((0, *r, 0), dtype=int, count=len(r) + 2)
    return float(d[path[:-1], path[1:]].sum())


def check_feasible(inst: Instance, sol: Solution, tol: float = 1e-6) -> tuple[bool, str]:
    """Every customer visited exactly once, no depot inside routes, capacity ok,
    cached cost consistent with a fresh recomputation."""
    seen = np.zeros(inst.n + 1, dtype=int)
    for r in sol.routes:
        for c in r:
            if c <= 0 or c > inst.n:
                return False, f"invalid node {c}"
            seen[c] += 1
        load = inst.demand[r].sum()
        if load > inst.capacity + tol:
            return False, f"capacity violated ({load} > {inst.capacity})"
    if (seen[1:] != 1).any():
        bad = np.where(seen[1:] != 1)[0] + 1
        return False, f"customers not visited exactly once: {bad[:10].tolist()}"
    fresh = sum(route_cost(inst, r) for r in sol.routes)
    if abs(fresh - sol.cost) > 1e-6 * max(1.0, fresh):
        return False, f"cached cost {sol.cost} != recomputed {fresh}"
    return True, "ok"


def clarke_wright(inst: Instance) -> Solution:
    """Parallel Clarke-Wright savings construction (capacity-feasible)."""
    n, d, q, Q = inst.n, inst.dist, inst.demand, inst.capacity
    route_of = {i: i for i in range(1, n + 1)}
    routes = {i: [i] for i in range(1, n + 1)}
    loads = {i: q[i] for i in range(1, n + 1)}
    i_idx, j_idx = np.triu_indices(n, k=1)
    i_idx, j_idx = i_idx + 1, j_idx + 1
    sav = d[0, i_idx] + d[0, j_idx] - d[i_idx, j_idx]
    for t in np.argsort(-sav, kind="stable"):
        if sav[t] <= 0:
            break
        i, j = int(i_idx[t]), int(j_idx[t])
        ri, rj = route_of[i], route_of[j]
        if ri == rj or loads[ri] + loads[rj] > Q:
            continue
        a, b = routes[ri], routes[rj]
        # i and j must be route endpoints; orient so that i is a's tail, j is b's head
        if a[-1] != i:
            if a[0] == i:
                a = a[::-1]
            else:
                continue
        if b[0] != j:
            if b[-1] == j:
                b = b[::-1]
            else:
                continue
        merged = a + b
        routes[ri] = merged
        loads[ri] += loads[rj]
        for c in b:
            route_of[c] = ri
        del routes[rj], loads[rj]
    return Solution(inst, list(routes.values()))
