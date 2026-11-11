"""CVRP instances: data structure, file loaders and an Uchoa-style generator.

All instances are treated as the Capacitated VRP (CVRP): one depot (node 0),
customers 1..n with demands, homogeneous vehicles of capacity Q, unlimited
fleet, objective = total travel distance. Time windows present in Solomon /
Gehring-Homberger files are read but ignored (CVRP projection), which is the
standard CVRP setting.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Instance:
    name: str
    coords: np.ndarray            # (n+1, 2), depot at index 0
    demand: np.ndarray            # (n+1,), demand[0] == 0
    capacity: float
    round_dist: bool = False      # True for CVRPLIB EUC_2D (nint rounding)
    meta: dict = field(default_factory=dict)
    dist_matrix: np.ndarray | None = None   # explicit (e.g. road-network) distances

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.demand = np.asarray(self.demand, dtype=float)
        if self.demand[0] != 0:
            raise ValueError("depot demand must be 0")
        if (self.demand[1:] > self.capacity).any():
            raise ValueError("a customer demand exceeds vehicle capacity")
        if self.dist_matrix is not None:
            d = np.asarray(self.dist_matrix, dtype=float)
            if not np.allclose(d, d.T):
                d = (d + d.T) / 2.0      # operators assume symmetric costs
            np.fill_diagonal(d, 0.0)
        else:
            diff = self.coords[:, None, :] - self.coords[None, :, :]
            d = np.sqrt((diff ** 2).sum(-1))
            if self.round_dist:
                d = np.floor(d + 0.5)
        self.dist = d
        self.n = len(self.demand) - 1
        # neighbour structure (excluding self), used by operators and features
        order = np.argsort(d + np.eye(len(d)) * 1e18, axis=1)
        self.neighbors = order[:, :-1]                       # self is last
        rank = np.empty_like(order)
        rows = np.arange(len(d))[:, None]
        rank[rows, order] = np.arange(len(d))[None, :]
        self.rank = rank                                     # rank[i, j]
        k = min(5, self.n - 1) if self.n > 1 else 1
        nn = d[1:, :][np.arange(self.n)[:, None], self.neighbors[1:, :k]]
        self.nn_scale = float(max(nn.mean(), 1e-9))          # mean 5-NN distance
        self.depot_scale = float(max(d[0, 1:].mean(), 1e-9)) # mean depot distance
        self.dmax = float(d.max())

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())

    def lower_bound_routes(self) -> int:
        return int(math.ceil(self.total_demand / self.capacity))


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_solomon(path: str, n_customers: int | None = None,
                 capacity: float | None = None, name: str | None = None) -> Instance:
    """Solomon / Gehring-Homberger text format (also header-less tables).

    The CVRP projection keeps coordinates and demands and drops time windows.
    If the file has no VEHICLE/CAPACITY header, ``capacity`` must be given
    (defaults to 200, the C1/RC1 value).
    """
    with open(path, "r", errors="ignore") as fh:
        lines = [ln.replace("\r", "") for ln in fh]
    cap = None
    for i, ln in enumerate(lines):
        if re.search(r"NUMBER\s+CAPACITY", ln, re.I):
            toks = lines[i + 1].split()
            if len(toks) >= 2:
                cap = float(toks[1])
            break
    rows = []
    for ln in lines:
        toks = ln.split()
        if len(toks) >= 7 and all(re.fullmatch(r"-?\d+(\.\d+)?", t) for t in toks[:7]):
            rows.append([float(t) for t in toks[:7]])
    arr = np.array(rows)
    if n_customers is not None:
        arr = arr[: n_customers + 1]
    if capacity is not None:
        cap = capacity
    if cap is None:
        cap = 200.0
    title = lines[0].strip() if lines and not lines[0].strip().upper().startswith("CUST") else ""
    nm = name or f"{os.path.splitext(os.path.basename(path))[0]}"
    return Instance(name=nm, coords=arr[:, 1:3], demand=arr[:, 3], capacity=cap,
                    round_dist=False,
                    meta={"source": os.path.basename(path), "title": title,
                          "format": "solomon", "time_windows_ignored": True})


def load_vrplib(path: str) -> Instance:
    """CVRPLIB (TSPLIB-style) .vrp file with EUC_2D distances."""
    with open(path) as fh:
        text = fh.read()
    header, coords, demands, cap, depot = {}, {}, {}, None, 1
    matrix_vals = []
    section = None
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if ":" in ln and re.fullmatch(r"[A-Za-z_ ]+", ln.split(":", 1)[0].strip()):
            key, val = [s.strip() for s in ln.split(":", 1)]
            header[key.upper()] = val
            if key.upper() == "CAPACITY":
                cap = float(val)
            section = None
            continue
        up = ln.upper()
        if up.startswith("NODE_COORD_SECTION"):
            section = "coord"; continue
        if up.startswith("DEMAND_SECTION"):
            section = "demand"; continue
        if up.startswith("DEPOT_SECTION"):
            section = "depot"; continue
        if up.startswith("EDGE_WEIGHT_SECTION"):
            section = "matrix"; continue
        if up.startswith("EOF"):
            break
        toks = ln.split()
        if section == "coord":
            coords[int(toks[0])] = (float(toks[1]), float(toks[2]))
        elif section == "demand":
            demands[int(toks[0])] = float(toks[1])
        elif section == "matrix":
            matrix_vals += [float(t) for t in toks]
        elif section == "depot":
            v = int(toks[0])
            if v > 0:
                depot = v
    ids = [depot] + [i for i in sorted(coords) if i != depot]
    xy = np.array([coords[i] for i in ids])
    dm = np.array([demands.get(i, 0.0) for i in ids])
    dm[0] = 0.0
    ew = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    dmat = None
    if ew == "EXPLICIT":
        if header.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper() != "FULL_MATRIX":
            raise ValueError("only FULL_MATRIX explicit weights are supported")
        m = len(ids)
        full = np.array(matrix_vals).reshape(m, m)
        order = [i - 1 for i in ids]                  # node ids are 1-based
        dmat = full[np.ix_(order, order)]
    return Instance(name=header.get("NAME", os.path.basename(path)).strip(),
                    coords=xy, demand=dm, capacity=cap, round_dist=(ew == "EUC_2D"),
                    meta={"source": os.path.basename(path), "format": "vrplib"},
                    dist_matrix=dmat)


def load_any(path: str, **kw) -> Instance:
    return load_vrplib(path) if path.lower().endswith(".vrp") else load_solomon(path, **kw)


# ----------------------------------------------------------------------------
# Uchoa-style generator (Uchoa et al., 2017, EJOR) -- controllable shift
# ----------------------------------------------------------------------------
def generate_uchoa(n: int, depot: str = "R", customers: str = "R",
                   demand: str = "U", route_size: float | None = None,
                   seed: int = 0, grid: int = 1000) -> Instance:
    """Generate a CVRP instance following the X-set generator logic.

    depot:      'C' centre, 'E' eccentric corner, 'R' random
    customers:  'R' uniform random, 'C' clustered, 'RC' half/half
    demand:     'U' unitary, '1-10', '5-10', '1-100', '50-100',
                'Q' quadrant-dependent, 'SL' many small / few large
    route_size: average customers per route r; capacity = ceil(r*sum(d)/n)
    """
    rng = np.random.default_rng(seed)
    if depot == "C":
        dep = np.array([grid / 2, grid / 2])
    elif depot == "E":
        dep = np.array([0.0, 0.0])
    else:
        dep = rng.uniform(0, grid, 2)

    def clustered(m):
        s = int(rng.integers(3, 9))
        seeds = rng.uniform(0, grid, (s, 2))
        pts = []
        while len(pts) < m:
            p = rng.uniform(0, grid, 2)
            w = np.exp(-np.linalg.norm(seeds - p, axis=1) / 40.0).sum()
            if rng.random() < min(1.0, w):
                pts.append(p)
        return np.array(pts)

    if customers == "R":
        xy = rng.uniform(0, grid, (n, 2))
    elif customers == "C":
        xy = clustered(n)
    elif customers == "RC":
        k = n // 2
        xy = np.vstack([clustered(k), rng.uniform(0, grid, (n - k, 2))])
    else:
        raise ValueError(customers)
    xy = np.round(xy)

    if demand == "U":
        dm = np.ones(n)
    elif demand in ("1-10", "5-10", "1-100", "50-100"):
        lo, hi = map(int, demand.split("-"))
        dm = rng.integers(lo, hi + 1, n).astype(float)
    elif demand == "Q":
        quad = ((xy[:, 0] >= dep[0]) ^ (xy[:, 1] >= dep[1]))
        dm = np.where(quad, rng.integers(51, 101, n), rng.integers(1, 51, n)).astype(float)
    elif demand == "SL":
        small = rng.random(n) < rng.uniform(0.7, 0.95)
        dm = np.where(small, rng.integers(1, 11, n), rng.integers(50, 101, n)).astype(float)
    else:
        raise ValueError(demand)

    r = route_size if route_size is not None else float(rng.uniform(3, 25))
    cap = float(max(dm.max(), math.ceil(r * dm.sum() / n)))
    coords = np.vstack([np.round(dep), xy])
    dem = np.concatenate([[0.0], dm])
    nm = f"G-n{n}-{depot}{customers}-{demand}-r{r:.0f}-s{seed}"
    return Instance(name=nm, coords=coords, demand=dem, capacity=cap, round_dist=True,
                    meta={"format": "generated", "depot": depot, "customers": customers,
                          "demand_type": demand, "route_size": r, "seed": seed})
