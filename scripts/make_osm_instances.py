"""Build CVRP instances on REAL road networks of sprawled cities (run locally).

Requires internet access to OpenStreetMap and:  pip install osmnx networkx
(The cloud machine used to develop this package cannot reach OSM.)

For each city a drivable street network within ``--radius`` metres of a
central point is downloaded; ``--n`` customers are sampled among street
nodes (so customer density follows the real urban fabric, sprawled or
compact), the depot is placed at a peripheral node (typical distribution
centre on the city edge), and the full shortest-path distance matrix in
metres is written to a CVRPLIB file with EDGE_WEIGHT_TYPE: EXPLICIT.
Simple sprawl descriptors are written to data/osm/city_descriptors.csv:
  circuity      mean (network distance / straight-line distance)
  node_density  street intersections per km^2
  cv_nn         coefficient of variation of nearest-neighbour distances
                (high = clustered/patchy, low = regular)
You can add your own district sprawl index (Marino 2020) as a column there.

    python scripts/make_osm_instances.py --n 100 --per-city 4
    python scripts/run_benchmarks.py --instances "data/osm/*.vrp" \
        --methods PALNS,VAE-PALNS:uniform,RACE:uniform --vae uniform=models/vae_uniform.npz \
        --tag osm --out results/bench_osm.csv
"""
import argparse
import os

import numpy as np

CITIES = {   # name: (lat, lon) of a central point
    "Lima": (-12.0464, -77.0428),
    "Bogota": (4.6097, -74.0817),
    "MexicoCity": (19.4326, -99.1332),
    "Santiago": (-33.4489, -70.6693),
    "SaoPaulo": (-23.5505, -46.6333),
    "Jakarta": (-6.2088, 106.8456),
    "Manila": (14.5995, 120.9842),
}


def build(city, lat, lon, n, radius, seed, out_dir, demand_range=(1, 100), route_size=8):
    import networkx as nx
    import osmnx as ox
    G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")
    G = ox.project_graph(G)
    G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    nodes = list(G.nodes)
    xy_all = np.array([[G.nodes[v]["x"], G.nodes[v]["y"]] for v in nodes])
    rng = np.random.default_rng(seed)
    centre = xy_all.mean(0)
    far = np.argsort(-np.linalg.norm(xy_all - centre, axis=1))[: max(10, len(nodes) // 50)]
    depot = int(rng.choice(far))
    cust = rng.choice([i for i in range(len(nodes)) if i != depot], size=n, replace=False)
    idx = [depot] + list(cust)
    sel = [nodes[i] for i in idx]
    D = np.zeros((n + 1, n + 1))
    for a, s in enumerate(sel):
        lengths = nx.single_source_dijkstra_path_length(G, s, weight="length")
        D[a] = [lengths.get(t, np.inf) for t in sel]
    if not np.isfinite(D).all():
        raise RuntimeError("disconnected nodes sampled")
    xy = xy_all[idx]
    dem = np.concatenate([[0], rng.integers(demand_range[0], demand_range[1] + 1, n)])
    cap = int(max(dem.max(), np.ceil(route_size * dem.sum() / n)))
    name = f"OSM-{city}-n{n}-s{seed}"
    path = os.path.join(out_dir, name + ".vrp")
    with open(path, "w") as fh:
        fh.write(f"NAME : {name}\nCOMMENT : OpenStreetMap road network, metres\nTYPE : CVRP\n"
                 f"DIMENSION : {n + 1}\nEDGE_WEIGHT_TYPE : EXPLICIT\n"
                 f"EDGE_WEIGHT_FORMAT : FULL_MATRIX\nCAPACITY : {cap}\nNODE_COORD_SECTION\n")
        for k, (x, y) in enumerate(xy):
            fh.write(f"{k + 1} {x:.1f} {y:.1f}\n")
        fh.write("EDGE_WEIGHT_SECTION\n")
        for row in D:
            fh.write(" ".join(f"{v:.1f}" for v in row) + "\n")
        fh.write("DEMAND_SECTION\n")
        for k, q in enumerate(dem):
            fh.write(f"{k + 1} {int(q)}\n")
        fh.write("DEPOT_SECTION\n1\n-1\nEOF\n")
    eu = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    mask = eu > 0
    nnd = np.sort(eu + np.eye(len(eu)) * 1e18, axis=1)[:, 0]
    area_km2 = np.pi * (radius / 1000) ** 2
    desc = {"instance": name, "city": city, "circuity": float((D[mask] / eu[mask]).mean()),
            "node_density": len(nodes) / area_km2, "cv_nn": float(nnd.std() / nnd.mean())}
    return path, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default=",".join(CITIES))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-city", type=int, default=4)
    ap.add_argument("--radius", type=int, default=6000)
    ap.add_argument("--out", default="data/osm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for city in a.cities.split(","):
        lat, lon = CITIES[city]
        for k in range(a.per_city):
            path, desc = build(city, lat, lon, a.n, a.radius, 100 + k, a.out)
            rows.append(desc)
            print("wrote", path, {k: round(v, 3) for k, v in desc.items() if isinstance(v, float)})
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "city_descriptors.csv"), index=False)


if __name__ == "__main__":
    main()
