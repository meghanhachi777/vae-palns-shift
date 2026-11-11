"""Build CVRP instances on real OpenStreetMap road networks (offline).

Data (in data/roads/):
  * <city>_roads.parquet: OSM road centre-lines (LineStrings, WGS84) from the
    OpenSidewalkMap city nodes (MIT); Lima's is a Geofabrik Peru extract,
    snapshot 2026-10-14. Road data (c) OpenStreetMap contributors, ODbL.
  * lima_callao_distritos.geojson: district boundaries of Lima and Callao (IGN),
    used to define Lima zones.

Street graph: every LineString vertex is a node and consecutive vertices are
joined by an edge with its length in metres (local UTM projection). OSM ways
share nodes at junctions, so shared coordinates give the network topology.
The graph is treated as undirected (one-way information is not in the
extract). Distances are shortest paths (Dijkstra) in metres.

Zones ("conditions"):
  Lima-core   consolidated, planned centre (Cercado, Brena, La Victoria, Lince,
              Jesus Maria, San Isidro, Miraflores, Surquillo, San Borja,
              Pueblo Libre, Magdalena, San Miguel, Barranco)
  Lima-north  northern expansion cone (Comas, Carabayllo, Puente Piedra)
  Lima-south  southern expansion cone (Villa El Salvador, Villa Maria del
              Triunfo, San Juan de Miraflores, Pachacamac, Lurin)
  Lima-east   San Juan de Lurigancho (largest district, hillside settlements)
  Milan, Oslo, Hiroshima  compact-city contrasts: disc of the same area as
              Lima-core around the city centre

Customers are sampled uniformly among street-network nodes of degree != 2
(junctions and dead ends) inside the zone, so their spatial density follows
the real urban fabric. The depot is a random node among the 5% of zone nodes
farthest from the zone centroid (an edge-of-zone distribution centre).
Demands U[1,100]; capacity for ~8 customers per route (as in the shift study).

Sprawl descriptors written to data/roads/instances/descriptors.csv:
  circuity        mean network / straight-line distance over customer pairs
  junction_density  junctions (degree >= 3) per km^2 in the zone
  dead_end_share  dead ends / (dead ends + junctions); high = fragmented,
                  cul-de-sac or informal street patterns
  cv_nn           coefficient of variation of customer nearest-neighbour distances
  zone_km2        zone area

    python scripts/make_road_instances.py --per-zone 6 --n 100
"""
import argparse
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "roads")

LIMA_ZONES = {
    "Lima-core": ["LIMA", "BREÑA", "LA VICTORIA", "LINCE", "JESUS MARIA", "SAN ISIDRO",
                  "MIRAFLORES", "SURQUILLO", "SAN BORJA", "PUEBLO LIBRE", "MAGDALENA DEL MAR",
                  "SAN MIGUEL", "BARRANCO"],
    "Lima-north": ["COMAS", "CARABAYLLO", "PUENTE PIEDRA"],
    "Lima-south": ["VILLA EL SALVADOR", "VILLA MARIA DEL TRIUNFO", "SAN JUAN DE MIRAFLORES",
                   "PACHACAMAC", "LURIN"],
    "Lima-east": ["SAN JUAN DE LURIGANCHO"],
}
CENTRES = {"Milan": (9.1900, 45.4642), "Oslo": (10.7522, 59.9139), "Hiroshima": (132.4553, 34.3853)}


def load_graph(city):
    g = gpd.read_parquet(os.path.join(DATA, f"{city.lower()}_roads.parquet"))
    utm = g.estimate_utm_crs()
    coords_list = shapely.get_coordinates(g.geometry.values, return_index=True)
    xy_ll, part = coords_list
    key = np.round(xy_ll, 7)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    same = part[1:] == part[:-1]
    a, b = inv[:-1][same], inv[1:][same]
    pts = gpd.GeoSeries(gpd.points_from_xy(uniq[:, 0], uniq[:, 1]), crs=4326).to_crs(utm)
    xy = np.column_stack([pts.x.values, pts.y.values])
    keep = a != b
    a, b = a[keep], b[keep]
    w = np.linalg.norm(xy[a] - xy[b], axis=1)
    n = len(uniq)
    A = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    A.sum_duplicates()
    deg = np.diff((A > 0).astype(np.int8).tocsr().indptr)
    return A, xy, deg, utm


def zone_polygon(name, utm):
    if name in LIMA_ZONES:
        d = gpd.read_file(os.path.join(DATA, "lima_callao_distritos.geojson"))
        poly = d[d.distrito.isin(LIMA_ZONES[name])].to_crs(utm).union_all()
        return poly
    lon, lat = CENTRES[name]
    c = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(utm).iloc[0]
    core = zone_polygon("Lima-core", gpd.read_file(os.path.join(DATA, "lima_callao_distritos.geojson"))
                        .estimate_utm_crs())
    r = np.sqrt(core.area / np.pi)
    return c.buffer(r)


def build_zone(zone, city, n, per_zone, seed0, out_dir, route_size=8):
    A, xy, deg, utm = load_graph(city)
    poly = zone_polygon(zone, utm)
    buf = poly.buffer(2000)
    inbuf = shapely.contains_xy(buf, xy[:, 0], xy[:, 1])
    idx_buf = np.flatnonzero(inbuf)
    sub = A[idx_buf][:, idx_buf]
    ncomp, lab = connected_components(sub, directed=False)
    big = np.bincount(lab).argmax()
    keep = idx_buf[lab == big]
    pos = {v: i for i, v in enumerate(keep)}
    S = A[keep][:, keep]
    in_zone = shapely.contains_xy(poly, xy[keep, 0], xy[keep, 1])
    cand = np.flatnonzero(in_zone & (deg[keep] != 2))
    zxy = xy[keep][in_zone]
    centroid = np.array([poly.centroid.x, poly.centroid.y])
    zone_nodes = np.flatnonzero(in_zone)
    far = zone_nodes[np.argsort(-np.linalg.norm(xy[keep][zone_nodes] - centroid, axis=1))]
    far = far[: max(20, len(far) // 20)]
    km2 = poly.area / 1e6
    zdeg = deg[keep][in_zone]
    junc, dead = int((zdeg >= 3).sum()), int((zdeg == 1).sum())
    rows = []
    for k in range(per_zone):
        rng = np.random.default_rng(seed0 + k)
        depot = int(rng.choice(far))
        cust = rng.choice(cand[cand != depot], size=n, replace=False)
        sel = np.r_[depot, cust]
        D = dijkstra(S, directed=False, indices=sel)[:, sel]
        if not np.isfinite(D).all():
            raise RuntimeError("disconnected sample")
        P = xy[keep][sel]
        dem = np.r_[0, rng.integers(1, 101, n)]
        cap = int(max(dem.max(), np.ceil(route_size * dem.sum() / n)))
        name = f"OSM-{zone}-n{n}-s{seed0 + k}"
        with open(os.path.join(out_dir, name + ".vrp"), "w") as fh:
            fh.write(f"NAME : {name}\nCOMMENT : OpenStreetMap road network (ODbL), metres, "
                     f"undirected shortest paths\nTYPE : CVRP\nDIMENSION : {n + 1}\n"
                     f"EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n"
                     f"CAPACITY : {cap}\nNODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(P):
                fh.write(f"{i + 1} {x:.1f} {y:.1f}\n")
            fh.write("EDGE_WEIGHT_SECTION\n")
            for row in D:
                fh.write(" ".join(f"{v:.1f}" for v in row) + "\n")
            fh.write("DEMAND_SECTION\n")
            for i, q in enumerate(dem):
                fh.write(f"{i + 1} {int(q)}\n")
            fh.write("DEPOT_SECTION\n1\n-1\nEOF\n")
        eu = np.linalg.norm(P[:, None] - P[None], axis=2)
        m = eu > 1.0
        nnd = np.sort(eu[1:, 1:] + np.eye(n) * 1e18, axis=1)[:, 0]
        rows.append({"instance": name, "zone": zone, "city": city, "n": n, "seed": seed0 + k,
                     "circuity": float(np.median(D[m] / eu[m])),
                     "junction_density": junc / km2, "dead_end_share": dead / max(1, dead + junc),
                     "cv_nn": float(nnd.std() / nnd.mean()), "zone_km2": km2,
                     "mean_depot_km": float(D[0, 1:].mean() / 1000)})
        print(f"  {name}: circuity {rows[-1]['circuity']:.2f}, depot-dist {rows[-1]['mean_depot_km']:.1f} km",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-zone", type=int, default=6)
    ap.add_argument("--zones", default="Lima-core,Lima-north,Lima-south,Lima-east,Milan,Oslo,Hiroshima")
    ap.add_argument("--out", default=os.path.join(DATA, "instances"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for zi, z in enumerate(a.zones.split(",")):
        city = "Lima" if z.startswith("Lima") else z
        print(f"{z} ({city})", flush=True)
        rows += build_zone(z, city, a.n, a.per_zone, 20_000 + 100 * zi, a.out)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "descriptors.csv"), index=False)
    print(pd.DataFrame(rows).groupby("zone")[["circuity", "junction_density", "dead_end_share",
                                             "cv_nn", "zone_km2", "mean_depot_km"]].mean().round(3))


if __name__ == "__main__":
    main()
