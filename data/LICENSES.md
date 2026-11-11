# Data sources and licences

The code in this repository is MIT-licensed (see `LICENSE`). The data have their own terms:

| Path | Content | Source | Licence / terms |
|---|---|---|---|
| `data/cvrplib/X-*.vrp`, `CMT*.vrp` | Uchoa et al. (2017) X instances, Christofides–Mingozzi–Toth instances | copied from the HGS-CVRP repository (Vidal 2022) | public benchmark instances; HGS-CVRP repository MIT (`HGS-CVRP_LICENSE.txt`) |
| `data/cvrplib/X-n101-k25.sol` | optimal solution computed with HGS-CVRP | this work | MIT |
| `data/user_files/*` | Solomon (1987) and Gehring–Homberger (1999) benchmark files (subsets) | public benchmark sets | public benchmark instances |
| `data/reference_costs.csv` | HGS-CVRP reference costs and literature best-known values | this work / CVRPLIB | MIT (values) |
| `data/roads/*_roads.parquet` | OpenStreetMap road centre-lines for Lima, Milan, Oslo, Hiroshima | OpenSidewalkMap city nodes (github.com/opensidewalkmap, MIT); Lima from the Geofabrik Peru extract of 2026-10-14 | **© OpenStreetMap contributors, ODbL 1.0** (https://opendatacommons.org/licenses/odbl/) |
| `data/roads/instances/*.vrp`, `descriptors.csv` | CVRP instances and street descriptors derived from the OSM data | this work | **ODbL 1.0** (derived database; attribution "© OpenStreetMap contributors") |

## Not included: Lima/Callao district boundaries
`scripts/make_road_instances.py` needs `data/roads/lima_callao_distritos.geojson` (district polygons with a
`distrito` column, WGS84) only to **rebuild** the Lima zones; the built instances are already included.
Obtain official district boundaries from the Instituto Geográfico Nacional (IGN) / INEI of Peru, or from
the copy in https://github.com/Hardnyx/limarutas (`data/raw/ign/lima_callao_distritos.geojson`; no licence
stated there, so check terms before redistributing).

## Larger CVRPLIB X instances (`data/cvrplib_large/`)

X-n303-k21, X-n344-k43, X-n401-k29 and X-n502-k39 (Uchoa et al., 2017) with their best-known solutions, taken from the PyVRP instance collection (https://github.com/PyVRP/Instances, MIT licence; see `PyVRP-Instances_LICENSE.txt`).
