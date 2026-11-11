# Distribution-shift study: shift-aware policies

Runs: 576 (all feasible). Instances per condition: {'C': 8, 'E-depot': 8, 'ID': 8, 'RC': 8, 'long-routes': 8, 'n200': 8}; seeds: 4.
Mondrian conformal gate (per demand type), alpha=0.1.

Values = mean % cost change vs PALNS (negative = better). `worst` = largest per-instance mean change (robustness).

## Mean cost change vs PALNS (%)

| condition   |   flagged_OOD |   race_guided_wins |   always |   prior |   race |   combined |   oracle |
|:------------|--------------:|-------------------:|---------:|--------:|-------:|-----------:|---------:|
| ID          |          0    |               0.56 |    0.249 |   0.249 |  0.243 |      0.249 |   -0.005 |
| RC          |          0    |               0.59 |   -0.011 |  -0.011 |  0.007 |     -0.011 |   -0.055 |
| C           |          0.38 |               0.62 |    0.088 |   0.12  |  0.099 |      0.186 |   -0.036 |
| E-depot     |          0    |               0.59 |    0.033 |   0.033 |  0.064 |      0.033 |   -0.056 |
| long-routes |          0.62 |               0.53 |   -0.059 |   0.078 | -0.051 |     -0.046 |   -0.184 |
| n200        |          0    |               0.56 |   -0.16  |  -0.16  | -0.023 |     -0.16  |   -0.183 |
| ALL         |          0.17 |               0.58 |    0.023 |   0.052 |  0.057 |      0.042 |   -0.086 |

## Worst-instance cost change vs PALNS (%)

| condition   |   worst_always |   worst_prior |   worst_race |   worst_combined |   worst_oracle |
|:------------|---------------:|--------------:|-------------:|-----------------:|---------------:|
| ID          |          0.525 |         0.525 |        0.745 |            0.525 |              0 |
| RC          |          0.139 |         0.139 |        0.185 |            0.139 |              0 |
| C           |          0.639 |         0.639 |        0.502 |            0.639 |              0 |
| E-depot     |          0.599 |         0.599 |        0.267 |            0.599 |              0 |
| long-routes |          0.747 |         0.747 |        0.586 |            0.747 |              0 |
| n200        |          0.184 |         0.184 |        0.289 |            0.184 |              0 |
| ALL         |          0.747 |         0.747 |        0.745 |            0.747 |              0 |

## Holm-corrected Wilcoxon p-values (paired over instance x seed)

| condition   |   always_vs_PALNS |   prior_vs_PALNS |   race_vs_PALNS |   combined_vs_PALNS |   prior_vs_always |   race_vs_always |   combined_vs_always |
|:------------|------------------:|-----------------:|----------------:|--------------------:|------------------:|-----------------:|---------------------:|
| ID          |            0.1369 |           0.1369 |          0.208  |              0.1369 |            1      |           1      |               1      |
| RC          |            1      |           1      |          1      |              1      |            1      |           1      |               1      |
| C           |            0.5576 |           0.1452 |          0.4406 |              0.0146 |            0.9604 |           0.9604 |               0.2069 |
| E-depot     |            1      |           1      |          1      |              1      |            1      |           1      |               1      |
| long-routes |            1      |           1      |          1      |              1      |            1      |           1      |               1      |
| n200        |            0.8226 |           0.8226 |          1      |              0.8226 |            1      |           0.5084 |               1      |
| ALL         |            0.8161 |           0.8161 |          0.6194 |              0.8161 |            0.8161 |           0.8161 |               0.8161 |

## Mean regret vs per-instance oracle (%)

|          |   regret_% |
|:---------|-----------:|
| never    |      0.09  |
| always   |      0.11  |
| prior    |      0.14  |
| race     |      0.145 |
| combined |      0.129 |
