# Distribution-shift study: shift-aware policies

Runs: 576 (all feasible). Instances per condition: {'C': 8, 'E-depot': 8, 'ID': 8, 'RC': 8, 'long-routes': 8, 'n200': 8}; seeds: 4.
Mondrian conformal gate (per demand type), alpha=0.1.

Values = mean % cost change vs PALNS (negative = better). `worst` = largest per-instance mean change (robustness).

## Mean cost change vs PALNS (%)

| condition   |   flagged_OOD |   race_guided_wins |   always |   prior |   race |   combined |   oracle |
|:------------|--------------:|-------------------:|---------:|--------:|-------:|-----------:|---------:|
| ID          |          0    |               0.81 |   -0.231 |  -0.231 | -0.15  |     -0.231 |   -0.25  |
| RC          |          0    |               0.44 |    0.065 |   0.065 |  0.113 |      0.065 |   -0.075 |
| C           |          0.38 |               0.44 |   -0.008 |   0.04  |  0.003 |      0.064 |   -0.065 |
| E-depot     |          0    |               0.5  |   -0.151 |  -0.151 |  0.039 |     -0.151 |   -0.223 |
| long-routes |          0.62 |               0.56 |   -0.024 |  -0.001 |  0.048 |      0.062 |   -0.107 |
| n200        |          0    |               0.56 |    0.008 |   0.008 | -0.085 |      0.008 |   -0.097 |
| ALL         |          0.17 |               0.55 |   -0.056 |  -0.045 | -0.005 |     -0.03  |   -0.136 |

## Worst-instance cost change vs PALNS (%)

| condition   |   worst_always |   worst_prior |   worst_race |   worst_combined |   worst_oracle |
|:------------|---------------:|--------------:|-------------:|-----------------:|---------------:|
| ID          |          0.101 |         0.101 |        0.341 |            0.101 |              0 |
| RC          |          0.524 |         0.524 |        0.731 |            0.524 |              0 |
| C           |          0.349 |         0.349 |        0.577 |            0.577 |              0 |
| E-depot     |          0.424 |         0.424 |        0.866 |            0.424 |              0 |
| long-routes |          0.401 |         0.401 |        0.52  |            0.52  |              0 |
| n200        |          0.391 |         0.391 |        0.27  |            0.391 |              0 |
| ALL         |          0.524 |         0.524 |        0.866 |            0.577 |              0 |

## Holm-corrected Wilcoxon p-values (paired over instance x seed)

| condition   |   always_vs_PALNS |   prior_vs_PALNS |   race_vs_PALNS |   combined_vs_PALNS |   prior_vs_always |   race_vs_always |   combined_vs_always |
|:------------|------------------:|-----------------:|----------------:|--------------------:|------------------:|-----------------:|---------------------:|
| ID          |             0.304 |            0.304 |          0.7436 |               0.304 |                 1 |           1      |                    1 |
| RC          |             1     |            1     |          1      |               1     |                 1 |           1      |                    1 |
| C           |             1     |            1     |          1      |               1     |                 1 |           1      |                    1 |
| E-depot     |             1     |            1     |          1      |               1     |                 1 |           0.8924 |                    1 |
| long-routes |             1     |            1     |          1      |               1     |                 1 |           1      |                    1 |
| n200        |             1     |            1     |          1      |               1     |                 1 |           1      |                    1 |
| ALL         |             1     |            1     |          1      |               1     |                 1 |           1      |                    1 |

## Mean regret vs per-instance oracle (%)

|          |   regret_% |
|:---------|-----------:|
| never    |      0.138 |
| always   |      0.08  |
| prior    |      0.092 |
| race     |      0.131 |
| combined |      0.106 |
