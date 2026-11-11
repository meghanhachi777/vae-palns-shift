# Summary (bench_pilot.csv)

Metric: `dev_best_pct` (%; gap to reference = HGS/BKS, or deviation from best found)
Instances: 12, seeds per instance: 5

## all

| method            |   mean_gap |   sd_gap |   median_gap |   cpu_s |   wall_s |   routes |   vae_share |
|:------------------|-----------:|---------:|-------------:|--------:|---------:|---------:|------------:|
| PALNS             |      0.712 |    0.715 |        0.419 |  19.316 |   15.726 |   13.033 |       0     |
| VAE-PALNS:uniform |      0.641 |    0.621 |        0.385 |  20.607 |   17.017 |   13.067 |       0.304 |
| VAE-PALNS:mixed   |      0.528 |    0.519 |        0.374 |  20.503 |   16.819 |   12.983 |       0.309 |

Friedman chi2=3.447, p=0.1785 (blocks=12)

Paired Wilcoxon vs baseline (negative diff = better than baseline):

| method            | vs    |   pairs |   wins |   ties |   losses |   mean_rel_diff_pct |   p_raw |   p_holm |
|:------------------|:------|--------:|-------:|-------:|---------:|--------------------:|--------:|---------:|
| VAE-PALNS:uniform | PALNS |      60 |     28 |      8 |       24 |             -0.068  |  0.6456 |   0.6456 |
| VAE-PALNS:mixed   | PALNS |      60 |     30 |      6 |       24 |             -0.1788 |  0.175  |   0.3501 |

## layout=C

| method            |   mean_gap |   sd_gap |   median_gap |   cpu_s |   wall_s |   routes |   vae_share |
|:------------------|-----------:|---------:|-------------:|--------:|---------:|---------:|------------:|
| PALNS             |      0.337 |    0.394 |        0.189 |  19.518 |   16.162 |    13.25 |       0     |
| VAE-PALNS:uniform |      0.293 |    0.332 |        0.15  |  20.651 |   16.723 |    13.25 |       0.289 |
| VAE-PALNS:mixed   |      0.35  |    0.441 |        0.186 |  20.692 |   16.736 |    13.25 |       0.299 |

Friedman chi2=0.500, p=0.7788 (blocks=4)

Paired Wilcoxon vs baseline (negative diff = better than baseline):

| method            | vs    |   pairs |   wins |   ties |   losses |   mean_rel_diff_pct |   p_raw |   p_holm |
|:------------------|:------|--------:|-------:|-------:|---------:|--------------------:|--------:|---------:|
| VAE-PALNS:uniform | PALNS |      20 |     12 |      0 |        8 |             -0.042  |  0.654  |        1 |
| VAE-PALNS:mixed   | PALNS |      20 |     10 |      0 |       10 |              0.0139 |  0.9256 |        1 |

## layout=R

| method            |   mean_gap |   sd_gap |   median_gap |   cpu_s |   wall_s |   routes |   vae_share |
|:------------------|-----------:|---------:|-------------:|--------:|---------:|---------:|------------:|
| PALNS             |      0.648 |    0.53  |        0.598 |  19.051 |   15.211 |     12.6 |       0     |
| VAE-PALNS:uniform |      0.456 |    0.406 |        0.369 |  20.416 |   16.937 |     12.7 |       0.309 |
| VAE-PALNS:mixed   |      0.535 |    0.4   |        0.466 |  20.227 |   16.418 |     12.6 |       0.315 |

Friedman chi2=4.133, p=0.1266 (blocks=4)

Paired Wilcoxon vs baseline (negative diff = better than baseline):

| method            | vs    |   pairs |   wins |   ties |   losses |   mean_rel_diff_pct |   p_raw |   p_holm |
|:------------------|:------|--------:|-------:|-------:|---------:|--------------------:|--------:|---------:|
| VAE-PALNS:uniform | PALNS |      20 |      9 |      5 |        6 |             -0.1886 |  0.1819 |   0.3638 |
| VAE-PALNS:mixed   | PALNS |      20 |      8 |      5 |        7 |             -0.1108 |  0.4955 |   0.4955 |

## layout=RC

| method            |   mean_gap |   sd_gap |   median_gap |   cpu_s |   wall_s |   routes |   vae_share |
|:------------------|-----------:|---------:|-------------:|--------:|---------:|---------:|------------:|
| PALNS             |      1.15  |    0.893 |        1.151 |  19.379 |   15.804 |    13.25 |       0     |
| VAE-PALNS:uniform |      1.172 |    0.68  |        1.374 |  20.755 |   17.392 |    13.25 |       0.314 |
| VAE-PALNS:mixed   |      0.7   |    0.647 |        0.624 |  20.59  |   17.302 |    13.1  |       0.314 |

Friedman chi2=6.500, p=0.03877 (blocks=4)

Paired Wilcoxon vs baseline (negative diff = better than baseline):

| method            | vs    |   pairs |   wins |   ties |   losses |   mean_rel_diff_pct |   p_raw |   p_holm |
|:------------------|:------|--------:|-------:|-------:|---------:|--------------------:|--------:|---------:|
| VAE-PALNS:uniform | PALNS |      20 |      7 |      3 |       10 |              0.0265 |  0.4074 |   0.4074 |
| VAE-PALNS:mixed   | PALNS |      20 |     12 |      1 |        7 |             -0.4396 |  0.1262 |   0.2524 |
