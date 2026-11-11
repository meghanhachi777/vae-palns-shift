# Summary (bench_main_Xonly.csv)

Metric: `gap_pct` (%; gap to reference = HGS/BKS, or deviation from best found)
Instances: 8, seeds per instance: 5

## all

| method          |   mean_gap |   sd_gap |   median_gap |   cpu_s |   wall_s |   routes |   vae_share |
|:----------------|-----------:|---------:|-------------:|--------:|---------:|---------:|------------:|
| ALNS            |      1.356 |    0.749 |        1.401 |  28.573 |   38.322 |   23.325 |       0     |
| PALNS-indep     |      2.03  |    1.038 |        1.752 |  29.233 |   18.491 |   23.375 |       0     |
| PALNS           |      1.345 |    0.867 |        1.05  |  28.843 |   18.31  |   23.35  |       0     |
| VAE-PALNS:mixed |      1.187 |    0.92  |        0.881 |  30.708 |   19.542 |   23.375 |       0.301 |

Friedman chi2=10.950, p=0.012 (blocks=8)

Paired Wilcoxon vs baseline (negative diff = better than baseline):

| method          | vs    |   pairs |   wins |   ties |   losses |   mean_rel_diff_pct |   p_raw |   p_holm |
|:----------------|:------|--------:|-------:|-------:|---------:|--------------------:|--------:|---------:|
| ALNS            | PALNS |      40 |     17 |      1 |       22 |              0.0147 |  0.7377 |   0.8633 |
| PALNS-indep     | PALNS |      40 |      9 |      0 |       31 |              0.6781 |  0      |   0.0001 |
| VAE-PALNS:mixed | PALNS |      40 |     23 |      0 |       17 |             -0.1537 |  0.4317 |   0.8633 |
