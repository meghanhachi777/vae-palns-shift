# Summary (bench_main.csv)

Metric: `gap_pct` (%; gap to reference = HGS/BKS, or deviation from best found)
Instances: 14, seeds per instance: 5

## all

| method          |   mean_gap |   sd_gap |   median_gap |   cpu_s |   wall_s |   routes |   vae_share |
|:----------------|-----------:|---------:|-------------:|--------:|---------:|---------:|------------:|
| ALNS            |      0.905 |    0.94  |        0.667 |  19.967 |   30.505 |   15.057 |       0     |
| PALNS-indep     |      1.263 |    1.26  |        0.957 |  20.498 |   14.492 |   15.071 |       0     |
| PALNS           |      0.83  |    0.934 |        0.494 |  20.275 |   14.39  |   15.057 |       0     |
| VAE-PALNS:mixed |      0.795 |    0.94  |        0.46  |  21.66  |   15.42  |   15.071 |       0.307 |

Friedman chi2=9.400, p=0.02442 (blocks=14)

Paired Wilcoxon vs baseline (negative diff = better than baseline):

| method          | vs    |   pairs |   wins |   ties |   losses |   mean_rel_diff_pct |   p_raw |   p_holm |
|:----------------|:------|--------:|-------:|-------:|---------:|--------------------:|--------:|---------:|
| ALNS            | PALNS |      70 |     17 |     24 |       29 |              0.0765 |  0.5016 |   0.9173 |
| PALNS-indep     | PALNS |      70 |     10 |     26 |       34 |              0.4287 |  0      |   0.0001 |
| VAE-PALNS:mixed | PALNS |      70 |     23 |     26 |       21 |             -0.0333 |  0.4586 |   0.9173 |
