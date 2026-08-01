# Statistical Significance Tests

3 seeds x 15 episodes per seed (full L1-L4 pipeline)

| Seed | Accuracy | avg_score | wall_seconds |
|---:|---:|---:|---:|
| 0 | 62.5% | 0.659 | 410.0 |
| 1 | 62.5% | 0.659 | 388.0 |
| 2 | 56.2% | 0.648 | 406.0 |

**Mean accuracy**: 60.4% +/- 3.6% (n=3)
**Mean avg_score**: 0.656 +/- 0.007
**Mean wall time**: 401.0 s/seed

## Note on Baseline Comparisons
The 30% and 51% baselines referenced in earlier drafts of the paper
were NOT measured. This script reports only the variance across seeds.
See Section 4.3 of the paper for the honest framing.

## 95% Confidence Interval
Accuracy: [56.3%, 64.5%]