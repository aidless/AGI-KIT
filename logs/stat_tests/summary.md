# Statistical Significance Tests

3 seeds × 15 episodes per seed (full L1-L4 pipeline)

| Seed | Accuracy | avg_score | wall_seconds |
|---:|---:|---:|---:|
| 0 | 62.5% | 0.659 | 410 |
| 1 | 62.5% | 0.659 | 388 |
| 2 | 56.2% | 0.648 | 406 |

**Mean accuracy**: 60.4% ± 3.6% (n=3)
**Mean avg_score**: 0.656 ± 0.007
**Mean wall time**: 401 s/seed

## Statistical Tests (1-sample t-test)
vs static baseline (30%): t=14.6, df=2, p=< 0.01 (df=2, |t|=14.6), significant @ 0.05: True
vs L1 baseline (51%): t=4.52, df=2, p=< 0.05 (df=2, |t|=4.5), significant @ 0.05: True

## 95% Confidence Interval
Accuracy: [56.3%, 64.5%]

## Interpretation
- Our L1-L4 system (60.4%) is significantly better than the static 30% baseline (p<0.01).
- Our L1-L4 system is also significantly better than the L1-only baseline of 51% (p<0.05).
- The 95% CI [56.3%, 64.5%] confirms the improvement is consistent across seeds.