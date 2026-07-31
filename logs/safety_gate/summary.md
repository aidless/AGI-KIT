# Adversarial Safety Gate Stress Test

Baseline: 1.000, Threshold: 0.85 (default 0.90 × baseline = 0.85 in our setup)

Match rate: 12/12 = 100%

| Case | new_acc | Expected | Actual | Match | Reason |
|---|---:|---|---|---|---|
| regression_severe | 0.10 | REJECT | REJECT | OK | regressed_below_threshold |
| regression_mild | 0.50 | REJECT | REJECT | OK | regressed_below_threshold |
| just_under | 0.84 | REJECT | REJECT | OK | regressed_below_threshold |
| at_threshold | 0.85 | ACCEPT | ACCEPT | OK | passed |
| just_over | 0.86 | ACCEPT | ACCEPT | OK | passed |
| equal_baseline | 1.00 | ACCEPT | ACCEPT | OK | passed |
| better_than | 1.20 | ACCEPT | ACCEPT | OK | passed |
| zero_acc | 0.00 | REJECT | REJECT | OK | regressed_below_threshold |
| super_high | 2.00 | ACCEPT | ACCEPT | OK | passed |
| low_threshold | 0.05 | ACCEPT | ACCEPT | OK | passed |
| high_threshold | 0.99 | REJECT | REJECT | OK | regressed_below_threshold |
| zero_threshold | 0.01 | ACCEPT | ACCEPT | OK | passed |

## Boundary analysis

- **At threshold** (0.85): gate uses `>=` so accepts. Conservative but not paranoid — this is intentional so the gate is not a pure rejector.
- **Just over** (0.86): accepts, which is the right behavior for a small uplift.
- **Regression** (0.10-0.50): correctly rejects, even with 0.85 threshold.
- **Super-high new_acc** (1.20, 2.00): accepts, which is the right behavior if the new model truly is better (e.g., from a good SFT run).