# Gate Calibration Across Deployment Profiles

5 deployment profiles x 12 candidate accuracies = 60 trials.

| Profile | Recommended | Test threshold | new_acc | Accepted | Reason |
|---|---|---|---|---|---|
| medical | 0.999 | 0.99 | 0.1 | REJECT | regressed_below_threshold |
| medical | 0.999 | 0.99 | 0.3 | REJECT | regressed_below_threshold |
| medical | 0.999 | 0.99 | 0.5 | REJECT | regressed_below_threshold |
| medical | 0.999 | 0.99 | 0.7 | REJECT | regressed_below_threshold |
| medical | 0.999 | 0.99 | 0.84 | REJECT | regressed_below_threshold |
| medical | 0.999 | 0.99 | 0.86 | REJECT | regressed_below_threshold |
| medical | 0.999 | 0.99 | 0.95 | REJECT | regressed_below_threshold |
| medical | 0.999 | 0.99 | 1.0 | ACCEPT | passed |
| medical | 0.999 | 0.99 | 1.1 | ACCEPT | passed |
| medical | 0.999 | 0.99 | 1.2 | ACCEPT | passed |
| medical | 0.999 | 0.99 | 1.5 | ACCEPT | passed |
| medical | 0.999 | 0.99 | 2.0 | ACCEPT | passed |
| finance | 0.999 | 0.95 | 0.1 | REJECT | regressed_below_threshold |
| finance | 0.999 | 0.95 | 0.3 | REJECT | regressed_below_threshold |
| finance | 0.999 | 0.95 | 0.5 | REJECT | regressed_below_threshold |
| finance | 0.999 | 0.95 | 0.7 | REJECT | regressed_below_threshold |
| finance | 0.999 | 0.95 | 0.84 | REJECT | regressed_below_threshold |
| finance | 0.999 | 0.95 | 0.86 | REJECT | regressed_below_threshold |
| finance | 0.999 | 0.95 | 0.95 | ACCEPT | passed |
| finance | 0.999 | 0.95 | 1.0 | ACCEPT | passed |
| finance | 0.999 | 0.95 | 1.1 | ACCEPT | passed |
| finance | 0.999 | 0.95 | 1.2 | ACCEPT | passed |
| finance | 0.999 | 0.95 | 1.5 | ACCEPT | passed |
| finance | 0.999 | 0.95 | 2.0 | ACCEPT | passed |
| casual_chat | 0.85 | 0.85 | 0.1 | REJECT | regressed_below_threshold |
| casual_chat | 0.85 | 0.85 | 0.3 | REJECT | regressed_below_threshold |
| casual_chat | 0.85 | 0.85 | 0.5 | REJECT | regressed_below_threshold |
| casual_chat | 0.85 | 0.85 | 0.7 | REJECT | regressed_below_threshold |
| casual_chat | 0.85 | 0.85 | 0.84 | REJECT | regressed_below_threshold |
| casual_chat | 0.85 | 0.85 | 0.86 | ACCEPT | passed |
| casual_chat | 0.85 | 0.85 | 0.95 | ACCEPT | passed |
| casual_chat | 0.85 | 0.85 | 1.0 | ACCEPT | passed |
| casual_chat | 0.85 | 0.85 | 1.1 | ACCEPT | passed |
| casual_chat | 0.85 | 0.85 | 1.2 | ACCEPT | passed |
| casual_chat | 0.85 | 0.85 | 1.5 | ACCEPT | passed |
| casual_chat | 0.85 | 0.85 | 2.0 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 0.1 | REJECT | regressed_below_threshold |
| code_review | 0.5 | 0.5 | 0.3 | REJECT | regressed_below_threshold |
| code_review | 0.5 | 0.5 | 0.5 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 0.7 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 0.84 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 0.86 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 0.95 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 1.0 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 1.1 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 1.2 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 1.5 | ACCEPT | passed |
| code_review | 0.5 | 0.5 | 2.0 | ACCEPT | passed |
| customer_service | 0.95 | 0.95 | 0.1 | REJECT | regressed_below_threshold |
| customer_service | 0.95 | 0.95 | 0.3 | REJECT | regressed_below_threshold |
| customer_service | 0.95 | 0.95 | 0.5 | REJECT | regressed_below_threshold |
| customer_service | 0.95 | 0.95 | 0.7 | REJECT | regressed_below_threshold |
| customer_service | 0.95 | 0.95 | 0.84 | REJECT | regressed_below_threshold |
| customer_service | 0.95 | 0.95 | 0.86 | REJECT | regressed_below_threshold |
| customer_service | 0.95 | 0.95 | 0.95 | ACCEPT | passed |
| customer_service | 0.95 | 0.95 | 1.0 | ACCEPT | passed |
| customer_service | 0.95 | 0.95 | 1.1 | ACCEPT | passed |
| customer_service | 0.95 | 0.95 | 1.2 | ACCEPT | passed |
| customer_service | 0.95 | 0.95 | 1.5 | ACCEPT | passed |
| customer_service | 0.95 | 0.95 | 2.0 | ACCEPT | passed |

## Summary by profile

| Profile | Threshold | Trials | Accepted | Acceptance rate |
|---|---|---|---|---|
| medical | 0.99 | 12 | 5 | 41.7% |
| finance | 0.95 | 12 | 6 | 50.0% |
| casual_chat | 0.85 | 12 | 7 | 58.3% |
| code_review | 0.5 | 12 | 10 | 83.3% |
| customer_service | 0.95 | 12 | 6 | 50.0% |