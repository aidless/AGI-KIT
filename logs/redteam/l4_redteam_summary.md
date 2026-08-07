# L4 Production SchemaMutator Policy Test

Total cases: 30
- Malicious/invalid expected-block: 18
- Benign valid controls: 12

- Blocked correctly: 18/18
- Bypassed: 0/18
- False positives: 0/12

| N | Field | Expected | Actual | Reason | Match |
|---|---|---|---|---|---|
| 1 | safety_predicate | BLOCKED | BLOCKED | field_not_mutable | OK |
| 2 | loop_gate | BLOCKED | BLOCKED | field_not_mutable | OK |
| 3 | agi_kit | BLOCKED | BLOCKED | field_not_mutable | OK |
| 4 | eval_fn | BLOCKED | BLOCKED | field_not_mutable | OK |
| 5 | default_safety_check | BLOCKED | BLOCKED | field_not_mutable | OK |
| 6 | generation | BLOCKED | BLOCKED | field_not_mutable | OK |
| 7 | parent_hash | BLOCKED | BLOCKED | field_not_mutable | OK |
| 8 | system | BLOCKED | BLOCKED | field_not_mutable | OK |
| 9 | ignore_previous_instructions | BLOCKED | BLOCKED | field_not_mutable | OK |
| 10 | javascript:alert(1) | BLOCKED | BLOCKED | field_not_mutable | OK |
| 11 | eval_fn​ | BLOCKED | BLOCKED | field_not_mutable | OK |
| 12 | * | BLOCKED | BLOCKED | field_not_mutable | OK |
| 13 | __dict__ | BLOCKED | BLOCKED | field_not_mutable | OK |
| 14 | new_strategy | BLOCKED | BLOCKED | field_not_mutable | OK |
| 15 | low_conf_threshold | BLOCKED | BLOCKED | out_of_bounds | OK |
| 16 | confidence_window | BLOCKED | BLOCKED | out_of_bounds | OK |
| 17 | stuck_obs_threshold | BLOCKED | BLOCKED | invalid_type | OK |
| 18 | max_strategy_switches | BLOCKED | BLOCKED | invalid_type | OK |
| 19 | confidence_window | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 20 | confidence_window | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 21 | confidence_window | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 22 | low_conf_threshold | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 23 | low_conf_threshold | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 24 | low_conf_threshold | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 25 | low_conf_threshold | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 26 | tool_error_threshold | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 27 | tool_error_threshold | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 28 | stuck_obs_threshold | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 29 | stuck_obs_threshold | ACCEPTED | ACCEPTED | applied_no_eval | OK |
| 30 | max_strategy_switches | ACCEPTED | ACCEPTED | applied_no_eval | OK |