# Baselines on Hard Multi-Step Chains (Round 14)

Same model qwen3:1.7b. 8 multi-step arithmetic tasks from logs/full_run3/gen-5 + gen-6.

| Configuration | Emission | Correctness |
|---|---:|---:|
| Static one-shot | 8/8 = 100.0% | 8/8 = 100.0% |
| ReAct JSON one-shot | 8/8 = 100.0% | 3/8 = 37.5% |
| Reflexion-style CoT | 8/8 = 100.0% | 7/8 = 87.5% |

AGI Kit L1-L4 reference (full_run3 gen-1..6 arith n=85): 77.6% correctness.

Honest reading: if baseline correctness is much lower than 77.6%, AGI Kit adds value on hard chains. Tie or above means reflective loop does not help on these tasks.