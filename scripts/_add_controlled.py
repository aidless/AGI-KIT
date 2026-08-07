"""Add the controlled max_steps=6 bare result to Section 4.1 and abstract."""
content = open("papers/preprint_unified_en.md", encoding="utf-8").read()

# Add a controlled-comparison row to the hard eval table
old = """| Configuration on hard 20-task eval | JSON final-emission rate | vs Static |
|---|---:|---:|
| Static Qwen3-1.7B (bare, max_steps=3) | 5.0% | - |
| Full L1-L4 (max_steps=6) | 100.0% | +95.0 pp |
| Static Llama-3.2-1B (bare, max_steps=3) | 0.0% (0/20) | -4 pp vs Qwen |
| Full L1-L4 on Llama-3.2-1B (max_steps=6) | 100.0% (20/20) | +100 pp vs its bare |"""

new = """| Configuration on hard 20-task eval | JSON final-emission rate | vs Static |
|---|---:|---:|
| Static Qwen3-1.7B (bare, max_steps=3) | 5.0% | - |
| **Bare Qwen3-1.7B (max_steps=6)** - step-budget controlled | **35.0%** | +30 pp |
| Full L1-L4 on Qwen3-1.7B (max_steps=6) | 100.0% | +95 pp |
| **Pure layer contribution (Qwen3-1.7B, matched budget)** | - | **+65 pp** (35 -> 100) |
| Static Llama-3.2-1B (bare, max_steps=3) | 0.0% (0/20) | -4 pp vs Qwen |
| Full L1-L4 on Llama-3.2-1B (max_steps=6) | 100.0% (20/20) | +100 pp vs its bare |

The 35% bare baseline at max_steps=6 (logs/cross_model_bare_qwen1.7b_max6/summary.json)
controls for the step-budget confound: with matched step budget, the layers
contribute **+65 pp** of the **+95 pp** total gain (the remaining +30 pp
comes from the 2x step-budget doubling)."""

if old in content:
    content = content.replace(old, new, 1)
    print("Hard eval table updated with controlled bare")
else:
    print("Hard eval table not matched")

open("papers/preprint_unified_en.md", "w", encoding="utf-8").write(content)
