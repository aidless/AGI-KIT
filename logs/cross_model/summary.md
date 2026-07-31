# Cross-Model Evaluation

Same 20 arithmetic tasks on multiple Ollama models.

| Model | Description | Correct | Accuracy | Avg sec/q |
|---|---|---:|---:|---:|
| `qwen3:1.7b` | Qwen3 family, 2.0B params (Q4_K_M) | 1/20 | 5.0% | 5.71 |
| `llama3.2:1b` | LLaMA 3.2 family, 1.2B params (Q8_0) | 1/20 | 5.0% | 0.8 |
| `qwen2.5:3b` | Qwen2.5 family, 3.1B params (Q4_K_M) | 14/20 | 70.0% | 1.45 |
| `qwen3:0.6b` | Qwen3 family, 0.75B params (Q4_K_M) | 1/20 | 5.0% | 3.66 |

## Per-task results

| # | Category | Q | Gold |
|---:|---|---|---|
| 1 | arithmetic | Use calculator to compute 17 * 23, then final | 391 |
| 2 | arithmetic | Use calculator to compute 256 + 789, then final | 1045 |
| 3 | arithmetic | Use calculator to compute 2**10, then final | 1024 |
| 4 | arithmetic | Use calculator to compute 88 * 88, then final | 7744 |
| 5 | arithmetic | Use calculator to compute 100 % 7, then final | 2 |
| 6 | chained | Use calculator to compute 7*6, then echo the result, then fi... | 42 |
| 7 | chained | Use calculator to compute 1024-256, then echo, then final | 768 |
| 8 | arithmetic | Use calculator to compute 99*99, then final | 9801 |
| 9 | arithmetic | Use calculator to compute 11*11, then final | 121 |
| 10 | arithmetic | Use calculator to compute 13*13, then final | 169 |
| 11 | arithmetic | Use calculator to compute 1000-1, then final | 999 |
| 12 | chained | Use calculator to compute 3**4, then echo, then final | 81 |
| 13 | arithmetic | Use calculator to compute 100/4, then final | 25 |
| 14 | arithmetic | Use calculator to compute 50*40-100, then final | 1900 |
| 15 | arithmetic | Use calculator to compute (15+5)*3, then final | 60 |
| 16 | arithmetic | Use calculator to compute 9999 - 1234, then final | 8765 |
| 17 | arithmetic | Use calculator to compute 1234 + 5678, then final | 6912 |
| 18 | arithmetic | Use calculator to compute 144 / 12, then final | 12 |
| 19 | chained | Use calculator to compute 2**8, then echo, then final | 256 |
| 20 | chained | Use calculator to compute 100/4, then echo, then final | 25 |

## Per-model per-task results

| Task # | `qwen3:1.7b` | `llama3.2:1b` | `qwen2.5:3b` | `qwen3:0.6b` |
|---:|---:|---:|---:|---:||
| 1 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (2s) |
| 2 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (3s) |
| 3 | ✗ (3s) | ✗ (2s) | ✓ (1s) | ✓ (3s) |
| 4 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (3s) |
| 5 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (2s) |
| 6 | ✗ (3s) | ✗ (1s) | ✗ (3s) | ✗ (3s) |
| 7 | ✗ (3s) | ✗ (2s) | ✗ (3s) | ✗ (3s) |
| 8 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (3s) |
| 9 | ✗ (3s) | ✗ (2s) | ✓ (1s) | ✗ (3s) |
| 10 | ✗ (3s) | ✗ (2s) | ✗ (3s) | ✗ (3s) |
| 11 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (2s) |
| 12 | ✗ (3s) | ✓ (2s) | ✗ (3s) | ✗ (3s) |
| 13 | ✓ (2s) | ✗ (1s) | ✓ (1s) | ✗ (1s) |
| 14 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (1s) |
| 15 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (3s) |
| 16 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (3s) |
| 17 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (2s) |
| 18 | ✗ (3s) | ✗ (1s) | ✓ (1s) | ✗ (3s) |
| 19 | ✗ (3s) | ✗ (2s) | ✗ (3s) | ✗ (3s) |
| 20 | ✗ (3s) | ✗ (2s) | ✗ (3s) | ✗ (3s) |