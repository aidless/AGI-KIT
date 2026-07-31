# Release Notes — AGI Research Kit v1.0

**TMLR submission bundle** (2026-07-31).

## Highlights
- 5-paper bundle documenting L1-L4 self-improving agent architecture
- 50-episode end-to-end evaluation (68.5% success rate, +38.5pp over static)
- Real SFT validation on SmolLM2-135M (2 min on CPU)
- 5 matplotlib figures, 7 PDFs, full reproduction scripts

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
```

## Quick start
```powershell
.\.venv\Scripts\python.exe -u experiments/full_run3.py --n 50 --retrain-every 10 --no-sft
```

## Papers
1. `papers/paper1_l1_self_critique_en.pdf` — Self-Critique as a First-Class Abstraction
2. `papers/paper2_l2_meta_control_en.pdf` — Semantic Strategy Memory with Meta-Controller
3. `papers/paper3_l3_continual_loop_en.pdf` — Continual Learning Loop with A/B Safety Gate
4. `papers/paper4_l4_recursive_en.pdf` — Bounded Recursive Self-Modification
5. `papers/paper5_l1_l4_system_en.pdf` — End-to-End Self-Improving Architecture

## Reviewer simulation
`papers/reviews/` contains simulated reviewer critiques of all 5 papers.

## License
MIT (see LICENSE)
