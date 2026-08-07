# Release Notes — AGI Research Kit v1.0

**Unified preprint submission draft** (2026-08-01).

## Highlights
- Unified manuscript documenting the L1-L4 self-improving agent architecture
- Hard 20-task evaluation: 95% gold correctness after independent recheck;
  prompt, step-budget, and control-flow confounds are reported explicitly
- Real SFT plumbing validation on SmolLM2-135M (2 min on CPU)
- Five matplotlib figures, generated PDFs, and full reproduction scripts

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
ollama pull qwen2.5:3b
ollama pull llama3.2:1b
```

## Quick start
```powershell
.\.venv\Scripts\python.exe -u experiments/full_run3.py --n 50 --retrain-every 10 --no-sft
```

## Paper
`papers/preprint_unified_en.pdf` is the current submission draft.
The predecessor five-paper bundle is archived under `papers/_deprecated/`.

## Review artifacts
`papers/reviews/` contains heuristic review and artifact-presence checks.

## License
Apache License 2.0 (see LICENSE)
