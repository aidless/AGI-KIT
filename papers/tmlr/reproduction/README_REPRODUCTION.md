# AGI Kit - TMLR Minimal Reproduction Package (2026-08-08)

Manuscript: *AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on
Consumer Hardware - Empirical Observations*
Author: Zewen Liu (ORCID 0009-0003-2981-9888)
Repository: https://github.com/aidless/AGI-KIT (Apache-2.0)

## 1. Environment
- Windows 11 (tested), 64 GB RAM, no GPU; CPU-only inference via Ollama.
- Python 3.12 venv: `python -m venv .venv`, then `pip install -r requirements.txt`.
- Ollama tags: `qwen3:1.7b`, `qwen3:0.6b`, `qwen2.5:3b`, `llama3.2:1b`.
- Round 22 candidate (deployable via GGUF): `ollama create agi-sft-qwen3-0.6b:round22 -f logs/Modelfile_round22` (GGUF itself is too large for the paper package; produced from the clean merged model via llama.cpp converter).

## 2. Gates
```powershell
.venv\Scripts\python.exe -m pytest tests\unit -q
.venv\Scripts\python.exe scripts\preflight_submission.py --allow-dirty
```
Passed on 2026-08-08: pytest 43/43; preflight all OK.

## 3. Round 22 L3 acceptance audit (negative result)
```powershell
.venv\Scripts\python.exe scripts\audit_l3_swap_round22.py
```
Record in this package: candidate 0/80, base 0/80, gate rejected, no swap.

## 4. Canonical GAIA2 adapter (negative result, runnable)
```powershell
.venv\Scripts\python.exe scripts\run_gaia2_canonical.py --config ambiguity --limit 1 --model qwen2.5:3b --out logs\gaia2_canonical_repro
```
Requires ARE package and `data/gaia2/scenarios_local/` from the repository.
Status: tool invocation improved (matched `update_apartment`, `list_saved_apartments`) but no scenario reaches a valid full score; no GAIA2 accuracy number is claimed. Traces: `logs/canonical_summary.json` (debug3 = before fix, fix8 = after fix; scenario trace included).

## 5. Seeds and determinism (honest note)
- Protocol seeds: 13, 29, 47, 71, 101 (multi-seed protocol specified in `papers/tmlr/REVIEW_RESPONSE_TMLR.md`; not yet run).
- `eval_round22_ollama.py` passes a seed value to the backend; the current `OllamaBackend` does not consume it, so Round 22 numbers are deterministic-in-practice (temperature 0) but not seed-controlled in code.
- GAIA2 runs use temperature 0; generation latency and quantization can affect wall-clock behavior but not tool-call semantics.

## 6. Excluded
- Model weights (>50 MiB binaries are excluded from the release).
- Full 160-scenario GAIA2 dataset (in repository, not in the zip).
- Historical three-run traces (descriptive only; paper Section 4.3).
