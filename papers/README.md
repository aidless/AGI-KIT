# Paper Artifacts

This directory contains one unified submission draft:

- `preprint_unified_en.md`: authoritative manuscript source
- `preprint_unified_en.pdf`: generated PDF
- `docx/preprint_unified_en.docx`: generated editable manuscript
- `00_INDEX_en.md`: concise evidence and readiness index
- `COVER_LETTER.md`: TMLR cover-letter draft
- `PUBLISHING.md`: preflight and submission workflow
- `figures/`: generated plots; only Figures 1 and 5 are embedded in the
  manuscript. Figures 2-4 are historical-only and are excluded because
  they do not support a submission-level performance claim.
- `reviews/`: heuristic review output and disk-presence checks

The material in `_deprecated/` is the predecessor five-paper bundle.
It is retained for audit history and must not be submitted as part of
the current manuscript.

## Ollama Prerequisites

Complete reproduction requires the four Ollama model tags used by the
primary, scorer, cross-model, and cross-family runs. Pull them before
running the experiment scripts:

```powershell
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
ollama pull qwen2.5:3b
ollama pull llama3.2:1b
```

## Build

```powershell
python scripts\make_figures.py
python scripts\build_papers_pdf_en.py
python scripts\make_docx.py
python scripts\preflight_submission.py
```

The manuscript intentionally distinguishes structural final emission
from gold-answer correctness and labels historical repeated-run results
as descriptive only. The controlled L1 artifact is at
`logs/controlled_arithmetic/run-llama3b-20260802/`: it contains the
fixed task manifest hash, paired raw traces, strict and normalized JSON
outcomes, and an exact paired analysis. Do not restore the superseded
unmeasured baselines, significance claim, or internal self-score.

## Publication Status

The draft has not been submitted. Author metadata is set (Zewen Liu,
independent researcher); the public repository URL remains required
before arXiv or TMLR submission.
