# Publishing Guide (Unified Preprint v2)

The manuscript is a submission draft, not a published preprint. Use
this checklist before any arXiv or TMLR upload.

## 1. Required Metadata

- Replace `AGI Research Kit Contributors` with final author names.
- Add affiliations, contact author, and ORCID identifiers if available.
- Create the public repository and replace every `<org>` or `your-org`
  URL placeholder.
- Confirm that the author list, title, abstract, license, and repository
  URL agree across Markdown, PDF, DOCX, package metadata, and forms.

## 2. Reproducibility Gate

Start Ollama and pull every model tag referenced by the reported
experiments before running the gate:

```powershell
ollama serve
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
ollama pull qwen2.5:3b
ollama pull llama3.2:1b
```

Run from the repository root:

```powershell
python scripts\preflight_submission.py
python -m pytest
python experiments\stress_safety_gate.py
python experiments\redteam\l4_redteam.py
```

The historical `logs/stat_tests/results.json` summary is not a valid
inferential experiment. Before making seed-level statistical claims,
rerun with fresh per-run directories, explicit seeds for Python, NumPy,
PyTorch, and the model backend, a fixed task manifest, and at least ten
independent runs. Report effect sizes and uncertainty at the run level.

## 3. Build and Inspect

```powershell
python scripts\make_figures.py
python scripts\build_papers_pdf_en.py
python scripts\make_docx.py
```

Inspect every PDF and DOCX page. Confirm that Figures 1 and 5 render,
tables are present and readable, no corrupted punctuation, replacement glyphs, internal
self-scores, stale five-paper text, or placeholder URLs remain.

## 4. Repository Release

The repository uses the Apache License 2.0. Do not publish until the Git
worktree is clean, tests pass, no secrets are present, and no tracked
file exceeds the hosting limit. The Bash helper in `dist/push.sh` is
optional; `scripts/preflight_submission.py` is the cross-platform gate.

Create a tagged release for the exact manuscript snapshot and archive
it in a DOI-granting repository such as Zenodo before citing a DOI.

## 5. arXiv

- Primary category: `cs.AI`; optional cross-list: `cs.LG`.
- Upload the final PDF or a self-contained TeX source bundle.
- Use the exact title, authors, abstract, and public repository URL.
- Do not claim the paper is already published or has a DOI before arXiv
  processing is complete.

## 6. TMLR

- Submit through OpenReview at `https://openreview.net/group?id=TMLR`.
- Upload the venue-compliant manuscript format requested by the current
  TMLR instructions.
- Paste the finalized cover letter; remove all placeholder author text.
- Describe the 6-scenario prompt comparison as exploratory and the
  historical three-run summary as descriptive only.

## 7. Post-Submission

After a public identifier exists, update the README, manuscript,
package metadata, citation block, and release notes in one commit.
