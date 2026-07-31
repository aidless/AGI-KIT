# Publishing Guide (AGI Kit Unified Preprint v1)

This guide replaces the prior TMLR bundle workflow. We are now targeting
**arXiv** with a single unified preprint, not TMLR.

## 1. Repository Snapshot (as of 2026-08-01)

- `papers/preprint_unified_en.md` (and .pdf / .docx): the new preprint.
- `papers/_deprecated/`: the predecessor 5-paper TMLR bundle, preserved
  unmodified for archival/audit.
- `papers/COVER_LETTER.md`: the arXiv submission cover letter.
- `papers/00_INDEX_en.pdf`: short index page for the preprint.
- `papers/figures/`: 5 matplotlib figures, 200 DPI.
- `papers/reviews/`: heuristic reviewer simulator output
  (`summary.txt` + 1 per-paper review file).

## 2. arXiv Submission Workflow

### Step 1: Export to arXiv-compatible .tar.gz

```bash
cd papers
tar czf ../dist/preprint_unified_arxiv.tar.gz \
    preprint_unified_en.tex        # OR use the rendered PDF directly
# If you have a .tex export, ship the source. Otherwise the PDF is
# acceptable for the first round.
```

arXiv accepts either:
- a `.tar.gz` containing a single `.tex` file and figures, OR
- a single `.pdf` (with figures embedded).

Our `papers/preprint_unified_en.pdf` already has figures embedded.

### Step 2: Submit via https://arxiv.org/submit

- Primary subject: `cs.AI` (Artificial Intelligence)
- Cross-list: `cs.LG` (Machine Learning)
- Title: *AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on
  Consumer Hardware - Empirical Observations*
- Authors: AGI Research Kit Contributors
- Abstract: copy from `papers/preprint_unified_en.md` (front matter).
- Comments: include the source URL of the GitHub repo (after push).

### Step 3: Source-code supplementary bundle

Upload `dist/agi-research-kit.tar.gz` (~488 MB) as a "source code"
supplementary. arXiv hosts up to a few GB supplementary per paper.

## 3. GitHub Push

```bash
# Optional: regen the source bundle
python scripts/make_release_artifacts.py

# Run self-checks (no push if any guard fails)
bash dist/push.sh --check-only dummy dummy

# Actual push (requires personal access token)
bash dist/push.sh <your-github-username> agi-research-kit
```

`dist/push.sh` now runs six pre-push guards:

1. Working tree clean
2. `dist/agi-research-kit.tar.gz` not tracked
3. No file >50 MB in working tree
4. LICENSE, README.md, papers/README.md present
5. `.env` absent
6. `dist/push.sh` executable

## 4. Why arXiv Instead of TMLR?

| | TMLR submission | arXiv preprint |
|---|---|---|
| Review bar | typically 4.0+/5.0 | none |
| Acceptance time | 3-6 months | 24-72 hours |
| Our reviewer-sim score | 3.43/5 | n/a |
| Verdict for our work | Major Revision | acceptable as a system report |

We reframed the contribution honestly (Section 8 of the preprint lists
what we did not validate). For a venue that demands a benchmark-beating
result, this work is not yet ready; we publish this preprint so the
empirical artifacts and the consolidated lessons are available to the
community and to ourselves as a starting point for further work.

## 5. Future Work

The honest list of follow-ups is in preprint Section 8 (Limitations). In
short: full GAIA2 evaluation, head-to-head baseline runs against
Voyager/MetaGPT/Reflexion on identical hardware, larger N for the
statistical validation, longer continual-learning runs, and adversarial
testing of L4.
