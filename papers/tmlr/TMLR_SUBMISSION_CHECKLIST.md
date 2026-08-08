# TMLR Submission Checklist (2026-08-08)

Manuscript: *AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on
Consumer Hardware - Empirical Observations*
Author: Zewen Liu (independent researcher; ORCID 0009-0003-2981-9888)
Repository: https://github.com/aidless/AGI-KIT (Apache-2.0)

## 1. Files to upload

| Item | File |
|---|---|
| Manuscript PDF | `papers/tmlr/AGI_Kit_TMLR_submission.pdf` (33 pages, current as of 2026-08-08) |
| Cover letter | `papers/tmlr/COVER_LETTER_TMLR.md` (paste into the OpenReview field) |
| Supplementary zip (optional) | Build from `scripts/`, `src/agi_kit/`, `data/`, `logs/` summaries; keep under 50 MiB |

## 2. OpenReview steps (TMLR)

1. Log in / create an OpenReview account at https://openreview.net.
2. Open the TMLR submission form: https://openreview.net/group?id=TMLR -> "New Submission".
3. Title: exact title above. Abstract: paste from `papers/preprint_unified_en.md` (Section Abstract).
4. Authors: Zewen Liu; add ORCID 0009-0003-2981-9888.
5. Keywords: agent, tool-use, continual learning, safety gate, empirical study.
6. Upload `AGI_Kit_TMLR_submission.pdf`.
7. Paste the cover letter.
8. Add supplementary zip if the venue accepts one (reproduction scripts + summaries).
9. Submit; keep the preprint/DOI field blank unless a public identifier exists (none yet).

## 3. Statements required by the venue (already in manuscript and cover letter)

- The 6-scenario prompt-template comparison (Section 7.5) is exploratory and
  descriptive only; it is not a statistically supported architecture comparison.
- The historical three-run summary (Section 4.3) is descriptive only; no
  confidence-interval, power, or superiority claim is made from it.
- No canonical GAIA2 accuracy number is reported (Section 13); the native
  tool-calling adapter attempt is recorded there and in `scripts/run_gaia2_canonical.py`.

## 4. Pre-submission gates (all passed on 2026-08-08)

- `python -m pytest` : 43 passed.
- `python scripts/preflight_submission.py --allow-dirty` : passed.
- PDF/DOCX rebuilt and verified (33 pages; DOCX embeds 2 figures, 17 tables).
- Cover letter metadata matches pyproject.toml and the manuscript.

## 5. Known caveats

- Public `origin/main` is behind local commits; run `git push origin main`
  when github.com is reachable, then confirm the repository URL resolves.
- Worktree is not fully clean (unrelated untracked baseline files remain);
  use `--allow-dirty` for local gates.
