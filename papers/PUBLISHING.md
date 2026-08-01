# Publishing Guide (AGI Kit Unified Preprint v2)

This guide covers submission to **TMLR** (primary target) with
**arXiv** as the always-available pre-print venue. The venue priorities
reflect the Round 9 reviewer-sim result of **3.50 / 5.0 (Weak Accept)**
which is sufficient for TMLR submission but not for a typical TMLR
Accept decision; arXiv is the safer immediate target.

## 1. Repository Snapshot (as of 2026-08-01)

- `papers/preprint_unified_en.md` (and `.pdf` / `.docx`): the unified preprint.
- `papers/_deprecated/`: the predecessor 5-paper TMLR bundle, preserved
  unmodified for archival/audit (DO NOT submit; not the record of submission).
- `papers/COVER_LETTER.md`: TMLR cover letter (with arXiv fallback note).
- `papers/00_INDEX_en.pdf`: short index page for the preprint.
- `papers/figures/`: 5 matplotlib figures, 200 DPI.
- `papers/reviews/`: heuristic reviewer simulator output
  (`summary.txt` + `preprint_unified_en_review.txt`).
- `dist/agi-research-kit.tar.gz`: full bundle for supplementary upload.
- `REPORT.md`: 23 sections including Round 9 ceiling analysis.

## 2. TMLR Submission Workflow

### Step 1: Confirm Working Tree

```bash
cd /path/to/agi-research-kit
bash dist/push.sh --check-only dummy dummy
# Expected: All self-checks passed.
```

### Step 2: Push to GitHub

```bash
bash dist/push.sh <your-github-username> agi-research-kit
```

The script runs six pre-push guards (working tree clean, tarball
not tracked, no >50 MB files, LICENSE/README present, .env absent,
push.sh executable). If any fail, fix and retry.

### Step 3: Submit via OpenReview

- URL: https://openreview.net/group?id=TMLR
- New Submission
- Upload `papers/docx/preprint_unified_en.docx` (Times Roman 11pt, A4)
- Cover letter: paste contents of `papers/COVER_LETTER.md`
- Supplementary: `dist/agi-research-kit.tar.gz`

## 3. arXiv Submission Workflow (Fallback or Primary)

### Step 1: Re-Verify PDF

```bash
explorer papers/preprint_unified_en.pdf
# Confirm 8-12 pages A4 with figures embedded
```

### Step 2: Submit via arXiv

- URL: https://arxiv.org/submit
- Primary subject: `cs.AI` (Artificial Intelligence)
- Cross-list: `cs.LG` (Machine Learning)
- Title: *AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on
  Consumer Hardware �� Empirical Observations*
- Authors: AGI Research Kit Contributors
- Abstract: copy from `papers/preprint_unified_en.md` front matter.
- Comments: include the GitHub repo URL (after Round 10 push).
- File: `papers/preprint_unified_en.pdf` (or a `.tar.gz` of `.tex` source).

arXiv typically processes in 24-72 hours. Once accepted, the
preprint DOI is permanent and can be cited in the TMLR submission.

## 4. Why arXiv Works as Either Fallback or Primary

| | TMLR | arXiv |
|---|---|---|
| Review bar | typically 4.0+/5.0 | none |
| Acceptance time | 3-6 months | 24-72 hours |
| Our reviewer-sim score | 3.50 (Weak Accept) | n/a |
| Verdict for our work | needs minor revision; can be accepted with revisions | acceptable as system report |

We have reframed the contribution honestly (��8 Limitations). For a
venue that demands a benchmark-beating result, this work is not yet
ready; arXiv publication as a system report is appropriate. The
empirical artifacts and consolidated lessons are valuable to the
community.

## 5. Final Integrity Sweep (Run Before Submission)

Run all five from `experiments/`:

```bash
python experiments/stress_safety_gate.py
# Expected: 12/12 OK
python experiments/redteam/l4_redteam.py
# Expected: 18/18 blocked, 0/5 false positives
python experiments/gate_calibration.py
# Expected: 60-trial grid, monotone acceptance rates
python experiments/arith_eval.py qwen3:1.7b 10
# Expected: 10/10 correct (warm Ollama required, ~5 min wall clock)
bash dist/push.sh --check-only dummy dummy
# Expected: all six guards pass
```

If any output differs from expectation, file a Round 11 fix before
submitting.

## 6. Future Work After arXiv Submission

- Round 10 follow-up: real SFT loop with SmolLM2-135M (Stream A.5)
- Round 11 follow-up: 10-seed statistical robustness (Stream A.3)
- Round 12 follow-up: head-to-head baselines (Stream B.1)
- Beyond: complete the remaining 7 GAIA2 apps (currently 3/10 covered)
- Beyond: deployment simulation with adversarial user (Stream C.2)

The README and PUBLISHING should be updated with the arXiv DOI within
7 days of acceptance.

