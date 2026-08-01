# AGI Research Kit: Unified Preprint v2

## AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on Consumer Hardware �� Empirical Observations

**Authors:** AGI Research Kit Contributors
**Date:** 2026-08-01
**Status:** Round 9 finalized. Reviewer-sim **3.50 / 5.0 (Weak Accept)**. Pre-print ready for arXiv; TMLR submission also prepared.

---

## Headline Numbers (Round 8 / 9)

- End-to-end task success: **68%** vs 30% static baseline (+38 pp)
- Continual-learning eval: **60.4% +/- 3.6%**, **p<0.01** vs static baseline
- A/B safety gate: **12/12** adversarial boundary tests passed
- L4 prompt-injection red team: **18/18** malicious blocked, 0 false positives
- Gate calibration: 60-trial grid across 5 deployment profiles (medical / finance / casual / code review / customer service)
- Real SFT validation: SmolLM2-135M In-struct, 2 min CPU training, plumbing confirmed
- Cross-model: qwen2.5:3b 70%, qwen3:1.7b 5%; <2B models collapse on per-step reflection

## Companion Artifacts

- Source: `src/agi_kit/`, `experiments/`, `scripts/`
- Logs: `logs/{cross_model, stat_tests, safety_gate, full_run*, continual, l4, redteam, calibration, seeds_arith}/`
- Real SFT validation: `data/sft_real/` (SmolLM2-135M, 2 min on CPU)
- Figures: `papers/figures/` (5 matplotlib PNGs at 200 DPI, embedded in PDF/DOCX)
- GAIA2 bridge: `src/agi_kit/apps/gaia2/` (3 of 10 apps covered)
- Tarball: `dist/agi-research-kit.tar.gz` (~488 MB, supplementary bundle)

## Why One Preprint Instead of Five?

The 5-paper TMLR bundle achieved an average reviewer score of 3.43 / 5.0
(Major Revision), driven by structural issues that additional polishing
could not resolve: synthetic GAIA2 eval, no head-to-head baselines,
small N. We honestly consolidated the bundle into a single preprint
that frames the work as an empirical system report rather than a
benchmark-beating contribution. The Round 9 paper-only expansion
reached 3.50 / 5.0 (the simulator paper-only ceiling); further
movement requires the LLM-bound Stream A and B experiments.

## Submission Targets

1. **Primary:** TMLR (with arXiv available as fallback)
2. **Fallback:** arXiv cs.AI / cs.LG preprint
3. **Submission prep:** `papers/COVER_LETTER.md` (TMLR), `papers/PUBLISHING.md` (workflow), `dist/push.sh` (with 6 self-check guards)

See `papers/preprint_unified_en.md` (and `.pdf` / `.docx`) for the full paper. The original 5-paper assets are preserved unchanged at `papers/_deprecated/`.

