# AGI Research Kit: Unified Preprint v2

## AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on Consumer Hardware - Empirical Observations

**Authors:** Zewen Liu (刘泽文)  
**Date:** 2026-08-02  
**Status:** Submission draft; not yet submitted

**Affiliation:** Independent researcher

**ORCID:** https://orcid.org/0009-0003-2981-9888

## Evidence Summary

- Five synthetic ablation configurations: **77.8% final-emission for
  every configuration**; the evaluation is saturated and does not
  identify component effects.
- Hard 20-task arithmetic evaluation: bare Qwen3-1.7B scores **5.0%**
  at three steps and **35.0%** at six steps; the full configuration
  records 100.0% structural completion and **95.0% correctness** after
  an independent gold recheck. The 60-point matched-step difference
  remains confounded by prompt structure and control flow.
- Controlled L1 check: on a separate fixed 20-task held-out split with
  matched prompt/tool/budget/model-seed settings, Static scores
  **19/20** and L1 reflection-and-verification scores **20/20**. The
  only discordant pair is an auditable correction, but exact p=1.0;
  this is not a general superiority claim.
- Retroactive gold re-evaluation: **77.6% correctness** on the
  85-episode arithmetic subset of 138 historical episodes.
- Safety gate: **12/12** deterministic boundary cases passed.
- Guarded self-evolving loop (Section 7.7, 10 rounds): 8 conservative
  REJECTs (6 on genuinely injected rounds), 2 safety-pass PROMOTEs
  REJECTs, 2 safety-pass PROMOTEs (delta=0.0), zero net-positive rounds;
  placebo-controlled skill-specific gains (tool_03/tool_h3) and
  direction-specific harm (tool_04/state_h3/state_01) documented with
  fixed rechecks and conflict scans.
- Production SchemaMutator policy test: **18/18** invalid changes
  blocked and **12/12** valid in-range controls accepted.
- Historical repeated-run result: 62.5%, 62.5%, and 56.2% over three
  16-episode runs. This result is descriptive only because seed control
  and artifact provenance are incomplete.

## Primary Files

- `preprint_unified_en.md`: manuscript source
- `preprint_unified_en.pdf`: PDF build
- `docx/preprint_unified_en.docx`: editable submission draft
- `COVER_LETTER.md`: TMLR cover-letter draft
- `PUBLISHING.md`: submission preflight and workflow
- `figures/`: generated plots; the manuscript embeds only Figures 1
  and 5. Historical figures are retained for audit but not used as
  submission-level performance evidence.
- `reviews/artifact_checklist.txt`: disk-presence audit, not a quality score

## Submission Blockers

Before external submission, supply final author names and affiliations,
replace repository URL placeholders, run a clean repeated experiment
with explicit seeds and isolated output directories, and pass
`python scripts/preflight_submission.py` from a clean Git worktree.

The predecessor five-paper bundle is retained in `_deprecated/` for
history only and is not the submission of record.
