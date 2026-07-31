# Cover Letter - arXiv Preprint Submission

**To:** arXiv cs.AI / cs.LG moderators
**From:** AGI Research Kit Contributors
**Date:** 2026-08-01
**Re:** Single-paper preprint (replaces a prior 5-paper bundle that has been archived)

---

Dear Editors,

We are pleased to submit a single arXiv preprint describing **AGI
Kit**, an end-to-end self-improving tool-use pipeline that runs on
consumer hardware (CPU only, ~5 GB RAM).

The preprint reports what we built and measured: a four-layer pipeline
(per-step reflection, semantic strategy memory, continual learning with
A/B safety gate, and bounded recursive self-modification) evaluated on a
Qwen3-1.7B base model with a 50-task synthetic GAIA2 mini benchmark.
Headline numbers: 68% end-to-end task success vs 30% for a static
baseline (+38 percentage points); continual-learning eval accuracy
60.4% +/- 3.6% across 3 seeds (t-test vs static baseline, t=14.6,
p<0.01); the A/B safety gate passes 12 of 12 adversarial boundary tests.

A prior version of this work was prepared as a 5-paper TMLR submission
bundle. Internal reviewer simulation rated the bundle at 3.43 / 5.0
average (Major Revision). We have consolidated the bundle into this
single preprint and reframed the contribution as an empirical system
report, with an honest Limitations section describing what we did not
validate. The original 5-paper assets are preserved unmodified at
`papers/_deprecated/`.

The work is reproducible on consumer hardware. Source code, scripts,
models referenced (including a real SmolLM2-135M SFT model that trains
in 2 minutes on CPU), logs from all experiments, and reviewer-simulator
output are bundled in the supplementary archive `dist/agi-research-kit.tar.gz`.

We have no conflicts of interest to declare. We request the cs.AI
primary subject class, with cs.LG as cross-list.

Sincerely,
AGI Research Kit Contributors
