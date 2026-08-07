# Cover Letter - TMLR Submission Draft

**To:** TMLR Action Editor  
**From:** AGI Research Kit Contributors  
**Date:** 2026-08-02  
**Re:** Single-paper submission

Dear Editor,

We submit the manuscript *"AGI Kit: An End-to-End Self-Improving
Tool-Use Pipeline on Consumer Hardware - Empirical Observations"* for
consideration at TMLR. The author names, affiliations, and repository
URL will be inserted before submission.

The manuscript reports a four-layer tool-use pipeline comprising
per-step reflection, semantic strategy memory, continual learning with
an A/B safety gate, and bounded schema mutation. The system runs on
CPU-only consumer hardware with a Qwen3-1.7B primary model and a
Qwen3-0.6B scorer.

The paper's main contribution is a transparent empirical system report,
including negative results and identified confounds:

- A five-configuration synthetic ablation is saturated at 77.8% JSON
  final-emission for every configuration and does not identify a
  component effect.
- On a 20-task arithmetic evaluation, the bare model scores 5.0% at
  three steps and 35.0% at six steps, while the full configuration
  records 100.0% structural completion but 95.0% correctness after an
  independent gold recheck. We report the 60-point matched-step gap
  as a configuration difference, not a pure layer effect, because
  prompt structure and control flow also differ.
- A new prompt-, tool-, budget-, and seed-matched 20-task check makes
  an L1 tool-evidence correction executable: Static scores 19/20 and
  L1 scores 20/20. The single discordant pair has exact two-sided
  McNemar p=1.0, so we describe it as an auditable correction instance
  rather than a statistically supported general gain.
- Re-evaluation of 138 historical episodes separates structural final
  emission from answer correctness; correctness is 77.6% on the
  85-episode arithmetic subset.
- The A/B safety gate passes 12 deterministic boundary tests. The
  production SchemaMutator rejects 18 invalid field/value changes and
  accepts 12 valid in-range controls in a direct policy test.
- Prompt-template comparisons include both negative and exploratory
  positive results. They are not presented as architecture-level
  comparisons against full ReAct, Reflexion, Voyager, or MetaGPT
  implementations.

We explicitly invalidate the inferential interpretation of an earlier
three-run summary: the driver produced 16 rather than 15 episodes per
run, did not control every stochastic seed, and later reused output
directories. The revised manuscript reports those values descriptively
and makes no confidence-interval, power, or superiority claim.

The repository contains source code, task data, experiment scripts,
figures, and machine-readable result summaries. The manuscript also
states the principal limitations: no canonical end-to-end GAIA2
harness, no architecture-level baseline comparison, incomplete seed
control for the historical repeated runs, mock retraining in the
headline continual-learning experiment, and no real-world deployment.

We have no conflicts of interest to declare.

Sincerely,

AGI Research Kit Contributors
