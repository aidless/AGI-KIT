# Cover Letter — TMLR Submission

**To**: TMLR Action Editor
**From**: AGI Research Kit Contributors
**Date**: 2026-07-31
**Re**: Bundle submission of 5 papers

---

Dear Editor,

We are pleased to submit a **bundle of five interconnected papers** to
TMLR. Together they describe **AGI Kit**, an end-to-end self-improving
tool-use agent architecture that runs on a 1.7B-parameter language model
on consumer hardware (~5 GB RAM, no GPU). The bundle is unified by a
single architecture and a single 50-episode empirical evaluation.

## The Bundle

The five papers form a logical progression:

1. **Paper 1 (L1)** formalizes self-critique as a first-class
   programming abstraction (`Reflector` with `log`, `summarize_episode`,
   `recent`), using a two-layer scoring scheme that blends a fast
   rule heuristic with a slower LLM judgment.

2. **Paper 2 (L2)** introduces Semantic Strategy Memory (`Playbook`)
   paired with a rule-based `MetaController`. Together they form a
   closed-loop meta-layer that detects and recovers from stuck
   trajectories without LLM sampling at decision time.

3. **Paper 3 (L3)** proposes `ContinualLoop` — a state machine driving
   task → episode → buffer → retrain → A/B-gated generation
   transitions. The `default_safety_check` is, to our knowledge, the
   first A/B-style safety gate applied to self-improving LLM agents.

4. **Paper 4 (L4)** introduces bounded recursive self-modification
   via three classes — `SchemaMutator`, `ToolFactory`,
   `PromptMutator` — each guarded by sandbox, lineage, and gate.

5. **Paper 5 (System)** integrates all four layers into a single
   `full_run3.py` pipeline, demonstrating 68% success rate on
   GAIA2-style tasks (+38 pp over a static baseline) at ~5 GB RAM.

## Headline Result

- 50 GAIA2-style episodes (54 tasks including 4 ToolFactory triggers)
- Qwen3-1.7B (Ollama) + Qwen3-0.6B (fast scorer)
- **68.5% success rate** vs 30% static baseline (+38 pp, +127% relative)
- **Eval accuracy rises monotonically** across 7 generations
  (0.585 → 0.735)
- **A/B safety gate correctly rejects** all 7 generations, preserving
  the base model
- L4: 2 schema mutations accepted, 3 trigger-task tools synthesized,
  4 prompt versions tracked
- Wall-clock: 1005 seconds (~19 s/episode) on Windows 11 / CPU only

## Why TMLR?

1. **Novelty**: Each paper introduces a small, specific abstraction
   (Reflector API, MetaController rules, A/B gate, bounded recursive
   self-modification) that has not been formalized in prior work.
2. **Soundness**: Every claim is backed by either a unit test or a
   reproducible 50-episode experiment.
3. **Reproducibility**: All code, weights, and logs are released.
   The 50-episode run can be reproduced in ~17 minutes on a consumer
   laptop with one shell command.
4. **Practical relevance**: Self-improving agents have been
   demonstrated on GPT-4-class models in cloud settings; we show
   that the same loop is feasible at 1.7B parameters and ~5 GB RAM.

## Why a Bundle?

We believe the four layers compose into a single coherent system, and
that presenting them as a bundle strengthens each individual paper:

- Paper 1 (L1) motivates Paper 2 (L2): the `self_score` produced by
  L1 is the signal L2's MetaController consumes.
- Paper 2 (L2) motivates Paper 3 (L3): the strategies stored in the
  Playbook are written by Paper 3's `StrategyMiner`.
- Paper 3 (L3) motivates Paper 4 (L4): the safety gate of L3 is the
  same gate L4 mutators must satisfy.

We are happy to consider the papers separately if the bundle format is
incompatible with the venue's review process.

## Suggested Reviewers

We suggest reviewers with backgrounds in:

- LLM agents and ReAct-style loops
- Continual learning and catastrophic forgetting
- Programmatic abstractions for ML systems
- AI safety and alignment

Specific names: we leave to the editor''s discretion.

## Reproducibility

- Code: `https://github.com/<org>/agi-research-kit` (or equivalent)
- All experiments reproducible via:
  ```powershell
  .\.venv\Scripts\python.exe -u experiments\full_run3.py --n 50 --retrain-every 10 --tool-factory-every 12 --no-sft
  ```
- Logs, traces, and generation JSONs in `logs/full_run3/`

## Conflicts of Interest

None declared.

## Funding

This work was self-funded as part of an open-source AGI research
initiative.

---

Sincerely,

AGI Research Kit Contributors
agi-research@example.com
2026-07-31