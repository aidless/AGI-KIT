# Failure modes

Known limits of this kit. Numbers below are checked against committed traces by `tests/unit/test_l1_readme_drift.py`. They are not new measurements.

## Cherry-picked discordant pair

The README L1 sentence refers to one run directory:

`logs/controlled_arithmetic/run-llama3b-20260802`

Model `llama3.2:3b-instruct-q8_0`, seed 20260802, split `test`, 20 tasks drawn from `data/controlled_arithmetic_v1.json` (30 tasks: 10 dev, 20 test). On `assessment.normalized_correct`, Static is 19/20 and L1 is 20/20. `summary.json` stores those same integers in `configs.*.correct`.

The only discordant task is `ca020`. Static's final is `9003`; the gold answer is `9001`; L1 used one reflection retry and finished at `9001`. `paired_analysis.json` records `discordant_pairs` 1, `l1_wins` 1, `l1_losses` 0, and `two_sided_exact_mcnemar_p` 1.0. That recorded p-value is the exact test on a single discordant pair. It is not a significance claim and it does not support general superiority.

Two stricter columns on the same rows are weaker than the headline:

- `assessment.strict_correct` is 17/20 Static and 18/20 L1.
- `summary.json` `strict_fenced_completed` is 0 for both arms (no row passes `strict_assessment`).

A second committed test-split run on the same manifest does not show an L1 gain. `logs/controlled_arithmetic/run-qwen0.6b-20260802` (`qwen3:0.6b`, seed 20260802) is Static 9/20 and L1 6/20 on `normalized_correct`. Quoting only the llama pair drops this run.

<!-- l1-other-run
run: logs/controlled_arithmetic/run-qwen0.6b-20260802
static_normalized_correct: 9
l1_normalized_correct: 6
-->

## Schema mutation risk

`SchemaMutator.propose` (`src/agi_kit/recursive.py`) writes the new value onto the live config before any eval. If `eval_fn` is omitted, the change is kept and logged as `applied_no_eval`. Bounds block unknown fields and out-of-range numbers; they do not require a held-out check.

`ToolFactory` compiles model-written Python and runs it with `exec` inside the process after an AST allowlist. The allowlist is not a sandbox. A synthesized callable still shares the interpreter with the agent. History is a local JSONL file, not an external approval log.

## GAIA scaffold emptiness

`experiments/eval_gaia2.py` is the documented GAIA / GAIA2 entry point. It loads a Hugging Face dataset (or a single manual string) and scores the local ReAct agent, whose tools are calculator, files, shell, and RAG. It does not drive the GAIA2 app universe and it does not report a canonical pass rate. No GAIA leaderboard number is claimed.

`src/agi_kit/gaia2_apps.py` and `experiments/eval_gaia2_simulator.py` are a separate, partial adapter. Unit tests cover parsers and a few scripted actions. They are not a GAIA result. `data/gaia2/` is a large cached scenario dump; see `data/README.md` and `data/gaia2/SCHEMA.md`. The original schema note still holds for the public entry point: the kit's default tool list does not answer GAIA2 app calls.

`tests/test_c0a_*.py` import Meta's `are` package. They are not part of `make test` or CI.
