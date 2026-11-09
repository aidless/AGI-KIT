# AGI Research Kit

**STATUS: WORK-IN-PROGRESS — not A yet.** Gap list is below. This repository is not a grade-A result.

CPU tool-use research prototype for reflection, strategy memory, a continual-learning safety gate, and bounded schema mutation.
The controlled L1 result is one auditable correction (Static 19/20, L1 20/20 on the held-out half of a 30-task manifest), not general superiority.
The manuscript is an unsubmitted draft. Withdrawn, and not results: a `p<0.01` claim, a 30% static baseline, and an internal 4.5 score.

Those 19/20 and 20/20 figures are `normalized_correct` on `logs/controlled_arithmetic/run-llama3b-20260802` (`summary.json` field `correct`). Stricter readings of the same trace, and a second committed run that does not show an L1 gain, are in [FAILURE_MODES.md](FAILURE_MODES.md).

<!-- l1-evidence
run: logs/controlled_arithmetic/run-llama3b-20260802
manifest: data/controlled_arithmetic_v1.json
static_normalized_correct: 19
l1_normalized_correct: 20
held_out_tasks: 20
manifest_tasks: 30
discordant_pairs: 1
recorded_mcnemar_p: 1.0
-->

Source code is Apache-2.0. Model weights and external datasets keep their own licenses. The unified manuscript (`papers/preprint_unified_en.md`) has not been submitted. See `papers/PUBLISHING.md`.

## Offline tests

Unit tests do not download models or `data/`. From a fresh clone:

```bash
pip install -r requirements-test.txt
make test
```

`make test` runs `pytest tests/unit`. That is the CI gate (`.github/workflows/ci.yml`). `requirements.txt` and `pip install -e .` pull Torch and other live-stack packages; they are not required for this gate. `tests/test_c0a_*.py` import Meta's `are` package and are outside `make test`.

## Live experiments

Scripts under `experiments/` call a local model (Ollama or Transformers) or a dataset download. They are not part of `make test`, and their logs are not a substitute for the unit tests. See [experiments/README.md](experiments/README.md). Model downloads are optional and large (tens of GB). `scripts/setup_env.ps1` is a Windows helper for that live stack, not the offline test path.

## Large data

`data/` is committed and about 400 MB, mostly cached GAIA2-mini scenario JSON. It is not required to run unit tests and it is not a GAIA score. See [data/README.md](data/README.md).

## Failure modes

[FAILURE_MODES.md](FAILURE_MODES.md): the cited L1 pair is one discordant task, schema mutation can apply with no eval, and the GAIA2 entry-point scaffold does not score the benchmark.

## Gaps (why this stays work-in-progress)

- L1 evidence is one 20-task held-out split of a 30-task manifest, one discordant pair, recorded paired p=1.0. That is not a general superiority result.
- The same manifest's committed `qwen3:0.6b` run does not replicate an L1 gain.
- The tree is large: many experiment scripts, papers, and a committed `data/` cache. Canonical GAIA2 scoring is not what `experiments/eval_gaia2.py` does.
- Schema and tool mutation are bounded in code and still unsafe to treat as a closed safety case (`applied_no_eval`, in-process `exec`).
- Archived logs keep the workstation paths written at run time. They are records, not setup instructions.
