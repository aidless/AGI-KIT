# experiments/

Live scripts. They are not `make test` and they are not the CI gate.

Running them needs a local model server (Ollama), a Transformers/Torch install, a dataset download, or all three. Outputs under `logs/` are run records. They do not replace `tests/unit`.

The controlled L1 numbers in the README come from the already committed trace `logs/controlled_arithmetic/run-llama3b-20260802`, produced by `controlled_arithmetic_eval.py`. Re-running that script is a new live eval. It does not rewrite the committed trace unless you point it at that directory yourself.

`eval_gaia2.py` is a scaffold over the local ReAct agent. It does not score canonical GAIA2. See `FAILURE_MODES.md`.
