# data/

Committed local artifacts. Not required for `make test`. Not a leaderboard submission.

Approximate checkout size of this directory is 400 MB. Most of that is `data/gaia2/scenarios_local/`, cached GAIA2-mini scenario JSON. Those files are derivatives of an upstream dataset. Their license stays with the upstream release; this tree does not relicense them. They are not model inputs for the unit tests, and storing them here does not mean a GAIA evaluation was run. See `data/gaia2/SCHEMA.md` and `FAILURE_MODES.md`.

`data/controlled_arithmetic_v1.json` is the 30-task manifest (10 dev, 20 test) referenced by the README L1 sentence. The drift check reads this file. Do not edit task text or gold answers to match a write-up.

Other files under `data/` (SFT round dumps, playbook JSONL) are experiment leftovers. Treat them as archives.
