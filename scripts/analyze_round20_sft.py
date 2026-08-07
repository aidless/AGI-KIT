"""Merge the frozen Round 20 SFT evaluation without overstating deployment status."""
from __future__ import annotations

import json
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "logs" / "sft_round20" / name).read_text(encoding="utf-8"))


def main() -> None:
    candidate = load("candidate120e3_fp16_retry_test80.json")
    base_rows = []
    for index in range(1, 5):
        base_rows.extend(load(f"base_test80_part0{index}.json")["rows"])
    assert [row["id"] for row in candidate["rows"]] == [row["id"] for row in base_rows]
    wins = sum(c["correct"] and not b["correct"] for c, b in zip(candidate["rows"], base_rows))
    losses = sum(b["correct"] and not c["correct"] for c, b in zip(candidate["rows"], base_rows))
    discordant = wins + losses
    p_value = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    decision = {
        "round": 20, "protocol": "frozen 80-task arithmetic tool-use test; same Agent prompt, parser, and 3-step budget",
        "candidate": {"backend": "transformers", "correct": candidate["correct"], "n": candidate["n"]},
        "base": {"backend": "transformers", "correct": sum(row["correct"] for row in base_rows), "n": len(base_rows)},
        "paired": {"candidate_wins": wins, "candidate_losses": losses, "two_sided_exact_mcnemar_p": p_value},
        "deployment_decision": "not_accepted",
        "reason": "candidate is executable through Transformers but has not passed the required Ollama deployment path; do not treat this as a deployed continual-learning update",
    }
    (ROOT / "logs" / "sft_round20" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
