"""Summarize a completed paired controlled-arithmetic run without rerunning it."""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def two_sided_exact_binomial(wins: int, losses: int) -> float:
    """Exact paired sign/McNemar p-value for discordant outcomes."""
    discordant = wins + losses
    if not discordant:
        return 1.0
    lower = min(wins, losses)
    tail = sum(comb(discordant, k) for k in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = (ROOT / args.run_dir).resolve()
    rows = [json.loads(line) for line in (run_dir / "per_task.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    static = {row["task_id"]: row for row in rows if row["config"] == "static"}
    l1 = {row["task_id"]: row for row in rows if row["config"] == "l1_reflection"}
    if set(static) != set(l1):
        raise SystemExit("paired task ids differ between configurations")
    wins = sum(not static[key]["assessment"]["normalized_correct"] and l1[key]["assessment"]["normalized_correct"] for key in static)
    losses = sum(static[key]["assessment"]["normalized_correct"] and not l1[key]["assessment"]["normalized_correct"] for key in static)
    activations = sum(bool(row.get("reflection_retry_used")) for row in l1.values())
    causal_usable = activations > 0 or (wins + losses) == 0
    analysis = {
        "n_pairs": len(static),
        "l1_wins": wins,
        "l1_losses": losses,
        "discordant_pairs": wins + losses,
        "two_sided_exact_mcnemar_p": two_sided_exact_binomial(wins, losses),
        "l1_activations": activations,
        "causal_interpretation_usable": causal_usable,
        "interpretation": (
            "Descriptive only unless the exact paired result is adequately powered and statistically informative."
            if causal_usable else
            "Not causally interpretable: outcomes changed although the L1 intervention never activated; investigate model-service non-determinism."
        ),
    }
    (run_dir / "paired_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    markdown = ["# Controlled Arithmetic Paired Analysis", "", "- Pairs: " + str(analysis["n_pairs"]), "- L1 wins/losses: " + str(wins) + "/" + str(losses), "- Discordant pairs: " + str(analysis["discordant_pairs"]), "- L1 activations: " + str(activations), "- Two-sided exact McNemar/binomial p: " + str(analysis["two_sided_exact_mcnemar_p"]), "", "Interpretation: " + analysis["interpretation"]]
    (run_dir / "paired_analysis.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
