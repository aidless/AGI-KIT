"""Evaluate an SFT candidate using the exact deployment Agent prompt and parser."""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from full_agent import Agent, TransformersBackend


def normalized_equal(prediction: object, gold: object) -> bool:
    """Compare scalar finals without treating formatting as an arithmetic error."""
    left, right = str(prediction).strip(), str(gold).strip()
    if left == right:
        return True
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    tasks = json.loads((ROOT / "data" / "sft_round20" / (args.split + ".json")).read_text(encoding="utf-8"))[args.start:args.start + args.limit]
    model = TransformersBackend(args.model)
    rows = []
    for task in tasks:
        prediction = Agent(llm=model, max_steps=3, max_tokens=80).run(
            "Use calculator to compute " + task["expression"] + ". Return the result."
        )
        rows.append({"id": task["id"], "gold": task["gold"], "pred": str(prediction),
                     "correct": normalized_equal(prediction, task["gold"])})
    summary = {"round": 20, "split": args.split, "start": args.start, "model": args.model, "label": args.label, "n": len(rows),
               "correct": sum(row["correct"] for row in rows), "rows": rows}
    out = ROOT / "logs" / "sft_round20" / (args.label + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("round", "model", "label", "n", "correct")}, indent=2))


if __name__ == "__main__":
    main()
