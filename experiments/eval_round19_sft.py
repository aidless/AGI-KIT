"""Small held-out CPU evaluation for the Round 19 SFT candidate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from full_agent import Agent, TransformersBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    tasks = json.loads((ROOT / "data/sft_round19/test.json").read_text(encoding="utf-8"))[:args.limit]
    rows = []
    model = TransformersBackend(args.model)
    for task in tasks:
        answer = Agent(llm=model, max_steps=4, max_tokens=128).run("Use calculator to compute " + task["expression"] + ". Return the result.")
        rows.append({"id": task["id"], "gold": task["gold"], "pred": str(answer), "correct": str(answer).strip() == task["gold"]})
    out = ROOT / "logs/sft_round19"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "label": args.label, "n": len(rows), "correct": sum(x["correct"] for x in rows), "rows": rows}
    (out / (args.label + ".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
