"""Create a deterministic, deployment-aligned arithmetic tool-use SFT corpus."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from full_agent import render_system_prompt


def make_task(index: int, *, split: str) -> dict:
    """Generate a unique expression from a split-specific seeded stream."""
    seeds = {"train": 20260820, "dev": 20260821, "test": 20260822}
    rng = random.Random(seeds[split] + index)
    a, b, c = rng.randint(11, 997), rng.randint(3, 97), rng.randint(5, 499)
    op = index % 4
    if op == 0:
        expr, answer = f"({a}*{b})+{c}", str(a * b + c)
    elif op == 1:
        expr, answer = f"({a}+{b})*{c}", str((a + b) * c)
    elif op == 2:
        expr, answer = f"({a}*{b})-{c}", str(a * b - c)
    else:
        divisor = rng.randint(2, 31)
        value = divisor * rng.randint(11, 301)
        expr, answer = f"({value}/{divisor})+{c}", str(value // divisor + c)
    task = f"Use calculator to compute {expr}. Return the result."
    tool_call = json.dumps({"tool": "calculator", "args": {"expr": expr}}, separators=(",", ":"))
    final = json.dumps({"final": answer}, separators=(",", ":"))
    return {
        "id": f"r20-{split}-{index:04d}", "expression": expr, "gold": answer,
        "messages": [
            {"role": "system", "content": render_system_prompt()},
            {"role": "user", "content": task},
            {"role": "assistant", "content": "```json\n" + tool_call + "\n```"},
            {"role": "user", "content": "Observation: " + answer},
            {"role": "assistant", "content": "```json\n" + final + "\n```"},
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")


def main() -> None:
    out = ROOT / "data" / "sft_round20"
    out.mkdir(parents=True, exist_ok=True)
    train = [make_task(index, split="train") for index in range(480)]
    dev = [make_task(index, split="dev") for index in range(20)]
    test = [make_task(index, split="test") for index in range(80)]
    train_expressions = {row["expression"] for row in train}
    assert train_expressions.isdisjoint(row["expression"] for row in dev + test)
    assert {row["expression"] for row in dev}.isdisjoint(row["expression"] for row in test)
    write_jsonl(out / "train.jsonl", train)
    (out / "dev.json").write_text(json.dumps(dev, indent=2), encoding="utf-8")
    (out / "test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "round": 20, "seed_train": 20260820, "seed_dev": 20260821, "seed_test": 20260822,
        "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
        "deployment_prompt": "experiments.full_agent.render_system_prompt",
        "disjoint_expression_sets": True,
    }, indent=2), encoding="utf-8")
    print(f"wrote {len(train)} train, {len(dev)} dev, and {len(test)} held-out tasks")


if __name__ == "__main__":
    main()
