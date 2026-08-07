"""Create a deterministic tool-use SFT corpus with a disjoint held-out split."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are a tool-use assistant. Output exactly one fenced JSON object. "
    "Use calculator with {\"tool\":\"calculator\",\"args\":{\"expr\":\"...\"}}, "
    "then return {\"final\":\"<answer>\"}."
)


def example(index: int) -> dict:
    a = 11 + (index * 7) % 89
    b = 3 + (index * 11) % 47
    c = 5 + (index * 13) % 71
    expr = f"({a}*{b})+{c}"
    answer = str(a * b + c)
    return {"id": f"r19-{index:03d}", "expression": expr, "gold": answer, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Use calculator to compute " + expr + ". Return the result."},
        {"role": "assistant", "content": "```json\n{\"tool\":\"calculator\",\"args\":{\"expr\":\"" + expr + "\"}}\n```"},
        {"role": "user", "content": "Observation: " + answer},
        {"role": "assistant", "content": "```json\n{\"final\":\"" + answer + "\"}\n```"},
    ]}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")


def main() -> None:
    out = ROOT / "data" / "sft_round19"
    out.mkdir(parents=True, exist_ok=True)
    rows = [example(i) for i in range(160)]
    train, test = rows[:120], rows[120:]
    write_jsonl(out / "train.jsonl", train)
    (out / "test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"seed": 19, "train": [r["id"] for r in train], "test": [r["id"] for r in test]}, indent=2), encoding="utf-8")
    print("wrote", len(train), "train and", len(test), "held-out test tasks")


if __name__ == "__main__":
    main()
