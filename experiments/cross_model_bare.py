"""cross_model_bare.py - bare qwen3:1.7b / llama3.2:1b on the 20-task arithmetic eval.

This is the "no L1-L4" baseline that pairs with cross_model_with_layers.py.
Both scripts use the same TASKS list from cross_model_with_layers.

Usage:
  python experiments/cross_model_bare.py --model qwen3:1.7b --max-steps 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agi_kit.llms.ollama import OllamaBackend
from full_agent import TOOLS as FA_TOOLS, parse_action
from agi_kit.llms.base import LLMMessage, MessageRole

# Same TASKS as cross_model_with_layers.py
TASKS = [
    ("arithmetic", "Use calculator to compute 17 * 23, then final", "391"),
    ("arithmetic", "Use calculator to compute 256 + 789, then final", "1045"),
    ("arithmetic", "Use calculator to compute 2**10, then final", "1024"),
    ("arithmetic", "Use calculator to compute 88 * 88, then final", "7744"),
    ("arithmetic", "Use calculator to compute 100 % 7, then final", "2"),
    ("chained",    "Use calculator to compute 7*6, then echo the result, then final", "42"),
    ("chained",    "Use calculator to compute 1024-256, then echo the result, then final", "768"),
    ("arithmetic", "Use calculator to compute 99*99, then final", "9801"),
    ("arithmetic", "Use calculator to compute 11*11, then final", "121"),
    ("arithmetic", "Use calculator to compute 13*13, then final", "169"),
    ("arithmetic", "Use calculator to compute 1000-1, then final", "999"),
    ("chained",    "Use calculator to compute 3**4, then echo the result, then final", "81"),
    ("arithmetic", "Use calculator to compute 100/4, then final", "25"),
    ("arithmetic", "Use calculator to compute 50*40-100, then final", "1900"),
    ("arithmetic", "Use calculator to compute (15+5)*3, then final", "60"),
    ("arithmetic", "Use calculator to compute 9999 - 1234, then final", "8765"),
    ("arithmetic", "Use calculator to compute 1234 + 5678, then final", "6912"),
    ("arithmetic", "Use calculator to compute 144 / 12, then final", "12"),
    ("chained",    "Use calculator to compute 2**8, then echo the result, then final", "256"),
    ("chained",    "Use calculator to compute 100/4, then echo the result, then final", "25"),
]


def _norm(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace(",", "").replace(" ", "").replace(".", "")
    return s


def bare_run(llm, task, gold, max_steps):
    """No L1-L4. Like cross_model_eval.py's run_task."""
    sys_prompt = (
        "You are a math assistant. Tools: calculator (expr:str). "
        "Output JSON in triple-backtick json block. "
        "Tool call: {\"tool\":\"calculator\",\"args\":{...}}. "
        "Final: {\"final\":\"<answer>\"}."
    )
    history = [
        LLMMessage(role=MessageRole.SYSTEM, content=sys_prompt),
        LLMMessage(role=MessageRole.USER, content=task),
    ]
    for i in range(1, max_steps + 1):
        try:
            resp = llm.chat(history, max_tokens=200, temperature=0.0)
        except Exception as e:
            return {"pred": "[ERR]", "ok": False, "steps": i - 1}
        text = resp.content if hasattr(resp, "content") else str(resp)
        history.append(LLMMessage(role=MessageRole.ASSISTANT, content=text))
        action = parse_action(text)
        if action is None:
            history.append(LLMMessage(role=MessageRole.USER, content="Output JSON please"))
            continue
        if "final" in action:
            pred = str(action["final"])
            return {"pred": pred, "ok": _norm(pred) == _norm(gold), "steps": i}
        tname = action.get("tool")
        targs = action.get("args", {}) or {}
        if tname not in FA_TOOLS:
            obs = "unknown tool " + str(tname)
        else:
            try:
                obs = str(FA_TOOLS[tname]["fn"](**targs))[:500]
            except Exception as e:
                obs = "err: " + str(e)
        history.append(LLMMessage(role=MessageRole.USER, content="Observation: " + str(obs)))
    return {"pred": "[max_steps]", "ok": False, "steps": max_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--out", default="logs/cross_model_bare")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading " + args.model + "...")
    llm = OllamaBackend(model=args.model)

    print("\n=== BARE " + args.model + " on 20 tasks (max_steps=" + str(args.max_steps) + ") ===")
    results = []
    t0 = time.time()
    for i, (cat, q, gold) in enumerate(TASKS):
        r = bare_run(llm, q, gold, args.max_steps)
        results.append({"task": q, "gold": gold, "pred": r["pred"],
                        "ok": r["ok"], "steps": r["steps"]})
        mark = "OK" if r["ok"] else "X "
        print("  [" + str(i + 1) + "/20 " + cat + "] " + mark +
              " steps=" + str(r["steps"]) + " pred=" + str(r["pred"])[:30])
    correct = sum(1 for r in results if r["ok"])
    acc = correct / len(TASKS)
    wall = time.time() - t0

    summary = {
        "model": args.model,
        "max_steps": args.max_steps,
        "n_tasks": len(TASKS),
        "correct": correct,
        "accuracy": round(acc, 3),
        "wall_seconds": round(wall, 1),
        "per_task": results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n=== SUMMARY ===")
    print("Bare " + args.model + ": " + str(correct) + "/" + str(len(TASKS)) +
          " = " + str(acc * 100) + "% (wall=" + str(round(wall, 1)) + "s)")
    print("Wrote " + str(out_dir / "summary.json"))


if __name__ == "__main__":
    main()
