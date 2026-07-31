"""Cross-model evaluation: run the same 20 tasks on multiple models.

Outputs:
  logs/cross_model/results.json   - per-task accuracy per model
  logs/cross_model/summary.md     - human-readable summary
"""
from __future__ import annotations

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

TASKS = [
    ("arithmetic", "Use calculator to compute 17 * 23, then final", "391"),
    ("arithmetic", "Use calculator to compute 256 + 789, then final", "1045"),
    ("arithmetic", "Use calculator to compute 2**10, then final", "1024"),
    ("arithmetic", "Use calculator to compute 88 * 88, then final", "7744"),
    ("arithmetic", "Use calculator to compute 100 % 7, then final", "2"),
    ("chained",    "Use calculator to compute 7*6, then echo the result, then final", "42"),
    ("chained",    "Use calculator to compute 1024-256, then echo, then final", "768"),
    ("arithmetic", "Use calculator to compute 99*99, then final", "9801"),
    ("arithmetic", "Use calculator to compute 11*11, then final", "121"),
    ("arithmetic", "Use calculator to compute 13*13, then final", "169"),
    ("arithmetic", "Use calculator to compute 1000-1, then final", "999"),
    ("chained",    "Use calculator to compute 3**4, then echo, then final", "81"),
    ("arithmetic", "Use calculator to compute 100/4, then final", "25"),
    ("arithmetic", "Use calculator to compute 50*40-100, then final", "1900"),
    ("arithmetic", "Use calculator to compute (15+5)*3, then final", "60"),
    ("arithmetic", "Use calculator to compute 9999 - 1234, then final", "8765"),
    ("arithmetic", "Use calculator to compute 1234 + 5678, then final", "6912"),
    ("arithmetic", "Use calculator to compute 144 / 12, then final", "12"),
    ("chained",    "Use calculator to compute 2**8, then echo, then final", "256"),
    ("chained",    "Use calculator to compute 100/4, then echo, then final", "25"),
]

MODELS = [
    ("qwen3:1.7b",   "Qwen3 family, 2.0B params (Q4_K_M)"),
    ("llama3.2:1b",  "LLaMA 3.2 family, 1.2B params (Q8_0)"),
    ("qwen2.5:3b",   "Qwen2.5 family, 3.1B params (Q4_K_M)"),
    ("qwen3:0.6b",   "Qwen3 family, 0.75B params (Q4_K_M)"),
]


def _norm(s):
    if s is None: return ""
    s = str(s).strip().lower()
    s = s.replace(",", "").replace(" ", "").replace(".", "")
    return s


def run_task(llm, task, gold, max_steps=3):

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
            return {"pred": "[ERR] " + str(e), "ok": False, "steps": i-1}
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
    out_dir = ROOT / "logs" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_name, model_desc in MODELS:
        print(f"\n=== {model_name} ({model_desc}) ===")
        try:
            llm = OllamaBackend(model=model_name)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue
        per_task = []
        for i, (cat, q, gold) in enumerate(TASKS):
            t0 = time.time()
            r = run_task(llm, q, gold)
            dt = round(time.time() - t0, 1)
            mark = "OK" if r["ok"] else "X "
            print(f"  [{i+1:2d}/{len(TASKS)} {cat:10s}] {mark} steps={r['steps']} dt={dt}s  pred={str(r['pred'])[:30]}")
            per_task.append({"category": cat, "question": q, "gold": gold,
                            "pred": r["pred"], "ok": r["ok"],
                            "steps": r["steps"], "seconds": dt})
        correct = sum(1 for r in per_task if r["ok"])
        all_results[model_name] = {
            "description": model_desc,
            "n_tasks": len(TASKS),
            "n_correct": correct,
            "accuracy": round(correct / len(TASKS), 3),
            "avg_seconds": round(statistics.mean(r["seconds"] for r in per_task), 2),
            "per_task": per_task,
        }
        print(f"  >>> {model_name}: {correct}/{len(TASKS)} = {correct/len(TASKS)*100:.1f}%")

    # Save JSON
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    # Summary MD
    lines = ["# Cross-Model Evaluation\n"]
    lines.append("Same 20 arithmetic tasks on multiple Ollama models.\n")
    lines.append("| Model | Description | Correct | Accuracy | Avg sec/q |")
    lines.append("|---|---|---:|---:|---:|")
    for model_name, r in all_results.items():
        lines.append(f"| `{model_name}` | {r['description']} | {r['n_correct']}/{r['n_tasks']} | {r['accuracy']*100:.1f}% | {r['avg_seconds']} |")
    lines.append("")
    lines.append("## Per-task results\n")
    lines.append("| # | Category | Q | Gold |")
    lines.append("|---:|---|---|---|")
    for i, (cat, q, gold) in enumerate(TASKS):
        q_short = q[:60] + ("..." if len(q) > 60 else "")
        lines.append(f"| {i+1} | {cat} | {q_short} | {gold} |")
    lines.append("")
    lines.append("## Per-model per-task results\n")
    lines.append("| Task # | " + " | ".join(f"`{m}`" for m in all_results.keys()) + " |")
    lines.append("|---:|" + "---:|" * len(all_results) + "|")
    for i in range(len(TASKS)):
        row = [str(i+1)]
        for model_name, r in all_results.items():
            t = r["per_task"][i]
            mark = "✓" if t["ok"] else "✗"
            row.append(f"{mark} ({t['steps']}s)")
        lines.append("| " + " | ".join(row) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote:")
    print(f"  {out_dir / 'results.json'}")
    print(f"  {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()