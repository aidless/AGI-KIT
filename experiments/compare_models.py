"""compare_models.py - Compare two models on a fixed task set

Runs the same 20 tasks through both models, computes accuracy.
"""
from __future__ import annotations
import argparse, json, sys, time, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from full_agent import Agent, OllamaBackend

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# 20 个能力测试题(gold = 标准答案)
TASKS = [
    ("arithmetic", "Use calculator to compute 17 * 23, then final", "391"),
    ("arithmetic", "Use calculator to compute 256 + 789, then final", "1045"),
    ("arithmetic", "Use calculator to compute 9999 - 1234, then final", "8765"),
    ("arithmetic", "Use calculator to compute 144 / 12, then final", "12.0"),
    ("arithmetic", "Use calculator to compute 2**10, then final", "1024"),
    ("arithmetic", "Use calculator to compute 100 % 7, then final", "2"),
    ("arithmetic", "Use calculator to compute (15+5)*3, then final", "60"),
    ("arithmetic", "Use calculator to compute 50 * 40 - 100, then final", "1900"),
    ("arithmetic", "Use calculator to compute 1234 + 5678, then final", "6912"),
    ("arithmetic", "Use calculator to compute 88 * 88, then final", "7744"),
    ("chained",    "Use calculator to compute 2**8, then echo the result, then final", "256"),
    ("chained",    "Use calculator to compute 100/4, then echo the result, then final", "25.0"),
    ("chained",    "Use calculator to compute 7*6, then echo the result, then final", "42"),
    ("chained",    "Use calculator to compute 3**4, then echo the result, then final", "81"),
    ("chained",    "Use calculator to compute 1024-256, then echo the result, then final", "768"),
    ("arithmetic", "Use calculator to compute 5!**+0, then final", "120"),  # fact 5!=120
    ("arithmetic", "Use calculator to compute 11*11, then final", "121"),
    ("arithmetic", "Use calculator to compute 13*13, then final", "169"),
    ("arithmetic", "Use calculator to compute 99*99, then final", "9801"),
    ("arithmetic", "Use calculator to compute 1000-1, then final", "999"),
]


def normalize(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace(",", "")
    s = s.replace(" ", "")
    return s


def run_one(llm, task, gold, max_steps=4):
    agent = Agent(llm=llm, max_steps=max_steps)
    t0 = time.time()
    try:
        pred = agent.run(task)
    except Exception as e:
        pred = "[ERR] " + str(e)
    dt = time.time() - t0
    ok = normalize(pred) == normalize(gold)
    return {"pred": pred, "gold": gold, "ok": ok, "seconds": round(dt, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen3:0.6b", "qwen3:1.7b"])
    p.add_argument("--out", default=None)
    p.add_argument("--max-tasks", type=int, default=20)
    args = p.parse_args()

    out = args.out or str(LOG_DIR / ("compare-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"))
    print("[compare] models:", args.models)
    print("[compare] tasks:", min(args.max_tasks, len(TASKS)))
    print("[compare] log:", out)

    summary = {}
    for model in args.models:
        print("\n========== model:", model, "==========")
        llm = OllamaBackend(model=model)
        results = []
        for i, (cat, q, gold) in enumerate(TASKS[:args.max_tasks]):
            print("\n[" + str(i+1) + "/" + str(min(args.max_tasks, len(TASKS))) + " " + cat + "] " + q.encode("ascii", "replace").decode("ascii"))
            r = run_one(llm, q, gold)
            r["task"] = q
            r["category"] = cat
            r["model"] = model
            results.append(r)
            mark = "OK" if r["ok"] else "X "
            print("  " + mark + " pred=" + str(r["pred"])[:60].encode("ascii", "replace").decode("ascii") + "  gold=" + gold)
        correct = sum(1 for r in results if r["ok"])
        total = len(results)
        avg_sec = sum(r["seconds"] for r in results) / max(total, 1)
        summary[model] = {
            "correct": correct,
            "total": total,
            "acc": round(correct / max(total, 1) * 100, 1),
            "avg_seconds": round(avg_sec, 2),
            "results": results,
        }
        print("\n[summary " + model + "] " + str(correct) + "/" + str(total) + " = " + str(summary[model]["acc"]) + "%, avg " + str(summary[model]["avg_seconds"]) + "s/q")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== FINAL COMPARISON ==========")
    print("Model".ljust(20) + "Acc".ljust(10) + "AvgSec".ljust(10))
    for m, s in summary.items():
        print(m.ljust(20) + (str(s["acc"]) + "%").ljust(10) + str(s["avg_seconds"]).ljust(10))
    print("\nFull results saved to:", out)


if __name__ == "__main__":
    main()