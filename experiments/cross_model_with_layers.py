"""cross_model_with_layers.py - 20-task arithmetic eval WITH the full pipeline.

Compares qwen3:1.7b on 20 hard arithmetic tasks:
  - bare (no layers)        -> like cross_model_eval.py baseline (already done: 5%)
  - full L1-L4              -> like full_run3

If layers matter, we expect L1-L4 to improve over bare.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agi_kit.llms.ollama import OllamaBackend
from full_agent import TOOLS as FA_TOOLS
from full_run3 import make_real_run_episode
from agi_kit.reflect import Reflector
from agi_kit.playbook import Playbook
from agi_kit.meta import MetaController

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:1.7b",
                        help="primary model (also used as scorer if --scorer not set)")
    parser.add_argument("--scorer", default="qwen3:0.6b",
                        help="scorer model for L1 Reflector")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--out", default="logs/cross_model_layers",
                        help="output directory for summary.json")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    print("Loading " + model_name + "...")
    llm = OllamaBackend(model=model_name)
    fast_llm = OllamaBackend(model=args.scorer)

    # Full pipeline run
    print("\n=== FULL L1-L4 on 20 hard arithmetic tasks ===")
    reflector = Reflector(
        main_llm=llm, fast_llm=fast_llm,
        trace_path=str(out_dir / "trace_full.jsonl"),
        enabled=True,
    )
    pb = Playbook(path=str(ROOT / "data/playbook.jsonl"), auto_load=True)
    if pb.size() == 0:
        pb.add("Use calculator for arithmetic; use echo to verify.",
               "Prefer the simplest tool that solves the sub-problem.",
               success_rate=0.85)
        pb.save()
    meta = MetaController()
    run_episode_fn = make_real_run_episode(
        playbook=pb, meta=meta, reflector=reflector,
        strategy_miner_llm=None, tool_factory=None,
        fa_tools=FA_TOOLS, max_steps=args.max_steps,
    )

    full_results = []
    t0 = time.time()
    for i, (cat, q, gold) in enumerate(TASKS):
        r = run_episode_fn(q, gold=gold)
        ok = (r["verdict"] == "success")
        full_results.append({"task": q, "gold": gold, "pred": r["final"],
                            "ok": ok, "steps": len(r["steps"]), "avg_score": r["avg_score"]})
        mark = "OK" if ok else "X "
        print("  [" + str(i + 1) + "/20 " + cat + "] " + mark +
              " steps=" + str(len(r["steps"])) +
              " verdict=" + r["verdict"] +
              " pred=" + str(r["final"])[:30])
    full_correct = sum(1 for r in full_results if r["ok"])
    full_acc = full_correct / len(TASKS)
    full_wall = time.time() - t0

    # Compare to known bare baseline from logs/cross_model/results.json (qwen3:1.7b = 5%)
    bare_acc = 0.05
    bare_correct = 1

    summary = {
        "model": model_name,
        "scorer_model": args.scorer,
        "max_steps": args.max_steps,
        "n_tasks": len(TASKS),
        "bare_known_from_logs_cross_model": {
            "correct": bare_correct,
            "accuracy": bare_acc,
            "source": "logs/cross_model/results.json (no L1-L4, max_steps=3)",
        },
        "full_l1_l4": {
            "correct": full_correct,
            "accuracy": round(full_acc, 3),
            "wall_seconds": round(full_wall, 1),
            "per_task": full_results,
        },
        "delta_pp": round((full_acc - bare_acc) * 100, 1),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Wrote " + str(out_dir / "summary.json"))

    print("\n=== SUMMARY ===")
    print("Bare (known, no layers):  " + str(bare_correct) + "/" + str(len(TASKS)) + " = " + str(bare_acc * 100) + "%")
    print("Full L1-L4 (max_steps=6): " + str(full_correct) + "/" + str(len(TASKS)) + " = " + str(full_acc * 100) + "%")
    print("Delta:                    " + str(summary["delta_pp"]) + " percentage points")
    print("Wrote " + str(out_dir / "summary.json"))


if __name__ == "__main__":
    main()
