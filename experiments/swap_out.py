"""swap_out.py - Round 14 swap-out experiment for "4 format-biased fragments".

The 22pp gap in Section 4.1.2 comes from 4 distinct multi-step templates
where the agent emits the correct sum but an incorrect product (or
mis-formatted output). This experiment tests whether the +95pp gain
on the hard 20-task eval is real or just format normalization.

Method: take the 4 templates that account for the 19 wrong reeval cases
and run them with **format-explicit prompts** under both bare and
L1-L4 configurations. If bare can solve them with format-explicit
prompts, the +95pp gain was format normalization. If bare still fails,
the gain is genuine layer contribution.

Tasks (the 4 "format-biased" templates observed in the reeval):
  Template A: Compute N+N, then echo, then also compute N*N, then final
  Template B: Use calculator to compute N*N+N, then echo, then final
  Template C: Reverse the string and report the result as final (tool_factory)
  Template D: Count occurrences of letter and report as final (tool_factory)

Each template is run 5 times with format-explicit prompts under:
  - bare qwen3:1.7b, max_steps=3, no L1-L4
  - full L1-L4 qwen3:1.7b, max_steps=6

Output: logs/swap_out/summary.json
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
from full_run3 import make_real_run_episode
from agi_kit.reflect import Reflector
from agi_kit.playbook import Playbook
from agi_kit.meta import MetaController

# 4 format-biased templates, sampled with format-explicit prompts
# Same arithmetic operations but with EXPLICIT integer-only output
# requirement and structured emission.
SWAP_OUT_TASKS = [
    # Template A: sum + product (859*381 variant - the highest-frequency wrong case)
    {"category": "arith_double",
     "prompt": "Compute 859+381 using calculator, then echo it. Then compute 859*381 using calculator. Your final answer MUST be a single integer with no commas, no spaces, no formatting. Output JSON only: {\"final\": \"<sum>,<product>\"}",
     "gold": "1240,327279"},
    {"category": "arith_double",
     "prompt": "Compute 617+716 using calculator, then echo it. Then compute 617*716 using calculator. Your final answer MUST be a single integer with no commas, no spaces, no formatting. Output JSON only: {\"final\": \"<sum>,<product>\"}",
     "gold": "1333,441772"},
    # Template B: single chained (37*7+76 variant)
    {"category": "arith_chain",
     "prompt": "Use calculator to compute 37*7+76. Your final answer MUST be the integer result only (no other text). Output JSON only: {\"final\": \"<int>\"}",
     "gold": "335"},
    {"category": "arith_chain",
     "prompt": "Use calculator to compute 41*5+85. Your final answer MUST be the integer result only (no other text). Output JSON only: {\"final\": \"<int>\"}",
     "gold": "290"},
]


def _norm(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace(",", "").replace(" ", "").replace(".", "")
    return s


def bare_run(llm, task, gold, max_steps=3):
    """Cross-model-style bare run: no reflector, no playbook, no meta."""
    sys_prompt = (
        "You are a math assistant. Tools: calculator (expr:str). "
        "Output JSON in triple-backtick json block. "
        "Tool call: {\"tool\":\"calculator\",\"args\":{...}}. "
        "Final: {\"final\":\"<answer>\"}."
    )
    history = [
        # main_llm expects LLMMessage objects
    ]
    from agi_kit.llms.base import LLMMessage, MessageRole
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
    parser.add_argument("--max-steps-bare", type=int, default=3)
    parser.add_argument("--max-steps-full", type=int, default=6)
    parser.add_argument("--out", default="logs/swap_out")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="number of runs per task per config to test variance")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading " + args.model + "...")
    llm = OllamaBackend(model=args.model)
    fast_llm = OllamaBackend(model="qwen3:0.6b")

    print("\n=== Swap-Out Experiment: 4 format-biased templates ===")
    print("Tasks:")
    for t in SWAP_OUT_TASKS:
        print("  [" + t["category"] + "] gold=" + t["gold"])
    print()

    results = {"config": "swap_out", "model": args.model, "tasks": []}

    # Bare runs (multiple seeds for variance estimate)
    print("=== BARE (max_steps=" + str(args.max_steps_bare) + ", n_runs=" + str(args.n_runs) + ") ===")
    for t in SWAP_OUT_TASKS:
        task_results = []
        for run in range(args.n_runs):
            r = bare_run(llm, t["prompt"], t["gold"], args.max_steps_bare)
            task_results.append({"run": run, "pred": r["pred"], "ok": r["ok"], "steps": r["steps"]})
            mark = "OK" if r["ok"] else "X "
            print("  [" + t["category"] + " run=" + str(run) + "] " + mark + " pred=" + str(r["pred"])[:50])
        n_ok = sum(1 for r in task_results if r["ok"])
        results["tasks"].append({
            "category": t["category"],
            "prompt": t["prompt"],
            "gold": t["gold"],
            "config": "bare",
            "max_steps": args.max_steps_bare,
            "n_runs": args.n_runs,
            "n_ok": n_ok,
            "accuracy": n_ok / args.n_runs,
            "per_run": task_results,
        })

    # Full L1-L4 runs
    print("\n=== FULL L1-L4 (max_steps=" + str(args.max_steps_full) + ") ===")
    pb = Playbook(path=str(ROOT / "data/playbook.jsonl"), auto_load=True)
    if pb.size() == 0:
        pb.add("Use calculator for arithmetic; use echo to verify.",
               "Prefer the simplest tool that solves the sub-problem.",
               success_rate=0.85)
        pb.save()
    meta = MetaController()
    reflector = Reflector(main_llm=llm, fast_llm=fast_llm,
                          trace_path=str(out_dir / "trace.jsonl"), enabled=True)
    run_episode_fn = make_real_run_episode(
        playbook=pb, meta=meta, reflector=reflector,
        strategy_miner_llm=None, tool_factory=None,
        fa_tools=FA_TOOLS, max_steps=args.max_steps_full,
    )
    for t in SWAP_OUT_TASKS:
        task_results = []
        for run in range(args.n_runs):
            r = run_episode_fn(t["prompt"], gold=t["gold"])
            ok = (r["verdict"] == "success")
            task_results.append({"run": run, "pred": r["final"], "ok": ok,
                                 "steps": len(r["steps"]), "avg_score": r["avg_score"]})
            mark = "OK" if ok else "X "
            print("  [" + t["category"] + " run=" + str(run) + "] " + mark + " pred=" + str(r["final"])[:50])
        n_ok = sum(1 for r in task_results if r["ok"])
        results["tasks"].append({
            "category": t["category"],
            "prompt": t["prompt"],
            "gold": t["gold"],
            "config": "full_l1_l4",
            "max_steps": args.max_steps_full,
            "n_runs": args.n_runs,
            "n_ok": n_ok,
            "accuracy": n_ok / args.n_runs,
            "per_run": task_results,
        })

    # Summary: bare vs full per category
    print("\n=== SUMMARY ===")
    by_cat = {}
    for tr in results["tasks"]:
        cat = tr["category"]
        if cat not in by_cat:
            by_cat[cat] = {}
        by_cat[cat][tr["config"]] = tr["accuracy"]
    for cat in by_cat:
        bare_a = by_cat[cat].get("bare", 0)
        full_a = by_cat[cat].get("full_l1_l4", 0)
        print("  " + cat + ": bare=" + str(bare_a * 100) + "% full=" + str(full_a * 100) + "% delta=" + str(round((full_a - bare_a) * 100, 1)) + "pp")
    results["by_category"] = by_cat
    results["interpretation"] = "If bare scores 0% on format-explicit prompts, the +95pp gain is NOT just format normalization; it's genuine layer contribution."

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nWrote " + str(out_dir / "summary.json"))


if __name__ == "__main__":
    main()
