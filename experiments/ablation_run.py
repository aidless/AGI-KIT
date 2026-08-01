"""ablation_run.py - Real layer ablation for AGI Kit.

Disables L1-L4 individually and runs the same task set, producing
empirical numbers for the layer ablation table in the paper.

Configs (each --out goes to a separate log dir):
  static    -> no layers (no L1, no L2, no L3, no L4)
  l1_only   -> L1 only
  l1_l2     -> L1+L2
  l1_l2_l3  -> L1+L2+L3
  full      -> all four layers

Usage:
  .venv\Scripts\python.exe experiments\ablation_run.py --config static --n 16
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
from agi_kit.reflect import Reflector
from agi_kit.playbook import Playbook
from agi_kit.meta import MetaController
from agi_kit.loop import ContinualLoop, ExperienceBuffer
from agi_kit.gaia2_tasks import load_gaia2_tasks
from agi_kit.evals_arith import make_eval_fn
from agi_kit.real_retrain_v2 import mock_retrain_v2
from agi_kit.recursive import SchemaMutator, MetaControllerConfig, ToolFactory
from agi_kit.tool_factory_tasks import get_trigger_tasks
from full_agent import TOOLS as FA_TOOLS

from full_run3 import make_real_run_episode


def run_ablation(args):
    out_path = ROOT / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    # Build the same task set every config uses
    gaia_tasks = load_gaia2_tasks(n=args.n, prefer_real=False)
    trigger_tasks = get_trigger_tasks(n=max(args.n // args.tool_factory_every, 4))
    tasks = []
    ti = 0
    for i in range(args.n):
        tasks.append(gaia_tasks[i % len(gaia_tasks)])
        if (i + 1) % args.tool_factory_every == 0 and ti < len(trigger_tasks):
            tasks.append(trigger_tasks[ti])
            ti += 1
    print("[{}] Built {} tasks ({} GAIA2 + {} trigger)".format(
        args.config, len(tasks), len(gaia_tasks), len(trigger_tasks)))

    # Build layers based on disable flags
    main_llm = OllamaBackend(model=args.base_model)
    fast_llm = OllamaBackend(model=args.scorer_model)

    # L1: Reflector (per-step self-critique)
    reflector = None
    if not args.disable_l1:
        reflector = Reflector(
            main_llm=main_llm, fast_llm=fast_llm,
            trace_path=str(out_path / "trace.jsonl"),
            enabled=True,
        )

    # L2: Playbook + MetaController
    pb = None
    meta = None
    if not args.disable_l2:
        pb = Playbook(path=str(ROOT / "data/playbook.jsonl"), auto_load=True)
        if pb.size() == 0:
            pb.add("Use calculator for arithmetic; use echo to verify; use file tools for I/O.",
                   "Prefer the simplest tool that solves the sub-problem.",
                   success_rate=0.85)
            pb.save()
        meta = MetaController()

    # L4: SchemaMutator + ToolFactory
    schema_mutator = None
    tool_factory = None
    strategy_miner_llm = None
    if not args.disable_l4:
        schema_mutator = SchemaMutator(
            MetaControllerConfig(),
            eval_fn=None,
            history_path=str(out_path / "schema_history.jsonl"),
        )
        try:
            schema_mutator.propose("low_conf_threshold", 0.25, "init: more aggressive intervention")
            schema_mutator.propose("stuck_obs_threshold", 4, "init: tolerate more retries")
        except Exception:
            pass
        tool_factory = ToolFactory(
            llm=main_llm, tool_registry=FA_TOOLS,
            eval_fn=None,
            history_path=str(out_path / "tool_factory_history.jsonl"),
        )
        strategy_miner_llm = main_llm  # mine strategies into playbook

    # Build episode runner
    run_episode_fn = make_real_run_episode(
        playbook=pb, meta=meta, reflector=reflector,
        strategy_miner_llm=strategy_miner_llm, tool_factory=tool_factory,
        fa_tools=FA_TOOLS, max_steps=args.max_steps,
    )

    # L3: Continual loop wrapping
    if args.disable_l3:
        # Direct run, no retraining
        print("[{}] Running {} episodes (no L3 retraining)".format(args.config, len(tasks)))
        t0 = time.time()
        episodes = []
        for i, task in enumerate(tasks):
            gold = task.get("gold") if isinstance(task, dict) else None
            t_text = task.get("task") if isinstance(task, dict) else task
            r = run_episode_fn(t_text, gold=gold)
            episodes.append({
                "task": t_text,
                "verdict": r["verdict"],
                "steps": len(r["steps"]),
                "avg_score": r["avg_score"],
            })
            if (i + 1) % 4 == 0 or i == len(tasks) - 1:
                ok = sum(1 for e in episodes if e["verdict"] == "success")
                print("  [{}/{}] running success={}/{} = {:.1f}%".format(
                    i + 1, len(tasks), ok, i + 1, ok / (i + 1) * 100))
        wall = time.time() - t0
        success_count = sum(1 for e in episodes if e["verdict"] == "success")
        summary = {
            "config": args.config,
            "episodes": len(episodes),
            "tasks_total": len(tasks),
            "gaia2_tasks": len(gaia_tasks),
            "trigger_tasks": len(trigger_tasks),
            "success_count": success_count,
            "success_rate": success_count / max(len(episodes), 1),
            "wall_seconds": round(wall, 1),
            "disabled_l1": args.disable_l1,
            "disabled_l2": args.disable_l2,
            "disabled_l3": args.disable_l3,
            "disabled_l4": args.disable_l4,
        }
    else:
        # Wrap in ContinualLoop with mock retrain (no real SFT)
        eval_fn = make_eval_fn(n_tasks=5, max_steps=3, base_model=args.base_model)
        print("[{}] Measuring baseline accuracy...".format(args.config))
        baseline_acc = eval_fn(args.base_model)
        print("[{}] baseline accuracy: {}".format(args.config, baseline_acc))

        retrain_every = 10 ** 6 if args.disable_l3 else args.retrain_every
        retrain_min_buffer = 10 ** 6 if args.disable_l3 else 3

        buffer = ExperienceBuffer(path=str(ROOT / "logs/buffer/buffer.jsonl"), min_score=0.4)
        loop = ContinualLoop(
            run_episode_fn=run_episode_fn,
            reflector=reflector, playbook=pb, meta=meta, buffer=buffer,
            retrain_fn=lambda **kw: mock_retrain_v2(**kw),
            eval_fn=eval_fn,
            safety_threshold=args.threshold,
            retrain_every=retrain_every,
            retrain_min_buffer=retrain_min_buffer,
            out_dir=str(out_path),
            base_model=args.base_model,
        )
        t0 = time.time()
        result = loop.run(tasks, baseline_acc=baseline_acc)
        wall = time.time() - t0
        summary = {
            "config": args.config,
            "episodes": len(result["episodes"]),
            "tasks_total": len(tasks),
            "gaia2_tasks": len(gaia_tasks),
            "trigger_tasks": len(trigger_tasks),
            "success_count": sum(1 for e in result["episodes"] if e.get("verdict") == "success"),
            "success_rate": sum(1 for e in result["episodes"] if e.get("verdict") == "success") / max(len(result["episodes"]), 1),
            "wall_seconds": round(wall, 1),
            "baseline_acc": baseline_acc,
            "final_generation": result.get("final_generation"),
            "generations": [{k: getattr(g, k, None) for k in ["generation", "eval_new_acc", "train_samples", "accepted", "reason"]} for g in result.get("generations", [])],
            "disabled_l1": args.disable_l1,
            "disabled_l2": args.disable_l2,
            "disabled_l3": args.disable_l3,
            "disabled_l4": args.disable_l4,
        }

    # Save
    out_json = out_path / "summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("[{}] === Summary ===".format(args.config))
    print("  episodes:    {}".format(summary["episodes"]))
    print("  success:     {}".format(summary.get("success_count", "n/a")))
    print("  success_rate: {:.1f}%".format(summary["success_rate"] * 100))
    print("  wall_seconds: {}s".format(summary["wall_seconds"]))
    print("  Wrote {}".format(out_json))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   choices=["static", "l1_only", "l1_l2", "l1_l2_l3", "full"],
                   help="ablation config name")
    p.add_argument("--n", type=int, default=16, help="episodes (default 16 for speed)")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--retrain-every", type=int, default=15)
    p.add_argument("--tool-factory-every", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--base-model", default="qwen3:1.7b")
    p.add_argument("--scorer-model", default="qwen3:0.6b")
    p.add_argument("--out", default=None,
                   help="output directory (default: logs/ablation/<config>)")
    p.add_argument("--disable-l1", action="store_true")
    p.add_argument("--disable-l2", action="store_true")
    p.add_argument("--disable-l3", action="store_true")
    p.add_argument("--disable-l4", action="store_true")
    args = p.parse_args()

    # Defaults per config
    if args.out is None:
        args.out = "logs/ablation/" + args.config
    if args.config == "static":
        args.disable_l1 = args.disable_l2 = args.disable_l3 = args.disable_l4 = True
    elif args.config == "l1_only":
        args.disable_l2 = args.disable_l3 = args.disable_l4 = True
    elif args.config == "l1_l2":
        args.disable_l3 = args.disable_l4 = True
    elif args.config == "l1_l2_l3":
        args.disable_l4 = True
    # "full" leaves all enabled

    run_ablation(args)


if __name__ == "__main__":
    main()


