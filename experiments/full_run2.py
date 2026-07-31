"""full_run2.py - L1+L2+L3+L4 full integration."""
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
from agi_kit.llms.base import LLMMessage, MessageRole
from agi_kit.reflect import Reflector
from agi_kit.playbook import Playbook
from agi_kit.meta import MetaController, AgentState, update_state_from_step, ControlAction
from agi_kit.loop import ContinualLoop, ExperienceBuffer
from agi_kit.strategy_miner import extract_strategies_from_episode, write_extracted_to_playbook, ExtractedStrategy
from agi_kit.gaia2_tasks import load_gaia2_tasks
from agi_kit.evals_arith import make_eval_fn, eval_arithmetic, measure_baseline
from agi_kit.real_retrain import real_retrain, mock_retrain
from agi_kit.recursive import SchemaMutator, MetaControllerConfig, ToolFactory


def make_real_run_episode(playbook=None, meta=None, reflector=None,
                         strategy_miner_llm=None, tool_factory=None,
                         fa_tools=None, max_steps=12):
    main_llm = OllamaBackend(model="qwen3:1.7b")

    sys_prompt_template = (
        "You are an AGI research assistant. Local ReAct Agent.\n"
        "Tools (one per line):\n{tool_block}\n\n"
        "ALWAYS respond with a JSON tool call or final answer in ```json ... ``` block.\n"
        "For a tool call use keys: tool (string), args (object).\n"
        "For a final answer use key: final (string).\n"
        "After observing, decide the next step. Be concise."
    )

    def run_episode(task, gold=None):
        tool_lines = []
        for n, t in fa_tools.items():
            tool_lines.append("- " + n + ": " + t["desc"][:60] + " | params " + str(t["params"]))
        tool_block = "\n".join(tool_lines)
        sys_prompt = sys_prompt_template.format(tool_block=tool_block)
        if playbook is not None and playbook.size() > 0:
            ctx = playbook.as_system_context(task, k=2)
            if ctx:
                sys_prompt += "\n\nStrategies from past experience:\n" + ctx
        history = [
            LLMMessage(role=MessageRole.SYSTEM, content=sys_prompt),
            LLMMessage(role=MessageRole.USER, content=task),
        ]
        steps = []
        final = None
        verdict = "max_steps"
        state = AgentState(step=0, max_steps=max_steps, last_action={}, last_observation="")
        last_hint = ""
        consecutive_err = 0
        last_obs = ""
        hindsight_lines = []
        import re

        for i in range(1, max_steps + 1):
            if last_hint:
                history.append(LLMMessage(role=MessageRole.USER, content="Hint: " + last_hint))
                last_hint = ""
            try:
                resp_obj = main_llm.chat(history, max_tokens=300, temperature=0.0)
                resp_text = resp_obj.content if hasattr(resp_obj, "content") else str(resp_obj)
            except Exception as e:
                resp_text = "err: LLM failed - " + str(e)
                history.append(LLMMessage(role=MessageRole.ASSISTANT, content=resp_text))
                history.append(LLMMessage(role=MessageRole.USER, content="Please output valid JSON."))
                continue
            history.append(LLMMessage(role=MessageRole.ASSISTANT, content=resp_text))
            cand = None
            mm = re.search(r"```json\s*(\{.*?\})\s*```", resp_text, re.S)
            if mm:
                cand = mm.group(1)
            else:
                for m3 in re.finditer(r"\{[^{}]*\}", resp_text):
                    try:
                        obj = json.loads(m3.group(0))
                        if "tool" in obj or "final" in obj:
                            cand = m3.group(0)
                            break
                    except Exception:
                        continue
            try:
                action = json.loads(cand) if cand else None
            except Exception:
                action = None
            if action is None:
                history.append(LLMMessage(role=MessageRole.USER, content="Output JSON please"))
                continue
            if "final" in action:
                final = str(action["final"])
                verdict = "success"
                break
            tname = action.get("tool")
            targs = action.get("args", {}) or {}
            if tname not in fa_tools:
                obs = "err: unknown tool " + str(tname)
                consecutive_err += 1
            else:
                try:
                    obs = str(fa_tools[tname]["fn"](**targs))[:1500]
                    consecutive_err = 0 if "err" not in obs.lower() else consecutive_err + 1
                except Exception as e:
                    obs = "err: " + str(e)
                    consecutive_err += 1
            history.append(LLMMessage(role=MessageRole.USER, content="Observation: " + str(obs)))
            last_obs = str(obs)
            self_score = 0.5
            if reflector is not None:
                try:
                    self_score = reflector.log(i, action, obs)
                except Exception:
                    self_score = 0.5
            update_state_from_step(state, action, obs, self_score)
            if reflector is not None:
                rec = reflector.recent(1)
                if rec and rec[0].get("hindsight"):
                    hindsight_lines.append(rec[0]["hindsight"])
            steps.append({"action": action, "obs": str(obs)[:500], "self_score": self_score})
            if meta is not None:
                sig = meta.decide(state, playbook=playbook, query=task)
                if sig.action == ControlAction.SWITCH_STRATEGY and sig.hint:
                    last_hint = sig.hint
                elif sig.action == ControlAction.ASK_USER:
                    verdict = "stuck"
                    break
            if tool_factory is not None and consecutive_err >= 4:
                rec = tool_factory.try_synthesize(task, last_obs)
                if rec.get("accepted"):
                    consecutive_err = 0

        if verdict == "success" and strategy_miner_llm is not None and playbook is not None:
            try:
                strategies = extract_strategies_from_episode(
                    task=task, final=final or "",
                    hindsight_lines=hindsight_lines,
                    llm=strategy_miner_llm,
                )
                pattern = task[:80]
                to_add = [ExtractedStrategy(
                    pattern=pattern, strategy=s, success_rate=0.8,
                    source_episode=str(len(steps))
                ) for s in strategies]
                write_extracted_to_playbook(playbook, to_add)
            except Exception:
                pass

        if not steps and verdict == "success":
            avg = 0.85
        else:
            avg = sum(s.get("self_score", 0.5) for s in steps) / max(len(steps), 1)
        return {
            "final": final, "verdict": verdict,
            "steps": steps, "avg_score": avg,
        }
    return run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--retrain-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--no-sft", action="store_true", default=False)
    p.add_argument("--use-mock-retrain", action="store_true", default=False)
    p.add_argument("--out", default="logs/full_run2")
    args = p.parse_args()

    out_path = ROOT / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    tasks = load_gaia2_tasks(n=args.n, prefer_real=False)
    print("=== Loaded", len(tasks), "GAIA2-style tasks ===")

    from full_agent import TOOLS as FA_TOOLS
    pb = Playbook(path=str(ROOT / "data/playbook.jsonl"), auto_load=True)
    if pb.size() == 0:
        pb.add("Use calculator for arithmetic; use echo to verify; use file tools for I/O.",
               "Prefer the simplest tool that solves the sub-problem.",
               success_rate=0.85)
        pb.save()

    mc = MetaController()
    main_llm = OllamaBackend(model="qwen3:1.7b")
    fast_llm = OllamaBackend(model="qwen3:0.6b")
    miner_llm = OllamaBackend(model="qwen3:1.7b")

    reflector = Reflector(main_llm=main_llm, fast_llm=fast_llm,
                          trace_path=str(out_path / "trace.jsonl"),
                          enabled=True)
    schema_mutator = SchemaMutator(MetaControllerConfig(),
                                    eval_fn=None,
                                    history_path=str(out_path / "schema_history.jsonl"))
    schema_mutator.propose("low_conf_threshold", 0.25, "init: more aggressive intervention")
    schema_mutator.propose("stuck_obs_threshold", 5, "init: tolerate more retries")

    tool_factory = ToolFactory(llm=miner_llm, tool_registry=FA_TOOLS,
                                eval_fn=None,
                                history_path=str(out_path / "tool_factory_history.jsonl"))

    run_episode_fn = make_real_run_episode(
        playbook=pb, meta=mc, reflector=reflector,
        strategy_miner_llm=miner_llm, tool_factory=tool_factory,
        fa_tools=FA_TOOLS, max_steps=args.max_steps,
    )

    eval_fn = make_eval_fn(n_tasks=5, max_steps=3, base_model="qwen3:1.7b")
    print("[setup] measuring baseline accuracy...")
    baseline_acc = eval_fn("qwen3:1.7b")
    print("[setup] baseline accuracy:", baseline_acc)

    if args.use_mock_retrain:
        retrain_fn = lambda **kw: mock_retrain(**kw, no_sft=True)
    else:
        def _retrain(samples, base_model, out_dir, **kwargs):
            return real_retrain(samples, base_model, out_dir,
                                no_sft=args.no_sft,
                                epochs=1, batch_size=2)
        retrain_fn = _retrain

    buffer = ExperienceBuffer(path=str(ROOT / "logs/buffer/buffer.jsonl"),
                              min_score=0.4)

    loop = ContinualLoop(
        run_episode_fn=run_episode_fn,
        reflector=reflector, playbook=pb, meta=mc, buffer=buffer,
        retrain_fn=retrain_fn,
        eval_fn=eval_fn,
        safety_threshold=0.90,
        retrain_every=args.retrain_every,
        retrain_min_buffer=3,
        out_dir=str(out_path),
        base_model="qwen3:1.7b",
    )

    print("=== Full Run 2: " + str(args.n) + " episodes, retrain every " +
          str(args.retrain_every) + ", max_steps=" + str(args.max_steps) + " ===")
    t0 = time.time()
    summary = loop.run(tasks, baseline_acc=baseline_acc)
    dt = time.time() - t0

    out_json = {
        "episodes": len(summary["episodes"]),
        "final_generation": summary["final_generation"],
        "buffer_size": summary["buffer_size"],
        "playbook_size": pb.size(),
        "generations": [
            {
                "generation": g.generation,
                "accepted": g.accepted,
                "train_samples": g.train_samples,
                "eval_new_acc": g.eval_new_acc,
                "train_seconds": g.train_seconds,
                "reason": g.reason,
            } for g in summary["generations"]
        ],
        "current_model": summary["current_model"],
        "success_rate": sum(1 for e in summary["episodes"] if e.get("verdict") == "success") /
                        max(len(summary["episodes"]), 1),
        "avg_score": (sum(e.get("avg_score", 0) for e in summary["episodes"]) /
                      max(len(summary["episodes"]), 1)),
        "total_seconds": round(dt, 1),
        "schema_mutations": len(schema_mutator.history),
        "tool_factory_attempts": len(tool_factory.synthesized),
        "tool_factory_accepted": sum(1 for s in tool_factory.synthesized if s.get("accepted")),
        "baseline_acc": baseline_acc,
    }
    out_file = out_path / "summary.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print()
    print("=== SUMMARY ===")
    print(json.dumps(out_json, indent=2, ensure_ascii=False))
    print()
    print("Wrote", out_file)


if __name__ == "__main__":
    main()