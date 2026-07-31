"""continual_runner.py - L3 持续学习闭环 runner.

Drives N episodes through ContinualLoop, triggering SFT periodically.
Mock mode (--mock) runs without GPU/Ollama - simulates agent behavior
and SFT effect with deterministic math.

Usage:
  python experiments/continual_runner.py --mock --n 10 --retrain-every 3
  python experiments/continual_runner.py --n 50 --retrain-every 10   # real Ollama
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Force offline for embedder
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agi_kit.loop import (
    ExperienceBuffer, ContinualLoop, TraceRecord,
    format_trace_for_sft, default_safety_check,
)
from agi_kit.playbook import Playbook
from agi_kit.meta import MetaController, AgentState, update_state_from_step, ControlAction
from agi_kit.reflect import Reflector


# ============================================================
# Task pool (synthetic + GAIA2-style mini)
# ============================================================
TASK_KINDS = ["calc", "chained", "file", "logic", "lookup"]


def synth_tasks(n=20, seed=0):
    """Build n synthetic tasks of varying kinds."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = TASK_KINDS[i % len(TASK_KINDS)]
        if kind == "calc":
            a = rng.randint(2, 99)
            b = rng.randint(2, 99)
            expr = str(a) + "*" + str(b)
            out.append({"task": "Calculate " + expr + " using calculator.", "kind": kind,
                        "gold": str(a * b)})
        elif kind == "chained":
            a = rng.randint(2, 20)
            out.append({"task": "Calculate " + str(a) + "**3 then echo result, then final.",
                        "kind": kind, "gold": str(a ** 3)})
        elif kind == "file":
            out.append({"task": "Read the first 60 chars of C:/Windows/win.ini.", "kind": kind,
                        "gold": None})
        elif kind == "logic":
            out.append({"task": "If x=" + str(i + 2) + ", compute x*2+1 using calculator.",
                        "kind": kind, "gold": str((i + 2) * 2 + 1)})
        elif kind == "lookup":
            out.append({"task": "What is " + str(i + 100) + " + " + str(i + 200) + " using calculator?",
                        "kind": kind, "gold": str(i + 100 + i + 200)})
    return out


# ============================================================
# Mock LLM
# ============================================================
@dataclass
class _R:
    content: str


class MockLLM:
    """Cycle through a deterministic plan; success improves with generation."""
    def __init__(self, kind, success_rate=0.6):
        self.kind = kind
        self.success_rate = success_rate

    def chat(self, messages, max_tokens=256, temperature=0.0):
        user_count = sum(1 for m in messages if getattr(m, "role", "") == "user")
        s = user_count - 1
        # Pick a plan based on step
        plan = _plan_for(self.kind, success_rate=self.success_rate)
        if s < 0:
            s = 0
        if s >= len(plan):
            s = len(plan) - 1
        return _R(content=plan[s])


def _plan_for(kind, success_rate):
    """Return list of plan entries. Earlier entries are the 'first attempts';
    success_rate controls how often the first action is correct."""
    if kind == "calc":
        if success_rate > 0.8:
            return [
                '```json\n{"tool":"calculator","args":{"expr":"__EXPR__"}}\n```',
                '```json\n{"final":"__ANS__"}\n```',
            ]
        return [
            '```json\n{"tool":"compute","args":{"expr":"__EXPR__"}}\n```',  # wrong name
            '```json\n{"tool":"calculator","args":{"expr":"__EXPR__"}}\n```',
            '```json\n{"final":"__ANS__"}\n```',
        ]
    if kind == "chained":
        if success_rate > 0.8:
            return [
                '```json\n{"tool":"calculator","args":{"expr":"__EXPR__"}}\n```',
                '```json\n{"tool":"echo","args":{"msg":"__ANS__"}}\n```',
                '```json\n{"final":"__ANS__"}\n```',
            ]
        return [
            '```json\n{"tool":"calculator","args":{"expr":"__EXPR__"}}\n```',
            '```json\n{"tool":"speak","args":{"msg":"__ANS__"}}\n```',  # wrong name
            '```json\n{"tool":"echo","args":{"msg":"__ANS__"}}\n```',
            '```json\n{"final":"__ANS__"}\n```',
        ]
    if kind == "file":
        return [
            '```json\n{"tool":"read_file","args":{"path":"C:/Windows/win.ini","max_chars":60}}\n```',
            '```json\n{"final":"first line"}\n```',
        ]
    if kind == "logic":
        return [
            '```json\n{"tool":"calculator","args":{"expr":"__EXPR__"}}\n```',
            '```json\n{"final":"__ANS__"}\n```',
        ]
    if kind == "lookup":
        return [
            '```json\n{"tool":"calculator","args":{"expr":"__EXPR__"}}\n```',
            '```json\n{"final":"__ANS__"}\n```',
        ]
    return ['```json\n{"final":"ok"}\n```']


def _fill_placeholders(plan, task_meta):
    expr = "x*2+1"
    ans = "ok"
    if task_meta["kind"] in ("calc", "logic", "lookup"):
        m = re.search(r"(\d+)\s*\*\s*(\d+)", task_meta["task"])
        if m:
            expr = m.group(1) + "*" + m.group(2)
            ans = str(int(m.group(1)) * int(m.group(2)))
        else:
            m2 = re.search(r"(\d+)\s*\+\s*(\d+)", task_meta["task"])
            if m2:
                expr = m2.group(1) + "+" + m2.group(2)
                ans = str(int(m2.group(1)) + int(m2.group(2)))
            else:
                m3 = re.search(r"x=(\d+)", task_meta["task"])
                if m3:
                    x = int(m3.group(1))
                    expr = str(x) + "*2+1"
                    ans = str(x * 2 + 1)
    elif task_meta["kind"] == "chained":
        m = re.search(r"(\d+)\s*\*\*\s*3", task_meta["task"])
        if m:
            expr = m.group(1) + "**3"
            ans = str(int(m.group(1)) ** 3)
    out = []
    for step in plan:
        out.append(step.replace("__EXPR__", expr).replace("__ANS__", ans))
    return out


# ============================================================
# Mock episode runner (smoke-test version)
# ============================================================
@dataclass
class _M:
    role: str
    content: str


def make_mock_run_episode(playbook=None, meta=None, reflector=None):
    def run_episode(task, gold=None):
        task_kind = ""
        m = re.search(r"kind=(\w+)", task)
        if m:
            task_kind = m.group(1)
        else:
            for k in TASK_KINDS:
                if k in task.lower():
                    task_kind = k
                    break
        if not task_kind:
            task_kind = "calc"
        task_meta = {"kind": task_kind, "task": task, "gold": gold}
        llm = MockLLM(task_kind, success_rate=0.55)
        plan = _fill_placeholders(_plan_for(task_kind, success_rate=0.55), task_meta)
        llm.script = plan
        llm.i = 0
        # Override chat to use filled plan
        orig_chat = llm.chat

        def chat(messages, **kw):
            user_count = sum(1 for m in messages if getattr(m, "role", "") == "user")
            s = user_count - 1
            s = max(0, min(s, len(plan) - 1))
            return _R(content=plan[s])
        llm.chat = chat

        sys_prompt = (
            "You are an AGI research assistant. Output STRICT JSON. "
            "Tools:\n- calculator\n- echo\n- read_file"
        )
        history = [_M("system", sys_prompt), _M("user", task)]
        steps = []
        final = None
        verdict = "max_steps"
        state = AgentState(step=0, max_steps=8, last_action={}, last_observation="")
        last_hint = ""
        for i in range(1, 9):
            if last_hint:
                history.append(_M("user", "Hint: " + last_hint))
                last_hint = ""
            resp = llm.chat(history)
            text = resp.content if hasattr(resp, "content") else resp["content"]
            history.append(_M("assistant", text))
            mm = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
            cand = None
            if mm:
                cand = mm.group(1)
            else:
                mm2 = re.search(r"\{.*\}", text, re.S)
                cand = mm2.group(0) if mm2 else None
            try:
                action = json.loads(cand) if cand else None
            except Exception:
                action = None
            if action is None:
                history.append(_M("user", "Please output valid JSON."))
                continue
            if "final" in action:
                final = str(action["final"])
                verdict = "success"
                break
            tname = action.get("tool")
            targs = action.get("args", {}) or {}
            if tname == "calculator":
                expr = str(targs.get("expr", "0"))
                if re.fullmatch(r"[\d\s+\-*/().%^|&*]+", expr):
                    try:
                        obs = str(eval(expr, {"__builtins__": {}}))
                    except Exception as e:
                        obs = "err: " + str(e)
                else:
                    obs = "err: bad chars"
            elif tname == "echo":
                obs = str(targs.get("msg", ""))
            elif tname == "read_file":
                try:
                    with open(targs.get("path", ""), "r", encoding="utf-8", errors="ignore") as f:
                        obs = f.read(int(targs.get("max_chars", 60)))
                except Exception as e:
                    obs = "err: " + str(e)
            else:
                obs = "err: unknown tool " + str(tname)
            history.append(_M("user", "Observation: " + str(obs)[:1500]))
            mock_score = 0.2 if "err" in obs.lower() else 0.85
            update_state_from_step(state, action, obs, mock_score)
            steps.append({"action": action, "obs": str(obs)[:500], "self_score": mock_score})
            if meta is not None:
                sig = meta.decide(state, playbook=playbook, query=task)
                if sig.action == ControlAction.SWITCH_STRATEGY and sig.hint:
                    last_hint = sig.hint
        return {"final": final, "verdict": verdict, "steps": steps,
                "avg_score": sum(s.get("self_score", 0.5) for s in steps) / max(len(steps), 1)}
    return run_episode


# ============================================================
# Mock retrain + eval
# ============================================================
def mock_retrain(samples, base_model, out_dir):
    """No-op retrain: just save the formatted samples as 'model artifacts'."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    # Simulated training time scales with N
    sim_secs = min(2.0, len(samples) * 0.01)
    time.sleep(sim_secs)
    return {"out_dir": out_dir, "samples": len(samples), "seconds": sim_secs}


def mock_eval(model_dir):
    """Mock eval: returns a value that increases per generation
    to simulate the self-improvement loop working."""
    path = Path(model_dir)
    try:
        gen = int(path.name.split("-")[-1])
    except Exception:
        gen = 1
    # Fake accuracy: 0.65 -> 0.85 over 5 gens
    return min(0.95, 0.55 + 0.06 * gen)


# ============================================================
# Real episode runner (uses full_agent's tools + OllamaBackend)
# ============================================================
def make_real_run_episode(playbook=None, meta=None, reflector=None):
    sys.path.insert(0, str(ROOT / "experiments"))
    from full_agent import TOOLS as FA_TOOLS, OllamaBackend

    main_llm = OllamaBackend(model="qwen3:1.7b")

    def run_episode(task, gold=None):
        tool_lines = []
        for n, t in FA_TOOLS.items():
            tool_lines.append("- " + n + ": " + t["desc"] + " | params " + str(t["params"]))
        tool_block = "\n".join(tool_lines)
        sys_prompt = (
            "You are an AGI research assistant. Local ReAct Agent.\n"
            "Tools:\n" + tool_block + "\n\n"
            "Output strict JSON in triple-backtick json block.\n"
            'Tool call: {"tool":"<name>","args":{...}}.\n'
            'Final answer: {"final":"<answer>"}.\n'
            "Be concise. Use tools step by step."
        )
        if playbook is not None and playbook.size() > 0:
            ctx = playbook.as_system_context(task, k=2)
            if ctx:
                sys_prompt += "\n\n" + ctx
        history_dicts = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task},
        ]
        steps = []
        final = None
        verdict = "max_steps"
        state = AgentState(step=0, max_steps=8, last_action={}, last_observation="")
        last_hint = ""

        for i in range(1, 9):
            if last_hint:
                history_dicts.append({"role": "user", "content": "Hint: " + last_hint})
                last_hint = ""
            try:
                resp_text = main_llm.chat(history_dicts, max_tokens=300, temperature=0.0)
            except Exception as e:
                resp_text = "err: LLM failed - " + str(e)
                history_dicts.append({"role": "assistant", "content": resp_text})
                history_dicts.append({"role": "user", "content": "Please output valid JSON."})
                continue
            history_dicts.append({"role": "assistant", "content": resp_text})
            mm = re.search(r"```json\s*(\{.*?\})\s*```", resp_text, re.S)
            cand = None
            if mm:
                cand = mm.group(1)
            else:
                mm2 = re.search(r"\{[^{}]*\"tool\"[^{}]*\}|\{.*?\"final\".*?\}", resp_text, re.S)
                cand = mm2.group(0) if mm2 else None
            try:
                action = json.loads(cand) if cand else None
            except Exception:
                action = None
            if action is None:
                history_dicts.append({"role": "user", "content": "Output JSON please"})
                continue
            if "final" in action:
                final = str(action["final"])
                verdict = "success"
                break
            tname = action.get("tool")
            targs = action.get("args", {}) or {}
            if tname not in FA_TOOLS:
                obs = "err: unknown tool " + str(tname)
            else:
                try:
                    obs = str(FA_TOOLS[tname]["fn"](**targs))[:1500]
                except Exception as e:
                    obs = "err: " + str(e)
            history_dicts.append({"role": "user", "content": "Observation: " + str(obs)})
            self_score = 0.5
            if reflector is not None:
                try:
                    self_score = reflector.log(i, action, obs)
                except Exception:
                    self_score = 0.5
            update_state_from_step(state, action, obs, self_score)
            steps.append({"action": action, "obs": str(obs)[:500], "self_score": self_score})
            if meta is not None:
                sig = meta.decide(state, playbook=playbook, query=task)
                if sig.action == ControlAction.SWITCH_STRATEGY and sig.hint:
                    last_hint = sig.hint
                elif sig.action == ControlAction.ASK_USER:
                    verdict = "stuck"
                    break
        return {
            "final": final,
            "verdict": verdict,
            "steps": steps,
            "avg_score": (0.8 if (not steps and verdict == "success") else (sum(s.get("self_score", 0.5) for s in steps) / max(len(steps), 1))),
        }
    return run_episode


# ============================================================
# Real retrain (calls existing sft_train.py)
# ============================================================
def real_retrain(samples, base_model, out_dir):
    """Wrap existing sft_train.py with pre-formatted messages."""
    import subprocess
    # Convert messages -> chat-template text via tokenizer? Keep simple:
    # Save samples as a HuggingFace dataset jsonl, point sft_train at it.
    workdir = Path(out_dir) / "_prep"
    workdir.mkdir(parents=True, exist_ok=True)
    # sft_train.py reads "messages" field
    with (workdir / "data.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "sft_train.py"),
        "--model", str(base_model),
        "--max-samples", str(len(samples)),
        "--epochs", "1",
        "--out", str(out_dir),
    ]
    print(">> launching:", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    return {"out_dir": out_dir, "samples": len(samples),
            "seconds": dt, "stdout_tail": r.stdout[-300:],
            "stderr_tail": r.stderr[-300:], "rc": r.returncode}


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true",
                   help="no Ollama / no GPU; deterministic mock")
    p.add_argument("--n", type=int, default=10, help="number of episodes")
    p.add_argument("--no-sft", action="store_true", help="real LLM episodes but mock retrain (skip real SFT)")
    p.add_argument("--retrain-every", type=int, default=3)
    p.add_argument("--retrain-min-buffer", type=int, default=3)
    p.add_argument("--out", default="logs/continual")
    p.add_argument("--buffer-path", default="logs/buffer/buffer.jsonl")
    p.add_argument("--playbook-path", default="data/playbook.jsonl")
    args = p.parse_args()

    out_path = ROOT / args.out
    out_path.mkdir(parents=True, exist_ok=True)

    # Components
    pb = Playbook(path=str(ROOT / args.playbook_path), auto_load=True)
    if pb.size() == 0:
        pb.add("when calculator tool fails, retry with correct name",
               "Use 'calculator' (not 'compute' or 'calc').",
               success_rate=0.9)
        pb.save()
    mc = MetaController()

    # Mock reflector (no LLM call)
    class _M0:
        def chat(self, msgs, **kw):
            class R: content = ""
            return R()
        model = "mock"
    reflector = Reflector(main_llm=_M0(), trace_path=str(out_path / "trace.jsonl"),
                          fast_llm=_M0(), enabled=True)

    buffer = ExperienceBuffer(path=str(ROOT / args.buffer_path), min_score=0.5)


    if args.mock:
        run_episode_fn = make_mock_run_episode(playbook=pb, meta=mc, reflector=reflector)
        retrain_fn = mock_retrain
        eval_fn = mock_eval
    else:
        # Real Ollama runner
        from agi_kit.llms.ollama import OllamaBackend
        real_main = OllamaBackend(model="qwen3:1.7b")
        real_fast = OllamaBackend(model="qwen3:0.6b")
        # Real reflector (uses real LLMs for self_score + hindsight)
        reflector = Reflector(
            main_llm=real_main,
            fast_llm=real_fast,
            trace_path=str(out_path / "trace.jsonl"),
            enabled=True,
        )
        run_episode_fn = make_real_run_episode(playbook=pb, meta=mc, reflector=reflector)
        if args.no_sft:
            retrain_fn = mock_retrain
        else:
            retrain_fn = real_retrain
        eval_fn = None  # real A/B eval would call compare_models.py

    loop = ContinualLoop(
        run_episode_fn=run_episode_fn,
        reflector=reflector,
        playbook=pb,
        meta=mc,
        buffer=buffer,
        retrain_fn=retrain_fn,
        eval_fn=eval_fn,
        safety_threshold=0.95,
        retrain_every=args.retrain_every,
        retrain_min_buffer=args.retrain_min_buffer,
        out_dir=str(out_path),
        base_model=("mock-base" if args.mock else "qwen3:1.7b"),
    )

    tasks = synth_tasks(n=args.n)
    print("=== Continual Loop: " + str(args.n) + " episodes (mock=" +
          str(args.mock) + ", retrain_every=" + str(args.retrain_every) + ") ===")
    summary = loop.run(tasks, baseline_acc=0.55)

    # Persist summary
    out_json = {
        "episodes": len(summary["episodes"]),
        "final_generation": summary["final_generation"],
        "buffer_size": summary["buffer_size"],
        "generations": [
            {
                "generation": g.generation,
                "accepted": g.accepted,
                "train_samples": g.train_samples,
                "eval_new_acc": g.eval_new_acc,
                "train_seconds": g.train_seconds,
                "reason": g.reason,
            }
            for g in summary["generations"]
        ],
        "current_model": summary["current_model"],
        "success_rate": sum(1 for e in summary["episodes"] if e.get("verdict") == "success") /
                        max(len(summary["episodes"]), 1),
        "avg_score": (sum(e.get("avg_score", 0) for e in summary["episodes"]) /
                      max(len(summary["episodes"]), 1)),
    }
    out_file = out_path / "summary.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print()
    print("=== SUMMARY ===")
    print(json.dumps(out_json, indent=2, ensure_ascii=False))
    print()
    print("Wrote " + str(out_file))
    print("Buffer has " + str(buffer.size()) + " records at " + str(buffer.path))


if __name__ == "__main__":
    main()