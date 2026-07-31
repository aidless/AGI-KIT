"""l2_meta_smoke.py - L2 Playbook + MetaController smoke test.

Three scenarios:
  A) Tool error streak: agent calls bad tool twice; meta should RETRY/SWITCH
  B) Stuck detection: 3 identical observations; meta should SWITCH_STRATEGY with hint
  C) Recovery: after SWITCH, agent uses hint and succeeds

Compares completion rate and step count with vs without MetaController.
Mock mode (no Ollama needed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.meta import (
    MetaController, ControlAction, AgentState, update_state_from_step,
)
from agi_kit.playbook import Playbook


# ============================================================
# Tools
# ============================================================
TOOLS = {}

def tool(name, desc, params):
    def deco(fn):
        TOOLS[name] = {"desc": desc, "params": params, "fn": fn}
        return fn
    return deco

@tool("calculator", "evaluate arithmetic expression", {"expr": "str"})
def calculator(expr):
    import math
    if not re.fullmatch(r"[\d\s+\-*/().%^|&]+", expr):
        return "err: bad chars"
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return "err: " + str(e)

@tool("echo", "echo a string", {"msg": "str"})
def echo(msg):
    return msg


# ============================================================
# Mock LLM (plan-based, can simulate mistakes)
# ============================================================
@dataclass
class _R:
    content: str


class MockLLM:
    """Cycle through a script. Script entries are JSON action strings."""
    def __init__(self, script):
        self.script = script
        self.i = 0

    def chat(self, messages, max_tokens=256, temperature=0.0):
        user_count = sum(1 for m in messages if getattr(m, "role", "") == "user")
        # Each step adds 2 messages (assistant + user Observation)
        s = min(user_count - 1, len(self.script) - 1)
        return _R(content=self.script[max(0, s)])


# ============================================================
# Agent runner with optional Reflector + MetaController
# ============================================================
@dataclass
class _M:
    role: str
    content: str


@dataclass
class RunResult:
    success: bool
    steps: int
    signals: list
    final: str
    notes: str = ""


def run_task(llm, task, max_steps=8, meta=None, playbook=None,
             inject_hints=True):
    sys_prompt = (
        "You are an AGI research assistant. Output STRICT JSON in a "
        "triple-backtick json block. Tool call: "
        '{"tool":"<name>","args":{...}}. Final: {"final":"<answer>"}.\n'
        "Tools:\n" + "\n".join(
            "- " + n + ": " + t["desc"] for n, t in TOOLS.items()
        )
    )
    history = [_M("system", sys_prompt), _M("user", task)]
    steps = 0
    signals = []
    final = None

    state = AgentState(
        step=0, max_steps=max_steps,
        last_action={}, last_observation="",
    )

    last_hint = ""
    for i in range(1, max_steps + 1):
        # Inject hint if any
        if inject_hints and last_hint:
            history.append(_M("user", "Hint from past: " + last_hint))
            last_hint = ""
        resp = llm.chat(history, max_tokens=256, temperature=0.0)
        text = resp.content if hasattr(resp, "content") else resp["content"]
        history.append(_M("assistant", text))
        # Parse
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
        cand = None
        if m:
            cand = m.group(1)
        else:
            m2 = re.search(r"\{.*\}", text, re.S)
            cand = m2.group(0) if m2 else None
        try:
            action = json.loads(cand) if cand else None
        except Exception:
            action = None
        if action is None:
            history.append(_M("user", "Please output valid JSON."))
            continue
        if "final" in action:
            final = str(action["final"])
            steps = i
            return RunResult(True, steps, signals, final, "final_reached")
        # Execute
        tname = action.get("tool")
        targs = action.get("args", {}) or {}
        if tname not in TOOLS:
            obs = "err: unknown tool " + str(tname)
        else:
            try:
                obs = str(TOOLS[tname]["fn"](**targs))
            except Exception as e:
                obs = "err: " + str(e)
        history.append(_M("user", "Observation: " + str(obs)[:1500]))
        # Update state (no real self_score here, mock 0.7)
        mock_score = 0.2 if "err" in obs.lower() else 0.8
        update_state_from_step(state, action, obs, mock_score)
        steps = i
        # Meta decides
        if meta is not None:
            sig = meta.decide(state, playbook=playbook, query=task)
            signals.append(sig.to_dict())
            if sig.action == ControlAction.RETRY:
                history.append(_M("user",
                    "Previous step failed. Try again with corrected arguments."))
                continue
            if sig.action == ControlAction.SWITCH_STRATEGY:
                if sig.hint:
                    last_hint = sig.hint
                continue
            if sig.action == ControlAction.ASK_USER:
                return RunResult(False, steps, signals, None, "ask_user")
            if sig.action == ControlAction.CHANGE_PLAN:
                return RunResult(False, steps, signals, None, "change_plan")
    return RunResult(False, steps, signals, final, "max_steps")


# ============================================================
# Scenarios
# ============================================================
def scenario_A_tool_error():
    # First call: bad tool name; second: bad tool name; third: correct
    bad_then_good = MockLLM([
        'I will use the wrong tool.\n```json\n{"tool":"calc","args":{"expr":"2**10"}}\n```',
        '```json\n{"tool":"calc","args":{"expr":"2**10"}}\n```',
        '```json\n{"tool":"calculator","args":{"expr":"2**10"}}\n```',
        '```json\n{"final":"1024"}\n```',
    ])
    return ("A_tool_error", "Calculate 2**10.",
            bad_then_good, 8, "should_recover_with_meta")


def scenario_B_stuck():
    # Agent keeps asking for the same broken tool
    stuck = MockLLM([
        '```json\n{"tool":"web_search","args":{"q":"hello"}}\n```',
        '```json\n{"tool":"web_search","args":{"q":"hello"}}\n```',
        '```json\n{"tool":"web_search","args":{"q":"hello"}}\n```',
        '```json\n{"tool":"web_search","args":{"q":"hello"}}\n```',
        '```json\n{"tool":"calculator","args":{"expr":"1+1"}}\n```',
        '```json\n{"final":"2"}\n```',
    ])
    return ("B_stuck", "Find information about 1+1.",
            stuck, 8, "should_detect_stuck_and_switch")


def scenario_C_chain():
    # Simple chain that succeeds first time
    chain = MockLLM([
        '```json\n{"tool":"calculator","args":{"expr":"7*8"}}\n```',
        '```json\n{"final":"56"}\n```',
    ])
    return ("C_simple", "Calculate 7*8.",
            chain, 6, "control_both_ways_should_succeed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="logs/l2_smoke.jsonl")
    p.add_argument("--playbook-out", default="data/playbook.jsonl")
    args = p.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-seed playbook with one useful strategy
    pb = Playbook(path=str(ROOT / args.playbook_out), auto_load=True)
    if pb.size() == 0:
        pb.add(
            pattern="when calculator tool name returns unknown, use calculator instead",
            strategy="The correct tool name is 'calculator' (not 'calc' or 'compute').",
            success_rate=0.95,
        )
        pb.add(
            pattern="when web tool is unavailable for math, fall back to calculator",
            strategy="For arithmetic, prefer calculator over web_search.",
            success_rate=0.9,
        )
        pb.save()
        print("Seeded playbook with 2 strategies.")

    scenarios = [scenario_A_tool_error(), scenario_B_stuck(), scenario_C_chain()]
    results = []

    for name, task, llm, max_steps, expected in scenarios:
        print()
        print("== Scenario " + name + " ==")
        print("  task: " + task)
        for use_meta in (False, True):
            label = "WITH" if use_meta else "NO  "
            mc = MetaController() if use_meta else None
            mc.reset() if mc else None
            t0 = time.time()
            r = run_task(llm, task, max_steps=max_steps,
                         meta=mc, playbook=pb, inject_hints=use_meta)
            dt = round(time.time() - t0, 2)
            sig_summary = (
                "->".join([s["action"] for s in r.signals])
                if r.signals else "(no signals)"
            )
            print("  " + label + " meta: success=" + str(r.success) +
                  " steps=" + str(r.steps) +
                  " signals=[" + sig_summary + "]" +
                  " dt=" + str(dt) + "s note=" + r.notes)
            results.append({
                "scenario": name, "task": task,
                "meta": use_meta,
                "success": r.success,
                "steps": r.steps,
                "notes": r.notes,
                "signals": r.signals,
                "seconds": dt,
                "expected": expected,
            })

    # Summary
    by_meta = {"with": [], "without": []}
    for r in results:
        by_meta["with" if r["meta"] else "without"].append(r)
    summary = {}
    for tag, rs in by_meta.items():
        success = sum(1 for r in rs if r["success"])
        avg_steps = sum(r["steps"] for r in rs) / max(len(rs), 1)
        recovery = sum(1 for r in rs if r["success"] and r["scenario"] in ("A_tool_error", "B_stuck"))
        summary[tag] = {
            "success_rate": round(success / max(len(rs), 1), 3),
            "avg_steps": round(avg_steps, 2),
            "scenarios_recovered": recovery,
            "n_scenarios": len(rs),
        }
    print()
    print("========== SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print()
    print("Wrote " + str(len(results)) + " rows to " + str(out_path))
    print("Playbook has " + str(pb.size()) + " strategies at " + str(pb.path))


if __name__ == "__main__":
    main()