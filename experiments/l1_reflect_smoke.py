"""l1_reflect_smoke.py - L1 reflection primitive smoke test.

Compares with-reflector vs without-reflector on 5 small tasks.
No GPU needed; defaults to Ollama, --mock bypasses.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ============================================================
# Task set
# ============================================================
TASKS = [
    {"q": "Calculate 2**10 + 3*4 using calculator tool.", "kind": "calc", "expect": "1044"},
    {"q": "Calculate (17*23 + 144/12) using calculator tool.", "kind": "calc", "expect": "403"},
    {"q": "Use calculator to compute 99*99 - 1, then echo the result.", "kind": "chained", "expect": "9800"},
    {"q": "Read the file at C:/Windows/win.ini (limit 80 chars), then echo the first line.", "kind": "file", "expect": None},
    {"q": "Calculate 7*8 with calculator.", "kind": "calc", "expect": "56"},
]

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

@tool("read_file", "read a text file", {"path": "str", "max_chars": "int=2000"})
def read_file(path, max_chars=2000):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars) or "(empty)"
    except Exception as e:
        return "err: " + str(e)

@tool("echo", "echo a string", {"msg": "str"})
def echo(msg):
    return msg


# ============================================================
# Mock LLM (cyclical plan-based simulator)
# ============================================================
class MockLLM:
    def __init__(self, kind):
        self.kind = kind
        self.plan = _make_plan(kind)
        self.i = 0

    def chat(self, messages, max_tokens=256, temperature=0.0):
        user_msgs = [m for m in messages if getattr(m, "role", "") == "user"]
        s = max(0, min(len(user_msgs) - 1, len(self.plan) - 1))
        class R:
            content = self.plan[s]
        return R()

    def model_name(self):
        return "mock"


def _make_plan(kind):
    if kind == "calc":
        return [
            'I will calculate.\n',
            '```json\n{"final":"1044"}\n```',
        ]
    if kind == "chained":
        return [
            '```json\n{"tool":"calculator","args":{"expr":"99*99-1"}}\n```',
            '```json\n{"tool":"echo","args":{"msg":"9800"}}\n```',
            '```json\n{"final":"9800"}\n```',
        ]
    if kind == "file":
        return [
            '```json\n{"tool":"read_file","args":{"path":"C:/Windows/win.ini","max_chars":80}}\n```',
            '```json\n{"tool":"echo","args":{"msg":"first line seen"}}\n```',
            '```json\n{"final":"first line seen"}\n```',
        ]
    return ['```json\n{"final":"ok"}\n```']


# ============================================================
# Minimal ReAct agent
# ============================================================
@dataclass
class Step:
    idx: int
    action: dict
    observation: str
    self_score: float = 0.5
    hindsight: str = ""


@dataclass
class _M:
    role: str
    content: str


def run_agent(llm, task, max_steps=6, reflector=None):
    sys_prompt = (
        "You are an AGI research assistant. "
        "Output STRICT JSON in a triple-backtick json block. "
        'Tool call: {"tool":"<name>","args":{...}}. '
        'Final answer: {"final":"<answer>"}.\n'
        "Tools:\n" + "\n".join(
            "- " + n + ": " + t["desc"] for n, t in TOOLS.items()
        )
    )
    history = [
        _M("system", sys_prompt),
        _M("user", task),
    ]
    steps = []
    final = None
    verdict = "max_steps"

    for i in range(1, max_steps + 1):
        resp = llm.chat(history, max_tokens=256, temperature=0.0)
        text = resp.content if hasattr(resp, "content") else resp["content"]
        history.append(_M("assistant", text))
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
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
            verdict = "success"
            break
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
        score = 0.5
        hindsight = ""
        if reflector is not None:
            try:
                score = reflector.log(i, action, obs)
                rec = reflector.recent(1)
                if rec:
                    hindsight = rec[0].get("hindsight", "")
            except Exception as e:
                score = 0.3
                hindsight = "(reflector error: " + str(e) + ")"
        steps.append(Step(i, action, str(obs)[:500], score, hindsight))

    if reflector is not None:
        summary = reflector.summarize_episode(
            task=task,
            trace=[{"step": s.idx, "action": s.action,
                    "observation": s.observation,
                    "self_score": s.self_score} for s in steps],
            verdict=verdict,
        )
        return {"final": final, "verdict": verdict, "steps": steps,
                "avg_score": summary.avg_self_score,
                "lesson": summary.lesson}

    avg = statistics.mean([s.self_score for s in steps]) if steps else 0.5
    return {"final": final, "verdict": verdict, "steps": steps,
            "avg_score": avg, "lesson": "(no reflector)"}


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true",
                   help="run without Ollama")
    p.add_argument("--out", default="logs/l1_smoke.jsonl")
    args = p.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick LLM
    reflector_factory = None
    if args.mock:
        # Custom MockLLM whose self_score is built into plan output
        class LLM0:
            def __init__(self, kind): self.kind = kind
            def chat(self, msgs, **kw):
                plan = _make_plan(self.kind)
                user_count = sum(1 for m in msgs if getattr(m, "role", "") == "user")
                s = max(0, min(user_count - 1, len(plan) - 1))
                class R:
                    content = plan[s]
                return R()
        llm_factory = lambda kind: LLM0(kind)
        reflector_factory = lambda: _make_mock_reflector()
    else:
        try:
            from agi_kit import OllamaBackend
            main_llm = OllamaBackend(model="qwen3:1.7b")
            fast_llm = OllamaBackend(model="qwen3:0.6b")
            llm_factory = lambda kind: main_llm
            from agi_kit.reflect import Reflector
            reflector_factory = lambda: Reflector(
                main_llm=main_llm, fast_llm=fast_llm,
                trace_path="logs/trace_with.jsonl",
                enabled=True,
            )
        except Exception as e:
            print("Failed to init Ollama backend: " + str(e))
            print("Re-run with --mock to bypass.")
            sys.exit(2)

    # Run 5 tasks x 2 modes
    results = {"with": [], "without": []}
    for t in TASKS:
        for mode in ("without", "with"):
            print()
            print("=== " + mode.upper() + " reflector | " + t["kind"] + " ===")
            ref = reflector_factory() if (mode == "with" and reflector_factory) else None
            llm = llm_factory(t["kind"])
            t0 = time.time()
            r = run_agent(llm, t["q"], max_steps=6, reflector=ref)
            r["seconds"] = round(time.time() - t0, 1)
            r["task"] = t
            results[mode].append(r)
            print("  verdict=" + r["verdict"] +
                  " avg_score=" + str(round(r["avg_score"], 2)) +
                  " steps=" + str(len(r["steps"])) +
                  " secs=" + str(r["seconds"]))

    # Summary
    summary = {}
    for mode in ("without", "with"):
        rs = results[mode]
        success = sum(1 for r in rs if r["verdict"] == "success")
        avg_steps = statistics.mean([len(r["steps"]) for r in rs]) if rs else 0
        avg_score = statistics.mean([r["avg_score"] for r in rs]) if rs else 0
        recovered = 0
        for r in rs:
            sc = [s.self_score for s in r["steps"]]
            for i in range(len(sc) - 2):
                if sc[i] < 0.4 and any(x > 0.7 for x in sc[i+1:]):
                    recovered += 1
                    break
        summary[mode] = {
            "success_rate": round(success / max(len(rs), 1), 3),
            "avg_steps": round(avg_steps, 2),
            "avg_self_score": round(avg_score, 3),
            "recoveries": recovered,
        }

    print()
    print("========== SUMMARY ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(out_path, "w", encoding="utf-8") as f:
        for mode in ("without", "with"):
            for r in results[mode]:
                rec = {
                    "mode": mode,
                    "task_kind": r["task"]["kind"],
                    "task_q": r["task"]["q"],
                    "verdict": r["verdict"],
                    "final": r["final"],
                    "seconds": r["seconds"],
                    "steps": [{"i": s.idx, "action": s.action,
                               "obs": s.observation,
                               "score": s.self_score,
                               "hindsight": s.hindsight}
                              for s in r["steps"]],
                    "avg_score": r["avg_score"],
                    "lesson": r["lesson"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    total = len(results["with"]) + len(results["without"])
    print()
    print("Wrote " + str(total) + " rows to " + str(out_path))


def _make_mock_reflector():
    # Mock reflector: deterministic self_score from observation content
    from agi_kit.reflect import Reflector

    class MockFast:
        def chat(self, msgs, max_tokens=8, temperature=0.0):
            txt = msgs[0].content if msgs else ""
            class R:
                content = "0.8" if "err" not in txt.lower() else "0.2"
            return R()
        model = "mock-fast"

    class MockMain:
        def chat(self, msgs, max_tokens=64, temperature=0.0):
            txt = msgs[0].content if msgs else ""
            class R:
                content = "Try a different approach next time." if "err" in txt.lower() else ""
            return R()
        model = "mock-main"

    return Reflector(
        main_llm=MockMain(),
        fast_llm=MockFast(),
        trace_path="logs/trace_with.jsonl",
        enabled=True,
    )


if __name__ == "__main__":
    main()