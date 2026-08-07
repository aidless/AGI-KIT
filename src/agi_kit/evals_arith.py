"""Real A/B arithmetic evaluation for executable Ollama model names.

An unconverted Hugging Face checkpoint is not an executable deployment
candidate for this evaluator and must never receive an estimated score.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments"))


def _normalize(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace(",", "").replace(" ", "")
    return s


# 5-task arithmetic benchmark for fast A/B
SUB_TASKS = [
    ("arithmetic", "Use calculator to compute 17 * 23, then final", "391"),
    ("arithmetic", "Use calculator to compute 256 + 789, then final", "1045"),
    ("arithmetic", "Use calculator to compute 2**10, then final", "1024"),
    ("arithmetic", "Use calculator to compute 88 * 88, then final", "7744"),
    ("chained",    "Use calculator to compute 7*6, then echo the result, then final", "42"),
]


def _is_hf_path(model_id: str) -> bool:
    """Detect HF-style path (e.g. logs/continual/gen-1)."""
    p = Path(model_id)
    if p.exists() and p.is_dir():
        return True
    return False


def _read_gen_meta(model_dir: str) -> dict:
    """Read generation metadata if present."""
    meta_path = Path(model_dir) / "gen_meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def eval_arithmetic(model_id: str,
                    n_tasks: int = 5,
                    max_steps: int = 4,
                    timeout_per_task: float = 30.0,
                    base_model: str = "qwen3:1.7b") -> float:
    """Run the real arithmetic subset; return accuracy in [0, 1]."""
    if _is_hf_path(model_id):
        raise ValueError(
            "candidate checkpoint is not an executable Ollama model; "
            "convert and evaluate it before invoking the safety gate"
        )
    # Ollama branch: real run
    try:
        from full_agent import OllamaBackend, Agent
    except Exception as e:
        return 0.0
    try:
        llm = OllamaBackend(model=model_id)
    except Exception:
        return 0.0
    n = min(n_tasks, len(SUB_TASKS))
    correct = 0
    completed = 0
    for i, (cat, q, gold) in enumerate(SUB_TASKS[:n]):
        try:
            agent = Agent(llm=llm, max_steps=max_steps)
            t0 = time.time()
            pred = agent.run(q)
            dt = time.time() - t0
            if dt > timeout_per_task:
                continue
            completed += 1
            if _normalize(pred) == _normalize(gold):
                correct += 1
        except Exception:
            continue
    if completed == 0:
        return 0.0
    return correct / max(n, 1)


def make_eval_fn(n_tasks: int = 5,
                 max_steps: int = 4,
                 base_model: str = "qwen3:1.7b") -> Callable:
    """Build an eval_fn suitable for ContinualLoop.eval_fn."""
    def _eval(model_id):
        return eval_arithmetic(model_id, n_tasks=n_tasks,
                                max_steps=max_steps, base_model=base_model)
    return _eval


# Baseline measurement (called once at startup)
def measure_baseline(model_id: str = "qwen3:1.7b",
                     n_tasks: int = 5,
                     max_steps: int = 4) -> float:
    """Measure baseline accuracy; used as initial A/B reference."""
    return eval_arithmetic(model_id, n_tasks=n_tasks, max_steps=max_steps)
