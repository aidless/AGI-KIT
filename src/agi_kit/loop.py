"""L3 Continual Learning Loop.

Drives the AGI self-improvement cycle:
  Task -> Agent(L1+L2) -> successful traces -> ExperienceBuffer
                                                      |
                                          every N episodes / playbook +M strategies
                                                      |
                                              SFT(base_model, buffer)
                                                      |
                                          A/B gate (new >= 95% baseline?)
                                                      |
                                          YES -> new generation
                                          NO  -> keep previous

Generations tracked. Each retrain produces models/sft-gen{N}/.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agi_kit.observability import get_logger

log = get_logger(__name__)


# ============================================================
# Data structures
# ============================================================
@dataclass
class TraceRecord:
    """One successful episode stored for potential SFT."""

    episode_id: str
    task: str
    final: str
    avg_self_score: float
    steps: list
    ts: str = ""
    generation: int = 0
    task_kind: str = ""

    def to_jsonl(self):
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class GenerationRecord:
    """One generation transition (after SFT)."""

    generation: int
    base_model: str
    new_model: str
    episodes_used: int
    train_samples: int
    train_seconds: float
    eval_baseline_acc: float
    eval_new_acc: float
    accepted: bool
    reason: str = ""
    ts: str = ""


# ============================================================
# Experience buffer
# ============================================================
class ExperienceBuffer:
    """JSONL-backed store of high-quality successful traces."""

    def __init__(self, path="logs/buffer/buffer.jsonl",
                 min_score=0.7,
                 max_size=5000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_score = min_score
        self.max_size = max_size
        self.records = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.records.append(TraceRecord(**json.loads(line)))
                except Exception:
                    pass

    def save(self):
        with self.path.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(r.to_jsonl() + "\n")

    def add(self, record: TraceRecord):
        if record.avg_self_score < self.min_score:
            return False
        if not record.final:
            return False
        self.records.append(record)
        # Trim if too large (FIFO)
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]
        return True

    def size(self):
        return len(self.records)

    def sample(self, n=100):
        # Stratified sample by task_kind when possible
        if len(self.records) <= n:
            return list(self.records)
        # Group by task_kind
        groups = {}
        for r in self.records:
            groups.setdefault(r.task_kind, []).append(r)
        out = []
        kinds = list(groups.keys())
        per_kind = max(1, n // max(len(kinds), 1))
        for k in kinds:
            bucket = groups[k]
            if len(bucket) <= per_kind:
                out.extend(bucket)
            else:
                # Evenly spaced sample
                idx = [int(i * (len(bucket) - 1) / max(per_kind - 1, 1))
                       for i in range(per_kind)]
                for i in idx:
                    if i < len(bucket):
                        out.append(bucket[i])
        # Fill up to n if needed
        if len(out) < n:
            seen = set(id(r) for r in out)
            for r in self.records:
                if id(r) not in seen:
                    out.append(r)
                    if len(out) >= n:
                        break
        return out[:n]

    def diversity(self):
        kinds = set(r.task_kind for r in self.records if r.task_kind)
        return len(kinds)

    def clear(self):
        self.records = []

    def reset_for_new_generation(self):
        # Called after successful SFT: keep records but mark them old
        self.save()


# ============================================================
# Format trace -> SFT training examples
# ============================================================
def format_trace_for_sft(trace: TraceRecord):
    """Convert one successful trace into a chat-style training text.

    Returns a dict with 'messages' list (system/user/assistant),
    or None if trace is malformed.
    """
    msgs = []
    # System: replay what agent saw
    msgs.append({
        "role": "system",
        "content": (
            "You are an AGI research assistant. Output STRICT JSON in "
            "triple-backtick json block. Tool call: "
            '{"tool":"<name>","args":{...}}. '
            'Final answer: {"final":"<answer>"}.'
        ),
    })
    # User: the task
    msgs.append({"role": "user", "content": trace.task})
    # Assistant + tool calls: replay each step
    for step in trace.steps:
        if step.get("action"):
            act = step["action"]
            # Mock action text: the agent's thought + JSON tool call
            content = "I will use the " + str(act.get("tool", "?")) + " tool.\n"
            content += "```json\n" + json.dumps(act, ensure_ascii=False) + "\n```"
            msgs.append({"role": "assistant", "content": content})
        if step.get("obs") is not None:
            msgs.append({
                "role": "user",
                "content": "Observation: " + str(step["obs"])[:800],
            })
    # Final answer
    if trace.final is not None:
        msgs.append({
            "role": "assistant",
            "content": "```json\n" + json.dumps({"final": trace.final},
                                                ensure_ascii=False) + "\n```",
        })
    return {"messages": msgs}


# ============================================================
# Safety gate (A/B eval)
# ============================================================
def default_safety_check(new_model_dir: str,
                         baseline_acc: float,
                         eval_fn: Optional[Callable] = None,
                         threshold: float = 0.95) -> dict:
    """Run a tiny eval to decide whether new model replaces baseline.

    Returns: {"accepted": bool, "new_acc": float, "reason": str}
    """
    if eval_fn is None:
        # Without an eval_fn, accept by default (caller's responsibility)
        return {"accepted": True, "new_acc": baseline_acc,
                "reason": "no_eval_fn_passed"}
    try:
        new_acc = eval_fn(new_model_dir)
    except Exception as e:
        return {"accepted": False, "new_acc": 0.0,
                "reason": "eval_failed: " + str(e)}
    accepted = new_acc >= baseline_acc * threshold
    return {
        "accepted": accepted,
        "new_acc": new_acc,
        "reason": ("passed" if accepted else "regressed_below_threshold"),
    }


# ============================================================
# Continual loop driver
# ============================================================
class ContinualLoop:
    """Drives task -> agent -> buffer -> periodic retrain -> generation bump."""

    def __init__(self,
                 run_episode_fn: Callable,        # run_episode_fn(task, gold=None) -> {"final", "verdict", "steps", "avg_score", "trace"}
                 reflector=None,
                 playbook=None,
                 meta=None,
                 buffer: Optional[ExperienceBuffer] = None,
                 retrain_fn: Optional[Callable] = None,  # retrain_fn(samples, base_model, out_dir) -> {"out_dir", "samples", "seconds"}
                 eval_fn: Optional[Callable] = None,    # eval_fn(model_dir) -> accuracy (0..1)
                 safety_threshold: float = 0.95,
                 retrain_every: int = 5,                # episodes per generation
                 retrain_min_buffer: int = 5,
                 out_dir: str = "logs/continual",
                 base_model: str = "qwen3:1.7b"):
        self.run_episode = run_episode_fn
        self.reflector = reflector
        self.playbook = playbook
        self.meta = meta
        self.buffer = buffer or ExperienceBuffer()
        self.retrain_fn = retrain_fn
        self.eval_fn = eval_fn
        self.safety_threshold = safety_threshold
        self.retrain_every = retrain_every
        self.retrain_min_buffer = retrain_min_buffer
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.base_model = base_model
        self.generation = 0
        self.current_model = base_model
        self.history = []                # per-episode outcomes
        self.generations = []            # per-generation transitions

    # ---- per-episode ----
    def step(self, task, gold=None, task_kind=""):
        self.generation += 0  # episodes do not bump generation; only retrain does
        t0 = time.time()
        try:
            res = self.run_episode(task, gold=gold)
        except Exception as e:
            res = {"final": None, "verdict": "error",
                   "steps": [], "avg_score": 0.0, "error": str(e)}
        res["task"] = task
        res["gold"] = gold
        res["task_kind"] = task_kind
        res["seconds"] = round(time.time() - t0, 2)
        self.history.append(res)
        # Buffer high-quality success
        if res.get("verdict") == "success" and res.get("avg_score", 0) >= self.buffer.min_score:
            rec = TraceRecord(
                episode_id=str(len(self.history)),
                task=task,
                final=str(res.get("final", "")),
                avg_self_score=res.get("avg_score", 0.0),
                steps=res.get("steps", []),
                ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                generation=self.generation,
                task_kind=task_kind,
            )
            self.buffer.add(rec)
        return res

    # ---- retrain decision ----
    def should_retrain(self):
        episodes_since = len(self.history) - self._last_retrain_at
        enough_buffer = self.buffer.size() >= self.retrain_min_buffer
        return (
            episodes_since >= self.retrain_every and
            enough_buffer and
            self.retrain_fn is not None
        )

    _last_retrain_at = 0

    def maybe_retrain(self, baseline_acc=1.0):
        if not self.should_retrain():
            return None
        return self.retrain(baseline_acc=baseline_acc)

    def retrain(self, baseline_acc=1.0):
        if self.retrain_fn is None:
            return None
        samples = self.buffer.sample(n=min(self.buffer.size(), 200))
        sft_examples = [format_trace_for_sft(r) for r in samples]
        sft_examples = [e for e in sft_examples if e]
        self.generation += 1
        out_dir = self.out_dir / ("gen-" + str(self.generation))
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        info = self.retrain_fn(
            samples=sft_examples,
            base_model=self.current_model,
            out_dir=str(out_dir),
        )
        train_seconds = round(time.time() - t0, 2)
        # A/B safety gate
        gate = default_safety_check(
            new_model_dir=str(out_dir),
            baseline_acc=baseline_acc,
            eval_fn=self.eval_fn,
            threshold=self.safety_threshold,
        )
        accepted = gate["accepted"]
        new_model_path = str(out_dir)
        if accepted:
            self.current_model = new_model_path
        gen_record = GenerationRecord(
            generation=self.generation,
            base_model=self.current_model if accepted else self.current_model,
            new_model=new_model_path,
            episodes_used=len(self.history) - self._last_retrain_at,
            train_samples=len(sft_examples),
            train_seconds=train_seconds,
            eval_baseline_acc=baseline_acc,
            eval_new_acc=gate["new_acc"],
            accepted=accepted,
            reason=gate["reason"],
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self.generations.append(gen_record)
        self._last_retrain_at = len(self.history)
        # Persist
        with (self.out_dir / "generations.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(gen_record), ensure_ascii=False) + "\n")
        self.buffer.save()
        log.info("continual_retrain",
                 gen=self.generation,
                 accepted=accepted,
                 train_samples=len(sft_examples),
                 new_acc=gate["new_acc"])
        return gen_record

    # ---- full run ----
    def run(self, tasks, gold_per_task=None, baseline_acc=1.0,
            verbose=True):
        """Run all tasks, calling retrain periodically.

        tasks: list of task strings (or list of {task, kind} dicts)
        gold_per_task: optional list of expected answers
        """
        n = len(tasks)
        self._last_retrain_at = 0
        for i, t in enumerate(tasks):
            if isinstance(t, dict):
                task = t["task"]
                kind = t.get("kind", "")
            else:
                task = t
                kind = ""
            gold = gold_per_task[i] if gold_per_task else None
            res = self.step(task, gold=gold, task_kind=kind)
            if verbose:
                print("[" + str(i + 1) + "/" + str(n) + "] " +
                      "kind=" + kind +
                      " verdict=" + res.get("verdict", "?") +
                      " score=" + str(round(res.get("avg_score", 0), 2)) +
                      " buf=" + str(self.buffer.size()) +
                      " gen=" + str(self.generation))
            # Periodic retrain
            if self.should_retrain():
                rec = self.retrain(baseline_acc=baseline_acc)
                if rec and verbose:
                    print("  >> generation " + str(rec.generation) +
                          " accepted=" + str(rec.accepted) +
                          " new_acc=" + str(round(rec.eval_new_acc, 3)) +
                          " samples=" + str(rec.train_samples))
        # Final retrain at end if buffer still has data
        if self.buffer.size() >= self.retrain_min_buffer and self.retrain_fn:
            rec = self.retrain(baseline_acc=baseline_acc)
            if rec and verbose:
                print("  >> final generation " + str(rec.generation) +
                      " accepted=" + str(rec.accepted))
        return {
            "episodes": self.history,
            "generations": self.generations,
            "final_generation": self.generation,
            "current_model": self.current_model,
            "buffer_size": self.buffer.size(),
        }