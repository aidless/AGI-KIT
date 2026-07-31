"""L1 反思原语 - thinking log + self-critique + hindsight.

本模块实现 AGI 路线图 L1 阶段:每个 Agent step 后自动记录
{step, action, obs, self_score, hindsight} 到 JSONL trace。

两层模型设计(默认):
  - 主力 LLM(Qwen3-1.7B):hindsight 深度复盘、episode summary
  - 快速 LLM(Qwen3-0.6B):每步 self_score 打分(只 0/1 输出)

用法:
    reflector = Reflector(main_llm=..., fast_llm=..., trace_path="logs/trace.jsonl")
    reflector.log(step_idx=1, action={"tool":"x","args":{...}}, observation="...")
    lesson = reflector.summarize_episode(task="...", trace=trace_list)
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from agi_kit.llms.base import LLM, LLMMessage, MessageRole
from agi_kit.observability import get_logger

log = get_logger(__name__)

# ============================================================
# Data structures
# ============================================================
@dataclass
class ReflectionRecord:
    episode_id: str
    step: int
    ts: str
    action: dict
    observation: str
    rule_score: float
    llm_score: float | None
    self_score: float
    hindsight: str
    latency_ms: int

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class EpisodeSummary:
    episode_id: str
    task: str
    total_steps: int
    avg_self_score: float
    final_verdict: str
    lesson: str
    key_takeaways: list = field(default_factory=list)


# ============================================================
# Reflector
# ============================================================
class Reflector:
    SCORE_PROMPT = (
        "Rate the quality of this agent action/observation pair on a 0.0-1.0 scale.\n"
        "Output ONLY a single float, no other text.\n"
        "0.0 = clearly wrong / wasted step.  0.5 = progress but unclear.  1.0 = clearly correct.\n"
        "Action: {action}\n"
        "Observation: {obs}"
    )

    HINDSIGHT_PROMPT = (
        "You are a self-critique module. Look at one agent step and write ONE sentence (max 25 words)\n"
        "answering: \"If the agent could redo this step, what should it do differently?\"\n"
        "Be concrete and actionable. Skip if no improvement is possible.\n"
        "Action: {action}\n"
        "Observation: {obs}"
    )

    EPISODE_PROMPT = (
        "You are a post-mortem analyzer for an agent run. Given the task and the trace,\n"
        "write a short LESSON (one paragraph) and 2-4 KEY TAKEAWAYS (one short phrase each).\n"
        "Output STRICT JSON:\n"
        'ZJZ_JSON_FMT_ZJZ\n'
        "\nTask: {task}\n\nTrace:\n{trace}"
    )

    def __init__(self, main_llm, trace_path="logs/trace.jsonl",
                 fast_llm=None, episode_path=None, enabled=True):
        self.main = main_llm
        self.fast = fast_llm or main_llm
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        if episode_path:
            self.episode_path = Path(episode_path)
        else:
            self.episode_path = self.trace_path.parent / "episode_summary.jsonl"
        self.episode_path.parent.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._lock = Lock()
        self._episode_id = uuid.uuid4().hex[:12]
        log.info("reflector_init", episode_id=self._episode_id,
                 trace_path=str(self.trace_path))

    @staticmethod
    def _rule_score(action, observation):
        obs = (observation or "").lower().strip()
        if not obs:
            return 0.2
        if any(k in obs for k in ("err:", "error:", "exception",
                                   "traceback", "unknown tool",
                                   "tool error", "n/a", "none")):
            return 0.2
        if obs in ("(empty)", "(无输出)", "(no text)"):
            return 0.4
        if "final" in str(action).lower() or len(obs) <= 80:
            return 0.9
        return 0.7

    def _llm_score(self, action, observation):
        try:
            act_s = json.dumps(action, ensure_ascii=False)[:300]
            obs_s = (observation or "")[:500]
            prompt = self.SCORE_PROMPT.format(action=act_s, obs=obs_s)
            r = self.fast.chat(
                [LLMMessage(role=MessageRole.USER, content=prompt)],
                max_tokens=8, temperature=0.0
            )
            txt = r.content.strip()
            m = re.search(r"[01]?\.\d+|[01]", txt)
            if not m:
                return None
            return max(0.0, min(1.0, float(m.group(0))))
        except Exception as e:
            log.warning("llm_score_failed", error=str(e))
            return None

    def _hindsight(self, action, observation):
        try:
            act_s = json.dumps(action, ensure_ascii=False)[:300]
            obs_s = (observation or "")[:500]
            prompt = self.HINDSIGHT_PROMPT.format(action=act_s, obs=obs_s)
            r = self.main.chat(
                [LLMMessage(role=MessageRole.USER, content=prompt)],
                max_tokens=64, temperature=0.0
            )
            return r.content.strip().replace("\n", " ")[:300]
        except Exception as e:
            log.warning("hindsight_failed", error=str(e))
            return ""

    def log(self, step_idx, action, observation):
        if not self.enabled:
            return 0.5
        t0 = time.perf_counter()
        rule = self._rule_score(action, observation)
        llm = self._llm_score(action, observation)
        self_score = rule * 0.4 + (llm if llm is not None else rule) * 0.6
        hindsight = self._hindsight(action, observation) if self_score < 0.85 else ""
        latency_ms = int((time.perf_counter() - t0) * 1000)
        rec = ReflectionRecord(
            episode_id=self._episode_id,
            step=step_idx,
            ts=datetime.now(timezone.utc).isoformat(),
            action=dict(action),
            observation=(observation or "")[:2000],
            rule_score=rule,
            llm_score=llm,
            self_score=round(self_score, 3),
            hindsight=hindsight,
            latency_ms=latency_ms,
        )
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(rec.to_jsonl() + "\n")
        return self_score

    def summarize_episode(self, task, trace, verdict="unknown"):
        if not self.enabled:
            return EpisodeSummary(
                episode_id=self._episode_id, task=task,
                total_steps=len(trace), avg_self_score=0.5,
                final_verdict=verdict, lesson="(reflector disabled)"
            )
        scores = [t.get("self_score", 0.5) for t in trace]
        avg = sum(scores) / max(len(scores), 1)
        lesson = ""
        takeaways = []
        try:
            compact_lines = []
            for t in trace[-15:]:
                act = json.dumps(t.get("action"))[:80]
                obs = (t.get("observation") or "")[:120]
                compact_lines.append(
                    "  step " + str(t.get("step")) + ": action=" + act +
                    " obs=" + obs + " score=" + str(t.get("self_score"))
                )
            compact = "\n".join(compact_lines)
            json_fmt = '{"lesson": "<paragraph>", "takeaways": ["t1", "t2", ...]}'
            prompt = self.EPISODE_PROMPT.format(
                task=task[:500], trace=compact
            ).replace("ZJZ_JSON_FMT_ZJZ", json_fmt)
            r = self.main.chat(
                [LLMMessage(role=MessageRole.USER, content=prompt)],
                max_tokens=400, temperature=0.0
            )
            txt = r.content.strip()
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                data = json.loads(m.group(0))
                lesson = data.get("lesson", txt[:600])
                takeaways = list(data.get("takeaways") or [])[:5]
            else:
                lesson = txt[:600]
        except Exception as e:
            log.warning("episode_summary_failed", error=str(e))
            lesson = "(summary failed)"
        summary = EpisodeSummary(
            episode_id=self._episode_id,
            task=task,
            total_steps=len(trace),
            avg_self_score=round(avg, 3),
            final_verdict=verdict,
            lesson=lesson,
            key_takeaways=takeaways,
        )
        with self._lock:
            with self.episode_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(summary), ensure_ascii=False) + "\n")
        return summary

    def recent(self, n=5):
        if not self.trace_path.exists():
            return []
        with self.trace_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return out