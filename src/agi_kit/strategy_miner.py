"""L2 extension: auto-write successful hindsight patterns into Playbook.

After each successful episode, extract actionable lessons and add them as
strategies. This closes the L2 loop without manual seeding.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import List

from agi_kit.llms.base import LLM, LLMMessage, MessageRole
from agi_kit.observability import get_logger
from agi_kit.playbook import Playbook

log = get_logger(__name__)


EXTRACT_PROMPT = (
    "You are a learning agent that extracts reusable strategies from successful task traces.\n"
    "Given a successful task and its trace, output 0-2 SHORT STRATEGIES (one sentence each).\n"
    "Each strategy should be generalizable (not tied to one specific task) and actionable.\n"
    "Output STRICT JSON: {\"strategies\": [\"s1\", \"s2\"]}. Empty list if no generalizable lesson.\n"
    "\nTask: {task}\n"
    "Final answer: {final}\n"
    "Key hindsight from steps: {hints}\n"
)


@dataclass
class ExtractedStrategy:
    pattern: str
    strategy: str
    success_rate: float
    source_episode: str


def extract_strategies_from_episode(task: str,
                                    final: str,
                                    hindsight_lines: List[str],
                                    llm: LLM,
                                    max_tokens=200) -> List[ExtractedStrategy]:
    """Call LLM to extract generalizable strategies from one episode."""
    hints_text = " | ".join(h for h in hindsight_lines if h)[:500] or "(none)"
    prompt = EXTRACT_PROMPT.format(task=task[:300], final=final[:80], hints=hints_text)
    try:
        r = llm.chat(
            [LLMMessage(role=MessageRole.USER, content=prompt)],
            max_tokens=max_tokens, temperature=0.0,
        )
        txt = r.content.strip()
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return []
        data = json.loads(m.group(0))
        out = []
        for s in data.get("strategies", [])[:2]:
            s = s.strip()
            if 15 < len(s) < 300:
                out.append(s)
        return out
    except Exception as e:
        log.warning("extract_strategies_failed", error=str(e))
        return []


def write_extracted_to_playbook(playbook: Playbook,
                                strategies: List[ExtractedStrategy]) -> int:
    """Add strategies to playbook; dedup by similarity (simple hash dedup)."""
    if not strategies:
        return 0
    added = 0
    existing_hashes = set()
    for s in playbook.strategies:
        existing_hashes.add(_hash_strategy(s.pattern + s.strategy))
    for es in strategies:
        h = _hash_strategy(es.pattern + es.strategy)
        if h in existing_hashes:
            continue
        existing_hashes.add(h)
        playbook.add(
            pattern=es.pattern,
            strategy=es.strategy,
            success_rate=es.success_rate,
        )
        added += 1
    if added > 0:
        playbook.save()
    return added


def _hash_strategy(text):
    # Simple normalization for dedup
    return " ".join(text.lower().split())[:200]


def should_extract(verdict: str, avg_score: float, threshold=0.7):
    return verdict == "success" and avg_score >= threshold