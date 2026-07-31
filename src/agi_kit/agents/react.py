"""ReAct (Reason + Act) agent."""
from __future__ import annotations

import json
import re
from typing import Any

from agi_kit.agents.base import Agent, AgentStep
from agi_kit.exceptions import ParseError
from agi_kit.llms.base import LLMMessage, MessageRole

_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


class ReActAgent(Agent):
    """ReAct: Thought -> Action -> Observation -> repeat."""

    paradigm = "react"

    def step(self, task: str) -> AgentStep:
        resp = self.llm.chat(self.history, temperature=0.0)
        self.history.append(LLMMessage(role=MessageRole.ASSISTANT, content=resp.content))
        action = self._parse_action(resp.content)
        s = AgentStep(step=len(self.history) // 2, thought=resp.content)
        if action is None:
            self.history.append(LLMMessage(
                role=MessageRole.USER,
                content="Please output valid JSON in ```json ... ``` block."
            ))
            return s
        if "final" in action:
            s.final = str(action["final"])
            return s
        tname = action.get("tool")
        targs = action.get("args", {}) or {}
        s.action = tname
        s.args = targs
        if tname not in self.tools:
            obs = f"unknown tool: {tname}"
        else:
            try:
                obs = self.tools.get(tname)(**targs)
            except Exception as e:
                obs = f"tool error: {e}"
        s.observation = str(obs)[:2000]
        self.history.append(LLMMessage(
            role=MessageRole.USER,
            content=f"Observation: {obs}"
        ))
        return s

    def _parse_action(self, text: str) -> dict[str, Any] | None:
        m = _JSON_RE.search(text)
        if m:
            cand = m.group(1)
        else:
            m2 = re.search(r"\{.*?\}", text, re.S)
            if not m2:
                return None
            cand = m2.group(0)
        try:
            return json.loads(cand)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON from LLM: {e}\nText: {text[:200]}")