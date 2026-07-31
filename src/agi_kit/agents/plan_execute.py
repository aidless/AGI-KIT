"""Plan-and-Execute agent: plan first, then execute each step."""
from __future__ import annotations

import json
import re
from typing import Any

from agi_kit.agents.base import Agent, AgentStep
from agi_kit.llms.base import LLMMessage, MessageRole
from agi_kit.tools.base import ToolRegistry


class PlanExecuteAgent(Agent):
    """Plan the whole task first, then execute each step with a fresh LLM call."""

    paradigm = "plan_execute"

    PLAN_PROMPT = (
        "Decompose the task into a list of concrete steps.\n"
        "Each step is either: {\"tool\": \"<name>\", \"args\": {...}} or {\"final\": \"<answer>\"}.\n"
        "Output strict JSON: {\"plan\": [...]}"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._plan: list[dict[str, Any]] | None = None

    def step(self, task: str) -> AgentStep:
        if self._plan is None:
            self._plan = self._generate_plan(task)
            return AgentStep(step=1, thought=f"plan: {self._plan}")
        return self._execute_next()

    def _generate_plan(self, task: str) -> list[dict[str, Any]]:
        self.history.append(LLMMessage(role=MessageRole.USER, content=self.PLAN_PROMPT + "\n\nTask: " + task))
        r = self.llm.chat(self.history)
        m = re.search(r"\{.*\}", r.content, re.S)
        if not m:
            raise RuntimeError(f"Failed to parse plan: {r.content[:200]}")
        plan = json.loads(m.group(0)).get("plan", [])
        return plan if isinstance(plan, list) else []

    def _execute_next(self) -> AgentStep:
        if not self._plan:
            raise RuntimeError("empty plan")
        step_def = self._plan.pop(0)
        if "final" in step_def:
            return AgentStep(step=2, thought="final", final=str(step_def["final"]))
        tname = step_def.get("tool")
        targs = step_def.get("args", {}) or {}
        if tname not in self.tools:
            obs = f"unknown tool: {tname}"
        else:
            try:
                obs = self.tools.get(tname)(**targs)
            except Exception as e:
                obs = f"tool error: {e}"
        return AgentStep(step=2, action=tname, args=targs, observation=str(obs)[:2000])