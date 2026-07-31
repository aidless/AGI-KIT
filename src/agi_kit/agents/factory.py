"""Agent factory."""
from __future__ import annotations

from typing import Any

from agi_kit.agents.base import Agent
from agi_kit.llms.base import LLM
from agi_kit.tools.base import ToolRegistry


def create_agent(paradigm: str, llm: LLM, **kwargs: Any) -> Agent:
    if paradigm == "react":
        from agi_kit.agents.react import ReActAgent
        return ReActAgent(llm=llm, **kwargs)
    elif paradigm == "plan_execute":
        from agi_kit.agents.plan_execute import PlanExecuteAgent
        return PlanExecuteAgent(llm=llm, **kwargs)
    elif paradigm == "reflexion":
        from agi_kit.agents.react import ReActAgent
        # Reflexion = ReAct + self-critique; implemented as ReAct with longer max_steps for now
        return ReActAgent(llm=llm, max_steps=kwargs.get("max_steps", 20), **kwargs)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}. Use: react | plan_execute | reflexion")