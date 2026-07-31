"""Agent base class and shared types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agi_kit.exceptions import AgentError, MaxStepsExceededError
from agi_kit.llms.base import LLM, LLMMessage, MessageRole
from agi_kit.observability import get_logger, metrics
from agi_kit.tools.base import ToolRegistry

log = get_logger(__name__)


@dataclass
class AgentStep:
    """One step in an agent's reasoning."""

    step: int
    thought: str
    action: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    observation: str | None = None
    final: str | None = None


class Agent(ABC):
    """Abstract agent."""

    paradigm: str = "base"

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry | None = None,
        max_steps: int = 10,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.history: list[LLMMessage] = []

    def _default_system_prompt(self) -> str:
        return (
            "You are an AGI research assistant.\n"
            "Output STRICT JSON in a single ```json ... ``` block.\n"
            "Tool call: {\"tool\": \"<name>\", \"args\": {...}}\n"
            "Final answer: {\"final\": \"<answer>\"}\n"
            f"Available tools:\n{self.tools.to_system_prompt_block()}"
        )

    @abstractmethod
    def step(self, task: str) -> AgentStep:
        """Run one reasoning step."""

    def run(self, task: str) -> str:
        """Run the full agent loop and return final answer."""
        self.history = [
            LLMMessage(role=MessageRole.SYSTEM, content=self.system_prompt),
            LLMMessage(role=MessageRole.USER, content=task),
        ]
        verdict = "max_steps"
        try:
            for i in range(1, self.max_steps + 1):
                with metrics.timed_tool("agent_step"):
                    s = self.step(task)
                metrics.inc(metrics.agent_steps_total, {"paradigm": self.paradigm})
                if s.final is not None:
                    verdict = "success"
                    log.info("agent_finished", paradigm=self.paradigm, steps=i)
                    return s.final
            verdict = "max_steps"
            raise MaxStepsExceededError(
                f"Agent reached max_steps={self.max_steps} without final answer"
            )
        except MaxStepsExceededError:
            metrics.inc(metrics.agent_runs_total, {"paradigm": self.paradigm, "verdict": "max_steps"})
            return "[agent] max_steps reached"
        except Exception as e:
            metrics.inc(metrics.agent_runs_total, {"paradigm": self.paradigm, "verdict": "error"})
            log.exception("agent_failed", paradigm=self.paradigm, error=str(e))
            raise AgentError(str(e)) from e
        finally:
            if verdict == "success":
                metrics.inc(metrics.agent_runs_total, {"paradigm": self.paradigm, "verdict": "success"})