"""Abstract LLM interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """Single chat message."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class LLMResponse:
    """Standardized LLM response."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    raw: Any = None


class LLM(ABC):
    """Abstract base class for all LLM backends."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self._kwargs = kwargs

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages and get a response."""

    def is_available(self) -> bool:
        """Health check: can this backend be reached?"""
        try:
            r = self.chat(
                [LLMMessage(role=MessageRole.USER, content="ping")],
                max_tokens=4,
            )
            return bool(r.content)
        except Exception:
            return False