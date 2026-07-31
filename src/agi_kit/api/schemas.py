"""Pydantic API schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """Run an agent on a task."""

    task: str = Field(..., description="The task to execute", min_length=1)
    paradigm: str | None = Field(default=None, description="react | plan_execute | reflexion")
    model: str | None = Field(default=None, description="Override LLM model")
    max_steps: int | None = Field(default=None, ge=1, le=50)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    """Agent execution result."""

    final: str
    steps: int
    duration_s: float
    model: str
    paradigm: str
    history_tail: list[dict[str, Any]] = Field(default_factory=list)


class ToolCallRequest(BaseModel):
    """Directly invoke a tool."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    """Tool execution result."""

    tool: str
    result: str


class ToolSpec(BaseModel):
    """Tool metadata."""

    name: str
    description: str
    parameters: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check."""

    status: str
    version: str
    llm_backend: str
    llm_model: str
    tools_count: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None