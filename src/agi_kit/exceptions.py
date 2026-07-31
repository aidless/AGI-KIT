"""Custom exception hierarchy for AGI Kit.

All exceptions inherit from `AGIKitError` for easy catching.
"""

from __future__ import annotations


class AGIKitError(Exception):
    """Base exception for all AGI Kit errors."""


class ConfigError(AGIKitError):
    """Configuration is invalid or missing."""


class LLMError(AGIKitError):
    """LLM backend failure (timeout, connection, response)."""


class ToolError(AGIKitError):
    """Tool execution failed."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"[{tool_name}] {message}")
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Requested tool does not exist in the registry."""


class AgentError(AGIKitError):
    """Agent loop or reasoning failure."""


class MaxStepsExceededError(AgentError):
    """Agent exceeded maximum iteration steps."""


class ParseError(AgentError):
    """Failed to parse LLM output (e.g., invalid JSON action)."""


class RAGError(AGIKitError):
    """RAG store / retrieval failure."""


class EvalError(AGIKitError):
    """Evaluation pipeline failure."""


class TrainingError(AGIKitError):
    """Training pipeline failure."""


class APIError(AGIKitError):
    """API server failure."""


class AuthError(APIError):
    """Authentication / authorization failure."""