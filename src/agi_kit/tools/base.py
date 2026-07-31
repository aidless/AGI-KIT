"""Tool registry with JSON Schema-based parameter validation."""
from __future__ import annotations

import functools
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from agi_kit.exceptions import ToolError, ToolNotFoundError
from agi_kit.observability import get_logger, metrics

log = get_logger(__name__)


@dataclass
class Tool:
    """A registered tool with JSON Schema spec."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema dict
    fn: Callable[..., Any]
    is_async: bool = False
    tags: list[str] = field(default_factory=list)

    def to_openai_spec(self) -> dict[str, Any]:
        """Return OpenAI-compatible tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, **kwargs: Any) -> Any:
        with metrics.timed_tool(self.name):
            try:
                return self.fn(**kwargs)
            except Exception as e:
                log.exception("tool_failed", tool=self.name, error=str(e))
                raise ToolError(self.name, str(e)) from e


class ToolRegistry:
    """Process-wide tool registry (singleton)."""

    _instance: "ToolRegistry | None" = None

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            log.warning("tool_override", tool=tool.name)
        self._tools[tool.name] = tool
        log.debug("tool_registered", tool=tool.name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFoundError(name, f"Available: {list(self._tools.keys())}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_system_prompt_block(self) -> str:
        """Render tools as a system-prompt block for non-native tool-use LLMs."""
        lines = []
        for t in self._tools.values():
            props = ", ".join(t.parameters.get("properties", {}).keys())
            lines.append(f"- {t.name}({props}): {t.description}")
        return "\n".join(lines)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_spec() for t in self._tools.values()]


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as a Tool.

    If `parameters` is not given, they are inferred from the function signature.
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        params_schema = parameters or _infer_schema(fn)
        t = Tool(
            name=name,
            description=description,
            parameters=params_schema,
            fn=fn,
            is_async=inspect.iscoroutinefunction(fn),
            tags=tags or [],
        )
        ToolRegistry().register(t)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wrapper

    return deco


def _infer_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema object from function signature."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        ann = hints.get(pname, str)
        json_type = _python_to_json_type(ann)
        properties[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            properties[pname]["default"] = param.default
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _python_to_json_type(tp: Any) -> str:
    origin = getattr(tp, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    if tp in (int,):
        return "integer"
    if tp in (float,):
        return "number"
    if tp is bool:
        return "boolean"
    return "string"