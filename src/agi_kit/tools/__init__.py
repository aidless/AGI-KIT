"""Tool system with JSON Schema validation."""
from agi_kit.tools.base import Tool, ToolRegistry, tool
from agi_kit.tools.builtin import (
    calculator,
    echo,
    list_dir,
    read_file,
    read_pdf,
    shell,
)
from agi_kit.tools.web import web_fetch, web_search
from agi_kit.tools.rag_tools import rag_add, rag_clear, rag_search

__all__ = [
    "Tool", "ToolRegistry", "tool",
    "calculator", "echo", "list_dir", "read_file", "read_pdf", "shell",
    "web_fetch", "web_search",
    "rag_add", "rag_search", "rag_clear",
]