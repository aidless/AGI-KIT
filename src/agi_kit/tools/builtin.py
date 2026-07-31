"""Built-in tools: file, shell, math, etc."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from agi_kit.tools.base import tool


# ---------- calculator ----------

@tool(
    "calculator",
    "Evaluate an arithmetic expression. Supports +, -, *, /, **, %, ^, (), |, &",
    parameters={
        "type": "object",
        "properties": {
            "expr": {"type": "string", "description": "Math expression to evaluate"}
        },
        "required": ["expr"],
    },
)
def calculator(expr: str) -> str:
    """Evaluate arithmetic expression safely."""
    import math
    if not re.fullmatch(r"[\d\s+\-*/().%^|&]+", expr):
        return "error: expression contains illegal characters"
    try:
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"error: {e}"


# ---------- file tools ----------

@tool(
    "read_file",
    "Read a local text file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "default": 3000},
        },
        "required": ["path"],
    },
)
def read_file(path: str, max_chars: int = 3000) -> str:
    p = Path(path)
    if not p.exists():
        return f"error: file not found: {path}"
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception as e:
        return f"error: {e}"


@tool(
    "read_pdf",
    "Read a PDF file and return text",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_pages": {"type": "integer", "default": 5},
        },
        "required": ["path"],
    },
)
def read_pdf(path: str, max_pages: int = 5) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "error: pypdf not installed. pip install pypdf"
    p = Path(path)
    if not p.exists():
        return f"error: file not found: {path}"
    try:
        reader = PdfReader(str(p))
        chunks = []
        for i, page in enumerate(reader.pages[:max_pages]):
            chunks.append(f"[page {i+1}]\n{page.extract_text()}")
        return "\n".join(chunks) or "(no text)"
    except Exception as e:
        return f"error: {e}"


@tool(
    "list_dir",
    "List files in a directory matching a glob pattern",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "pattern": {"type": "string", "default": "*"},
        },
        "required": ["path"],
    },
)
def list_dir(path: str, pattern: str = "*") -> str:
    import glob
    p = Path(path)
    if not p.exists():
        return f"error: directory not found: {path}"
    try:
        files = glob.glob(str(p / pattern))
        return "\n".join(os.path.basename(f) for f in files[:100]) or "(empty)"
    except Exception as e:
        return f"error: {e}"


@tool(
    "shell",
    "Execute a short shell command (no pipes, redirections, or semicolons)",
    parameters={
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
        "required": ["cmd"],
    },
    tags=["dangerous"],
)
def shell(cmd: str) -> str:
    if any(x in cmd for x in ["|", ">", "<", "&", ";"]):
        return "error: disallowed characters in command"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return (r.stdout + r.stderr).strip()[:3000] or "(no output)"
    except Exception as e:
        return f"error: {e}"


# ---------- echo (testing) ----------

@tool(
    "echo",
    "Echo back the input (sanity check tool)",
    parameters={
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    },
)
def echo(msg: str) -> str:
    return msg