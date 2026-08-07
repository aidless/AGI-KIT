from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.llms.base import LLMMessage, MessageRole
from agi_kit.llms.ollama import OllamaBackend


def test_ollama_backend_forwards_optional_structured_format(monkeypatch):
    captured = {}

    class Client:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"message": {"content": "{}"}}

    backend = OllamaBackend(auto_pull=False)
    backend._client = Client()
    schema = {"type": "object"}
    backend.chat([LLMMessage(role=MessageRole.USER, content="x")], format=schema)
    assert captured["format"] == schema
