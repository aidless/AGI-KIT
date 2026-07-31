"""LLM backend factory."""
from __future__ import annotations

from typing import Any

from agi_kit.config import LLMSettings
from agi_kit.exceptions import LLMError
from agi_kit.llms.base import LLM


def create_llm(settings: LLMSettings | None = None, **overrides: Any) -> LLM:
    """Create an LLM backend from settings or kwargs."""
    if settings is None:
        from agi_kit.config import get_settings
        settings = get_settings().llm
    cfg = {**settings.model_dump(), **overrides}

    backend = cfg.pop("backend", "ollama")
    model = cfg.pop("model", "qwen3:1.7b")

    if backend == "ollama":
        from agi_kit.llms.ollama import OllamaBackend
        return OllamaBackend(model=model, **{k: v for k, v in cfg.items() if k in {"host", "timeout_s", "auto_pull"}})
    elif backend == "transformers":
        from agi_kit.llms.transformers_backend import TransformersBackend
        return TransformersBackend(model=model, **{k: v for k, v in cfg.items() if k in {"device", "dtype", "cache_dir"}})
    elif backend in ("openai", "openai_compat"):
        from agi_kit.llms.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model=model, **{k: v for k, v in cfg.items() if k in {"base_url", "api_key", "timeout_s"}})
    else:
        raise LLMError(f"Unknown backend: {backend}. Use: ollama | transformers | openai")