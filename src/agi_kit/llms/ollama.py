"""Ollama backend."""
from __future__ import annotations

import time
from typing import Any

from agi_kit.exceptions import LLMError
from agi_kit.llms.base import LLM, LLMMessage, LLMResponse
from agi_kit.observability import get_logger, metrics

log = get_logger(__name__)


class OllamaBackend(LLM):
    """Ollama local inference backend.

    Requires Ollama daemon running. See https://ollama.com.
    """

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 60.0,
        auto_pull: bool = True,
    ) -> None:
        super().__init__(model=model)
        self.host = host
        self.timeout_s = timeout_s
        self.auto_pull = auto_pull
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import ollama  # type: ignore[import-not-found]
            except ImportError as e:
                raise LLMError(
                    "ollama package not installed. pip install ollama>=0.4"
                ) from e
            self._client = ollama.Client(host=self.host, timeout=self.timeout_s)
            if self.auto_pull:
                try:
                    self._client.show(self.model)
                except Exception:
                    log.info("pulling_model", model=self.model)
                    self._client.pull(self.model)
        return self._client

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        msg_dicts = [m.to_dict() for m in messages]
        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        if stop:
            options["stop"] = stop

        with metrics.timed_llm(backend="ollama", model=self.model):
            t0 = time.perf_counter()
            try:
                r = client.chat(model=self.model, messages=msg_dicts, options=options)
            except Exception as e:
                raise LLMError(f"Ollama call failed: {e}") from e
            dt = time.perf_counter() - t0

        content = r.get("message", {}).get("content", "")
        prompt_t = r.get("prompt_eval_count", 0) or 0
        comp_t = r.get("eval_count", 0) or 0
        if prompt_t:
            metrics.inc(metrics.llm_tokens_total, {"backend": "ollama", "model": self.model, "direction": "prompt"}) if False else None
        return LLMResponse(
            content=content,
            model=self.model,
            prompt_tokens=int(prompt_t),
            completion_tokens=int(comp_t),
            total_tokens=int(prompt_t + comp_t),
            latency_s=dt,
            raw=r,
        )