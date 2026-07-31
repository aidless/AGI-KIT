"""OpenAI-compatible backend (works with HF Inference, OpenRouter, vLLM, etc)."""
from __future__ import annotations

import time
from typing import Any

from agi_kit.exceptions import LLMError
from agi_kit.llms.base import LLM, LLMMessage, LLMResponse
from agi_kit.observability import get_logger, metrics

log = get_logger(__name__)


class OpenAICompatBackend(LLM):
    """Any OpenAI-compatible HTTP API."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise LLMError("openai package required. pip install openai>=1.50") from e
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)
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
        with metrics.timed_llm(backend="openai", model=self.model):
            t0 = time.perf_counter()
            try:
                r = client.chat.completions.create(
                    model=self.model,
                    messages=msg_dicts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
            except Exception as e:
                raise LLMError(f"OpenAI call failed: {e}") from e
            dt = time.perf_counter() - t0

        msg = r.choices[0].message
        usage = r.usage
        return LLMResponse(
            content=msg.content or "",
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_s=dt,
            raw=r,
        )