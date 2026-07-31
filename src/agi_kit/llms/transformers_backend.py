"""Hugging Face Transformers backend (local model)."""
from __future__ import annotations

import time
from typing import Any

from agi_kit.exceptions import LLMError
from agi_kit.llms.base import LLM, LLMMessage, LLMResponse
from agi_kit.observability import get_logger, metrics

log = get_logger(__name__)


class TransformersBackend(LLM):
    """Local Transformers model (CPU/GPU)."""

    def __init__(
        self,
        model: str,
        device: str = "auto",
        dtype: str = "auto",
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(model=model)
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise LLMError(
                "transformers/torch required. pip install transformers torch"
            ) from e

        log.info("loading_transformers_model", model=self.model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, cache_dir=self.cache_dir)
        kwargs: dict[str, Any] = {"dtype": self.dtype}
        if self.device != "auto":
            kwargs["device_map"] = self.device
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model, cache_dir=self.cache_dir, **kwargs
        )

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import torch

        self._load()
        msg_dicts = [m.to_dict() for m in messages]
        prompt = self._tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with metrics.timed_llm(backend="transformers", model=self.model):
            t0 = time.perf_counter()
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = max(temperature, 0.01)
            if stop:
                gen_kwargs["stop_strings"] = stop
                gen_kwargs["tokenizer"] = self._tokenizer
            with torch.no_grad():
                out = self._model.generate(**inputs, **gen_kwargs)
            dt = time.perf_counter() - t0

        text = self._tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return LLMResponse(
            content=text,
            model=self.model,
            prompt_tokens=inputs.input_ids.shape[1],
            completion_tokens=out.shape[1] - inputs.input_ids.shape[1],
            total_tokens=out.shape[1],
            latency_s=dt,
        )