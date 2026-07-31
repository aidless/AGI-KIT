"""Prometheus metrics for AGI Kit operations."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from prometheus_client import Counter, Gauge, Histogram

# LLM metrics
llm_requests_total = Counter(
    "agi_kit_llm_requests_total",
    "Total LLM API calls",
    ["backend", "model", "status"],
)
llm_request_seconds = Histogram(
    "agi_kit_llm_request_seconds",
    "LLM call latency",
    ["backend", "model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)
llm_tokens_total = Counter(
    "agi_kit_llm_tokens_total",
    "Total tokens consumed",
    ["backend", "model", "direction"],  # direction: prompt | completion
)

# Agent metrics
agent_runs_total = Counter(
    "agi_kit_agent_runs_total",
    "Total agent runs",
    ["paradigm", "verdict"],  # verdict: success | max_steps | error
)
agent_steps_total = Counter(
    "agi_kit_agent_steps_total",
    "Total agent reasoning steps",
    ["paradigm"],
)
agent_step_seconds = Histogram(
    "agi_kit_agent_step_seconds",
    "Agent step latency",
    ["paradigm"],
)

# Tool metrics
tool_invocations_total = Counter(
    "agi_kit_tool_invocations_total",
    "Tool invocations",
    ["tool", "status"],  # status: ok | error
)
tool_seconds = Histogram(
    "agi_kit_tool_seconds",
    "Tool execution latency",
    ["tool"],
)

# RAG metrics
rag_chunks_indexed_total = Counter("agi_kit_rag_chunks_indexed_total", "Chunks indexed into RAG")
rag_search_total = Counter("agi_kit_rag_search_total", "RAG searches")
rag_search_hits = Histogram(
    "agi_kit_rag_search_hits",
    "Number of hits per search",
    buckets=(0, 1, 2, 3, 5, 10),
)


class _MetricsFacade:
    """Façade so users can do `from agi_kit.observability import metrics; metrics.llm(...)`."""

    @contextmanager
    def timed_llm(self, backend: str, model: str) -> Iterator[None]:
        start = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            llm_request_seconds.labels(backend=backend, model=model).observe(
                time.perf_counter() - start
            )
            llm_requests_total.labels(backend=backend, model=model, status=status).inc()

    @contextmanager
    def timed_tool(self, tool: str) -> Iterator[None]:
        start = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            tool_seconds.labels(tool=tool).observe(time.perf_counter() - start)
            tool_invocations_total.labels(tool=tool, status=status).inc()

    def inc(self, counter: Counter, labels: dict[str, str] | None = None) -> None:
        if labels:
            counter.labels(**labels).inc()
        else:
            counter.inc()

    def observe(self, hist: Histogram, value: float, labels: dict[str, str] | None = None) -> None:
        if labels:
            hist.labels(**labels).observe(value)
        else:
            hist.observe(value)


metrics = _MetricsFacade()


@contextmanager
def timed(name: str, **labels: Any) -> Iterator[None]:
    """Generic timer context manager."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name == "llm":
        metrics.observe(llm_request_seconds, elapsed, labels)
    elif name == "tool":
        metrics.observe(tool_seconds, elapsed, labels)