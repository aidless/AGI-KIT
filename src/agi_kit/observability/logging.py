"""Structured logging with structlog.

Provides JSON-formatted logs in production, pretty console logs in dev.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from agi_kit.config import LogFormat, LogLevel, get_settings

_configured = False


def configure_logging(
    level: LogLevel | None = None,
    fmt: LogFormat | None = None,
) -> None:
    """Configure structlog globally. Idempotent."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = level or settings.observability.log_level
    fmt = fmt or settings.observability.log_format

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    if fmt == LogFormat.JSON:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.value)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger, optionally bound with initial values."""
    configure_logging()
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    return logger.bind(**initial_values) if initial_values else logger