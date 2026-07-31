"""Configuration management with Pydantic Settings.

Supports three sources, in increasing priority:
1. Built-in defaults
2. YAML files in configs/ directory
3. Environment variables (prefix AGI_KIT_)
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agi_kit.exceptions import ConfigError


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


# --------------------------------------------------------------------------- #
# Nested config schemas
# --------------------------------------------------------------------------- #


class LLMSettings(BaseModel):
    """LLM backend configuration."""

    backend: str = Field(default="ollama", description="ollama | transformers | openai")
    model: str = Field(default="qwen3:1.7b")
    host: str | None = Field(default="http://127.0.0.1:11434")
    api_key: str | None = Field(default=None)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(default=60.0, gt=0)


class AgentSettings(BaseModel):
    """Agent behavior configuration."""

    paradigm: str = Field(default="react", description="react | plan_execute | reflexion")
    max_steps: int = Field(default=10, ge=1, le=100)
    enable_tracing: bool = Field(default=True)
    parser_max_retries: int = Field(default=2, ge=0, le=10)


class ToolSettings(BaseModel):
    """Tool system configuration."""

    enable_web: bool = Field(default=True)
    enable_shell: bool = Field(default=False)
    enable_rag: bool = Field(default=True)
    shell_timeout_s: float = Field(default=15.0, gt=0)


class RAGSettings(BaseModel):
    """RAG configuration."""

    embedder: str = Field(default="BAAI/bge-small-en-v1.5")
    chunk_size: int = Field(default=400, ge=64, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=1024)
    index_path: Path = Field(default=Path("./data/rag.index"))


class APISettings(BaseModel):
    """FastAPI server settings."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=32)
    secret_key: str = Field(default="change-me-in-production")
    rate_limit_per_min: int = Field(default=60, ge=1)
    enable_auth: bool = Field(default=True)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class ObservabilitySettings(BaseModel):
    """Logging / metrics / tracing."""

    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: LogFormat = Field(default=LogFormat.CONSOLE)
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    enable_tracing: bool = Field(default=False)
    otel_endpoint: str | None = Field(default=None)


class Settings(BaseSettings):
    """Root configuration container."""

    model_config = SettingsConfigDict(
        env_prefix="AGI_KIT_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # Meta
    env: str = Field(default="dev")
    data_dir: Path = Field(default=Path("./data"))
    model_cache_dir: Path = Field(default=Path("./models"))

    # Sub-configs
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    api: APISettings = Field(default_factory=APISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @field_validator("data_dir", "model_cache_dir", mode="before")
    @classmethod
    def _expand_path(cls, v: Any) -> Path:
        p = Path(os.path.expanduser(str(v)))
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# --------------------------------------------------------------------------- #
# Global settings registry (singleton with caching)
# --------------------------------------------------------------------------- #

_cached: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Return process-wide Settings (cached)."""
    global _cached
    if _cached is None or reload:
        # Try default config file
        default_cfg = Path("configs/default.yaml")
        if default_cfg.exists():
            _cached = Settings.from_yaml(default_cfg)
        else:
            _cached = Settings()
    return _cached


def reset_settings() -> None:
    """Clear cached settings (used in tests)."""
    global _cached
    _cached = None