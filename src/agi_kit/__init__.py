"""AGI Kit - Enterprise-grade AGI research platform.

This package provides:
- Pluggable LLM backends (Ollama, Transformers, OpenAI-compatible)
- Multiple agent paradigms (ReAct, Plan-Execute, Reflexion)
- Tool registry with JSON Schema validation
- RAG system with pluggable embedders and retrievers
- Evaluation framework (GAIA, custom benchmarks)
- FastAPI server with auth and observability
- Training utilities (SFT with TRL/PEFT)
- L1 Reflection primitive (thinking log + self-critique)
- L2 Playbook + MetaController (strategy memory + behavior control)
- L3 Continual Learning Loop (auto-SFT generation pipeline)
- L4 Recursive self-improvement (schema/tool/prompt mutation)

Quick start:
    >>> from agi_kit import Agent, OllamaBackend
    >>> from agi_kit import Reflector, Playbook, MetaController
    >>> from agi_kit import ContinualLoop, ExperienceBuffer
    >>> from agi_kit import SchemaMutator, ToolFactory, PromptMutator
"""

from agi_kit.__version__ import __version__
from agi_kit.agents.base import Agent, AgentStep
from agi_kit.agents.react import ReActAgent
from agi_kit.llms.base import LLM, LLMMessage, LLMResponse, MessageRole
from agi_kit.llms.ollama import OllamaBackend
from agi_kit.reflect import Reflector, ReflectionRecord, EpisodeSummary
from agi_kit.playbook import Playbook, Strategy
from agi_kit.meta import MetaController, ControlSignal, ControlAction, AgentState
from agi_kit.loop import (
    ExperienceBuffer, ContinualLoop, TraceRecord, GenerationRecord,
    format_trace_for_sft, default_safety_check,
)
from agi_kit.strategy_miner import (
    extract_strategies_from_episode, write_extracted_to_playbook,
    should_extract, ExtractedStrategy,
)
from agi_kit.gaia2_tasks import load_gaia2_tasks
from agi_kit.evals_arith import eval_arithmetic, make_eval_fn
from agi_kit.recursive import (
    SchemaMutator, MetaControllerConfig,
    ToolFactory, ToolSpec,
    PromptMutator, PromptTemplate,
)

__all__ = [
    "__version__",
    "Agent", "AgentStep", "ReActAgent",
    "LLM", "LLMMessage", "LLMResponse", "MessageRole", "OllamaBackend",
    "Reflector", "ReflectionRecord", "EpisodeSummary",
    "Playbook", "Strategy",
    "MetaController", "ControlSignal", "ControlAction", "AgentState",
    "ExperienceBuffer", "ContinualLoop", "TraceRecord", "GenerationRecord",
    "format_trace_for_sft", "default_safety_check",
    "extract_strategies_from_episode", "write_extracted_to_playbook",
    "should_extract", "ExtractedStrategy",
    "load_gaia2_tasks",
    "eval_arithmetic", "make_eval_fn",
    "SchemaMutator", "MetaControllerConfig",
    "ToolFactory", "ToolSpec",
    "PromptMutator", "PromptTemplate",
]