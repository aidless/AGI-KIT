"""L2 元控制器 - monitors Agent behavior and decides control signals.

Pure rule-based (no LLM call), runs at every step:
- confidence: rolling mean of recent self_scores from Reflector
- progress: step / max_steps
- stuck detection: 3 consecutive identical observations
- bad tool detection: repeated tool errors

Outputs ControlSignal which the Agent loop should act on:
  - continue: keep going
  - retry: redo last step (e.g. with different args)
  - switch_strategy: inject a Playbook hint into the system prompt
  - ask_user: stop and ask for human input
  - change_plan: abort and request a new plan
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque

from agi_kit.observability import get_logger

log = get_logger(__name__)


class ControlAction(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    SWITCH_STRATEGY = "switch_strategy"
    ASK_USER = "ask_user"
    CHANGE_PLAN = "change_plan"


@dataclass
class ControlSignal:
    action: ControlAction
    reason: str
    confidence: float = 0.0
    step: int = 0
    hint: str = ""  # injected into system prompt when switch_strategy

    def to_dict(self):
        return {
            "action": self.action.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "step": self.step,
            "hint": self.hint[:200] if self.hint else "",
        }


@dataclass
class AgentState:
    """State passed into MetaController.decide each step."""

    step: int
    max_steps: int
    last_action: dict
    last_observation: str
    self_score_history: list = field(default_factory=list)  # last N scores
    tool_error_streak: int = 0
    obs_hash_streak: int = 0
    playbook_hints_used: int = 0


class MetaController:
    """Rule-based meta-controller for L2 self-improvement."""

    def __init__(self,
                 confidence_window=5,
                 low_conf_threshold=0.35,
                 tool_error_threshold=2,
                 stuck_obs_threshold=3,
                 max_strategy_switches=2):
        self.confidence_window = confidence_window
        self.low_conf_threshold = low_conf_threshold
        self.tool_error_threshold = tool_error_threshold
        self.stuck_obs_threshold = stuck_obs_threshold
        self.max_strategy_switches = max_strategy_switches
        self._switch_count = 0

    def reset(self):
        self._switch_count = 0

    def decide(self, state: AgentState, playbook=None, query: str = "") -> ControlSignal:
        # 1) Budget check
        if state.step >= state.max_steps:
            return ControlSignal(
                ControlAction.CHANGE_PLAN,
                reason="max_steps reached",
                confidence=_safe_mean(state.self_score_history),
                step=state.step,
            )

        # 2) Stuck on same observation?
        if state.obs_hash_streak >= self.stuck_obs_threshold:
            hint = ""
            if playbook and query and self._switch_count < self.max_strategy_switches:
                hits = playbook.search(query, k=1)
                if hits:
                    hint = "Try a different approach: " + hits[0][0].strategy
                    self._switch_count += 1
            return ControlSignal(
                ControlAction.SWITCH_STRATEGY if hint else ControlAction.ASK_USER,
                reason="stuck: " + str(state.obs_hash_streak) +
                       " identical observations in a row",
                confidence=_safe_mean(state.self_score_history),
                step=state.step,
                hint=hint,
            )

        # 3) Tool errors accumulating?
        if state.tool_error_streak >= self.tool_error_threshold:
            return ControlSignal(
                ControlAction.RETRY if state.tool_error_streak == self.tool_error_threshold
                else ControlAction.SWITCH_STRATEGY,
                reason="tool_error_streak=" + str(state.tool_error_streak),
                confidence=_safe_mean(state.self_score_history),
                step=state.step,
                hint="Check tool name spelling and required arguments.",
            )

        # 4) Low confidence?
        recent = state.self_score_history[-self.confidence_window:]
        conf = _safe_mean(recent)
        if conf < self.low_conf_threshold and len(recent) >= self.confidence_window:
            hint = ""
            if playbook and query and self._switch_count < self.max_strategy_switches:
                hits = playbook.search(query, k=1)
                if hits:
                    hint = "Try this known strategy: " + hits[0][0].strategy
                    self._switch_count += 1
            return ControlSignal(
                ControlAction.SWITCH_STRATEGY if hint else ControlAction.RETRY,
                reason="low_confidence=" + str(round(conf, 2)),
                confidence=conf,
                step=state.step,
                hint=hint,
            )

        # 5) Default
        return ControlSignal(
            ControlAction.CONTINUE,
            reason="ok",
            confidence=conf,
            step=state.step,
        )


def _safe_mean(xs):
    if not xs:
        return 0.5
    return sum(xs) / len(xs)


def update_state_from_step(state: AgentState,
                           action: dict,
                           observation: str,
                           self_score: float) -> AgentState:
    """Convenience: update streaks after each step."""
    state.step += 1
    state.last_action = dict(action)
    state.last_observation = observation
    state.self_score_history.append(self_score)
    # Keep history bounded
    if len(state.self_score_history) > 50:
        state.self_score_history = state.self_score_history[-50:]
    # Tool error streak
    obs_low = (observation or "").lower()
    is_err = ("err:" in obs_low or "error" in obs_low or
              "unknown tool" in obs_low or "exception" in obs_low)
    state.tool_error_streak = state.tool_error_streak + 1 if is_err else 0
    # Observation hash streak (for stuck detection)
    h = _obs_hash(observation)
    if state.last_observation and _obs_hash(state.last_observation) == h:
        state.obs_hash_streak += 1
    else:
        state.obs_hash_streak = 1
    return state


def _obs_hash(obs):
    # Normalize: lowercase, collapse whitespace, strip
    return " ".join((obs or "").lower().split())[:200]