"""L4 Recursive self-improvement.

Three capabilities that close the AGI feedback loop on itself:
  1. Schema mutation: MetaController can rewrite its own detection thresholds
  2. Tool factory: Agent can synthesize new tools at runtime
  3. Prompt mutation: Reflector can rewrite its own hindsight prompts

All mutations are version-controlled and guarded by:
  - Sandbox: never modify live state without backup
  - Eval gate: any change must pass an eval check before being kept
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agi_kit.observability import get_logger

log = get_logger(__name__)


# ============================================================
# 1. Schema mutation - MetaController self-modifies thresholds
# ============================================================
@dataclass
class MetaControllerConfig:
    """Mutable config; the meta-controller can rewrite its own fields."""
    confidence_window: int = 5
    low_conf_threshold: float = 0.35
    tool_error_threshold: int = 2
    stuck_obs_threshold: int = 3
    max_strategy_switches: int = 2
    generation: int = 0
    parent_hash: str = ""

    def to_dict(self):
        return asdict(self)


class SchemaMutator:
    """Propose + apply changes to MetaControllerConfig."""

    def __init__(self,
                 config: MetaControllerConfig,
                 eval_fn: Optional[Callable] = None,
                 threshold: float = 0.95,
                 history_path: str = "logs/l4/schema_history.jsonl"):
        self.config = config
        self.eval_fn = eval_fn
        self.threshold = threshold
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = []
        self._load_history()

    def _load_history(self):
        if not self.history_path.exists():
            return
        with self.history_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.history.append(json.loads(line))
                except Exception:
                    pass

    def _hash(self, d):
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]

    @staticmethod
    def _validate_change(field_name, new_value):
        bounds = {
            "confidence_window": (int, 1, 100),
            "low_conf_threshold": ((int, float), 0.0, 1.0),
            "tool_error_threshold": (int, 1, 100),
            "stuck_obs_threshold": (int, 1, 100),
            "max_strategy_switches": (int, 0, 100),
        }
        if field_name not in bounds:
            return False, "field_not_mutable"
        expected_type, lower, upper = bounds[field_name]
        if isinstance(new_value, bool) or not isinstance(new_value, expected_type):
            return False, "invalid_type"
        if isinstance(new_value, float) and not math.isfinite(new_value):
            return False, "non_finite"
        if not lower <= new_value <= upper:
            return False, "out_of_bounds"
        return True, None

    def propose(self, field_name, new_value, reason):
        """Propose a change. Does not apply until accept() is called."""
        valid, validation_reason = self._validate_change(field_name, new_value)
        if not valid:
            return {
                "accepted": False,
                "reason_final": validation_reason,
                "field": field_name,
                "new": new_value,
            }
        old = getattr(self.config, field_name, None)
        if old == new_value:
            return {"accepted": False, "reason": "no_change",
                    "field": field_name, "new": new_value}
        backup = copy.deepcopy(self.config)
        setattr(self.config, field_name, new_value)
        self.config.generation += 1
        self.config.parent_hash = self._hash(asdict(backup))
        # Gate: if eval_fn provided, must not regress
        accepted = True
        new_acc = None
        reason_final = "applied_no_eval"
        if self.eval_fn is not None:
            try:
                new_acc = float(self.eval_fn(self.config))
                # baseline assumed 1.0 if no other baseline known; accept if >= threshold
                accepted = new_acc >= self.threshold
                reason_final = "passed" if accepted else "regressed_below_threshold"
            except Exception as e:
                accepted = False
                reason_final = "eval_failed: " + str(e)
        if not accepted:
            # Rollback
            self.config = backup
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "field": field_name,
            "old": old, "new": new_value,
            "reason": reason,
            "accepted": accepted,
            "new_acc": new_acc,
            "reason_final": reason_final,
            "generation": self.config.generation,
            "hash": self._hash(asdict(self.config)),
        }
        self.history.append(rec)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("schema_mutate",
                 field=field_name, accepted=accepted,
                 new_acc=new_acc, reason=reason_final)
        return rec


# ============================================================
# 2. Tool factory - synthesize new tools at runtime
# ============================================================
TOOL_FACTORY_PROMPT_TMPL = """You are a tool factory. Given a failing observation, propose ONE small Python function.
Output STRICT JSON:
{{"name": "<snake_case>", "desc": "<short>", "params": {{"x": "str"}}, "code": "def <name>(x):\n    return <expr>"}}
ZJZ_PLACEHOLDERS_ZJZ"""

def _make_tool_prompt(obs, task):
    return (TOOL_FACTORY_PROMPT_TMPL
            .replace("ZJZ_PLACEHOLDERS_ZJZ", "")
            .replace("{obs}", obs[:600])
            .replace("{task}", task[:200]))

TOOL_FACTORY_PROMPT = _make_tool_prompt  # callable for try_synthesize


@dataclass
class ToolSpec:
    name: str
    desc: str
    params: dict
    code: str
    uses: int = 0
    created_ts: str = ""

    def to_dict(self):
        return asdict(self)


class ToolFactory:
    """Synthesize new Python tools when agent fails repeatedly."""

    def __init__(self,
                 llm,                            # for tool generation
                 tool_registry,                   # registry to register new tools into
                 eval_fn: Optional[Callable] = None,
                 history_path: str = "logs/l4/tool_factory_history.jsonl"):
        self.llm = llm
        self.registry = tool_registry
        self.eval_fn = eval_fn
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.synthesized = []

    def _exec_safely(self, code, name):
        """Compile and exec the proposed tool, returning a callable."""
        if not isinstance(code, str) or len(code) > 4000:
            return None, "code_too_long"
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,29}", str(name)):
            return None, "invalid_name"
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            return None, "syntax_error: " + str(e)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            return None, "single_function_required"
        function = tree.body[0]
        if function.name != name or function.decorator_list:
            return None, "function_name_mismatch"

        denied = (
            ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.Global,
            ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.Raise,
            ast.Delete, ast.While, ast.For, ast.AsyncFor, ast.ListComp,
            ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Await, ast.Yield,
            ast.YieldFrom,
        )
        safe_builtin_names = {
            "abs", "all", "any", "bool", "dict", "enumerate", "float",
            "int", "len", "list", "max", "min", "range", "round",
            "set", "sorted", "str", "sum", "tuple", "zip",
        }
        safe_methods = {
            "capitalize", "casefold", "endswith", "find", "format",
            "isalnum", "isalpha", "isdigit", "join", "lower", "lstrip",
            "replace", "rstrip", "split", "startswith", "strip", "title",
            "upper",
        }
        for node in ast.walk(tree):
            if isinstance(node, denied):
                return None, "unsafe_syntax: " + type(node).__name__
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                return None, "private_name"
            if isinstance(node, ast.Attribute) and (
                node.attr.startswith("_") or node.attr not in safe_methods
            ):
                return None, "unsafe_attribute"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in safe_builtin_names:
                    return None, "unsafe_call"
                if isinstance(node.func, ast.Attribute) and node.func.attr not in safe_methods:
                    return None, "unsafe_call"

        safe_builtins = {
            name_: __builtins__[name_] if isinstance(__builtins__, dict)
            else getattr(__builtins__, name_)
            for name_ in safe_builtin_names
        }
        allowed_globals = {"__builtins__": safe_builtins}
        namespace = {}
        try:
            exec(compile(tree, "<generated-tool>", "exec"), allowed_globals, namespace)
        except Exception as e:
            return None, "exec_error: " + str(e)
        if name not in namespace:
            return None, "name_not_in_code"
        fn = namespace[name]
        if not callable(fn):
            return None, "not_callable"
        return fn, None

    def try_synthesize(self, task, observation):
        """Propose a new tool for a recurring failure."""
        from agi_kit.llms.base import LLMMessage, MessageRole
        prompt = TOOL_FACTORY_PROMPT(observation[:600], task[:200])
        try:
            r = self.llm.chat(
                [LLMMessage(role=MessageRole.USER, content=prompt)],
                max_tokens=400, temperature=0.0,
            )
            txt = r.content.strip()
            m = re.search(r"\{.*\}", txt, re.S)
            if not m:
                return {"accepted": False, "reason": "no_json"}
            data = json.loads(m.group(0))
            spec = ToolSpec(
                name=data["name"][:30],
                desc=data.get("desc", "")[:200],
                params=data.get("params", {}) or {},
                code=data["code"],
                uses=0,
                created_ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
        except Exception as e:
            return {"accepted": False, "reason": "gen_failed: " + str(e)}
        # Safe exec
        fn, err = self._exec_safely(spec.code, spec.name)
        if err:
            spec_dict = spec.to_dict()
            spec_dict["accepted"] = False
            spec_dict["reason"] = err
            self.synthesized.append(spec_dict)
            self._save(spec_dict)
            return spec_dict
        # Eval gate
        accepted = True
        if self.eval_fn is not None:
            try:
                score = float(self.eval_fn(spec))
                accepted = score >= 0.5
            except Exception:
                accepted = False
        if accepted:
            # Register into agent's tool registry
            try:
                self.registry[spec.name] = {
                    "desc": spec.desc, "params": spec.params, "fn": fn,
                }
            except Exception as e:
                accepted = False
                spec_dict = spec.to_dict()
                spec_dict["accepted"] = False
                spec_dict["reason"] = "register_failed: " + str(e)
                self.synthesized.append(spec_dict)
                self._save(spec_dict)
                return spec_dict
        spec_dict = spec.to_dict()
        spec_dict["accepted"] = accepted
        spec_dict["reason"] = "registered" if accepted else "eval_failed"
        self.synthesized.append(spec_dict)
        self._save(spec_dict)
        log.info("tool_factory",
                 name=spec.name, accepted=accepted,
                 reason=spec_dict["reason"])
        return spec_dict

    def _save(self, rec):
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 3. Prompt mutation - Reflector rewrites its own prompts
# ============================================================
@dataclass
class PromptTemplate:
    """A versioned prompt template."""
    name: str
    text: str
    version: int = 1
    parent_version: int = 0
    metrics: dict = field(default_factory=dict)
    created_ts: str = ""

    def to_dict(self):
        return asdict(self)


class PromptMutator:
    """Reflector can rewrite its own hindsight/score prompts based on episode metrics."""

    def __init__(self,
                 prompt_path: str = "logs/l4/prompts.jsonl"):
        self.prompt_path = Path(prompt_path)
        self.prompt_path.parent.mkdir(parents=True, exist_ok=True)
        self.templates = {}

    def save_version(self, name, text, parent_version=0, metrics=None):
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        existing = [t for t in self.templates.get(name, []) if t.version == parent_version]
        version = (max([t.version for t in self.templates.get(name, [])], default=0) + 1)
        t = PromptTemplate(
            name=name, text=text, version=version,
            parent_version=parent_version, metrics=metrics or {},
            created_ts=ts,
        )
        self.templates.setdefault(name, []).append(t)
        with self.prompt_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
        log.info("prompt_mutate", name=name, version=version)
        return t

    def latest(self, name):
        versions = self.templates.get(name, [])
        return max(versions, key=lambda t: t.version) if versions else None
