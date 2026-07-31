"""L4 recursive smoke test - 验证 schema mutator / tool factory / prompt mutator
不需要 Ollama;用 mock LLM。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agi_kit.meta import MetaController, AgentState
from agi_kit.recursive import (
    SchemaMutator, MetaControllerConfig,
    ToolFactory, ToolSpec,
    PromptMutator,
)


# ============================================================
# Mock LLM
# ============================================================
@dataclass
class _R:
    content: str


class MockLLM:
    def chat(self, messages, **kw):
        # If asked to generate a tool, return a valid tool spec
        prompt = messages[0].content if messages else ""
        if "tool factory" in prompt.lower() or "Tool factory" in prompt:
            return _R(content=json.dumps({
                "name": "repeat_text",
                "desc": "Repeat a string N times",
                "params": {"text": "str", "n": "int"},
                "code": "def repeat_text(text, n=2):\n    return text * int(n)",
            }))
        return _R(content="mock response")


# ============================================================
# 1. Schema mutation test
# ============================================================
def test_schema_mutation():
    print("=== Test 1: Schema mutation ===")
    base_config = MetaControllerConfig()
    # Mock eval: returns "score" = how aggressive the config is
    def mock_eval(cfg):
        # Reward lower thresholds (more aggressive intervention)
        return 1.0 - (cfg.low_conf_threshold / 1.0) * 0.5
    mutator = SchemaMutator(base_config, eval_fn=mock_eval, threshold=0.5)
    # Propose: lower threshold from 0.35 to 0.20 (more aggressive)
    rec = mutator.propose("low_conf_threshold", 0.20, "want more aggressive intervention")
    print("  propose lower threshold:", rec["accepted"], "new_acc=", rec["new_acc"])
    # Propose: raise it back (should be rejected because eval_fn returns lower score)
    rec2 = mutator.propose("low_conf_threshold", 0.50, "try raising it")
    print("  propose raise threshold:", rec2["accepted"], "new_acc=", rec2["new_acc"])
    print("  final config:", json.dumps(mutator.config.to_dict(), indent=2))
    return mutator


# ============================================================
# 2. Tool factory test
# ============================================================
def test_tool_factory():
    print()
    print("=== Test 2: Tool factory ===")
    # Simple registry
    registry = {"calc": {"fn": lambda x: x}}
    llm = MockLLM()
    factory = ToolFactory(llm, registry)
    # Try synthesizing for a "I need to repeat text" failure
    rec = factory.try_synthesize(
        task="Repeat HELLO 3 times",
        observation="err: no tool found for repeat operation"
    )
    print("  synthesize result:", json.dumps({k: v for k, v in rec.items() if k != "code"}, indent=2))
    print("  registry now has:", list(registry.keys()))
    if rec["accepted"] and "repeat_text" in registry:
        # Test the new tool
        fn = registry["repeat_text"]["fn"]
        print("  test repeat_text('ab', 3) =", fn("ab", 3))
    return factory


# ============================================================
# 3. Prompt mutation test
# ============================================================
def test_prompt_mutation():
    print()
    print("=== Test 3: Prompt mutation ===")
    pm = PromptMutator()
    v1 = pm.save_version(
        "hindsight",
        "Look at this step. What could be improved?\nAction: {action}\nObs: {obs}",
        metrics={"expected_quality": 0.5},
    )
    print("  v1 version:", v1.version)
    v2 = pm.save_version(
        "hindsight",
        "Be more concrete. What SPECIFIC change would help?\nAction: {action}\nObs: {obs}",
        parent_version=1,
        metrics={"expected_quality": 0.7},
    )
    print("  v2 version:", v2.version, "parent:", v2.parent_version)
    # Show lineage
    for n in ["hindsight"]:
        versions = sorted(pm.templates.get(n, []), key=lambda t: t.version)
        print("  lineage", n, ":", [v.version for v in versions])
    return pm


# ============================================================
# Run all
# ============================================================
def main():
    mutator = test_schema_mutation()
    factory = test_tool_factory()
    pm = test_prompt_mutation()
    # Persist summary
    summary = {
        "schema_mutation": {
            "final_config": mutator.config.to_dict(),
            "history_len": len(mutator.history),
        },
        "tool_factory": {
            "synthesized_count": len(factory.synthesized),
            "registered": [s["name"] for s in factory.synthesized if s["accepted"]],
        },
        "prompt_mutation": {
            "templates": {n: [t.version for t in v] for n, v in pm.templates.items()},
        },
    }
    out = ROOT / "logs" / "l4_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print("Wrote", out)


if __name__ == "__main__":
    main()