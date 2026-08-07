from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.recursive import MetaControllerConfig, SchemaMutator, ToolFactory


def test_schema_mutator_rejects_unknown_and_out_of_bounds_fields(tmp_path):
    mutator = SchemaMutator(
        MetaControllerConfig(), history_path=str(tmp_path / "history.jsonl")
    )
    unknown = mutator.propose("eval_fn", "override", "test")
    invalid = mutator.propose("low_conf_threshold", -0.1, "test")
    assert unknown["accepted"] is False
    assert unknown["reason_final"] == "field_not_mutable"
    assert invalid["accepted"] is False
    assert invalid["reason_final"] == "out_of_bounds"
    assert not hasattr(mutator.config, "eval_fn")


def test_schema_mutator_accepts_bounded_configuration_change(tmp_path):
    mutator = SchemaMutator(
        MetaControllerConfig(), history_path=str(tmp_path / "history.jsonl")
    )
    result = mutator.propose("low_conf_threshold", 0.2, "test")
    assert result["accepted"] is True
    assert mutator.config.low_conf_threshold == 0.2


def test_tool_factory_executes_simple_whitelisted_function():
    factory = object.__new__(ToolFactory)
    function, error = factory._exec_safely(
        "def uppercase_text(x):\n    return str(x).upper()",
        "uppercase_text",
    )
    assert error is None
    assert function("hello") == "HELLO"


def test_tool_factory_rejects_imports_private_attributes_and_loops():
    factory = object.__new__(ToolFactory)
    cases = [
        "def unsafe(x):\n    import os\n    return os.getcwd()",
        "def unsafe(x):\n    return x.__class__",
        "def unsafe(x):\n    while True:\n        pass",
        "def unsafe(x):\n    return open(x).read()",
    ]
    for code in cases:
        function, error = factory._exec_safely(code, "unsafe")
        assert function is None
        assert error is not None
