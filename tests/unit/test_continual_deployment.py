from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.evals_arith import eval_arithmetic
from agi_kit.loop import (
    ContinualLoop,
    ExperienceBuffer,
    TraceRecord,
    default_safety_check,
)


def test_safety_gate_rejects_candidates_without_held_out_evaluator():
    result = default_safety_check("candidate", baseline_acc=1.0)
    assert result == {
        "accepted": False,
        "new_acc": 0.0,
        "reason": "eval_fn_required",
    }


def test_unconverted_checkpoint_cannot_receive_an_estimated_eval_score(tmp_path):
    with pytest.raises(ValueError, match="not an executable Ollama model"):
        eval_arithmetic(str(tmp_path))


def test_loop_rejects_candidate_without_executable_model_name(tmp_path):
    buffer = ExperienceBuffer(path=str(tmp_path / "buffer.jsonl"), min_score=0.0)
    buffer.add(TraceRecord("one", "task", "answer", 1.0, []))
    evaluated = False

    def eval_fn(_model):
        nonlocal evaluated
        evaluated = True
        return 1.0

    loop = ContinualLoop(
        run_episode_fn=lambda _task, gold=None: {},
        buffer=buffer,
        retrain_fn=lambda **_kwargs: {"trained": False, "ollama_model": None},
        eval_fn=eval_fn,
        out_dir=str(tmp_path / "runs"),
    )
    record = loop.retrain(baseline_acc=1.0)
    assert record is not None
    assert record.accepted is False
    assert record.reason == "candidate_not_executable"
    assert evaluated is False


def test_generation_record_preserves_base_model_after_acceptance(tmp_path):
    buffer = ExperienceBuffer(path=str(tmp_path / "buffer.jsonl"), min_score=0.0)
    buffer.add(TraceRecord("one", "task", "answer", 1.0, []))

    loop = ContinualLoop(
        run_episode_fn=lambda _task, gold=None: {},
        buffer=buffer,
        retrain_fn=lambda **_kwargs: {"ollama_model": "candidate-model"},
        eval_fn=lambda _model: 1.0,
        out_dir=str(tmp_path / "runs"),
        base_model="base-model",
    )
    record = loop.retrain(baseline_acc=1.0)

    assert record.accepted is True
    assert record.base_model == "base-model"
    assert record.new_model == "candidate-model"
    assert loop.current_model == "candidate-model"


def test_paired_gate_drives_generation_record(tmp_path):
    buffer = ExperienceBuffer(path=str(tmp_path / "buffer.jsonl"), min_score=0.0)
    buffer.add(TraceRecord("one", "task", "answer", 1.0, []))
    calls = []

    def paired_gate(base_model, candidate_model):
        calls.append((base_model, candidate_model))
        return {
            "accepted": True,
            "baseline_acc": 0.6,
            "new_acc": 0.8,
            "reason": "paired_gate_passed",
            "evaluation": {"mean_difference": 0.2, "n_tasks": 4},
        }

    loop = ContinualLoop(
        run_episode_fn=lambda _task, gold=None: {},
        buffer=buffer,
        retrain_fn=lambda **_kwargs: {"ollama_model": "candidate-model"},
        eval_fn=lambda _model: 0.1,
        paired_eval_fn=paired_gate,
        out_dir=str(tmp_path / "runs"),
        base_model="base-model",
    )
    record = loop.retrain(baseline_acc=1.0)

    assert calls == [("base-model", "candidate-model")]
    assert record.eval_baseline_acc == 0.6
    assert record.eval_new_acc == 0.8
    assert record.evaluation["mean_difference"] == 0.2


def test_round18_sft_record_is_an_explicit_rejection():
    import json

    record = json.loads((ROOT / "logs/sft_validation/round18.json").read_text(encoding="utf-8"))
    assert record["deployment_decision"] == "reject"
    assert record["transformers_cpu_smoke"]["correct"] is False
