from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.evaluation_protocol import (
    HeldOutEvaluator,
    assess_answer,
    extract_final,
    numeric_equal,
)
from agi_kit.paired_evaluation import PairedEvaluator


def test_held_out_evaluator_is_fixed_and_auditable():
    evaluator = HeldOutEvaluator(
        tasks=[
            {"id": "a", "task": "one", "gold": 1},
            {"id": "b", "task": "two", "gold": 2},
        ],
        run_fn=lambda model, task, gold, budget: {
            "correct": model == "candidate" and budget == 3 and task == "one"
        },
        budget=3,
    )

    assert evaluator("candidate") == 0.5
    first = evaluator.last_result
    assert first["n_tasks"] == 2
    assert first["budget"] == 3
    assert first["task_set_id"] == evaluator.task_set_id
    assert [item["task_id"] for item in first["results"]] == ["a", "b"]


def test_paired_evaluator_uses_same_budget_and_records_ci():
    calls = []

    def run(model, task, gold, budget):
        calls.append((model, task, budget))
        return model == "candidate" or task == "easy"

    evaluator = PairedEvaluator(
        tasks=[
            {"id": "easy", "task": "easy", "gold": 1},
            {"id": "hard", "task": "hard", "gold": 2},
        ],
        run_fn=run,
        budget={"max_steps": 4},
        bootstrap_samples=200,
        max_regression=0.0,
    )
    result = evaluator("baseline", "candidate")

    assert result["accepted"] is True
    assert result["evaluation"]["mean_difference"] == 0.5
    assert len(result["evaluation"]["bootstrap_ci_95"]) == 2
    assert all(call[2] == {"max_steps": 4} for call in calls)
from scripts.analyze_controlled_arithmetic import two_sided_exact_binomial


def test_strict_protocol_accepts_one_final_block_and_normalizes_numbers():
    result = assess_answer('```json\n{"final":"25.0"}\n```', "25")
    assert result.structural_completion is True
    assert result.normalized_correct is True
    assert result.strict_correct is False


def test_strict_protocol_rejects_extra_final_blocks_and_prose_numbers():
    assert extract_final('The candidates are 2 and 5.') == (None, "no_single_final_json_block")
    assert extract_final('```json\n{"final":"2"}\n```\n```json\n{"final":"5"}\n```') == (
        None,
        "multiple_final_json_blocks",
    )


def test_normalized_transport_accepts_only_a_complete_unfenced_json_object():
    normalized = assess_answer('{"final":391}', "391", allow_unfenced=True)
    prose = assess_answer('answer: {"final":391}', "391", allow_unfenced=True)
    assert normalized.structural_completion is True
    assert normalized.normalized_correct is True
    assert prose.structural_completion is False


def test_exact_paired_test_does_not_overstate_one_discordant_pair():
    assert two_sided_exact_binomial(1, 0) == 1.0
    assert two_sided_exact_binomial(0, 0) == 1.0


def test_strict_protocol_rejects_nonfinal_json_and_wrong_number():
    malformed = assess_answer('```json\n{"answer":"25"}\n```', "25")
    wrong = assess_answer('```json\n{"final":"5"}\n```', "2")
    assert malformed.structural_completion is False
    assert wrong.structural_completion is True
    assert wrong.normalized_correct is False
    assert numeric_equal("12", 12)
