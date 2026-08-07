"""Strict, auditable answer extraction for agent evaluations.

The protocol keeps transport success (a parseable JSON final) separate from
semantic correctness.  It deliberately does not search arbitrary prose for a
number: doing so can turn an explanation containing several candidate answers
into a false positive.
"""
from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Sequence


JSON_FINAL_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class AnswerAssessment:
    structural_completion: bool
    final: str | None
    normalized_correct: bool
    strict_correct: bool
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeldOutEvaluator:
    """Evaluate models on one immutable task set with a fixed budget.

    ``run_fn`` receives ``(model, task, gold, budget)`` and returns either a
    boolean, a numeric score, or a mapping containing ``correct``. The latest
    evaluation summary is available through ``last_result`` for audit logs.
    """

    def __init__(
        self,
        tasks: Sequence[dict[str, Any]],
        run_fn: Callable[[str, Any, Any, Any], object],
        budget: Any,
        task_set_id: str | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("held-out task set must not be empty")
        self.tasks = tuple(dict(task) for task in tasks)
        self.run_fn = run_fn
        self.budget = budget
        self.task_set_id = task_set_id or self._hash_tasks(self.tasks)
        self.last_result: dict[str, Any] = {}

    @staticmethod
    def _hash_tasks(tasks: Sequence[dict[str, Any]]) -> str:
        payload = json.dumps(list(tasks), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __call__(self, model: str) -> float:
        results = []
        for index, item in enumerate(self.tasks):
            task_id = str(item.get("id", index))
            value = self.run_fn(model, item.get("task"), item.get("gold"), self.budget)
            if isinstance(value, dict):
                correct = bool(value.get("correct", False))
                score = value.get("score")
            else:
                correct = bool(value) if isinstance(value, bool) else float(value) >= 1.0
                score = float(value) if isinstance(value, (int, float)) else None
            results.append({"task_id": task_id, "correct": correct, "score": score})
        accuracy = sum(result["correct"] for result in results) / len(results)
        self.last_result = {
            "task_set_id": self.task_set_id,
            "n_tasks": len(results),
            "budget": self.budget,
            "model": model,
            "accuracy": accuracy,
            "results": results,
        }
        return accuracy


def numeric_equal(left: object, right: object) -> bool:
    """Compare finite decimal answers without accepting arbitrary expressions."""
    try:
        a = Decimal(str(left).strip())
        b = Decimal(str(right).strip())
        return a.is_finite() and b.is_finite() and a == b
    except (InvalidOperation, ValueError):
        return str(left).strip().casefold() == str(right).strip().casefold()


def extract_final(text: str, *, allow_unfenced: bool = False) -> tuple[str | None, str | None]:
    """Return exactly one JSON ``final`` field or a reason it was rejected.

    ``allow_unfenced`` is deliberately narrow: it accepts only a complete JSON
    object, never a JSON-looking substring from prose.
    """
    candidates: list[str] = []
    for match in JSON_FINAL_RE.finditer(text or ""):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == {"final"}:
            candidates.append(str(value["final"]))
    if not candidates and allow_unfenced:
        try:
            value = json.loads((text or "").strip())
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and set(value) == {"final"}:
            candidates.append(str(value["final"]))
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, "no_single_final_json_block"
    return None, "multiple_final_json_blocks"


def assess_answer(text: str, gold: object, *, allow_unfenced: bool = False) -> AnswerAssessment:
    final, error = extract_final(text, allow_unfenced=allow_unfenced)
    if error:
        return AnswerAssessment(False, None, False, False, error)
    assert final is not None
    return AnswerAssessment(
        structural_completion=True,
        final=final,
        normalized_correct=numeric_equal(final, gold),
        strict_correct=final == str(gold),
    )
