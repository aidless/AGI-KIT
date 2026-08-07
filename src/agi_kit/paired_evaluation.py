from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable, Sequence


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


class PairedEvaluator:
    """Compare two models on identical tasks and an identical budget."""

    def __init__(
        self,
        tasks: Sequence[dict[str, Any]],
        run_fn: Callable[[str, Any, Any, Any], object],
        budget: Any,
        min_effect: float = 0.0,
        max_regression: float = 0.0,
        bootstrap_samples: int = 2000,
        seed: int = 0,
        task_set_id: str | None = None,
    ) -> None:
        if len(tasks) < 2:
            raise ValueError("paired evaluation needs at least two tasks")
        if bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        self.tasks = tuple(dict(task) for task in tasks)
        self.run_fn = run_fn
        self.budget = budget
        self.min_effect = min_effect
        self.max_regression = max_regression
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        payload = json.dumps(list(self.tasks), sort_keys=True, ensure_ascii=False)
        self.task_set_id = task_set_id or hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.last_result: dict[str, Any] = {}

    @staticmethod
    def _correct(value: object) -> bool:
        if isinstance(value, dict):
            return bool(value.get("correct", False))
        if isinstance(value, bool):
            return value
        return float(value) >= 1.0

    def __call__(self, baseline_model: str, candidate_model: str) -> dict[str, Any]:
        rows = []
        for index, item in enumerate(self.tasks):
            task_id = str(item.get("id", index))
            baseline = self._correct(self.run_fn(
                baseline_model, item.get("task"), item.get("gold"), self.budget))
            candidate = self._correct(self.run_fn(
                candidate_model, item.get("task"), item.get("gold"), self.budget))
            rows.append({
                "task_id": task_id,
                "baseline_correct": baseline,
                "candidate_correct": candidate,
                "difference": int(candidate) - int(baseline),
            })
        differences = [float(row["difference"]) for row in rows]
        rng = random.Random(self.seed)
        bootstrap_means = []
        for _ in range(self.bootstrap_samples):
            sample = [rng.choice(differences) for _ in differences]
            bootstrap_means.append(sum(sample) / len(sample))
        baseline_accuracy = sum(row["baseline_correct"] for row in rows) / len(rows)
        candidate_accuracy = sum(row["candidate_correct"] for row in rows) / len(rows)
        mean_difference = sum(differences) / len(differences)
        ci = [
            _percentile(bootstrap_means, 0.025),
            _percentile(bootstrap_means, 0.975),
        ]
        accepted = (
            mean_difference >= self.min_effect and
            ci[0] >= -self.max_regression
        )
        self.last_result = {
            "task_set_id": self.task_set_id,
            "n_tasks": len(rows),
            "budget": self.budget,
            "baseline_model": baseline_model,
            "candidate_model": candidate_model,
            "baseline_accuracy": baseline_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "mean_difference": mean_difference,
            "bootstrap_ci_95": ci,
            "min_effect": self.min_effect,
            "max_regression": self.max_regression,
            "seed": self.seed,
            "accepted": accepted,
            "results": rows,
        }
        return {
            "accepted": accepted,
            "new_acc": candidate_accuracy,
            "baseline_acc": baseline_accuracy,
            "reason": "paired_gate_passed" if accepted else "paired_gate_failed",
            "evaluation": self.last_result,
        }
