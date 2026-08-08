"""Run the five-criterion L3 acceptance audit and record an audited swap."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exact_mcnemar_two_sided(n_discordant: int) -> float:
    return 1.0 if n_discordant == 0 else 2 * 0.5 ** n_discordant


def main() -> None:
    base_model = "qwen3:0.6b"
    candidate_model = "agi-sft-qwen3-0.6b:round22"
    out_dir = ROOT / "logs" / "sft_round22"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Executable via Ollama: smoke test
    smoke = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "eval_round22_ollama.py"),
            candidate_model,
            "--label",
            "candidate_smoke",
            "--split",
            "dev",
            "--limit",
            "2",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    smoke_ok = smoke.returncode == 0

    # 2 + 3. Frozen 80-task held-out eval for candidate and base
    candidate_eval = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "eval_round22_ollama.py"),
            candidate_model,
            "--label",
            "candidate_test80",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    base_eval = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "eval_round22_ollama.py"),
            base_model,
            "--label",
            "base_test80",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )

    candidate = json.loads((out_dir / "candidate_test80.json").read_text(encoding="utf-8"))
    base = json.loads((out_dir / "base_test80.json").read_text(encoding="utf-8"))
    candidate_correct = candidate["correct"]
    base_correct = base["correct"]
    n = candidate["n"]

    # Paired discordant counts on aligned task ids
    cand_map = {row["id"]: row["correct"] for row in candidate["rows"]}
    base_map = {row["id"]: row["correct"] for row in base["rows"]}
    discordant = sum(
        1
        for task_id in cand_map
        if task_id in base_map and cand_map[task_id] != base_map[task_id]
    )
    candidate_wins = sum(
        1
        for task_id in cand_map
        if task_id in base_map and cand_map[task_id] and not base_map[task_id]
    )
    base_wins = sum(
        1
        for task_id in cand_map
        if task_id in base_map and not cand_map[task_id] and base_map[task_id]
    )
    p_value = exact_mcnemar_two_sided(discordant)

    # 4. A/B gate: accept only with a non-degenerate evidence base.
    #    If the base scores 0 on the frozen test, an "infinite" ratio must
    #    not be treated as a pass; the candidate has not demonstrated a gain.
    threshold = 0.85
    ratio = candidate_correct / base_correct if base_correct else None
    gate_accepted = (
        base_correct > 0
        and candidate_correct > 0
        and ratio is not None
        and ratio >= threshold
    )

    # 5. Audited swap record
    audit = {
        "round": 22,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate": candidate_model,
        "base": base_model,
        "criterion_1_executable_ollama": {
            "smoke_returncode": smoke.returncode,
            "smoke_stdout_tail": smoke.stdout[-300:],
            "smoke_stderr_tail": smoke.stderr[-300:],
            "passed": smoke_ok,
        },
        "criterion_2_frozen_heldout": {
            "split": "test",
            "n": n,
            "candidate_correct": candidate_correct,
            "base_correct": base_correct,
            "manifest": "data/sft_round20/test.json",
        },
        "criterion_3_statistical": {
            "discordant_pairs": discordant,
            "candidate_wins": candidate_wins,
            "base_wins": base_wins,
            "two_sided_exact_mcnemar_p": p_value,
        },
        "criterion_4_gate": {
            "threshold": threshold,
            "ratio": None if ratio is None else round(ratio, 4),
            "accepted": gate_accepted,
        },
        "criterion_5_audited_swap": {
            "swap_executed": gate_accepted,
            "running_model_after": candidate_model if gate_accepted else base_model,
            "audit_file": str(out_dir / "swap_audit.json"),
        },
    }
    (out_dir / "swap_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Persistent current-model state
    state = {
        "round": 22,
        "running_model": audit["criterion_5_audited_swap"]["running_model_after"],
        "updated_at": audit["timestamp"],
    }
    (out_dir / "current_model.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
