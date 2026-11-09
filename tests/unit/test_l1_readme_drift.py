"""Lock README / FAILURE_MODES L1 sentences to committed traces.

This does not rerun models and does not invent a p-value. It reads the
JSONL already in the tree and the p-value already stored in
paired_analysis.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.analyze_controlled_arithmetic import two_sided_exact_binomial


ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
FAILURES = (ROOT / "FAILURE_MODES.md").read_text(encoding="utf-8")

DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:(?:\\|/)")


def _block(text: str, name: str) -> dict[str, str]:
    match = re.search(rf"<!-- {name}\n(.*?)-->", text, re.S)
    assert match, name
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _rows(relative_run: str) -> list[dict]:
    path = ROOT / relative_run / "per_task.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalized(rows: list[dict], config: str) -> int:
    return sum(
        1
        for row in rows
        if row["config"] == config and row["assessment"]["normalized_correct"]
    )


def test_readme_l1_block_matches_llama_trace():
    fields = _block(README, "l1-evidence")
    rows = _rows(fields["run"])
    static = _normalized(rows, "static")
    l1 = _normalized(rows, "l1_reflection")
    assert static == int(fields["static_normalized_correct"]) == 19
    assert l1 == int(fields["l1_normalized_correct"]) == 20
    assert "Static 19/20, L1 20/20" in README
    assert "not general superiority" in README

    summary = json.loads((ROOT / fields["run"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["configs"]["static"]["correct"] == static
    assert summary["configs"]["l1_reflection"]["correct"] == l1
    assert summary["n_tasks"] == 20
    assert summary["split"] == "test"

    manifest = json.loads((ROOT / fields["manifest"]).read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    assert len(tasks) == int(fields["manifest_tasks"]) == 30
    assert sum(task["split"] == "test" for task in tasks) == int(fields["held_out_tasks"]) == 20

    paired = json.loads((ROOT / fields["run"] / "paired_analysis.json").read_text(encoding="utf-8"))
    assert paired["discordant_pairs"] == int(fields["discordant_pairs"]) == 1
    assert paired["l1_wins"] == 1 and paired["l1_losses"] == 0
    assert paired["two_sided_exact_mcnemar_p"] == float(fields["recorded_mcnemar_p"]) == 1.0
    assert two_sided_exact_binomial(paired["l1_wins"], paired["l1_losses"]) == paired[
        "two_sided_exact_mcnemar_p"
    ]

    by_id = {
        (row["config"], row["task_id"]): row
        for row in rows
    }
    static_row = by_id[("static", "ca020")]
    l1_row = by_id[("l1_reflection", "ca020")]
    assert static_row["assessment"]["normalized_correct"] is False
    assert l1_row["assessment"]["normalized_correct"] is True
    assert static_row["assessment"]["final"] == "9003"
    assert l1_row["assessment"]["final"] == "9001"
    assert static_row["gold"] == "9001"
    assert l1_row["reflection_retry_used"] is True


def test_failure_modes_match_stricter_and_other_run():
    fields = _block(README, "l1-evidence")
    rows = _rows(fields["run"])
    strict = {
        config: sum(
            1
            for row in rows
            if row["config"] == config and row["assessment"]["strict_correct"]
        )
        for config in ("static", "l1_reflection")
    }
    assert strict == {"static": 17, "l1_reflection": 18}
    assert "17/20 Static and 18/20 L1" in FAILURES
    summary = json.loads((ROOT / fields["run"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["configs"]["static"]["strict_fenced_completed"] == 0
    assert summary["configs"]["l1_reflection"]["strict_fenced_completed"] == 0
    assert "strict_fenced_completed" in FAILURES
    assert "`ca020`" in FAILURES

    other = _block(FAILURES, "l1-other-run")
    other_rows = _rows(other["run"])
    assert _normalized(other_rows, "static") == int(other["static_normalized_correct"]) == 9
    assert _normalized(other_rows, "l1_reflection") == int(other["l1_normalized_correct"]) == 6
    assert "Static 9/20 and L1 6/20" in FAILURES


def test_withdrawn_claims_stay_withdrawn():
    assert "Withdrawn, and not results" in README
    assert "p<0.01" in README
    assert "30% static baseline" in README
    assert "4.5 score" in README
    assert "not been submitted" in README
    assert "STATUS: WORK-IN-PROGRESS" in README
    # The withdrawn tokens appear in the withdrawal sentence only.
    body = README.split("Withdrawn, and not results", 1)[1]
    withdrawal, _, rest = body.partition("\n")
    assert "p<0.01" in withdrawal
    assert "p<0.01" not in rest
    assert "4.5 score" not in rest


def test_runnable_docs_and_code_have_no_drive_paths():
    roots = [
        ROOT / "src",
        ROOT / "scripts",
        ROOT / "experiments",
        ROOT / "tests",
        ROOT / "papers",
    ]
    files = [
        ROOT / "README.md",
        ROOT / "FAILURE_MODES.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "REPORT.md",
        ROOT / "PAPER_DRAFT.md",
        ROOT / "Makefile",
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / "requirements.txt",
        ROOT / "requirements-test.txt",
        ROOT / "data" / "README.md",
        ROOT / "data" / "gaia2" / "SCHEMA.md",
    ]
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file())
    offenders: list[str] = []
    for path in files:
        if path.suffix.lower() not in {".py", ".md", ".ps1", ".yml", ".yaml", ".txt", ""}:
            continue
        if path.name in {"Makefile"} or path.suffix.lower() in {".py", ".md", ".ps1", ".yml", ".yaml", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if DRIVE_PATH.search(text):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
