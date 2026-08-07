from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.preflight_submission import numerically_equal


ROOT = Path(__file__).resolve().parents[2]


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_numeric_answer_comparison_normalizes_equivalent_values():
    assert numerically_equal("25.0", "25")
    assert numerically_equal(" 12 ", 12)
    assert not numerically_equal("5", "2")


def test_full_configuration_is_19_of_20_by_gold_recheck():
    rows = load_json("logs/cross_model_layers/summary.json")["full_l1_l4"]["per_task"]
    correct = sum(numerically_equal(row["pred"], row["gold"]) for row in rows)
    assert len(rows) == 20
    assert correct == 19
    assert any(row["ok"] and not numerically_equal(row["pred"], row["gold"])
               for row in rows)


def test_historical_repeated_runs_are_described_as_non_inferential():
    rows = load_json("logs/stat_tests/results.json")
    assert [row["episodes"] for row in rows] == [16, 16, 16]
    manuscript = (ROOT / "papers/preprint_unified_en.md").read_text(encoding="utf-8")
    assert "Historical Repeated-Run Summary (Descriptive Only)" in manuscript
    assert "95% confidence interval" not in manuscript
    assert "3 seeds x 15 episodes" not in manuscript


def test_manuscript_reports_rechecked_hard_eval_result():
    manuscript = (ROOT / "papers/preprint_unified_en.md").read_text(encoding="utf-8")
    assert "95.0% correctness" in manuscript
    assert "independent gold recheck" in manuscript
    assert "pure layer effect" in manuscript
    assert "100.0% (20/20)" not in manuscript


def test_generated_docx_contains_figures_and_tables():
    path = ROOT / "papers/docx/preprint_unified_en.docx"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        document_xml = archive.read("word/document.xml")
    assert sum(name.startswith("word/media/") for name in names) >= 2
    assert document_xml.count(b"<w:tbl>") >= 8


def test_license_metadata_is_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache-2.0" in pyproject
    assert "Apache License" in license_text
