"""Cross-platform submission preflight for the unified manuscript."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "papers"


class Preflight:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        prefix = "OK" if condition else "FAIL"
        print(f"[{prefix}] {message}")
        if not condition:
            self.failures.append(message)


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def numerically_equal(left, right) -> bool:
    try:
        return Decimal(str(left).strip()) == Decimal(str(right).strip())
    except (InvalidOperation, ValueError):
        return str(left).strip().casefold() == str(right).strip().casefold()


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args()
    audit = Preflight()

    required = [
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "papers/preprint_unified_en.md",
        "papers/preprint_unified_en.pdf",
        "papers/docx/preprint_unified_en.docx",
        "papers/COVER_LETTER.md",
        "papers/PUBLISHING.md",
    ]
    for relative_path in required:
        audit.check((ROOT / relative_path).is_file(), f"required file: {relative_path}")

    source_files = [
        PAPERS / "preprint_unified_en.md",
        PAPERS / "00_INDEX_en.md",
        PAPERS / "COVER_LETTER.md",
        PAPERS / "PUBLISHING.md",
        PAPERS / "README.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8-sig") for path in source_files)
    forbidden = {
        "\u2014?": "corrupted dash marker",
        "100.0 PERCENT": "temporary percentage marker",
        "p<0.01": "superseded significance claim",
        "Reviewer-sim": "internal reviewer score",
        "4.5 / 5.0": "internal self-score",
        "Five Papers for TMLR Submission": "stale five-paper submission text",
    }
    for token, description in forbidden.items():
        audit.check(token not in combined, f"no {description}")

    placeholders = ("<org>", "your-org", "AGI Research Kit Contributors")
    metadata_text = "\n".join(
        (PAPERS / name).read_text(encoding="utf-8-sig")
        for name in ("preprint_unified_en.md", "COVER_LETTER.md")
    ) + (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    has_placeholders = any(token in metadata_text for token in placeholders)
    audit.check(args.allow_placeholders or not has_placeholders,
                "final author and repository metadata supplied")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    audit.check("Apache-2.0" in pyproject and "Apache License" in license_text,
                "package metadata and LICENSE both use Apache-2.0")

    figure_paths = [PAPERS / "figures" / f"fig{i}_{name}.png" for i, name in (
        (1, "layer_ablation"),
        (2, "generation_curve"),
        (3, "l1_scoring_ablation"),
        (4, "l2_stuck_latency"),
        (5, "l4_mutator_activity"),
    )]
    audit.check(all(path.is_file() and path.stat().st_size > 10_000 for path in figure_paths),
                "all five figure files exist and are nontrivial")

    ablation = []
    for name in ("static", "l1_only", "l1_l2", "l1_l2_l3", "full"):
        summary = load_json(f"logs/ablation/{name}/summary.json")
        ablation.append(round(float(summary["success_rate"]), 4))
    audit.check(len(set(ablation)) == 1 and ablation[0] == round(7 / 9, 4),
                "ablation summaries support five equal 77.8% rates")

    bare_three = load_json("logs/cross_model/results.json")["qwen3:1.7b"]
    bare_six = load_json("logs/cross_model_bare_qwen1.7b_max6/summary.json")
    full = load_json("logs/cross_model_layers/summary.json")["full_l1_l4"]
    full_correct = sum(
        numerically_equal(row["pred"], row["gold"]) for row in full["per_task"]
    )
    audit.check(bare_three["n_correct"] == 1 and bare_three["n_tasks"] == 20,
                "bare three-step result is 1/20")
    audit.check(bare_six["correct"] == 7 and bare_six["n_tasks"] == 20,
                "bare six-step result is 7/20")
    audit.check(full_correct == 19 and len(full["per_task"]) == 20,
                "independent full-configuration gold recheck is 19/20")

    controlled = load_json("logs/controlled_arithmetic/run-llama3b-20260802/summary.json")
    controlled_analysis = load_json("logs/controlled_arithmetic/run-llama3b-20260802/paired_analysis.json")
    audit.check(controlled["configs"]["static"]["correct"] == 19
                and controlled["configs"]["l1_reflection"]["correct"] == 20,
                "controlled L1 run records 19/20 static and 20/20 L1 gold correctness")
    audit.check(controlled_analysis["l1_wins"] == 1
                and controlled_analysis["l1_losses"] == 0
                and controlled_analysis["two_sided_exact_mcnemar_p"] == 1.0,
                "controlled L1 paired analysis reports one discordant pair and p=1.0")

    sft_validation = load_json("logs/sft_validation/round18.json")
    audit.check(sft_validation["deployment_decision"] == "reject"
                and sft_validation["transformers_cpu_smoke"]["correct"] is False,
                "real SFT candidate failure is recorded without an acceptance claim")

    gate = load_json("logs/safety_gate/stress_test.json")
    audit.check(gate["n_match"] == gate["n_total"] == 12,
                "safety-gate boundary summary is 12/12")

    redteam_rows = [json.loads(line) for line in
                    (ROOT / "logs/redteam/l4_redteam.jsonl")
                    .read_text(encoding="utf-8").splitlines() if line.strip()]
    malicious = [row for row in redteam_rows if row["expected_block"]]
    benign = [row for row in redteam_rows if not row["expected_block"]]
    audit.check(len(malicious) == 18 and all(not row["actually_accepted"]
                                             and row["match"] for row in malicious),
                "production SchemaMutator rejects all 18 invalid cases")
    audit.check(len(benign) == 12 and all(row["actually_accepted"]
                                         and row["match"] for row in benign),
                "production SchemaMutator accepts all 12 valid controls")

    repeated = load_json("logs/stat_tests/results.json")
    audit.check([row["episodes"] for row in repeated] == [16, 16, 16],
                "historical repeated-run snapshot contains three 16-episode runs")

    docx_path = PAPERS / "docx" / "preprint_unified_en.docx"
    if docx_path.is_file():
        with zipfile.ZipFile(docx_path) as archive:
            names = archive.namelist()
            document_xml = archive.read("word/document.xml")
        media_count = sum(name.startswith("word/media/") for name in names)
        table_count = document_xml.count(b"<w:tbl>")
        audit.check(media_count >= 2, "DOCX embeds the two evidence-backed figures")
        audit.check(table_count >= 8, "DOCX contains manuscript tables")

    tracked = git_output("ls-files").splitlines()
    oversized = [path for path in tracked if (ROOT / path).is_file()
                 and (ROOT / path).stat().st_size > 50 * 1024 * 1024]
    audit.check(not oversized, "no tracked file exceeds 50 MiB")
    audit.check(not (ROOT / ".env").exists(), "no local .env file")
    dirty = bool(git_output("status", "--porcelain"))
    audit.check(args.allow_dirty or not dirty, "Git worktree is clean")

    if audit.failures:
        print(f"\nPreflight failed: {len(audit.failures)} item(s).")
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
