"""artifact_checklist.py - honest inventory of shipped evidence.

This script replaces scripts/real_reviewer.py (Round 11) which was a
self-dealing rubric designed to score 4.5. That score was engineering,
not measurement. This replacement is a checklist with no axes, no
weights, and no score. It only reports which evidence artifacts exist
on disk and which are missing.

If you want a 4-5 scale, write scripts/external_reviewer.py and have
someone outside this repo run it. The text inside this repo cannot
honestly award itself a score.
"""
"""
Imports block: keep docstring at top is from the first Set-Content.
"""
import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
REVIEWS = ROOT / "papers" / "reviews"
REVIEWS.mkdir(parents=True, exist_ok=True)


CHECKLIST = [
    ("Preprint (markdown)",            "papers/preprint_unified_en.md",          "file"),
    ("Preprint (PDF)",                  "papers/preprint_unified_en.pdf",          "file"),
    ("Preprint (DOCX)",                 "papers/docx/preprint_unified_en.docx",   "file"),
    ("Paper INDEX",                     "papers/00_INDEX_en.pdf",                  "file"),
    ("Cover letter",                    "papers/COVER_LETTER.md",                  "file"),
    ("Publishing guide",                "papers/PUBLISHING.md",                    "file"),
    ("Figure 1 - layer ablation",        "papers/figures/fig1_layer_ablation.png", "file"),
    ("Figure 2 - generation curve",      "papers/figures/fig2_generation_curve.png", "file"),
    ("Figure 3 - L1 scoring",            "papers/figures/fig3_l1_scoring_ablation.png", "file"),
    ("Figure 4 - L2 stuck latency",      "papers/figures/fig4_l2_stuck_latency.png", "file"),
    ("Figure 5 - L4 mutator activity",   "papers/figures/fig5_l4_mutator_activity.png", "file"),
    ("A/B safety gate boundary tests",   "logs/safety_gate/stress_test.json",       "file"),
    ("A/B safety gate summary",          "logs/safety_gate/summary.md",             "file"),
    ("L4 red-team trace",                "logs/redteam/l4_redteam.jsonl",           "file"),
    ("L4 red-team summary",              "logs/redteam/l4_redteam_summary.md",      "file"),
    ("Gate calibration grid",            "logs/calibration/gate_calibration.json", "file"),
    ("Gate calibration summary",         "logs/calibration/gate_calibration_summary.md", "file"),
    ("Statistical tests summary",        "logs/stat_tests/summary.md",              "file"),
    ("Cross-model summary",              "logs/cross_model/summary.md",            "file"),
    ("Arithmetic eval JSON",             "logs/seeds_arith/arith_qwen3_1_7b_n10.json", "file"),
    ("Real GAIA2-mini scenarios",        "data/gaia2/validation.jsonl",             "file"),
    ("GAIA2 schema notes",               "data/gaia2/SCHEMA.md",                   "file"),
    ("GAIA2 app shim - Calendar",        "src/agi_kit/apps/gaia2/calendar.py",     "file"),
    ("GAIA2 app shim - Emails",          "src/agi_kit/apps/gaia2/emails.py",       "file"),
    ("GAIA2 app shim - Shopping",        "src/agi_kit/apps/gaia2/shopping.py",     "file"),
    ("Ablation - static",                "logs/ablation/static",                   "dir"),
    ("Ablation - l1_only",               "logs/ablation/l1_only",                  "dir"),
    ("Ablation - l1_l2",                 "logs/ablation/l1_l2",                    "dir"),
    ("Ablation - l1_l2_l3",              "logs/ablation/l1_l2_l3",                 "dir"),
    ("LICENSE",                          "LICENSE",                                "file"),
    ("Project README",                   "README.md",                              "file"),
    ("Papers README",                    "papers/README.md",                       "file"),
    ("Tarball (off-band)",               "dist/agi-research-kit.tar.gz",           "file"),
    ("GitHub Actions CI",                ".github/workflows/ci.yml",               "file"),
    ("push.sh with self-checks",         "dist/push.sh",                           "file"),
    ("L1 Reflector",                     "src/agi_kit/reflect.py",                 "file"),
    ("L2 Meta-Controller",               "src/agi_kit/meta.py",                    "file"),
    ("L3 Continual Loop + safety gate",  "src/agi_kit/loop.py",                    "file"),
    ("L4 Recursive Modifier",            "src/agi_kit/recursive.py",               "file"),
    ("Safety gate boundary test driver", "experiments/stress_safety_gate.py",      "file"),
    ("L4 red-team test driver",          "experiments/redteam/l4_redteam.py",      "file"),
    ("Gate calibration test driver",     "experiments/gate_calibration.py",        "file"),
]


def _check(expected_path, kind):
    p = ROOT / expected_path
    if not p.exists():
        return False, "missing"
    if kind == "dir":
        children = list(p.glob("*"))
        if children:
            return True, str(len(children)) + " child file(s)"
        return False, "empty dir"
    if kind == "file":
        sz = p.stat().st_size
        return True, str(sz) + " bytes"
    return False, "unknown kind"


def run():
    n_present = 0
    n_missing = 0
    out = []
    out.append("# AGI Kit Artifact Checklist")
    out.append("")
    out.append("Generated: " + datetime.datetime.now().isoformat())
    out.append("")
    out.append("This is an honest inventory. No score is calculated.")
    out.append("Each artifact is either present on disk or missing.")
    out.append("")
    out.append("Round 11 replaced a self-dealing scoring tool (real_reviewer.py,")
    out.append("which calibrated a rubric to award 4.5) with this checklist.")
    out.append("")
    out.append("## Items")
    out.append("")
    out.append("| # | Item | Expected path | Status |")
    out.append("|---|---|---|---|")
    for i, (label, path, kind) in enumerate(CHECKLIST, 1):
        ok, detail = _check(path, kind)
        status = "PRESENT (" + detail + ")" if ok else "MISSING"
        if ok:
            n_present += 1
        else:
            n_missing += 1
        out.append("| " + str(i) + " | " + label + " | `" + path + "` | " + status + " |")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("Total items: " + str(len(CHECKLIST)))
    out.append("Present:    " + str(n_present))
    out.append("Missing:    " + str(n_missing))
    out.append("")
    text = chr(10).join(out)
    (REVIEWS / "artifact_checklist.txt").write_text(text, encoding="utf-8")
    summary = ("Total items: " + str(len(CHECKLIST)) + chr(10) +
               "Present:    " + str(n_present) + chr(10) +
               "Missing:    " + str(n_missing) + chr(10))
    (REVIEWS / "artifact_checklist_summary.txt").write_text(summary, encoding="utf-8")
    print(text)
    print(chr(10).join(["", "Wrote papers/reviews/artifact_checklist.txt",
                       "Wrote papers/reviews/artifact_checklist_summary.txt"]))


if __name__ == "__main__":
    run()
