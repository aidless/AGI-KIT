# Gate calibration demo
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
LOG_DIR = ROOT / "logs" / "calibration"
LOG_DIR.mkdir(parents=True, exist_ok=True)

from agi_kit.loop import default_safety_check


# 5 deployment profiles, each with a recommended threshold
PROFILES = [
    ("medical",    0.999, 0.99, "stricter; safety-critical"),
    ("finance",    0.999, 0.95, "near-monotonic; rare regression allowed"),
    ("casual_chat",0.85,  0.85, "default"),
    ("code_review",0.50,  0.50, "lenient; quality gain more important"),
    ("customer_service", 0.95, 0.95, "between"),
]


# synthetic candidates per profile: 12 trials, mix of regressed and improved
CANDIDATES = [0.10, 0.30, 0.50, 0.70, 0.84, 0.86, 0.95, 1.00, 1.10, 1.20, 1.50, 2.00]


def run():
    rows = []
    for profile, recommended, threshold_test, note in PROFILES:
        for new_acc in CANDIDATES:
            out = default_safety_check(
                new_model_dir="synthetic/candidate",
                baseline_acc=1.0,
                eval_fn=lambda _: new_acc,
                threshold=threshold_test,
            )
            rows.append({
                "profile": profile,
                "new_acc": new_acc,
                "threshold": threshold_test,
                "recommended_threshold": recommended,
                "accepted": out.get("accepted"),
                "reason": out.get("reason"),
            })

    out_json = LOG_DIR / "gate_calibration.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    md = []
    md.append("# Gate Calibration Across Deployment Profiles")
    md.append("")
    md.append("5 deployment profiles x 12 candidate accuracies = 60 trials.")
    md.append("")
    md.append("| Profile | Recommended | Test threshold | new_acc | Accepted | Reason |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        act = "ACCEPT" if r["accepted"] else "REJECT"
        md.append("| " + r["profile"] + " | " + str(r["recommended_threshold"]) + " | " + str(r["threshold"]) + " | " + str(r["new_acc"]) + " | " + act + " | " + str(r["reason"]) + " |")

    summary = []
    for profile, _, t, _ in PROFILES:
        sub = [r for r in rows if r["profile"] == profile]
        n = len(sub)
        a = sum(1 for r in sub if r["accepted"])
        summary.append((profile, t, n, a))

    md.append("")
    md.append("## Summary by profile")
    md.append("")
    md.append("| Profile | Threshold | Trials | Accepted | Acceptance rate |")
    md.append("|---|---|---|---|---|")
    for profile, t, n, a in summary:
        md.append("| " + profile + " | " + str(t) + " | " + str(n) + " | " + str(a) + " | " + str(round(100*a/n, 1)) + "% |")

    out_md = LOG_DIR / "gate_calibration_summary.md"
    out_md.write_text(chr(10).join(md), encoding="utf-8")
    print(chr(10).join(md))
    print(chr(10).join(["", "Wrote " + str(out_json), "Wrote " + str(out_md)]))


if __name__ == "__main__":
    run()
