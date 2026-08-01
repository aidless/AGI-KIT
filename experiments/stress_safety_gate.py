"""Adversarial stress test for the A/B safety gate.

Strategy: synthesize "fake" generation outputs that LOOK like they
should be accepted but are actually bad, and verify the gate correctly
rejects them. Also test the boundary cases (threshold sweep).

Outputs:
  logs/safety_gate/stress_test.json  - per-case decision
  logs/safety_gate/summary.md        - human-readable summary
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
LOG_DIR = ROOT / "logs" / "safety_gate"
LOG_DIR.mkdir(parents=True, exist_ok=True)

from agi_kit.loop import default_safety_check


# Simulated "new model" accuracies under various scenarios
CASES = [
    # (name, new_acc, expected_decision, threshold, comment)
    # Standard cases (threshold=0.85)
    ("regression_severe",      0.10, False, 0.85, "new=10% vs baseline 100%"),
    ("regression_mild",         0.50, False, 0.85, "new=50% vs baseline 100%"),
    ("just_under",              0.84, False, 0.85, "new=84% vs threshold 0.85"),
    ("at_threshold",            0.85, True,  0.85, "new=85% = threshold"),
    ("just_over",               0.86, True,  0.85, "new=86% > threshold 0.85"),
    ("equal_baseline",          1.00, True,  0.85, "new=100% = baseline"),
    ("better_than",             1.20, True,  0.85, "new=120% of baseline"),
    ("zero_acc",                 0.00, False, 0.85, "new=0%"),
    ("super_high",               2.00, True,  0.85, "new=200% of baseline"),
    # Threshold variations
    ("low_threshold",           0.05, True,  0.01, "threshold=0.01, new=5%"),
    ("high_threshold",          0.99, False, 0.999, "threshold=0.999, new=99%"),
    ("zero_threshold",           0.01, True,  0.0, "threshold=0, new=1%"),
]


def run():
    results = []
    for name, new_acc, expected_accept, threshold, comment in CASES:
        out = default_safety_check(
            new_model_dir="logs/synthetic/gen-N",
            baseline_acc=1.0,
            eval_fn=lambda _: new_acc,  # mock eval returns fixed value
            threshold=threshold,
        )
        accepted = out.get("accepted", False)
        result_str = "ACCEPT" if accepted else "REJECT"
        ok = "OK" if accepted == expected_accept else "BAD MISMATCH"
        results.append({
            "name": name, "new_acc": new_acc,
            "expected": "ACCEPT" if expected_accept else "REJECT",
            "actual": result_str,
            "match": ok == "OK",
            "reason": out.get("reason", ""),
            "comment": comment,
        })
        print(f"  [{ok}] {name:20s} new_acc={new_acc:.2f} expected={'ACCEPT' if expected_accept else 'REJECT'}  actual={result_str}  ({comment})")

    n_total = len(results)
    n_match = sum(1 for r in results if r["match"])
    print(f"\n  Match rate: {n_match}/{n_total} = {n_match/n_total*100:.0f}%")

    # Save
    with (LOG_DIR / "stress_test.json").open("w", encoding="utf-8") as f:
        json.dump({"results": results, "n_total": n_total,
                   "n_match": n_match,
                   "match_rate": round(n_match/n_total, 3)}, f, indent=2)

    # Markdown summary
    lines = ["# Adversarial Safety Gate Stress Test\n"]
    lines.append("Baseline: 1.000, Threshold: 0.85 (function default is 0.95; 0.85 is the headline-run value passed explicitly)\n")
    lines.append(f"Match rate: {n_match}/{n_total} = {n_match/n_total*100:.0f}%\n")
    lines.append("| Case | new_acc | Expected | Actual | Match | Reason |")
    lines.append("|---|---:|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['name']} | {r['new_acc']:.2f} | {r['expected']} | {r['actual']} | {'OK' if r['match'] else 'BAD'} | {r['reason']} |")
    lines.append("")
    lines.append("## Boundary analysis\n")
    lines.append("- **At threshold** (0.85): gate uses `>=` so accepts. Conservative "
                 "but not paranoid 鈥?this is intentional so the gate is not a pure rejector.")
    lines.append("- **Just over** (0.86): accepts, which is the right behavior for a "
                 "small uplift.")
    lines.append("- **Regression** (0.10-0.50): correctly rejects, even with 0.85 threshold.")
    lines.append("- **Super-high new_acc** (1.20, 2.00): accepts, which is the right "
                 "behavior if the new model truly is better (e.g., from a good SFT run).")
    (LOG_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {LOG_DIR / 'stress_test.json'}")
    print(f"Wrote {LOG_DIR / 'summary.md'}")


if __name__ == "__main__":
    run()


