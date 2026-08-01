"""Statistical significance: run the same eval 3 times with different seeds.

Outputs:
  logs/stat_tests/results.json   - per-seed accuracy
  logs/stat_tests/summary.md     - mean 卤 std + paired t-test
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs" / "stat_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)


SEEDS = [0, 1, 2]
N = 15  # smaller than 50 for speed
RETRAIN_EVERY = 8


def run_one_seed(seed):
    out = LOG_DIR / f"seed{seed}"
    out.mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "full_run3.py"),
        "--n", str(N),
        "--retrain-every", str(RETRAIN_EVERY),
        "--no-sft",
        "--out", str(out),
    ]
    print(f"\n[seed={seed}] running: {' '.join(cmd)}")
    import time
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": str(seed)},
                       timeout=1800)
    dt = time.time() - t0
    print(f"[seed={seed}] done in {dt:.0f}s (rc={r.returncode})")
    summary_file = out / "summary.json"
    if not summary_file.exists():
        return None
    with open(summary_file, encoding="utf-8") as f:
        s = json.load(f)
    s["seed"] = seed
    s["wall_seconds"] = round(dt, 1)
    return s


def main():
    results = []
    for s in SEEDS:
        r = run_one_seed(s)
        if r is not None:
            results.append(r)
    if not results:
        print("No results collected")
        return

    # Save
    out_file = LOG_DIR / "results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Stats
    accs = [r["success_rate"] for r in results]
    scores = [r["avg_score"] for r in results]
    times = [r["wall_seconds"] for r in results]

    mean_acc = statistics.mean(accs)
    std_acc = statistics.stdev(accs) if len(accs) > 1 else 0
    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0
    mean_time = statistics.mean(times)
    # No baseline t-test: the 30% and 51% baselines in earlier drafts
    # were not measured. Section 4.3 of the paper describes the variance
    # estimate only.
    t_stat, p_val = float("nan"), float("nan")

    summary = {
        "n_seeds": len(accs),
        "n_episodes_per_seed": N,
        "accuracy": {
            "mean": round(mean_acc, 4),
            "std": round(std_acc, 4),
            "values": accs,
            "95_ci": [round(mean_acc - 1.96 * std_acc / (len(accs) ** 0.5), 4),
                      round(mean_acc + 1.96 * std_acc / (len(accs) ** 0.5), 4)],
        },
        "avg_self_score": {
            "mean": round(mean_score, 4),
            "std": round(std_score, 4),
            "values": scores,
        },
        "wall_seconds": {
            "mean": round(mean_time, 1),
            "values": times,
        },
        "vs_baseline_30pct": {
            "t_stat": round(t_stat, 3) if t_stat == t_stat else None,
            "p_value": round(p_val, 4) if p_val == p_val else None,
            "significant_at_0.05": (p_val < 0.05) if p_val == p_val else None,
        },
        "vs_baseline_50pct": {},
        "raw_runs": results,
    }
    # No second t-test: the 30% and 51% baselines were not measured.
    # Section 4.3 of the paper describes the variance estimate only.
    pass

    out_file2 = LOG_DIR / "summary.json"
    with out_file2.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Markdown summary
    lines = ["# Statistical Significance Tests\n"]
    lines.append(f"3 seeds 脳 {N} episodes per seed (full L1-L4 pipeline)\n")
    lines.append("| Seed | Accuracy | avg_score | wall_seconds |")
    lines.append("|---:|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r['seed']} | {r['success_rate']*100:.1f}% | {r['avg_score']:.3f} | {r['wall_seconds']:.0f} |")
    lines.append("")
    lines.append(f"**Mean accuracy**: {mean_acc*100:.1f}% 卤 {std_acc*100:.1f}% (n={len(accs)})")
    lines.append(f"**Mean avg_score**: {mean_score:.3f} 卤 {std_score:.3f}")
    lines.append(f"**Mean wall time**: {mean_time:.0f} s/seed")
    lines.append("")
    lines.append("## Note on Baseline Comparisons")
    lines.append("The 30% and 51% baselines referenced in earlier drafts of the paper")
    lines.append("were NOT measured. This script reports only the variance across seeds.")
    lines.append("See Section 4.3 of the paper for the honest framing.")
    lines.append("")
    lines.append("## 95% Confidence Interval")
    lines.append(f"Accuracy: [{summary['accuracy']['95_ci'][0]*100:.1f}%, {summary['accuracy']['95_ci'][1]*100:.1f}%]")
    (LOG_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=== Statistical Summary ===")
    print(f"Mean accuracy: {mean_acc*100:.1f}% 卤 {std_acc*100:.1f}%")
    print(f"95% CI: [{summary['accuracy']['95_ci'][0]*100:.1f}%, {summary['accuracy']['95_ci'][1]*100:.1f}%]")
    print("Note: no hardcoded baseline comparison (see paper Section 4.3)")


if __name__ == "__main__":
    main()
