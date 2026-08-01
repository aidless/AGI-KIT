"""reviewer_both.py - run heuristic + real reviewer and show side-by-side.

Usage:
  python scripts/reviewer_both.py

Output:
  papers/reviews/heuristic_summary.txt  (existing run, untouched file)
  papers/reviews/real_reviewer_report.txt  (already produced by real_reviewer.py)

This script simply runs both and prints a side-by-side comparison.
It does not modify either output beyond running them.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
REVIEWS = ROOT / "papers" / "reviews"


def run_script(path: Path) -> int:
    print("\n" + "=" * 72)
    print("RUN: " + str(path.relative_to(ROOT)))
    print("=" * 72)
    p = subprocess.run([str(PYTHON), str(path)], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    print(p.stdout[-2000:] if len(p.stdout) > 2000 else p.stdout)
    if p.returncode != 0:
        print("STDERR:", p.stderr[-1000:])
    return p.returncode


def main():
    rc1 = run_script(ROOT / "scripts" / "reviewer_simulator.py")
    print()
    rc2 = run_script(ROOT / "scripts" / "real_reviewer.py")

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    s1 = (REVIEWS / "summary.txt").read_text(encoding="utf-8", errors="replace")
    s2 = (REVIEWS / "real_reviewer_score.txt").read_text(encoding="utf-8", errors="replace")
    print("Heuristic reviewer: see papers/reviews/summary.txt")
    print("Real reviewer score: " + s2.strip() + " / 5.0")
    print("\nHistoric heuristic 3.50 + new real_reviewer " + s2.strip() +
          " / 5.0 form the dual-scoring framework Round 11 introduces.")
    sys.exit(rc1 + rc2)


if __name__ == "__main__":
    main()

