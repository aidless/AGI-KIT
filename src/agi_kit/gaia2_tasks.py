"""GAIA2-style task source.

Generates multi-step tasks that mimic GAIA2's profile:
  - Multi-step reasoning required
  - Cross-tool usage (calculator + echo + file + shell)
  - Numerical + string answers

If a real GAIA2 dataset is locally cached, can be loaded instead.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).parent.parent.parent


def _try_load_real_gaia2(split="validation", limit=20):
    """Try to load real GAIA2 from local cache or HF."""
    cache_paths = [
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub",
        Path("F:/hf_cache"),
    ]
    for base in cache_paths:
        if not base.exists():
            continue
        for d in base.glob("datasets--*gaia*"):
            snapshot = d / "snapshots"
            if not snapshot.exists():
                continue
            # Look for parquet/jsonl inside snapshots
            for ext in ("*.parquet", "*.jsonl"):
                for f in snapshot.rglob(ext):
                    try:
                        if f.suffix == ".parquet":
                            import pyarrow.parquet as pq
                            tbl = pq.read_table(f)
                            df = tbl.to_pandas()
                            return df.head(limit).to_dict("records")
                        else:
                            rows = []
                            with open(f, encoding="utf-8") as fh:
                                for line in fh:
                                    rows.append(json.loads(line))
                                    if len(rows) >= limit:
                                        break
                            return rows
                    except Exception:
                        continue
    return None


def synth_gaia2_tasks(n=15, seed=42):
    """Build GAIA2-style multi-step tasks.

    Each task forces 2-4 tool calls and a final answer.
    """
    rng = random.Random(seed)
    tasks = []
    categories = ["arith_chain", "file_calc", "shell_read", "double_lookup", "word_count"]
    for i in range(n):
        cat = categories[i % len(categories)]
        if cat == "arith_chain":
            a = rng.randint(10, 99)
            b = rng.randint(2, 9)
            c = rng.randint(50, 200)
            expr = str(a) + "*" + str(b) + "+" + str(c)
            ans = a * b + c
            tasks.append({
                "task": ("Use calculator to compute " + expr +
                         ", then echo the result, then give the final answer."),
                "kind": "gaia2_arith_chain",
                "gold": str(ans),
                "category": cat,
            })
        elif cat == "file_calc":
            # Read a real file, count chars, multiply
            path = "README.md"
            target = 3
            tasks.append({
                "task": ("Read the file " + path + " (limit 500 chars), then count the "
                         "number of lines, then multiply by " + str(target) +
                         " using calculator, then final."),
                "kind": "gaia2_file_calc",
                "gold": None,  # dynamic
                "category": cat,
            })
        elif cat == "shell_read":
            tasks.append({
                "task": ("Run shell command 'echo HELLO_WORLD', then calculator "
                         "to multiply the length of the output by 7, then final."),
                "kind": "gaia2_shell",
                "gold": str(len("HELLO_WORLD") * 7),
                "category": cat,
            })
        elif cat == "double_lookup":
            a = rng.randint(100, 999)
            b = rng.randint(100, 999)
            tasks.append({
                "task": ("Compute " + str(a) + "+" + str(b) + " with calculator, then echo it, "
                         "then also compute " + str(a) + "*" + str(b) + ", then final should "
                         "include both."),
                "kind": "gaia2_double",
                "gold": str(a + b) + " and " + str(a * b),
                "category": cat,
            })
        else:  # word_count
            tasks.append({
                "task": ("Use calculator to compute 50*40-100, then echo it, then "
                         "state the number of digits in the result, then final."),
                "kind": "gaia2_word_count",
                "gold": "4",
                "category": cat,
            })
    return tasks


def load_gaia2_tasks(n=15, prefer_real=True, seed=42) -> List[dict]:
    """Try real GAIA2 first, fall back to synthetic."""
    if prefer_real:
        real = _try_load_real_gaia2(limit=n)
        if real:
            # Normalize field names
            out = []
            for r in real:
                q = r.get("question") or r.get("query") or r.get("task") or ""
                g = r.get("answer") or r.get("final_answer") or r.get("gold") or None
                if q:
                    out.append({"task": q, "kind": "gaia2_real", "gold": g,
                                "category": r.get("category", "real")})
            if out:
                return out[:n]
    return synth_gaia2_tasks(n=n, seed=seed)


if __name__ == "__main__":
    tasks = load_gaia2_tasks(n=10, prefer_real=False)
    for t in tasks:
        print("[" + t["kind"] + "] " + t["task"][:80] +
              " ... gold=" + str(t.get("gold")))