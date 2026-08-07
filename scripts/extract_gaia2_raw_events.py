"""Extract reproducible GAIA2 mini scenarios including initial state and events.

The compact validation.jsonl contains only oracle actions.  This tool reads
the original Arrow cache and writes the complete scenario JSON needed by a
stateful simulator; it never treats oracle actions as agent predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arrow", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.ipc as ipc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with ipc.open_stream(args.arrow) as reader, args.out.open("w", encoding="utf-8") as out:
        for batch in reader:
            for row in batch.to_pylist():
                scenario = json.loads(row["data"])
                record = {
                    "id": row["id"],
                    "scenario_id": row["scenario_id"],
                    "category": row["category"],
                    "initial_apps": scenario["apps"],
                    "events": scenario["events"],
                    "version": scenario.get("version"),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print("extracted", count, "full scenarios to", args.out)


if __name__ == "__main__":
    main()
