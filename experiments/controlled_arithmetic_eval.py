"""Paired, prompt-matched L1 reflection evaluation.

Static and L1 runs receive identical task text, system prompt, calculator,
step budget, model and generation seed.  The L1 arm alone receives a retry
message when its final conflicts with the most recent calculator observation.
No gold answer is exposed to either arm during execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from agi_kit.evaluation_protocol import assess_answer, extract_final, numeric_equal
from agi_kit.llms.base import LLMMessage, MessageRole
from agi_kit.llms.ollama import OllamaBackend


SYSTEM_PROMPT = """You are a tool-use assistant. You have one tool:
- calculator(expr: string): evaluates an arithmetic expression.
Respond only with one fenced JSON object. For a tool call use
{\"tool\":\"calculator\",\"args\":{\"expr\":\"...\"}}. After a tool observation,
return {\"final\":\"<answer>\"}."""


def parse_action(text: str) -> dict[str, Any] | None:
    import re
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        candidate = (text or "").strip()
    else:
        candidate = match.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def calculate(expr: str) -> str:
    from full_agent import calculator
    return str(calculator(expr))


def run_task(llm: OllamaBackend, task: dict[str, str], config: str, max_steps: int) -> dict[str, Any]:
    history = [
        LLMMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        LLMMessage(role=MessageRole.USER, content=task["prompt"]),
    ]
    trace: list[dict[str, Any]] = []
    latest_numeric_observation: str | None = None
    raw_final = ""
    correction_used = False
    for step in range(1, max_steps + 1):
        try:
            response = llm.chat(history, max_tokens=160, temperature=0.0)
        except Exception as exc:
            trace.append({"step": step, "transport_error": str(exc)})
            return {
                "task_id": task["id"], "split": task["split"], "gold": task["gold"],
                "assessment": assess_answer("", task["gold"]).to_dict(),
                "steps": len(trace), "reflection_retry_used": correction_used,
                "trace": trace, "run_error": "llm_transport_error",
            }
        raw = response.content
        history.append(LLMMessage(role=MessageRole.ASSISTANT, content=raw))
        action = parse_action(raw)
        event: dict[str, Any] = {"step": step, "raw": raw, "action": action}
        if action is None:
            trace.append(event)
            history.append(LLMMessage(role=MessageRole.USER, content="Return one valid fenced JSON object."))
            continue
        if set(action) == {"final"}:
            raw_final = raw
            final, _ = extract_final(raw, allow_unfenced=True)
            if config == "l1_reflection" and not correction_used and final is not None and latest_numeric_observation is not None and not numeric_equal(final, latest_numeric_observation):
                correction_used = True
                event["reflection"] = "final_conflicted_with_tool_observation"
                trace.append(event)
                history.append(LLMMessage(
                    role=MessageRole.USER,
                    content=("Reflection: your final answer conflicts with the calculator observation "
                             + latest_numeric_observation + ". Check the observation and return one corrected JSON final."),
                ))
                continue
            trace.append(event)
            break
        if set(action) == {"tool", "args"} and action.get("tool") == "calculator" and isinstance(action.get("args"), dict):
            expr = action["args"].get("expr")
            observation = calculate(str(expr)) if isinstance(expr, str) else "err: invalid expr"
            event["observation"] = observation
            if not observation.startswith("err:"):
                latest_numeric_observation = observation
            history.append(LLMMessage(role=MessageRole.USER, content="Observation: " + observation))
        else:
            event["observation"] = "err: unsupported action"
            history.append(LLMMessage(role=MessageRole.USER, content="Use calculator or return a final JSON object."))
        trace.append(event)
    assessment = assess_answer(raw_final, task["gold"], allow_unfenced=True)
    strict_assessment = assess_answer(raw_final, task["gold"])
    return {
        "task_id": task["id"],
        "split": task["split"],
        "gold": task["gold"],
        "assessment": assessment.to_dict(),
        "strict_assessment": strict_assessment.to_dict(),
        "steps": len(trace),
        "reflection_retry_used": correction_used,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--limit", type=int, default=0, help="nonzero for a smoke test only")
    parser.add_argument("--out", default="", help="fresh run directory, default uses a UTC timestamp")
    args = parser.parse_args()

    manifest_path = ROOT / "data" / "controlled_arithmetic_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [x for x in manifest["tasks"] if x["split"] == args.split]
    if args.limit:
        tasks = tasks[:args.limit]
    run_name = args.out or ("logs/controlled_arithmetic/" + datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"))
    out_dir = ROOT / run_name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit("refusing to overwrite nonempty run directory: " + str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    all_rows: list[dict[str, Any]] = []
    checkpoint_path = out_dir / "per_task.jsonl"
    run_meta = {
        "run_name": run_name, "model": args.model, "seed": args.seed,
        "max_steps": args.max_steps, "timeout_s": args.timeout_s, "split": args.split,
        "n_tasks": len(tasks), "manifest_sha256": manifest_hash,
        "system_prompt_sha256": prompt_hash,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    for config in ("static", "l1_reflection"):
        for index, task in enumerate(tasks):
            llm = OllamaBackend(model=args.model, seed=args.seed + index, timeout_s=args.timeout_s)
            started = time.perf_counter()
            row = run_task(llm, task, config, args.max_steps)
            row.update({"config": config, "seed": args.seed + index, "seconds": round(time.perf_counter() - started, 3)})
            all_rows.append(row)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = "OK" if row["assessment"]["normalized_correct"] else "X"
            print(config, task["id"], status, "steps=" + str(row["steps"]))
    summary: dict[str, Any] = {**run_meta, "configs": {}}
    for config in ("static", "l1_reflection"):
        rows = [x for x in all_rows if x["config"] == config]
        summary["configs"][config] = {
            "correct": sum(x["assessment"]["normalized_correct"] for x in rows),
            "normalized_transport_completed": sum(x["assessment"]["structural_completion"] for x in rows),
            "strict_fenced_completed": sum(x["strict_assessment"]["structural_completion"] for x in rows),
            "mean_steps": round(sum(x["steps"] for x in rows) / max(1, len(rows)), 3),
            "reflection_retries": sum(x["reflection_retry_used"] for x in rows),
        }
    static = {x["task_id"]: x for x in all_rows if x["config"] == "static"}
    l1 = {x["task_id"]: x for x in all_rows if x["config"] == "l1_reflection"}
    win = sum(not static[k]["assessment"]["normalized_correct"] and l1[k]["assessment"]["normalized_correct"] for k in static)
    loss = sum(static[k]["assessment"]["normalized_correct"] and not l1[k]["assessment"]["normalized_correct"] for k in static)
    summary["paired_comparison"] = {"l1_wins": win, "l1_losses": loss, "discordant": win + loss, "note": "Exact paired inference requires the full held-out run; smoke runs are descriptive only."}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
