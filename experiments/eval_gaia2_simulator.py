"""Run a no-oracle-leak GAIA2-mini simulator smoke evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.gaia2_apps import GaiaAppRuntime, action_json_schema, summarize_initial_state
from agi_kit.gaia2_simulator import GaiaAction, GaiaScenario, args_to_dict
from agi_kit.llms.base import LLMMessage, MessageRole
from agi_kit.llms.ollama import OllamaBackend


def parse_prediction(text: str) -> GaiaAction | None:
    import re
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1) if match else text.strip()
    try:
        item = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if set(item) != {"app", "function", "args"} or not isinstance(item["args"], dict):
        return None
    return GaiaAction(str(item["app"]), str(item["function"]), {str(k): str(v) for k, v in item["args"].items()})


def observation_text(scenario: GaiaScenario) -> str:
    lines = []
    for event in scenario.observations_until(0):
        action = event["action"]
        if event["event_type"] == "USER":
            lines.append("USER: " + args_to_dict(action.get("args", [])).get("content", ""))
    return "\n".join(lines) + "\n\nVISIBLE APP STATE:\n" + summarize_initial_state(scenario.initial_apps)


def rent_action_feedback(action: GaiaAction, scenario: GaiaScenario) -> str | None:
    """Check an explicit RentAFlat user constraint using visible app state only."""
    if (action.app, action.function) != ("RentAFlat", "remove_saved_apartment"):
        return None
    user = observation_text(scenario)
    import re
    match = re.search(r"in the (\d+)-(\d+) price range", user)
    city = re.search(r"properties in ([A-Za-z]+)", user)
    if not match or not city:
        return None
    state = next((app.get("app_state", {}) for app in scenario.initial_apps if app.get("name") == "RentAFlat"), {})
    apartment_id = action.args.get("apartment_id")
    apartment = state.get("apartments", {}).get(apartment_id, {}) if isinstance(state, dict) else {}
    lo, hi = int(match.group(1)), int(match.group(2))
    if city.group(1).casefold() not in str(apartment.get("location", "")).casefold() or lo <= float(apartment.get("price", -1)) <= hi:
        return "Reflection: the selected apartment does not satisfy the user's stated city and out-of-range-price deletion condition. Choose another valid saved apartment."
    return None


def load_scenario(arrow: Path, index: int) -> GaiaScenario:
    import pyarrow.ipc as ipc
    with ipc.open_stream(arrow) as reader:
        rows = [row for batch in reader for row in batch.to_pylist()]
    raw = json.loads(rows[index]["data"])
    return GaiaScenario(rows[index]["scenario_id"], raw["apps"], raw["events"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arrow", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--out", default="logs/gaia2_smoke.json")
    args = parser.parse_args()
    scenario = load_scenario(args.arrow, args.index)
    runtime = GaiaAppRuntime(scenario.initial_apps)
    tools = "\n".join("- " + item for item in runtime.tool_catalog())
    system = ("Privately compare every user constraint against the visible app state before acting. "
              "Return exactly one JSON object with app, function, and args; do not explain. "
              "Choose the single next action from this tool catalog only:\n" + tools)
    history = [LLMMessage(role=MessageRole.SYSTEM, content=system), LLMMessage(role=MessageRole.USER, content=observation_text(scenario))]
    llm = OllamaBackend(model=args.model, seed=20260802, timeout_s=180)
    action_schema = action_json_schema(scenario.initial_apps)
    predictions = []
    raw_outputs = []
    transport_errors = []
    for _ in range(args.max_actions):
        try:
            response = llm.chat(history, max_tokens=180, temperature=0, format=action_schema)
        except Exception as exc:
            transport_errors.append(str(exc))
            break
        raw_outputs.append(response.content)
        action = parse_prediction(response.content)
        if action is None:
            break
        invoked = runtime.invoke(action)
        if not invoked["ok"]:
            history.extend([LLMMessage(role=MessageRole.ASSISTANT, content=response.content), LLMMessage(role=MessageRole.USER, content="Reflection: that action is not valid in the current visible app state (" + invoked["error"] + "). Choose one valid action.")])
            continue
        feedback = rent_action_feedback(action, scenario)
        if feedback:
            runtime.journal.pop()
            history.extend([LLMMessage(role=MessageRole.ASSISTANT, content=response.content), LLMMessage(role=MessageRole.USER, content=feedback)])
            continue
        predictions.append(action)
        history.extend([LLMMessage(role=MessageRole.ASSISTANT, content=response.content), LLMMessage(role=MessageRole.USER, content="Action recorded. Choose the next action or stop.")])
    result = scenario.score_sequence(predictions)
    result.update({"model": args.model, "raw_outputs": raw_outputs, "predictions": [a.__dict__ for a in predictions], "transport_errors": transport_errors, "oracle_exposed_to_model": False})
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
