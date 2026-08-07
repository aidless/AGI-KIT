"""Development-only routed GAIA2 action probe.

This is not an official GAIA2 evaluation.  It exposes only selected visible
app state and a narrowed action schema, then scores after generation against
the oracle sequence for diagnostics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import re
from pathlib import Path

import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.gaia2_apps import GaiaAppRuntime, action_json_schema, summarize_initial_state
from agi_kit.gaia2_simulator import GaiaAction, GaiaScenario, args_to_dict
from agi_kit.llms.base import LLMMessage, MessageRole
from agi_kit.llms.ollama import OllamaBackend


KEYWORDS = {
    "RentAFlat": ("apartment", "property", "rent", "flat"),
    "Messages": ("message", "text"), "Chats": ("chat", "conversation"),
    "Calendar": ("calendar", "event", "meeting"), "Emails": ("email",),
    "Shopping": ("cart", "order", "buy", "purchase"), "Cabs": ("cab", "ride"),
    "Contacts": ("contact",), "Files": ("file", "folder"),
}


def parse_action(text: str) -> GaiaAction | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return GaiaAction(str(data["app"]), str(data["function"]), {str(k): str(v) for k, v in data["args"].items()})
    except (KeyError, TypeError, json.JSONDecodeError):
        return None


def constrain_rent_state(apps: list[dict], user_text: str) -> list[dict]:
    """Expose only user-eligible deletion IDs; derived solely from visible state."""
    constrained = deepcopy(apps)
    match = re.search(r"in the (\d+)-(\d+) price range", user_text, re.I)
    city = re.search(r"properties in ([A-Za-z]+)", user_text, re.I)
    if not match or not city:
        return constrained
    low, high = map(float, match.groups())
    city_name = city.group(1).casefold()
    for app in constrained:
        if app["name"] != "RentAFlat":
            continue
        state = app.get("app_state", {})
        apartments = state.get("apartments", {})
        state["_eligible_remove"] = [
            apartment_id for apartment_id in state.get("saved_apartments", [])
            if (details := apartments.get(apartment_id, {}))
            and city_name in str(details.get("location", "")).casefold()
            and not low <= float(details.get("price", -1)) <= high
        ]
        state["_eligible_save"] = [
            apartment_id for apartment_id, details in apartments.items()
            if apartment_id not in state.get("saved_apartments", [])
            and city_name in str(details.get("location", "")).casefold()
            and low <= float(details.get("price", -1)) <= high
        ]
    friend = re.search(r"friend ([A-Z][A-Za-z]+ [A-Z][A-Za-z]+)", user_text)
    if friend:
        target = friend.group(1).casefold()
        for app in constrained:
            if app["name"] == "Messages":
                state = app.get("app_state", {})
                names = state.get("id_to_name", {})
                reverse = state.get("name_to_id", {})
                ids = [uid for uid, name in names.items() if str(name).casefold() == target]
                ids.extend([uid for name, uid in reverse.items() if str(name).casefold() == target])
                state["_eligible_user_ids"] = list(dict.fromkeys(str(uid) for uid in ids))
    return constrained


def phase_schema(schema: dict, apps: list[dict], completed_saves: set[str], initial_save_ids: set[str], dynamic_events: int) -> tuple[str, dict]:
    rent = next((app.get("app_state", {}) for app in apps if app["name"] == "RentAFlat"), {})
    if rent.get("_eligible_remove"):
        phase, allowed = "remove", {("RentAFlat", "remove_saved_apartment")}
    elif initial_save_ids - completed_saves:
        phase, allowed = "save", {("RentAFlat", "save_apartment")}
    else:
        phase, allowed = "notify", {("Messages", "send_message")}
    branches = [branch for branch in schema["oneOf"] if (branch["properties"]["app"]["const"], branch["properties"]["function"]["const"]) in allowed]
    return phase, {"oneOf": branches or schema["oneOf"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arrow", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--max-actions", type=int, default=4)
    args = parser.parse_args()
    rows = [row for batch in ipc.open_stream(args.arrow) for row in batch.to_pylist()]
    raw = json.loads(rows[args.index]["data"])
    scenario = GaiaScenario(rows[args.index]["scenario_id"], raw["apps"], raw["events"])
    user_text = "\n".join(args_to_dict(event["action"].get("args", [])).get("content", "") for event in scenario.observations_until(0) if event["event_type"] == "USER")
    lower = user_text.casefold()
    selected_names = {"AgentUserInterface"}
    selected_names.update(app for app, words in KEYWORDS.items() if any(word in lower for word in words))
    if "friend" in lower:
        selected_names.update({"Messages", "Contacts"})
    selected = [app for app in raw["apps"] if app["name"] in selected_names]
    llm = OllamaBackend(model=args.model, seed=20260802, timeout_s=120)
    runtime = GaiaAppRuntime(selected)
    initial_schema_apps = constrain_rent_state([{"name": name, "app_state": state} for name, state in runtime.state.items()], user_text)
    initial_rent = next((app["app_state"] for app in initial_schema_apps if app["name"] == "RentAFlat"), {})
    initial_save_ids = set(initial_rent.get("_eligible_save", []))
    predictions, raw_outputs, invocation_results, phases = [], [], [], []
    applied_events: set[str] = set()
    for step in range(args.max_actions):
        relative_time = (step + 1) * 60
        for event in scenario.ready_observations(relative_time, applied_events):
            if event["event_type"] == "ENV" and event.get("action", {}).get("app") in runtime.state:
                runtime.apply_environment_event(event)
            applied_events.add(str(event.get("event_id", "")))
        ready_turns = scenario.ready_agent_event_ids(relative_time, applied_events)
        if not ready_turns:
            continue
        live_apps = [{"name": name, "app_state": state} for name, state in runtime.state.items()]
        schema_apps = constrain_rent_state(live_apps, user_text)
        completed_saves = {action.args.get("apartment_id", "") for action in predictions if action.app == "RentAFlat" and action.function == "save_apartment"}
        completed_replies = {action.args.get("email_id", "") for action in predictions if action.app == "Emails" and action.function == "reply_to_email"}
        for app in schema_apps:
            if app["name"] == "Emails":
                folders = app.get("app_state", {}).get("folders", {})
                app["app_state"]["_eligible_email_ids"] = [
                    str(email.get("email_id", email.get("id")))
                    for folder in folders.values() if isinstance(folder, dict)
                    for email in folder.get("emails", []) if isinstance(email, dict)
                    and str(email.get("email_id", email.get("id"))) not in completed_replies
                ]
        for app in schema_apps:
            if app["name"] == "RentAFlat":
                app["app_state"]["_eligible_save"] = sorted(initial_save_ids - completed_saves)
        phase, schema = phase_schema(action_json_schema(schema_apps), schema_apps, completed_saves, initial_save_ids, len(runtime.environment_events))
        phases.append(phase)
        prompt = ("Return exactly one JSON object with app, function, and args. Do not explain. "
                  "Complete the current phase before any later phase. Current phase: " + phase + ". "
                  "Choose the next valid action; do not repeat an action already completed. Completed actions: " +
                  json.dumps([action.__dict__ for action in predictions], separators=(",", ":")) + "\nUSER:\n" + user_text +
                  "\nVISIBLE STATE:\n" + summarize_initial_state(live_apps, max_chars=3500, per_app_chars=700))
        response = llm.chat([LLMMessage(role=MessageRole.USER, content=prompt)], max_tokens=160, temperature=0, format=schema)
        raw_outputs.append(response.content)
        action = parse_action(response.content)
        invoked = runtime.invoke(action) if action is not None else {"ok": False, "error": "parse_failure"}
        invocation_results.append(invoked)
        if action is None or not invoked.get("ok"):
            break
        predictions.append(action)
        completed_turn = ready_turns[0]
        runtime.bind_agent_event(completed_turn)
        applied_events.add(completed_turn)
    result = scenario.score_sequence(predictions, runtime.reference_bindings)
    result.update({"development_only": True, "oracle_exposed_to_model": False, "selected_apps": [app["name"] for app in selected], "user_text": user_text, "phases": phases, "raw_outputs": raw_outputs, "invocation_results": invocation_results, "environment_events_applied": len(runtime.environment_events), "predictions": [action.__dict__ for action in predictions]})
    out = ROOT / "logs" / "gaia2_routed_dev" / f"scenario_{args.index}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
