# Messages app - GAIA2 app simulation.
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent.parent))
from agi_kit.apps.gaia2 import load_state, save_state


def list_messages(scenario_id, args):
    state = load_state(scenario_id, "messages")
    return {"messages": state.get("messages", [])}


def get_message(scenario_id, args):
    state = load_state(scenario_id, "messages")
    mid = args.get("message_id")
    messages = state.get("messages", [])
    if mid is not None and 0 <= int(mid) < len(messages):
        return {"message": messages[int(mid)]}
    return {"message": None}


def send_message(scenario_id, args):
    state = load_state(scenario_id, "messages")
    msg = {
        "to": args.get("to", ""),
        "body": args.get("body", ""),
    }
    state.setdefault("messages", []).append(msg)
    save_state(scenario_id, "messages", state)
    return {"sent": msg, "id": len(state["messages"]) - 1}


def delete_message(scenario_id, args):
    state = load_state(scenario_id, "messages")
    mid = args.get("message_id")
    messages = state.get("messages", [])
    if mid is not None and 0 <= int(mid) < len(messages):
        removed = messages.pop(int(mid))
        save_state(scenario_id, "messages", state)
        return {"deleted": removed}
    return {"deleted": None}


FUNCTIONS = {
    "list_messages": list_messages,
    "get_message": get_message,
    "send_message": send_message,
    "delete_message": delete_message,
}


def call(scenario_id, function, args):
    if function not in FUNCTIONS:
        return {"error": "unknown function"}
    return FUNCTIONS[function](scenario_id, args)

