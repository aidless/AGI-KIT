# Calendar app - GAIA2 app simulation.
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent.parent))
from agi_kit.apps.gaia2 import load_state, save_state


def list_events(scenario_id, args):
    state = load_state(scenario_id, "calendar")
    return {"events": state.get("events", [])}


def get_current_date(scenario_id, args):
    state = load_state(scenario_id, "calendar")
    return {"current_date": state.get("current_date", "2025-01-01")}


def create_event(scenario_id, args):
    state = load_state(scenario_id, "calendar")
    ev = {
        "title": args.get("title", "untitled"),
        "start": args.get("start", ""),
        "end": args.get("end", ""),
        "attendees": args.get("attendees", []),
    }
    state.setdefault("events", []).append(ev)
    save_state(scenario_id, "calendar", state)
    return {"created": ev, "id": len(state["events"])}


def delete_event(scenario_id, args):
    state = load_state(scenario_id, "calendar")
    eid = args.get("event_id")
    events = state.get("events", [])
    if eid is not None and 0 <= int(eid) < len(events):
        removed = events.pop(int(eid))
        save_state(scenario_id, "calendar", state)
        return {"deleted": removed}
    return {"deleted": None, "reason": "no such event"}


FUNCTIONS = {
    "list_events": list_events,
    "get_current_date": get_current_date,
    "create_event": create_event,
    "delete_event": delete_event,
}


def call(scenario_id, function, args):
    if function not in FUNCTIONS:
        return {"error": "unknown function"}
    return FUNCTIONS[function](scenario_id, args)

