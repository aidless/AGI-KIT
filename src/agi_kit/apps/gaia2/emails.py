# Emails app - GAIA2 app simulation.
from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent.parent))
from agi_kit.apps.gaia2 import load_state, save_state


def list_emails(scenario_id, args):
    state = load_state(scenario_id, "emails")
    return {"emails": state.get("emails", [])}


def get_email(scenario_id, args):
    state = load_state(scenario_id, "emails")
    eid = args.get("email_id")
    emails = state.get("emails", [])
    if eid is not None and 0 <= int(eid) < len(emails):
        return {"email": emails[int(eid)]}
    return {"email": None}


def send_email(scenario_id, args):
    state = load_state(scenario_id, "emails")
    email = {
        "to": args.get("to", ""),
        "subject": args.get("subject", ""),
        "body": args.get("body", ""),
    }
    state.setdefault("emails", []).append(email)
    save_state(scenario_id, "emails", state)
    return {"sent": email, "id": len(state["emails"]) - 1}


def delete_email(scenario_id, args):
    state = load_state(scenario_id, "emails")
    eid = args.get("email_id")
    emails = state.get("emails", [])
    if eid is not None and 0 <= int(eid) < len(emails):
        removed = emails.pop(int(eid))
        save_state(scenario_id, "emails", state)
        return {"deleted": removed}
    return {"deleted": None}


FUNCTIONS = {
    "list_emails": list_emails,
    "get_email": get_email,
    "send_email": send_email,
    "delete_email": delete_email,
}


def call(scenario_id, function, args):
    if function not in FUNCTIONS:
        return {"error": "unknown function"}
    return FUNCTIONS[function](scenario_id, args)

