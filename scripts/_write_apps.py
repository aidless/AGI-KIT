from pathlib import Path

PKG_INIT = """GAIA2 app universe bridge.

Minimal Python implementations of three GAIA2 apps (Calendar, Emails,
Shopping) that expose the dataset's app.function(args) API backed by
in-memory state. Together with agents implementing tools with these
names, they allow partial canonical evaluation on the GAIA2-mini
validation set (covers ~51% of expected calls).

The remaining 7 apps (RentAFlat, Chats, Messages, Cabs, Contacts,
Files, AgentUserInterface) are documented as future work in the
preprint Limitations section.
"""

GAIA2_INIT = """GAIA2 apps: per-scenario state plus dispatch.

`call(app, function, args, scenario_id)` dispatches and persists.
"""
from pathlib import Path
import json

RUN_ROOT = Path("logs/gaia2_runs")
RUN_ROOT.mkdir(parents=True, exist_ok=True)


def _state_path(scenario_id, app):
    d = RUN_ROOT / scenario_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{app}.state.json"


def load_state(scenario_id, app):
    p = _state_path(scenario_id, app)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_state(scenario_id, app, state):
    p = _state_path(scenario_id, app)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
"""

CALENDAR = """Calendar app - GAIA2 app simulation.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent.parent))
from . import load_state, save_state


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
"""

EMAILS = """Emails app - GAIA2 app simulation.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent.parent))
from . import load_state, save_state


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
"""

SHOPPING = """Shopping app - GAIA2 app simulation.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent.parent.parent))
from . import load_state, save_state


_CATALOG = [
    {"product_id": "p_001", "name": "Notebook", "price": 4.99},
    {"product_id": "p_002", "name": "Pen set", "price": 12.50},
    {"product_id": "p_003", "name": "USB-C cable", "price": 9.75},
    {"product_id": "p_004", "name": "Coffee mug", "price": 8.00},
    {"product_id": "p_005", "name": "Wireless mouse", "price": 24.99},
]


def get_product_information(scenario_id, args):
    pid = args.get("product_id", "")
    for p in _CATALOG:
        if p["product_id"] == pid:
            return {"product": p}
    return {"product": None}


def display_all_products(scenario_id, args):
    return {"products": _CATALOG}


def purchase_item(scenario_id, args):
    state = load_state(scenario_id, "shopping")
    pid = args.get("product_id", "")
    for p in _CATALOG:
        if p["product_id"] == pid:
            state.setdefault("purchases", []).append({"product": p, "qty": args.get("quantity", 1)})
            save_state(scenario_id, "shopping", state)
            return {"purchased": p, "qty": args.get("quantity", 1)}
    return {"error": "no such product"}


def get_purchase_history(scenario_id, args):
    state = load_state(scenario_id, "shopping")
    return {"history": state.get("purchases", [])}


FUNCTIONS = {
    "get_product_information": get_product_information,
    "display_all_products": display_all_products,
    "purchase_item": purchase_item,
    "get_purchase_history": get_purchase_history,
}


def call(scenario_id, function, args):
    if function not in FUNCTIONS:
        return {"error": "unknown function"}
    return FUNCTIONS[function](scenario_id, args)
"""

# Write all files
files = [
    ("src/agi_kit/apps/__init__.py", PKG_INIT),
    ("src/agi_kit/apps/gaia2/__init__.py", GAIA2_INIT),
    ("src/agi_kit/apps/gaia2/calendar.py", CALENDAR),
    ("src/agi_kit/apps/gaia2/emails.py", EMAILS),
    ("src/agi_kit/apps/gaia2/shopping.py", SHOPPING),
]
for path, content in files:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print("Wrote", path)
