"""Whitelisted stateful adapter for the GAIA2-mini app action space.

The adapter models the benchmark's observable side effects as per-app action
journals. It intentionally accepts only the 29 action names observed in the
cached mini validation split; arbitrary method dispatch is not permitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
import json
from typing import Any

from agi_kit.gaia2_simulator import GaiaAction


SUPPORTED: dict[str, set[str]] = {
    "AgentUserInterface": {"send_message_to_user"},
    "Calendar": {"add_calendar_event", "delete_calendar_event"},
    "Emails": {"send_email", "reply_to_email", "delete_email", "move_email", "forward_email"},
    "Shopping": {"add_to_cart", "checkout", "cancel_order", "remove_from_cart"},
    "Messages": {"send_message", "send_message_to_group_conversation", "create_group_conversation"},
    "RentAFlat": {"save_apartment", "remove_saved_apartment"},
    "Cabs": {"order_ride", "user_cancel_ride"},
    "Chats": {"send_message", "send_message_to_group_conversation", "create_group_conversation", "remove_participant_from_conversation"},
    "Contacts": {"delete_contact", "edit_contact", "add_new_contact"},
    "Files": {"rm", "mv", "mkdir"},
}

# Explicit contracts added only after inspecting the app API, never from the
# target scenario's oracle call. Remaining operations stay permissive until
# their contracts are independently recovered.
REQUIRED_ARGUMENTS: dict[tuple[str, str], set[str]] = {
    ("AgentUserInterface", "send_message_to_user"): {"content"},
    ("Calendar", "add_calendar_event"): {"title", "start_datetime", "end_datetime"},
    ("Calendar", "delete_calendar_event"): {"event_id"},
    ("Emails", "send_email"): {"recipients", "subject", "content"},
    ("Emails", "reply_to_email"): {"email_id", "folder_name", "content"},
    ("Emails", "delete_email"): {"email_id", "folder_name"},
    ("Emails", "move_email"): {"email_id", "source_folder_name", "dest_folder_name"},
    ("Emails", "forward_email"): {"email_id", "folder_name", "recipients"},
    ("Messages", "send_message"): {"user_id", "content"},
    ("Messages", "send_message_to_group_conversation"): {"conversation_id", "content"},
    ("Messages", "create_group_conversation"): {"title", "user_ids"},
    ("RentAFlat", "save_apartment"): {"apartment_id"},
    ("RentAFlat", "remove_saved_apartment"): {"apartment_id"},
    ("Shopping", "add_to_cart"): {"item_id", "quantity"},
    ("Shopping", "remove_from_cart"): {"item_id", "quantity"},
    ("Shopping", "cancel_order"): {"order_id"},
    ("Cabs", "order_ride"): {"start_location", "end_location", "ride_time", "service_type"},
    ("Contacts", "delete_contact"): {"contact_id"},
    ("Contacts", "edit_contact"): {"contact_id", "updates"},
    ("Files", "rm"): {"path", "recursive"},
    ("Files", "mv"): {"path1", "path2"},
    ("Files", "mkdir"): {"path", "create_parents"},
}


def action_json_schema(initial_apps: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON Schema exposing only APIs available in this scenario."""
    available = {str(app["name"]) for app in initial_apps}
    choices = []
    for app in sorted(available & set(SUPPORTED)):
        for function in sorted(SUPPORTED[app]):
            required = REQUIRED_ARGUMENTS.get((app, function), set())
            arg_properties = {name: {"type": "string"} for name in required}
            if (app, function) == ("RentAFlat", "remove_saved_apartment"):
                state = next((item.get("app_state", {}) for item in initial_apps if item.get("name") == "RentAFlat"), {})
                saved = state.get("_eligible_remove", state.get("saved_apartments", [])) if isinstance(state, dict) else []
                if saved:
                    arg_properties["apartment_id"] = {"type": "string", "enum": [str(value) for value in saved]}
            if (app, function) == ("RentAFlat", "save_apartment"):
                state = next((item.get("app_state", {}) for item in initial_apps if item.get("name") == "RentAFlat"), {})
                apartments = state.get("apartments", {}) if isinstance(state, dict) else {}
                saved = set(state.get("saved_apartments", [])) if isinstance(state, dict) else set()
                choices_ids = state.get("_eligible_save", [str(value) for value in apartments if value not in saved]) if isinstance(state, dict) else []
                if choices_ids:
                    arg_properties["apartment_id"] = {"type": "string", "enum": choices_ids}
            if (app, function) == ("Messages", "send_message"):
                state = next((item.get("app_state", {}) for item in initial_apps if item.get("name") == "Messages"), {})
                ids = state.get("_eligible_user_ids", []) if isinstance(state, dict) else []
                if ids:
                    arg_properties["user_id"] = {"type": "string", "enum": [str(value) for value in ids]}
            if (app, function) == ("Emails", "reply_to_email"):
                state = next((item.get("app_state", {}) for item in initial_apps if item.get("name") == "Emails"), {})
                ids = state.get("_eligible_email_ids", []) if isinstance(state, dict) else []
                if ids:
                    arg_properties["email_id"] = {"type": "string", "enum": [str(value) for value in ids]}
            choices.append({
                "type": "object",
                "additionalProperties": False,
                "required": ["app", "function", "args"],
                "properties": {
                    "app": {"const": app},
                    "function": {"const": function},
                    "args": {
                        "type": "object",
                        "required": sorted(required),
                        "properties": arg_properties,
                    },
                },
            })
    return {"oneOf": choices}


def summarize_initial_state(initial_apps: list[dict[str, Any]], max_chars: int = 8000, per_app_chars: int = 900) -> str:
    """Render only declared app state, never hidden oracle events."""
    chunks = []
    for app in initial_apps:
        name = str(app.get("name", "unknown"))
        if name == "Files":
            continue
        raw_state = app.get("app_state", {})
        if name == "RentAFlat" and isinstance(raw_state, dict):
            apartments = raw_state.get("apartments", {})
            compact = []
            if isinstance(apartments, dict):
                for apartment_id, details in apartments.items():
                    if isinstance(details, dict):
                        compact.append({"id": apartment_id, "location": details.get("location"), "price": details.get("price")})
            state = repr({"apartments": compact, "saved_apartments": raw_state.get("saved_apartments", [])})
        else:
            state = repr(raw_state)
            state = state[:per_app_chars]
        chunks.append(name + ": " + state)
    return "\n".join(chunks)[:max_chars]


@dataclass
class GaiaAppRuntime:
    initial_apps: list[dict[str, Any]]
    journal: list[GaiaAction] = field(default_factory=list)
    environment_events: list[dict[str, Any]] = field(default_factory=list)
    reference_bindings: dict[str, str] = field(default_factory=dict)
    last_created_email_id: str | None = field(default=None, init=False)
    state: dict[str, dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.state = {
            str(app["name"]): deepcopy(app.get("app_state", {}))
            for app in self.initial_apps
            if isinstance(app.get("app_state", {}), dict)
        }

    def apply_environment_event(self, event: dict[str, Any]) -> None:
        if event.get("event_type") != "ENV":
            raise ValueError("only ENV events may be applied as environment state")
        self.environment_events.append(event)
        action = event.get("action", {})
        if action.get("app") == "RentAFlat" and action.get("function") == "add_new_apartment":
            args = _event_args(action)
            apartment_id = str(action.get("action_id", "")).rsplit("-", 1)[-1]
            self.state.setdefault("RentAFlat", {}).setdefault("apartments", {})[apartment_id] = args
        elif action.get("app") == "Emails":
            self._apply_email_environment_event(event)

    def _apply_email_environment_event(self, event: dict[str, Any]) -> None:
        """Advance the visible mail state for an independently observed ENV event."""
        action = event.get("action", {})
        function = action.get("function")
        args = _event_args(action)
        event_id = str(event.get("event_id", ""))
        if not event_id:
            raise ValueError("email environment event is missing event_id")
        folders = self._email_folders()
        timestamp = float(event.get("event_time", event.get("event_relative_time", 0.0)))

        if function in {"create_and_add_email", "send_email_to_user_only"}:
            folder = str(args.get("folder_name", "INBOX"))
            recipients = args.get("recipients")
            if not isinstance(recipients, list):
                recipients = [self.state.get("Emails", {}).get("user_email", "")] if function == "send_email_to_user_only" else []
            email = {
                "sender": str(args.get("sender", "")),
                "recipients": recipients,
                "subject": str(args.get("subject", "")),
                "content": str(args.get("content", "")),
                "email_id": event_id,
                "parent_id": None,
                "cc": [],
                "attachments": {},
                "timestamp": timestamp,
                "is_read": False,
            }
            self._add_email(folders, folder, email)
            self.reference_bindings[event_id] = event_id
        elif function == "reply_to_email_from_user":
            parent_id = self._resolve_email_reference(str(args.get("email_id", "")))
            parent = self._find_email(folders, parent_id)
            if parent is None:
                raise ValueError("reply_parent_email_not_found")
            email = {
                "sender": str(args.get("sender", "")),
                "recipients": [self.state.get("Emails", {}).get("user_email", "")],
                "subject": "Re: " + str(parent.get("subject", "")),
                "content": str(args.get("content", "")),
                "email_id": event_id,
                "parent_id": parent_id,
                "cc": [],
                "attachments": {},
                "timestamp": timestamp,
                "is_read": False,
            }
            self._add_email(folders, "INBOX", email)
            self.reference_bindings[event_id] = event_id

    def _email_folders(self) -> dict[str, list[dict[str, Any]]]:
        email_state = self.state.setdefault("Emails", {})
        folders = email_state.setdefault("folders", {})
        normalized: dict[str, list[dict[str, Any]]] = {}
        for name, value in list(folders.items()):
            key = str(name).upper()
            emails = value.get("emails", []) if isinstance(value, dict) else value
            normalized[key] = emails if isinstance(emails, list) else []
        folders.clear()
        folders.update({name: {"folder_name": name, "emails": emails} for name, emails in normalized.items()})
        return normalized

    def _add_email(self, folders: dict[str, list[dict[str, Any]]], folder: str, email: dict[str, Any]) -> None:
        target = folders.setdefault(folder.upper(), [])
        target.append(email)
        target.sort(key=lambda item: float(item.get("timestamp", 0.0)), reverse=True)
        self.state.setdefault("Emails", {})["folders"] = {
            name: {"folder_name": name, "emails": emails} for name, emails in folders.items()
        }

    @staticmethod
    def _find_email(folders: dict[str, list[dict[str, Any]]], email_id: str, folder: str | None = None) -> dict[str, Any] | None:
        candidates = [folders.get(folder.upper(), [])] if folder else folders.values()
        return next((email for emails in candidates for email in emails if str(email.get("email_id", email.get("id"))) == email_id), None)

    def _resolve_email_reference(self, value: str) -> str:
        if value.startswith("{{") and value.endswith("}}"):
            return self.reference_bindings.get(value[2:-2], value)
        return value

    def invoke(self, action: GaiaAction) -> dict[str, Any]:
        self.last_created_email_id = None
        if action.app not in SUPPORTED or action.function not in SUPPORTED[action.app]:
            return {"ok": False, "error": "unsupported_gaia2_action"}
        app_state = self.state.get(action.app, {})
        if (action.app, action.function) == ("RentAFlat", "remove_saved_apartment"):
            saved = app_state.get("saved_apartments", [])
            if action.args.get("apartment_id") not in {str(value) for value in saved}:
                return {"ok": False, "error": "apartment_not_saved"}
        if action.app == "RentAFlat":
            saved = app_state.setdefault("saved_apartments", [])
            apartment_id = action.args.get("apartment_id")
            if action.function == "save_apartment" and apartment_id:
                if apartment_id not in saved:
                    saved.append(apartment_id)
            elif action.function == "remove_saved_apartment" and apartment_id:
                saved.remove(apartment_id)
        elif action.app == "Shopping":
            cart = app_state.setdefault("cart", {})
            item_id, quantity = action.args.get("item_id"), int(action.args.get("quantity", "1"))
            if action.function == "add_to_cart" and item_id:
                cart[item_id] = int(cart.get(item_id, 0)) + quantity
            elif action.function == "remove_from_cart" and item_id:
                if item_id not in cart or int(cart[item_id]) < quantity:
                    return {"ok": False, "error": "item_not_in_cart"}
                cart[item_id] = int(cart[item_id]) - quantity
                if not cart[item_id]:
                    del cart[item_id]
            elif action.function == "checkout":
                app_state.setdefault("orders", []).append({"items": deepcopy(cart), "discount_code": action.args.get("discount_code")})
                cart.clear()
        elif action.app == "Calendar" and action.function == "delete_calendar_event":
            event_id = action.args.get("event_id")
            events = app_state.get("events", [])
            kept = [event for event in events if str(event.get("event_id", event.get("id"))) != event_id]
            if len(kept) == len(events):
                return {"ok": False, "error": "calendar_event_not_found"}
            app_state["events"] = kept
        elif action.app == "Calendar" and action.function == "add_calendar_event":
            app_state.setdefault("events", []).append(dict(action.args))
        elif action.app in {"Messages", "Chats"} and action.function.startswith("send_message"):
            app_state.setdefault("sent_messages", []).append(dict(action.args))
        elif action.app == "Emails":
            folders = self._email_folders()
            folder = action.args.get("folder_name", "INBOX").upper()
            email_id = action.args.get("email_id")
            if action.function in {"delete_email", "move_email", "reply_to_email", "forward_email"} and not email_id:
                return {"ok": False, "error": "email_id_required"}
            if action.function == "send_email":
                email = self._new_agent_email(action.args)
                self._add_email(folders, "SENT", email)
                self.last_created_email_id = str(email["email_id"])
            elif action.function == "delete_email":
                email = self._find_email(folders, email_id, folder)
                if email is None:
                    return {"ok": False, "error": "email_not_found"}
                folders[folder].remove(email)
                self._add_email(folders, "TRASH", email)
            elif action.function == "move_email":
                source = action.args.get("source_folder_name", folder).upper()
                destination = action.args.get("dest_folder_name", "INBOX").upper()
                email = self._find_email(folders, email_id, source)
                if email is None:
                    return {"ok": False, "error": "email_not_found"}
                folders[source].remove(email)
                self._add_email(folders, destination, email)
            elif action.function in {"reply_to_email", "forward_email"}:
                source = self._find_email(folders, email_id, folder)
                if source is None:
                    return {"ok": False, "error": "email_not_found"}
                subject_prefix = "Re: " if action.function == "reply_to_email" else "FWD: "
                content = action.args.get("content", "") if action.function == "reply_to_email" else "> " + str(source.get("content", ""))
                recipients = action.args.get("recipients") if action.function == "forward_email" else [str(source.get("sender", ""))]
                email = self._new_agent_email(action.args, subject=subject_prefix + str(source.get("subject", "")), content=content, recipients=recipients, parent_id=email_id)
                self._add_email(folders, "SENT", email)
                self.last_created_email_id = str(email["email_id"])
        elif action.app == "Contacts":
            contacts = app_state.setdefault("contacts", [])
            contact_id = action.args.get("contact_id")
            if action.function == "add_new_contact":
                contacts.append(dict(action.args))
            elif action.function == "delete_contact":
                app_state["contacts"] = [x for x in contacts if str(x.get("contact_id", x.get("id"))) != contact_id]
            elif action.function == "edit_contact":
                for contact in contacts:
                    if str(contact.get("contact_id", contact.get("id"))) == contact_id:
                        updates = action.args.get("updates", {})
                        if isinstance(updates, dict):
                            contact.update(updates)
        elif action.app == "Cabs":
            rides = app_state.setdefault("ride_history", [])
            if action.function == "order_ride":
                rides.append(dict(action.args))
            elif action.function == "user_cancel_ride" and rides:
                rides[-1]["cancelled"] = True
        elif action.app == "Files":
            app_state.setdefault("operations", []).append(dict(action.args) | {"function": action.function})
        elif action.app == "AgentUserInterface":
            app_state.setdefault("messages", []).append({"role": "agent", "content": action.args.get("content", "")})
        self.journal.append(action)
        return {"ok": True, "app": action.app, "function": action.function, "args": action.args}

    def bind_agent_event(self, event_id: str) -> None:
        """Bind an oracle-event reference to the concrete mail ID just emitted.

        The scheduler calls this after an agent turn is attempted.  It keeps
        later ENV replies executable without exposing oracle action arguments.
        """
        if self.last_created_email_id is not None:
            self.reference_bindings[str(event_id)] = self.last_created_email_id

    def _new_agent_email(self, args: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        folders = self._email_folders()
        next_number = sum(len(emails) for emails in folders.values()) + 1
        email = {
            "sender": self.state.get("Emails", {}).get("user_email", "user@meta.com"),
            "recipients": args.get("recipients", []),
            "subject": args.get("subject", ""),
            "content": args.get("content", ""),
            "email_id": f"agent-email-{next_number}",
            "parent_id": None,
            "cc": args.get("cc", []),
            "attachments": {},
            "timestamp": float(next_number),
            "is_read": True,
        }
        email.update(overrides)
        if isinstance(email["recipients"], str):
            email["recipients"] = [email["recipients"]]
        return email

    def tool_catalog(self) -> list[str]:
        available = {str(app["name"]) for app in self.initial_apps}
        return [app + "." + function for app in sorted(available & set(SUPPORTED)) for function in sorted(SUPPORTED[app])]


def _event_args(action: dict[str, Any]) -> dict[str, Any]:
    """Decode typed values stored as JSON text in the raw GAIA2 event stream."""
    parsed: dict[str, Any] = {}
    for arg in action.get("args", []):
        value = arg.get("value")
        if isinstance(value, str) and arg.get("value_type", "").startswith(("list", "dict")):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        parsed[str(arg["name"])] = value
    return parsed
