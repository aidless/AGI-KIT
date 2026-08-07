from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.gaia2_apps import GaiaAppRuntime, action_json_schema, summarize_initial_state
from agi_kit.gaia2_simulator import GaiaAction


def test_adapter_whitelists_real_actions_and_records_stateful_journal():
    runtime = GaiaAppRuntime([{"name": "Calendar"}, {"name": "Messages"}])
    assert "Calendar.add_calendar_event" in runtime.tool_catalog()
    assert runtime.invoke(GaiaAction("Calendar", "add_calendar_event", {"title": "Meeting"}))["ok"]
    assert runtime.journal == [GaiaAction("Calendar", "add_calendar_event", {"title": "Meeting"})]
    assert runtime.invoke(GaiaAction("Calendar", "drop_database", {}))["ok"] is False


def test_adapter_only_accepts_environment_events_as_environment_state():
    runtime = GaiaAppRuntime([])
    runtime.apply_environment_event({"event_type": "ENV", "action": {}})
    assert len(runtime.environment_events) == 1


def test_state_summary_reads_only_initial_app_state():
    summary = summarize_initial_state([{"name": "Contacts", "app_state": {"contacts": ["Ada"]}}])
    assert "Contacts" in summary and "Ada" in summary


def test_rent_a_flat_summary_keeps_all_property_ids_and_prices():
    summary = summarize_initial_state([{"name": "RentAFlat", "app_state": {"apartments": {"a": {"location": "Stockholm", "price": 900}, "b": {"location": "Stockholm", "price": 2000}}, "saved_apartments": ["a"]}}])
    assert "'id': 'a'" in summary and "'price': 2000" in summary


def test_action_schema_exposes_only_available_whitelisted_calls():
    schema = action_json_schema([{"name": "Calendar"}])
    calls = {(branch["properties"]["app"]["const"], branch["properties"]["function"]["const"]) for branch in schema["oneOf"]}
    assert calls == {("Calendar", "add_calendar_event"), ("Calendar", "delete_calendar_event")}


def test_known_contract_requires_apartment_identifier():
    schema = action_json_schema([{"name": "RentAFlat"}])
    branch = next(x for x in schema["oneOf"] if x["properties"]["function"]["const"] == "remove_saved_apartment")
    assert branch["properties"]["args"]["required"] == ["apartment_id"]


def test_remove_contract_limits_ids_to_visible_saved_apartments():
    schema = action_json_schema([{"name": "RentAFlat", "app_state": {"saved_apartments": ["one", "two"]}}])
    branch = next(x for x in schema["oneOf"] if x["properties"]["function"]["const"] == "remove_saved_apartment")
    assert branch["properties"]["args"]["properties"]["apartment_id"]["enum"] == ["one", "two"]


def test_runtime_rejects_removal_of_unsaved_apartment():
    runtime = GaiaAppRuntime([{"name": "RentAFlat", "app_state": {"saved_apartments": ["saved"]}}])
    assert runtime.invoke(GaiaAction("RentAFlat", "remove_saved_apartment", {"apartment_id": "other"}))["error"] == "apartment_not_saved"


def test_email_environment_events_advance_visible_state_and_bind_event_id():
    runtime = GaiaAppRuntime([{"name": "Emails", "app_state": {"user_email": "me@example.com", "folders": {"INBOX": {"emails": []}, "SENT": {"emails": []}}}}])
    runtime.apply_environment_event({
        "event_type": "ENV", "event_id": "env-email-1", "event_time": 10,
        "action": {"app": "Emails", "function": "create_and_add_email", "args": [
            {"name": "sender", "value": "ada@example.com", "value_type": "str"},
            {"name": "recipients", "value": "[\"me@example.com\"]", "value_type": "list"},
            {"name": "subject", "value": "Hello", "value_type": "str"},
            {"name": "content", "value": "Need a reply", "value_type": "str"},
            {"name": "folder_name", "value": "INBOX", "value_type": "str"},
        ]},
    })
    inbox = runtime.state["Emails"]["folders"]["INBOX"]["emails"]
    assert inbox[0]["email_id"] == "env-email-1"
    assert inbox[0]["recipients"] == ["me@example.com"]
    assert runtime.reference_bindings == {"env-email-1": "env-email-1"}


def test_email_actions_move_delete_and_reply_without_leaving_duplicate_messages():
    initial = {"name": "Emails", "app_state": {"user_email": "me@example.com", "folders": {"INBOX": {"emails": [{"email_id": "one", "sender": "ada@example.com", "subject": "Question", "content": "Hi"}]}, "SENT": {"emails": []}, "TRASH": {"emails": []}}}}
    runtime = GaiaAppRuntime([initial])
    assert runtime.invoke(GaiaAction("Emails", "move_email", {"email_id": "one", "source_folder_name": "INBOX", "dest_folder_name": "SENT"}))["ok"]
    assert not runtime.state["Emails"]["folders"]["INBOX"]["emails"]
    assert [item["email_id"] for item in runtime.state["Emails"]["folders"]["SENT"]["emails"]] == ["one"]
    assert runtime.invoke(GaiaAction("Emails", "reply_to_email", {"email_id": "one", "folder_name": "SENT", "content": "Answer"}))["ok"]
    sent = runtime.state["Emails"]["folders"]["SENT"]["emails"]
    reply = next(item for item in sent if item["email_id"] != "one")
    assert reply["recipients"] == ["ada@example.com"] and reply["parent_id"] == "one"
    assert runtime.invoke(GaiaAction("Emails", "delete_email", {"email_id": "one", "folder_name": "SENT"}))["ok"]
    assert [item["email_id"] for item in runtime.state["Emails"]["folders"]["TRASH"]["emails"]] == ["one"]


def test_agent_email_binding_enables_later_environment_reply():
    runtime = GaiaAppRuntime([{"name": "Emails", "app_state": {"user_email": "me@example.com", "folders": {"SENT": {"emails": []}}}}])
    assert runtime.invoke(GaiaAction("Emails", "send_email", {"recipients": "ada@example.com", "subject": "Status", "content": "Update"}))["ok"]
    runtime.bind_agent_event("agent-turn-1")
    runtime.apply_environment_event({
        "event_type": "ENV", "event_id": "env-reply-1", "event_time": 20,
        "action": {"app": "Emails", "function": "reply_to_email_from_user", "args": [
            {"name": "sender", "value": "ada@example.com", "value_type": "str"},
            {"name": "email_id", "value": "{{agent-turn-1}}", "value_type": "str"},
            {"name": "content", "value": "Received", "value_type": "str"},
        ]},
    })
    inbox = runtime.state["Emails"]["folders"]["INBOX"]["emails"]
    assert inbox[0]["parent_id"] == runtime.reference_bindings["agent-turn-1"]
    assert inbox[0]["subject"] == "Re: Status"
