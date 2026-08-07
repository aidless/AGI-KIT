from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.gaia2_simulator import GaiaAction, GaiaScenario


def event(kind, at, app, function, args):
    return {"event_type": kind, "event_relative_time": at, "action": {"app": app, "function": function, "args": args}}


def test_oracle_actions_are_not_exposed_as_observations_and_score_exactly():
    user = event("USER", 0, "AgentUserInterface", "send_message_to_agent", [{"name": "content", "value": "book a cab"}])
    env = event("ENV", 2, "Cabs", "add_driver", [{"name": "name", "value": "Ada"}])
    gold = event("AGENT", 3, "Cabs", "order_ride", [{"name": "end_location", "value": "Library"}])
    scenario = GaiaScenario("test", [], [user, env, gold])
    assert [x["event_type"] for x in scenario.observations_until(3)] == ["USER", "ENV"]
    expected = scenario.oracle_actions()
    assert expected == [GaiaAction("Cabs", "order_ride", {"end_location": "Library"})]
    assert scenario.score_sequence(expected)["exact_sequence"] is True
    assert scenario.score_sequence([])["exact_sequence"] is False


def test_ready_observations_respects_dependencies_without_oracle_access():
    user = event("USER", 0, "AgentUserInterface", "send_message_to_agent", [])
    user["event_id"] = "u1"
    env = event("ENV", 2, "Cabs", "add_driver", [])
    env["event_id"] = "e1"
    env["dependencies"] = ["u1"]
    scenario = GaiaScenario("test", [], [user, env])
    assert [item["event_id"] for item in scenario.ready_observations(2, set())] == ["u1"]
    assert [item["event_id"] for item in scenario.ready_observations(2, {"u1"})] == ["e1"]


def test_score_resolves_environment_object_references_before_comparing_actions():
    gold = event("AGENT", 1, "Emails", "reply_to_email", [{"name": "email_id", "value": "{{event-1}}"}])
    scenario = GaiaScenario("test", [], [gold])
    prediction = GaiaAction("Emails", "reply_to_email", {"email_id": "visible-email"})
    assert scenario.score_sequence([prediction], {"event-1": "visible-email"})["exact_sequence"] is True


def test_agent_turn_scheduler_returns_only_ids_after_dependencies_are_visible():
    user = event("USER", 0, "AgentUserInterface", "send_message_to_agent", [])
    user["event_id"] = "user-1"
    agent = event("AGENT", 1, "Emails", "send_email", [])
    agent["event_id"] = "agent-1"
    agent["dependencies"] = ["user-1"]
    scenario = GaiaScenario("test", [], [user, agent])
    assert scenario.ready_agent_event_ids(1, set()) == []
    assert scenario.ready_agent_event_ids(1, {"user-1"}) == ["agent-1"]
