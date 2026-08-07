from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from eval_gaia2_simulator import parse_prediction, rent_action_feedback
from agi_kit.gaia2_simulator import GaiaAction, GaiaScenario


def test_gaia2_prediction_parser_requires_exact_action_schema():
    action = parse_prediction('```json\n{"app":"Calendar","function":"add_calendar_event","args":{"title":"x"}}\n```')
    assert action is not None and action.app == "Calendar"
    assert parse_prediction('{"app":"Calendar","function":"add_calendar_event"}') is None
    assert parse_prediction('{"final":"done"}') is None


def test_rent_reflection_uses_visible_state_not_oracle_labels():
    scenario = GaiaScenario("x", [{"name": "RentAFlat", "app_state": {"apartments": {"a": {"location": "Stockholm", "price": 1500}, "b": {"location": "Stockholm", "price": 900}}}}], [{"event_type": "USER", "event_relative_time": 0, "action": {"args": [{"name": "content", "value": "Delete properties in Stockholm not in the 1000-1800 price range"}]}}])
    assert rent_action_feedback(GaiaAction("RentAFlat", "remove_saved_apartment", {"apartment_id": "a"}), scenario)
    assert rent_action_feedback(GaiaAction("RentAFlat", "remove_saved_apartment", {"apartment_id": "b"}), scenario) is None
