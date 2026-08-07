"""Stateful event scheduling and action scoring for cached GAIA2 scenarios.

This module deliberately separates scenario observations from oracle labels.
`USER`/`ENV` events are exposed to an agent; `AGENT` oracle events are used
only after a prediction has been submitted for exact scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def args_to_dict(args: list[dict[str, Any]]) -> dict[str, str]:
    return {str(arg["name"]): str(arg.get("value", "")) for arg in args}


@dataclass(frozen=True)
class GaiaAction:
    app: str
    function: str
    args: dict[str, str]

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "GaiaAction":
        action = event["action"]
        return cls(str(action["app"]), str(action["function"]), args_to_dict(action.get("args", [])))


@dataclass
class GaiaScenario:
    scenario_id: str
    initial_apps: list[dict[str, Any]]
    events: list[dict[str, Any]]

    def observations_until(self, relative_time: float) -> list[dict[str, Any]]:
        visible = []
        delivered: set[str] = set()
        for event in sorted(self.events, key=lambda item: (float(item["event_relative_time"]), str(item.get("event_id", "")))):
            if event["event_type"] not in {"USER", "ENV"} or float(event["event_relative_time"]) > relative_time:
                continue
            if set(event.get("dependencies", [])) - delivered:
                continue
            visible.append(event)
            delivered.add(str(event.get("event_id", "")))
        return visible

    def observation_times(self) -> list[float]:
        """Times at which a USER or ENV observation may become visible."""
        return sorted({float(event["event_relative_time"]) for event in self.events if event["event_type"] in {"USER", "ENV"}})

    def ready_observations(self, relative_time: float, completed_event_ids: set[str]) -> list[dict[str, Any]]:
        """Return newly eligible USER/ENV events without consulting oracle actions."""
        ready = []
        for event in self.events:
            if event["event_type"] not in {"USER", "ENV"}:
                continue
            event_id = str(event.get("event_id", ""))
            if event_id in completed_event_ids or float(event["event_relative_time"]) > relative_time:
                continue
            if set(event.get("dependencies", [])) <= completed_event_ids:
                ready.append(event)
        return sorted(ready, key=lambda item: (float(item["event_relative_time"]), str(item.get("event_id", ""))))

    def ready_agent_event_ids(self, relative_time: float, completed_event_ids: set[str]) -> list[str]:
        """Return schedulable agent turn IDs without exposing oracle action data."""
        ready = []
        for event in self.events:
            if event["event_type"] != "AGENT":
                continue
            event_id = str(event.get("event_id", ""))
            if event_id in completed_event_ids or float(event["event_relative_time"]) > relative_time:
                continue
            if set(event.get("dependencies", [])) <= completed_event_ids:
                ready.append(event_id)
        return sorted(ready)

    def oracle_actions(self) -> list[GaiaAction]:
        return [GaiaAction.from_event(event) for event in self.events if event["event_type"] == "AGENT"]

    def score_sequence(
        self,
        predictions: list[GaiaAction],
        reference_bindings: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Score actions after resolving IDs emitted by environment events.

        GAIA2 records a later action's reference to an environment-created
        object as ``{{Event-ENV-...}}``.  An agent cannot emit that template:
        it observes the concrete ID in the app state.  The runtime therefore
        supplies the event-to-object bindings it created while advancing time.
        """
        bindings = reference_bindings or {}

        def resolve(value: str) -> str:
            if value.startswith("{{") and value.endswith("}}"):
                return bindings.get(value[2:-2], value)
            return value

        gold = [
            GaiaAction(action.app, action.function, {name: resolve(value) for name, value in action.args.items()})
            for action in self.oracle_actions()
        ]
        matches = [pred == expected for pred, expected in zip(predictions, gold)]
        return {
            "scenario_id": self.scenario_id,
            "gold_actions": len(gold),
            "predicted_actions": len(predictions),
            "step_matches": sum(matches),
            "exact_sequence": len(predictions) == len(gold) and all(matches),
        }
