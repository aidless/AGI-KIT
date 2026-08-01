"""GAIA2 apps: per-scenario state plus dispatch.

call(app, function, args, scenario_id) dispatches and persists state.
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

