"""Profile GAIA2 scenarios: app distribution and runnable subset."""
import json
from pathlib import Path
ROOT = Path(".").resolve()
fn = ROOT / "data/gaia2/validation.jsonl"
apps_seen = {}
scenarios_by_app = {}
candidate_scenarios = []
for line in fn.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    scenario_id = r.get("scenario_id", "unknown")
    data_str = r.get("data", "{}")
    apps_in_scenario = []
    try:
        data = json.loads(data_str) if isinstance(data_str, str) else data_str
        for a in data.get("apps", []):
            apps_in_scenario.append(a.get("name", "?"))
    except Exception:
        pass
    if not apps_in_scenario:
        for ev in r.get("expected_actions", []):
            action = ev.get("action", {})
            app_name = action.get("app")
            if app_name:
                apps_in_scenario.append(app_name)
    for app in apps_in_scenario:
        apps_seen[app] = apps_seen.get(app, 0) + 1
    if apps_in_scenario:
        key = ",".join(sorted(set(apps_in_scenario)))
        scenarios_by_app.setdefault(key, []).append(scenario_id)
    runnable = {"Calendar", "Emails", "Shopping"}
    if all(a in runnable for a in apps_in_scenario):
        candidate_scenarios.append((scenario_id, r))

print("Total apps in scenarios:")
for a, n in sorted(apps_seen.items(), key=lambda x: -x[1]):
    print("  " + a + ": " + str(n))
print()
print("App-combinations:")
for k, v in sorted(scenarios_by_app.items(), key=lambda x: -len(x[1])):
    print("  " + str(len(v)) + " scenarios: " + k[:80])
print()
print("Runnable with our 3 apps (Calendar/Emails/Shopping only): " + str(len(candidate_scenarios)))
if candidate_scenarios:
    sid = candidate_scenarios[0][0]
    print("Example runnable scenario id: " + sid)

