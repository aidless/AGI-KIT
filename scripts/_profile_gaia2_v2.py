"""Profile runnable GAIA2 scenarios."""
import json
from pathlib import Path
ROOT = Path(".").resolve()
fn = ROOT / "data/gaia2/validation.jsonl"
runnable = []
total = 0
ok_set = {"Calendar", "Emails", "Shopping"}
for line in fn.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    total += 1
    actions = r.get("expected_actions", [])
    apps_used = set()
    for a in actions:
        app_name = a.get("app")
        if app_name:
            apps_used.add(app_name)
    if apps_used and apps_used.issubset(ok_set):
        runnable.append((r.get("scenario_id"), list(apps_used), len(actions)))

print("Total scenarios: " + str(total))
print("Runnable with our 3 apps (Calendar/Emails/Shopping): " + str(len(runnable)))
for sid, apps, n in runnable[:25]:
    print("  " + str(sid) + " apps=" + str(apps) + " actions=" + str(n))

