# GAIA2-mini Schema and Bridge Notes

## Source

- **Dataset**: `meta-agents-research-environments___gaia2`, **mini config**
- **Split**: validation (160 scenarios)
- **Path on disk**: `F:\hf_cache\datasets\meta-agents-research-environments___gaia2\mini\0.0.0\<snapshot-hash>\`
- **Format**: Apache Arrow IPC stream (`gaia2-validation.arrow`)
- **Cached size**: 408 MB

## Schema

Each scenario is a virtual multi-turn agent interaction. Fields:

- `id`: scenario hash, e.g. `0741_0uetwq6bp7jrofwsyy9uel4puv0c3kb2`
- `scenario_id`: e.g. `scenario_universe_21_xvc7uo`
- `category`: one of {`time`, `search`, `execution`, `ambiguity`, `adaptability`}; 32 each
- `tags`: scenario meta-tags from `metadata.definition.tags`
- `available_apps`: list of `{name, class_name}` for each app instance
- `expected_actions`: list of `{app, function, args}` describing what an oracle agent should do at each `OracleEvent/AGENT`
- `n_events`: total events (mix of AGENT, ENV, USER)

## 10 Apps in the Universe

| App | Class | Expected calls |
|---|---|---:|
| Calendar | CalendarAppV2 | 250 |
| Emails | EmailAppV2 | 166 |
| Shopping | ShoppingAppV2 | 146 |
| AgentUserInterface | AgentUserInterface | 136 |
| Messages | MessagesAppV2 | 104 |
| RentAFlat | RentAFlatAppV2 | 97 |
| Chats | ChatsAppV2 | 54 |
| Cabs | CabsAppV2 | 43 |
| Contacts | ContactsAppV2 | 33 |
| Files | SandboxLocalFileSystem | 13 |

Total expected AGENT calls: ~1,042 across the 160 scenarios.

## Raw Event Availability

The original Arrow cache includes a `data` JSON field containing initial app
state and the complete `USER`, `ENV`, and oracle `AGENT` event stream. The
tracked `validation.jsonl` is a compact derivative that retains only oracle
actions. Use `scripts/extract_gaia2_raw_events.py` on the Arrow cache to
regenerate complete scenarios for a simulator; never use oracle actions as
the model input or prediction.

## Why AGI Kit Cannot Evaluate On This Data Without New Implementation

AGI Kit's `full_agent.py` exposes 11 tools (calculator, read_file, read_pdf,
echo, list_dir, shell, web_search, web_fetch, rag_add, rag_search,
rag_clear). **None of these overlap with the 10 GAIA2 apps.** Every
expected `app.function(args)` in GAIA2 would be unanswerable by AGI Kit
without first implementing the 10 apps as Python tool classes.

For example:
- Expected `Calendar__create_event(...)` -> AGI Kit has no Calendar tool.
- Expected `Emails__send_email(...)` -> AGI Kit has no Emails tool.
- Expected `Shopping__purchase_item(...)` -> AGI Kit has no Shopping tool.

## What Would Be Required for Canonical GAIA2 Evaluation

1. Implement each of the 10 apps as Python classes exposing the
   `app.function(args)` API of the dataset. Each app likely needs
   persistent state across the scenario (messages, calendar entries,
   etc.) - a small in-memory or sqlite-backed store.
2. Implement the GAIA2 simulator harness: load the scenario, alternate
   AGENT/ENV/USER events, present the agent with the agent's prompt at
   each AGENT event, capture the agent's action, score exact match or
   per-step success.
3. Implement the canonical scoring methodology (`pass_rate` over the
   full scenario, not just first-step tool selection).
4. Run the full pipeline. Estimated wall clock on this hardware
   (Qwen3-1.7B): ~6-10 hours for the 160 scenarios, plus LLM
   generation, plus app simulation.

Estimated total implementation effort: **1 to 2 weeks** of focused
engineering for one engineer.

## What We Did Instead (Round 7)

- Confirmed the dataset is real (`meta-agents-research-environments___gaia2`
  mini validation, 160 scenarios, 408 MB).
- Confirmed the 10-app universe the dataset expects.
- Confirmed AGI Kit cannot evaluate without those apps.
- Extracted `validation.jsonl` (160 records, ~140 KB) so future work can
  pick it up without re-reading the arrow file.
- Documented the bridge gap in `papers/preprint_unified_en.md` Section 12.
- Added an explicit open-item to `Limitations` and `Future Work`.

## Files This Round

- `data/gaia2/validation.jsonl` - 160 records, extracted schema
- `data/gaia2/SCHEMA.md` - this file
- `scripts/load_gaia2.py` - the extractor
- `papers/preprint_unified_en.md` - new Section 12 "Bridging Real GAIA2"
