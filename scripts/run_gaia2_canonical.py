"""Run the official ARE GAIA2 harness with a native tool-calling adapter.

The ARE default agent expects a ReAct-style "Thought:/Action:" text loop, which
small local models follow poorly (all earlier runs produced zero tool calls).
This adapter keeps the official environment/scenario/validation pipeline but
replaces the LLM engine: the model is invoked with native OpenAI-style function
calling, and each tool call is rendered back into the ReAct JSON format the ARE
agent loop can execute.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[1]

from are.simulation.agents.agent_builder import AgentBuilder
from are.simulation.benchmark.scenario_executor import (
    multiply_scenarios_iterator,
    preprocess_scenarios_iterator,
)
from are.simulation.benchmark.scenario_loader import setup_scenarios_iterator
from are.simulation.multi_scenario_runner import MultiScenarioRunner
from are.simulation.scenarios.config import MultiScenarioRunnerConfig
from are.simulation.utils.countable_iterator import CountableIterator
from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


class NativeToolEngine:
    """LLM engine adapter: native function calling -> ReAct JSON text."""

    APP_KEYWORDS = {
        "RentAFlat": ("apartment", "property", "rent", "flat"),
        "Messages": ("message", "text"),
        "Chats": ("chat", "conversation"),
        "Calendar": ("calendar", "event", "meeting"),
        "Emails": ("email",),
        "Shopping": ("cart", "order", "buy", "purchase"),
        "Cabs": ("cab", "ride"),
        "Contacts": ("contact",),
        "Files": ("file", "folder", "document"),
        "City": ("city", "place", "location"),
        "InternalContacts": ("colleague", "employee"),
    }


    def __init__(
        self,
        tools_provider: Callable[[], dict[str, Any]],
        endpoint: str,
        model: str,
        timeout_s: float = 240.0,
        progress_path: str | None = None,
        observation_transform: str = "decision_fields",
        tool_catalog_path: str | None = None,
    ):
        self.tools_provider = tools_provider
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        if observation_transform not in {"raw_truncate_3500", "legacy_id_name", "decision_fields"}:
            raise ValueError(f"unsupported_observation_transform:{observation_transform}")
        self.progress_path = progress_path
        self.observation_transform = observation_transform
        self.tool_catalog_path = tool_catalog_path
        self._tool_catalog_written = False
        self.client = httpx.Client(timeout=timeout_s)
        self._current_messages: list[dict[str, Any]] = []
        self._last_observation_raw_chars = 0
        self._last_observation_transformed_chars = 0

    def _dump_tool_catalog(self) -> None:
        """Authoritative runtime tool catalog from the initialized ARE tool
        OBJECTS, written once before the first model request. This is the
        single source of truth for tool_count / sorted SHA-256; regex-based
        extraction is debug-only and never enters research conclusions."""
        import hashlib
        import subprocess
        from pathlib import Path as _Path

        names = sorted((self.tools_provider() or {}).keys())
        commit = "unknown"
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=str(ROOT)).stdout.strip() or "unknown"
        except Exception:
            pass
        catalog = {
            "tool_count": len(names),
            "tools": names,
            "sha256": "sha256:" + hashlib.sha256(json.dumps(names).encode("utf-8")).hexdigest(),
            "source": "are-initialized runtime tool objects (react_agent.tools)",
            "runner_commit": commit,
        }
        try:
            _Path(self.tool_catalog_path).parent.mkdir(parents=True, exist_ok=True)
            _Path(self.tool_catalog_path).write_text(
                json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
            self._tool_catalog_written = True
        except Exception as exc:  # pragma: no cover
            print(f"WARN tool catalog dump failed: {exc}")

    def _tool_schema(self) -> list[dict[str, Any]]:
        all_tools = list(self.tools_provider().values())
        selected = all_tools
        selected = [
            tool
            for tool in selected
            if tool.name != "SystemApp__get_current_time"
        ]
        schema = []
        for tool in selected:
            inputs = getattr(tool, "inputs", {}) or {}
            if isinstance(inputs, dict):
                properties = {
                    key: {"type": "string", "description": str(value)}
                    for key, value in inputs.items()
                }
                required = list(inputs.keys())
            else:
                properties = {}
                required = []
            schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": getattr(tool, "description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return schema

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        self._last_observation_raw_chars = 0
        self._last_observation_transformed_chars = 0
        out = []
        role_map = {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
            "tool_call": "assistant",
            "tool_response": "user",
            "tool-response": "user",
            "tool": "tool",
        }
        for message in messages:
            role = role_map.get(str(message.get("role", "user")), "user")
            content = message.get("content", "")
            if content is None:
                content = ""
            if role == "user" and "Observation" in str(content):
                raw_content = str(content)
                content = self._transform_observation(raw_content)
                self._last_observation_raw_chars += len(raw_content)
                self._last_observation_transformed_chars += len(content)
            out.append({"role": role, "content": str(content)})
        return out

    def _transform_observation(self, content: str) -> str:
        if self.observation_transform == "raw_truncate_3500":
            return content[:3500]
        if self.observation_transform == "legacy_id_name":
            return self._legacy_compact_tool_response(content)
        return self._compact_tool_response(content)

    @staticmethod
    def _legacy_compact_tool_response(content: str) -> str:
        """Retain the fix8 lossier transform for the pre-registered diagnostic."""
        entries = re.findall(
            r"'([0-9a-fA-F]{8,})': Apartment\(name='([^']*)'",
            content,
        )
        if entries:
            lines = [f"{apartment_id} | {name}" for apartment_id, name in entries]
            return (
                "STATE SUMMARY (saved/all apartments):\n"
                + "\n".join(lines)
                + "\n(original observation truncated; use the exact IDs above)"
            )
        return content[:3500]

    @staticmethod
    def _compact_tool_response(content: str) -> str:
        """Compress RentAFlat observations without dropping action-decision fields.

        The ARE Apartment dataclass repr is large, but reducing it to only an ID
        and name prevents a model from checking location, price, or saved status
        before choosing a save/remove action. Preserve a stable, bounded subset
        of those fields for every recognized apartment. Fall back to the legacy
        character cap only when the input is not an ARE Apartment observation.
        """
        apartment_pattern = re.compile(
            r"'(?P<apartment_id>[0-9a-fA-F]{8,})': Apartment\("
            r"name=(?P<name>.*?), "
            r"location=(?P<location>.*?), "
            r"zip_code=(?P<zip_code>.*?), "
            r"price=(?P<price>-?\d+(?:\.\d+)?), "
            r"bedrooms=(?P<bedrooms>\d+), "
            r"bathrooms=(?P<bathrooms>\d+), "
            r"property_type=(?P<property_type>.*?), "
            r".*? saved=(?P<saved>True|False)\)",
            re.DOTALL,
        )
        entries = list(apartment_pattern.finditer(content))
        if entries:
            lines = [
                "{apartment_id} | name={name} | location={location} | "
                "zip_code={zip_code} | price={price} | bedrooms={bedrooms} | "
                "bathrooms={bathrooms} | property_type={property_type} | saved={saved}".format(
                    **entry.groupdict()
                )
                for entry in entries
            ]
            return (
                "STATE SUMMARY (all apartments; decision fields preserved):\n"
                + "\n".join(lines)
                + "\n(use the exact IDs and visible decision fields above)"
            )
        return content[:3500]

    def _render_react_action(self, name: str, args: dict[str, Any]) -> str:
        return (
            "Thought: I will call the "
            + name
            + " tool with the required arguments.\nAction:\n"
            + json.dumps(
                {"action": name, "action_input": args},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "<end_action>"
        )

    @staticmethod
    def _extract_action(content: str) -> tuple[str, dict[str, Any]] | None:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return str(data["action"]), dict(data.get("action_input") or {})
        except (KeyError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _trailing_repeat_count(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], int]:
        """Return the trailing run of identical assistant tool calls."""
        calls = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            parsed = NativeToolEngine._extract_action(str(message.get("content", "")))
            if parsed is not None:
                calls.append(parsed)
        if not calls:
            return "", {}, 0
        name, args = calls[-1]
        count = 0
        for item in reversed(calls):
            if item[0] == name and json.dumps(item[1], sort_keys=True) == json.dumps(args, sort_keys=True):
                count += 1
            else:
                break
        return name, args, count

    def _force_progress(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], str] | None:
        """Break degenerate loops by rewriting a repeated call to a state tool."""
        name, args, count = self._trailing_repeat_count(messages)
        if count < 3 or not name:
            return None
        note = (
            "STOP repeating the same call. It has already been attempted "
            + str(count)
            + " times. Inspect the latest observation and choose a different action that advances the task."
        )
        if name == "SystemApp__get_current_time":
            return "SystemApp__wait_for_notification", {"timeout": 60}, note
        if name.endswith("remove_saved_apartment"):
            prefix = name.split("__", 1)[0]
            list_tool = prefix + "__list_saved_apartments"
            all_tools = list(self.tools_provider().values())
            if any(tool.name == list_tool for tool in all_tools):
                return list_tool, {}, note + " First call " + list_tool + " to obtain the exact saved IDs."
        if name == "SystemApp__wait_for_notification":
            note = (
                "STOP waiting. Scenario events have already been delivered; "
                "execute the user's task now with the available tools."
            )
            inspection = self._inspection_tool()
            if inspection is not None:
                return inspection, {}, note + " Inspect the app state with " + inspection + " first."
        recent = [call for call in self._recent_tool_calls(messages)][-4:]
        if (
            name.startswith("SystemApp__")
            and recent
            and all(call[0].startswith("SystemApp__") for call in recent)
        ):
            inspection = self._inspection_tool()
            if inspection is not None:
                return inspection, {}, note + " Inspect the app state with " + inspection + " first."
        return None, None, note

    @staticmethod
    def _recent_tool_calls(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        calls = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            parsed = NativeToolEngine._extract_action(str(message.get("content", "")))
            if parsed is not None:
                calls.append(parsed)
        return calls

    @staticmethod
    def _app_mutation_flags(tools: list[Any]) -> dict[str, bool]:
        """Which apps expose state-changing tools (save/create/remove/send/buy)."""
        flags: dict[str, bool] = {}
        for tool in tools:
            name = tool.name.lower()
            app = tool.name.split("__", 1)[0]
            if app == "SystemApp":
                continue
            if any(
                marker in name
                for marker in ("save", "create", "add", "remove", "delete", "send", "buy", "purchase", "update", "edit", "book")
            ):
                flags[app] = True
            else:
                flags.setdefault(app, False)
        return flags

    def _ranked_inspection_tools(self) -> list[str]:
        """No-arg read-only tools, task-app-first."""
        tools = self._tool_schema_tools()
        mutation = self._app_mutation_flags(tools)
        scored = []
        for tool in tools:
            name = tool.name
            lower = name.lower()
            if name.startswith("SystemApp__") or lower in ("agentuserinterface__send_message_to_user",):
                continue
            inputs = getattr(tool, "inputs", {}) or {}
            if inputs:
                continue
            if not any(marker in lower for marker in ("list_", "get_", "read_", "search_")):
                continue
            app = name.split("__", 1)[0]
            score = 0
            if mutation.get(app):
                score += 100
            if "saved" in lower or "all_" in lower or "current" in lower:
                score += 20
            if lower.startswith("list_"):
                score += 10
            if "contact" in lower:
                score -= 30  # Contacts is rarely the task's central state
            scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in scored]

    def _inspection_tool(self) -> str | None:
        """Prefer a no-argument list/search tool from a relevant app."""
        ranked = self._ranked_inspection_tools()
        return ranked[0] if ranked else None

    def _first_turn_state_bootstrap(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
        """On the very first step, force one read-only query so real state and
        legal IDs enter the context before the model acts."""
        for message in messages:
            if message.get("role") in ("tool_response", "tool-response") and "Observation" in str(message.get("content", "")):
                return None
        tool_name = self._inspection_tool()
        if tool_name is None:
            return None
        return tool_name, {}

    def _tool_schema_tools(self) -> list[Any]:
        all_tools = list(self.tools_provider().values())
        return [tool for tool in all_tools if tool.name != "SystemApp__get_current_time"]

    def __call__(
        self,
        messages: list[dict[str, Any]],
        stop_sequences=[],
        schema=None,
        additional_trace_tags=None,
    ) -> tuple[str, dict[str, Any]]:
        started = time.time()
        self._current_messages = messages
        if self.tool_catalog_path and not self._tool_catalog_written:
            self._dump_tool_catalog()
        self._last_observation_raw_chars = 0
        self._last_observation_transformed_chars = 0
        bootstrap = self._first_turn_state_bootstrap(messages)
        if bootstrap is not None:
            tool_name, tool_args = bootstrap
            if self.progress_path:
                with open(self.progress_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "t": time.time(),
                                "tool": tool_name,
                                "args": tool_args,
                                "requested_tool": None,
                                "state_bootstrap": True,
                                "observation_transform": self.observation_transform,
                                "observation_chars_raw": 0,
                                "observation_chars_transformed": 0,
                                "tokens": 0,
                                "usage_available": False,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            return self._render_react_action(tool_name, tool_args), {
                "completion_duration": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "state_bootstrap": True,
            }
        converted = self._convert_messages(messages)
        has_business_state = any(
            message.get("role") in ("tool_response", "tool-response")
            and "Observation" in str(message.get("content", ""))
            and not str(message.get("content", "")).startswith("ERROR")
            for message in messages
        )
        if converted and converted[-1]["role"] == "user":
            instructions = (
                "\n\n[SYSTEM]: You must call a tool in this step unless you are "
                "completely done. If done, call "
                "AgentUserInterface__send_message_to_user with the final answer. "
                "Before acting, inspect the current state with search/list tools. "
                "Never invent IDs or addresses; use values returned by tools. "
                "To advance the simulation clock call SystemApp__wait_for_notification; "
                "SystemApp__get_current_time does not advance time and should be called at most once."
            )
            if has_business_state:
                instructions += (
                    " You have already received the required app state. "
                    "Do not call time/status tools again unless a new event was just delivered. "
                    "Execute the remaining steps of the user's task now, using the exact IDs returned by the tools."
                )
            converted[-1]["content"] += instructions
        forced = self._force_progress(messages)
        if forced is not None and converted and converted[-1]["role"] == "user":
            converted[-1]["content"] += "\n[SYSTEM]: " + forced[2]
        payload = {
            "model": self.model,
            "messages": converted,
            "tools": self._tool_schema(),
            "temperature": 0,
            "max_tokens": 256,
        }
        response = self.client.post(
            self.endpoint + "/chat/completions", json=payload
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        metadata = {
            "completion_duration": time.time() - started,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "usage_available": bool(usage),
            "observation_transform": self.observation_transform,
            "observation_chars_raw": self._last_observation_raw_chars,
            "observation_chars_transformed": self._last_observation_transformed_chars,
        }
        tool_calls = choice.get("tool_calls") or []
        if tool_calls:
            call = tool_calls[0]
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            forced = self._force_progress(messages)
            forced_name, forced_args = None, None
            if forced is not None and forced[0]:
                forced_name, forced_args = forced[0], forced[1]
            final_name = forced_name if forced_name else name
            final_args = forced_args if forced_args is not None else args
            if self.progress_path:
                with open(self.progress_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "t": time.time(),
                                "tool": final_name,
                                "args": final_args,
                                "requested_tool": name if forced_name else None,
                                "forced_rewrite": bool(forced_name),
                                "observation_transform": self.observation_transform,
                                "observation_chars_raw": self._last_observation_raw_chars,
                                "observation_chars_transformed": self._last_observation_transformed_chars,
                                "tokens": metadata["total_tokens"],
                                "usage_available": metadata["usage_available"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            return self._render_react_action(final_name, final_args), metadata
        content = choice.get("content") or ""
        if "Action:" not in content:
            fallback = self._inspection_tool() or "SystemApp__wait_for_notification"
            fallback_args = {} if fallback.startswith("RentAFlat__") or "list_" in fallback else {"timeout": 10}
            if self.progress_path:
                with open(self.progress_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "t": time.time(),
                                "tool": fallback,
                                "args": fallback_args,
                                "requested_tool": None,
                                "fallback": True,
                                "observation_transform": self.observation_transform,
                                "observation_chars_raw": self._last_observation_raw_chars,
                                "observation_chars_transformed": self._last_observation_transformed_chars,
                                "tokens": metadata["total_tokens"],
                                "usage_available": metadata["usage_available"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            return self._render_react_action(fallback, fallback_args), metadata
        return content, metadata


class NativeToolAgentBuilder(AgentBuilder):
    def __init__(
        self,
        endpoint: str,
        model: str,
        progress_path: str | None = None,
        observation_transform: str = "decision_fields",
        tool_catalog_path: str | None = None,
    ):
        super().__init__()
        self.endpoint = endpoint
        self.model = model
        self.progress_path = progress_path
        self.observation_transform = observation_transform
        self.tool_catalog_path = tool_catalog_path

    def build(self, agent_config, env=None, mock_responses=None):
        agent = super().build(agent_config, env=env, mock_responses=mock_responses)

        def tools_provider():
            react = getattr(agent, "react_agent", None)
            return getattr(react, "tools", {}) or {}

        agent.react_agent.llm_engine = NativeToolEngine(
            tools_provider=tools_provider,
            endpoint=self.endpoint,
            model=self.model,
            progress_path=self.progress_path,
            observation_transform=self.observation_transform,
            tool_catalog_path=self.tool_catalog_path,
        )
        agent.max_iterations = 40
        agent.react_agent.max_iterations = 40
        agent.react_agent.total_iterations = 120
        return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data" / "gaia2" / "scenarios_local"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--out", default=str(ROOT / "logs" / "gaia2_canonical"))
    parser.add_argument("--scenario-timeout", type=int, default=600)
    parser.add_argument(
        "--observation-transform",
        choices=["raw_truncate_3500", "legacy_id_name", "decision_fields"],
        default="decision_fields",
        help="Observation transform for the C0A-08 diagnostic; keep fixed within a run.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = str(out_dir / "llm_progress.jsonl")

    scenarios_iterator = setup_scenarios_iterator(
        dataset_path=args.dataset,
        dataset_config=args.config,
        dataset_split=None,
        hf=None,
        hf_revision=None,
        load_completed_events=False,
        limit=args.limit,
    )

    def set_scenario_metadata():
        for scenario, completed_events in scenarios_iterator:
            if not hasattr(scenario, "config") or scenario.config is None:
                scenario.config = args.config
            if not hasattr(scenario, "has_a2a_augmentation"):
                scenario.has_a2a_augmentation = False
            yield scenario, completed_events

    scenarios_with_metadata = CountableIterator(
        set_scenario_metadata(), scenarios_iterator.total_count
    )
    multiplied = (
        multiply_scenarios_iterator(scenarios_with_metadata, args.num_runs)
        if args.num_runs > 1
        else scenarios_with_metadata
    )

    runner_config = MultiScenarioRunnerConfig(
        model=args.model,
        model_provider="local",
        agent="default",
        oracle=False,
        export=True,
        output_dir=str(out_dir),
        trace_dump_format="lite",
        endpoint=args.endpoint,
        max_concurrent_scenarios=1,
        executor_type="thread",
        enable_caching=False,
        judge_only=False,
        timeout_seconds=args.scenario_timeout,
        a2a_app_prop=0.0,
        use_custom_logger=True,
        simulated_generation_time_mode="measured",
        judge_engine_config=LLMEngineConfig(
            model_name=args.model,
            provider="local",
            endpoint=args.endpoint,
        ),
        log_level="INFO",
    )

    final_scenarios = preprocess_scenarios_iterator(multiplied, runner_config)
    runner = MultiScenarioRunner(
        agent_builder=NativeToolAgentBuilder(
            args.endpoint,
            args.model,
            progress_path=progress_path,
            observation_transform=args.observation_transform,
            tool_catalog_path=str(out_dir / "tool_catalog.json"),
        )
    )
    result = runner.run_with_events(
        runner_config,
        final_scenarios,
        progress_description=f"Running {args.config} scenarios (native-tool adapter)",
    )

    rows = []
    for (scenario_id, run_number), scenario_result in result.scenario_results.items():
        rows.append(
            {
                "scenario_id": scenario_id,
                "run_number": run_number,
                "success": scenario_result.success,
                "rationale": scenario_result.rationale,
                "exception": (
                    str(scenario_result.exception)
                    if scenario_result.exception
                    else None
                ),
                "duration_s": scenario_result.duration,
            }
        )
    summary = {
        "config": args.config,
        "model": args.model,
        "endpoint": args.endpoint,
        "observation_transform": args.observation_transform,
        "n_scenarios": len(rows),
        "n_success": sum(1 for r in rows if r["success"] is True),
        "n_failed": sum(1 for r in rows if r["success"] is False),
        "success_rate": (
            round(
                sum(1 for r in rows if r["success"] is True) / len(rows), 4
            )
            if rows
            else None
        ),
        "rows": rows,
    }
    (out_dir / "canonical_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
