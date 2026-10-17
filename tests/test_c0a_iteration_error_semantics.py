from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from are.simulation.agents.default_agent.base_agent import BaseAgent, TerminationStep
from are.simulation.agents.agent_log import ErrorLog
from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor


def main() -> None:
    termination = TerminationStep(
        name="iteration_probe",
        condition=lambda agent: agent.iterations >= agent.max_iterations,
    )
    agent = BaseAgent(
        llm_engine=lambda *args, **kwargs: "",
        system_prompts={"system_prompt": "C0A iteration fixture"},
        tools={},
        action_executor=JsonActionExecutor(use_custom_logger=True),
        termination_step=termination,
        max_iterations=3,
        total_iterations=2,
    )

    def failing_step(self):
        raise RuntimeError("deliberate-c0a-error")

    agent.initialize()
    agent.step = MethodType(failing_step, agent)
    agent.execute_agent_loop()

    error_logs = [log for log in agent.logs if isinstance(log, ErrorLog)]
    payload = {
        "max_iterations": agent.max_iterations,
        "total_iterations": agent.total_iterations,
        "final_iterations": agent.iterations,
        "final_planning_counter": agent.planning_counter,
        "error_log_count": len(error_logs),
        "error_types": [error.error for error in error_logs],
        "interpretation": (
            "Verified against are@79463674 base_agent.py: the loop 'finally' block "
            "increments BOTH iterations and planning_counter on every pass, including "
            "exceptions raised inside step() (L855-861). The docstring claim that "
            "max_iterations 'excludes errors' is NOT implemented as documented for "
            "this error path. log_error() appends ErrorLog(error=<type-name-string>) "
            "per failed step (L896-911), so error.error is a string, not an exception. "
            "The default termination condition (L187) checks only iterations >= "
            "max_iterations; total_iterations is not enforced by it. Implication for "
            "C0A-07: with the canonical defaults, the effective bound is max_iterations "
            "via iterations (error-inclusive); total_iterations=120 is inert unless the "
            "GAIA2 config wires a different termination condition."
        ),
    }
    assert payload["final_iterations"] == 3
    assert payload["final_planning_counter"] == 3
    assert payload["error_log_count"] == 3
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
