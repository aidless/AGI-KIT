"""ToolFactory trigger tasks - deliberately design tasks that force
the agent to repeat the same wrong tool name, triggering L4 ToolFactory.

Examples:
  - "Convert text to uppercase" -> ToolFactory should produce 'uppercase_text'
  - "Reverse a string" -> 'reverse_text'
  - "Count occurrences of 'a' in 'banana'" -> 'count_char'

These tasks have no existing tool in the registry, so Qwen3 will fail
repeatedly and ToolFactory should synthesize new ones.
"""
from __future__ import annotations

import random
from typing import List


TRIGGER_TASKS = [
    {
        "task": "Convert the text 'hello world' to uppercase and give the final answer.",
        "kind": "tool_factory_upper",
        "expected_tool": "uppercase_text",
        "gold": "HELLO WORLD",
    },
    {
        "task": "Reverse the string 'agentic' and report the result as final.",
        "kind": "tool_factory_reverse",
        "expected_tool": "reverse_text",
        "gold": "citenega",
    },
    {
        "task": "Count how many times the letter 'a' appears in the word 'banana', then final.",
        "kind": "tool_factory_count",
        "expected_tool": "count_char",
        "gold": "3",
    },
    {
        "task": "Calculate the square root of 144 using a tool, then final.",
        "kind": "tool_factory_sqrt",
        "expected_tool": "sqrt_value",
        "gold": "12",
    },
    {
        "task": "Get the length of the string 'supercalifragilisticexpialidocious' as final.",
        "kind": "tool_factory_length",
        "expected_tool": "string_length",
        "gold": "34",
    },
    {
        "task": "Repeat the word 'echo' 5 times and give the final answer.",
        "kind": "tool_factory_repeat",
        "expected_tool": "repeat_text",
        "gold": "echoechoechoechoecho",
    },
    {
        "task": "Check if 17 is a prime number using a tool, then final.",
        "kind": "tool_factory_isprime",
        "expected_tool": "is_prime",
        "gold": "true",
    },
    {
        "task": "Compute the factorial of 6 with a tool, then final.",
        "kind": "tool_factory_factorial",
        "expected_tool": "factorial",
        "gold": "720",
    },
]


def get_trigger_tasks(n: int = 8, seed: int = 0) -> List[dict]:
    rng = random.Random(seed)
    pool = list(TRIGGER_TASKS)
    rng.shuffle(pool)
    return pool[:n]