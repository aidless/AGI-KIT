"""Architecture-level baselines on the 20-task hard arithmetic eval.

Lightweight but faithful reproductions of three architectural patterns, all on
the same 20 tasks, the same Ollama model (qwen3:1.7b), the same calculator tool,
and the same 6-step budget:

- Reflexion: verbal RL with up to 3 attempts and an explicit reflection step.
- Voyager-lite: skill library with embedding retrieval; skills added after success.
- MetaGPT-lite: PM -> Engineer -> Reviewer role loop with corrective feedback.

The goal is a head-to-head *architecture* comparison on identical hardware and
tasks, not a claim that these match the original papers' full implementations.
"""
from __future__ import annotations

import json
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agi_kit.llms.base import LLMMessage, MessageRole
from agi_kit.llms.ollama import OllamaBackend

from cross_model_with_layers import TASKS


def normalized_equal(prediction, gold) -> bool:
    left, right = str(prediction).strip(), str(gold).strip()
    if left == right:
        return True
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def calculator(expr: str) -> str:
    import math

    if not re.fullmatch(r"[\d\s+\-*/().%^|&]+", expr):
        return "err: bad chars"
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math}))
    except Exception as exc:
        return "err: " + str(exc)


def parse_action(text: str) -> tuple[str, dict] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if "final" in data:
            return ("final", {"final": str(data["final"])})
        if "tool" in data and isinstance(data.get("args"), dict):
            return ("tool", {"tool": str(data["tool"]), "args": data["args"]})
    return None


SYSTEM_TOOL = (
    "You are a local tool-use agent. You have one tool:\n"
    "- calculator(expr): evaluate an arithmetic expression "
    "(supports + - * / ** % ( ) ^).\n"
    "Reply with exactly one JSON object, either "
    '{"tool":"calculator","args":{"expr":"..."}} or {"final":"..."}.'
)


def task_prompt(task_text: str) -> str:
    return task_text + "\nReturn the final numeric answer."


def run_attempt(
    llm,
    task_text: str,
    max_steps: int = 6,
    extra_system: str = "",
    extra_history: list[LLMMessage] | None = None,
    max_tokens: int = 512,
) -> dict:
    history = list(extra_history or [])
    if not history or history[0].role != MessageRole.SYSTEM:
        history.insert(0, LLMMessage(role=MessageRole.SYSTEM, content=SYSTEM_TOOL + extra_system))
    history.append(LLMMessage(role=MessageRole.USER, content=task_prompt(task_text)))
    steps = []
    for _ in range(max_steps):
        raw = ""
        for _retry in range(3):
            response = llm.chat(history, max_tokens=max_tokens, temperature=0)
            raw = response.content or ""
            if raw.strip():
                break
        if not raw.strip():
            steps.append({"raw": "", "parsed": None, "retried": True})
            return {"final": None, "steps": steps, "ok": False, "reason": "empty_response"}
        parsed = parse_action(raw)
        steps.append({"raw": raw[:500], "parsed": parsed})
        if parsed is None:
            return {"final": raw.strip(), "steps": steps, "ok": False}
        kind, payload = parsed
        if kind == "tool":
            if payload["tool"] != "calculator":
                obs = "err: unknown tool"
            else:
                obs = calculator(str(payload["args"].get("expr", "")))
            history.append(LLMMessage(role=MessageRole.ASSISTANT, content=raw))
            history.append(
                LLMMessage(role=MessageRole.USER, content="Observation: " + obs)
            )
            continue
        final = payload["final"]
        return {"final": final, "steps": steps, "ok": True}
    return {"final": None, "steps": steps, "ok": False, "reason": "max_steps"}


def run_static(llm, task_text: str) -> dict:
    return run_attempt(llm, task_text)


def run_reflexion(llm, task_text: str) -> dict:
    history: list[LLMMessage] = []
    reflections: list[str] = []
    for attempt in range(3):
        extra = ""
        if reflections:
            extra = "\n\nPrevious reflections:\n- " + "\n- ".join(reflections)
        result = run_attempt(
            llm, task_text, extra_system=extra, extra_history=history[:1] if history else None
        )
        result["attempt"] = attempt + 1
        if result.get("ok") and result.get("final") is not None:
            result["reflections"] = reflections
            return result
        # Generate reflection
        history = history[:1] if history else []
        trace = "\n".join(
            f"step {i}: {step['raw']}" for i, step in enumerate(result["steps"])
        )
        prompt = (
            f"Task: {task_text}\nYour previous attempt did not produce a correct final answer.\n"
            f"Trace:\n{trace}\nFinal: {result.get('final')}\n"
            "Reflect briefly: what went wrong and what should you do differently? "
            "Keep it to two sentences."
        )
        reflection = llm.chat(
            [LLMMessage(role=MessageRole.USER, content=prompt)], max_tokens=512
        ).content.strip()
        reflections.append(reflection)
    result["reflections"] = reflections
    result["ok"] = False
    return result


class SkillLibrary:
    """Voyager-lite skill library.

    Retrieval uses sentence-transformers embeddings when the frozen
    environment provides a working install; otherwise it falls back to
    TF-IDF cosine retrieval (sklearn) so the run is reproducible on this
    machine without a network fetch. The retrieval mechanism is not the
    claim being tested; the skill-library loop is.
    """

    def __init__(self):
        self.skills: list[str] = []
        self._model = None
        self._encoder = None
        self._vectorizer = None

    def _ensure_encoder(self):
        if self._encoder is None and self._vectorizer is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(
                    "BAAI/bge-small-en-v1.5", device="cpu"
                )
            except Exception as exc:  # broken env: fall back to TF-IDF
                self._encoder = False
                print(f"[voyager] embedding unavailable ({exc.__class__.__name__}); using TF-IDF retrieval", flush=True)

    def _embed(self, texts: list[str]):
        self._ensure_encoder()
        if self._encoder:
            return self._encoder.encode(texts, normalize_embeddings=True)
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer()
            self._skill_matrix = self._vectorizer.fit_transform(self.skills)
        else:
            self._skill_matrix = self._vectorizer.transform(self.skills)
        return self._vectorizer.transform(texts)

    def retrieve(self, task_text: str, k: int = 2) -> list[str]:
        if not self.skills:
            return []
        if self._encoder is None and self._vectorizer is None:
            self._ensure_encoder()
        if self._encoder:
            query_emb = self._embed([task_text])[0]
            skill_embs = self._embed(self.skills)
            scores = skill_embs @ query_emb
        else:
            q = self._embed([task_text])
            import numpy as np

            qd = q.toarray()[0]
            sd = self._skill_matrix.toarray()
            scores = np.asarray([float(qd @ s) for s in sd])
        order = scores.argsort()[::-1][:k]
        return [self.skills[i] for i in order]

    def add(self, skill: str):
        skill = skill.strip()
        if skill and skill not in self.skills:
            self.skills.append(skill)


def run_voyager(llm, task_text: str, library: SkillLibrary) -> dict:
    skills = library.retrieve(task_text)
    extra_system = ""
    if skills:
        extra_system = "\n\nReusable skills from your library:\n- " + "\n- ".join(skills)
    result = run_attempt(llm, task_text, extra_system=extra_system)
    result["skills_used"] = skills
    if result.get("ok") and result.get("final") is not None:
        skill_prompt = (
            f"Task: {task_text}\nYou solved it with final answer {result['final']}.\n"
            "Write one short reusable skill (one sentence) capturing the strategy "
            "that worked."
        )
        skill = llm.chat(
            [LLMMessage(role=MessageRole.USER, content=skill_prompt)], max_tokens=512
        ).content.strip()
        library.add(skill)
        result["skill_added"] = skill
    return result


def run_metagpt(llm, task_text: str) -> dict:
    plan = llm.chat(
        [
            LLMMessage(
                role=MessageRole.SYSTEM,
                content="You are the Product Manager of a tiny agent team. "
                "Write a 2-3 step plan for the task using the calculator tool.",
            ),
            LLMMessage(role=MessageRole.USER, content=task_text),
        ],
        max_tokens=512,
    ).content.strip()
    reviewer_feedback = ""
    for cycle in range(2):
        extra = (
            "\n\nPlan from PM:\n" + plan
            + ("\n\nReviewer feedback from last cycle:\n" + reviewer_feedback if reviewer_feedback else "")
        )
        result = run_attempt(llm, task_text, extra_system=extra)
        result["plan"] = plan
        result["cycle"] = cycle + 1
        final = result.get("final")
        if result.get("ok") and final is not None:
            review = llm.chat(
                [
                    LLMMessage(
                        role=MessageRole.SYSTEM,
                        content="You are the Reviewer. Given the plan and final answer, "
                        "reply with exactly PASS if the final answer is a number, "
                        "otherwise give corrective feedback in one sentence.",
                    ),
                    LLMMessage(
                        role=MessageRole.USER,
                        content=f"Plan:\n{plan}\nFinal answer: {final}",
                    ),
                ],
                max_tokens=512,
            ).content.strip()
            result["review"] = review
            if "PASS" in review.upper() and final is not None:
                return result
            reviewer_feedback = review
    result["ok"] = False
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=("static", "reflexion", "voyager", "metagpt"), required=True)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "logs" / "baselines_architectural"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / f"progress_{args.config}.log"
    llm = OllamaBackend(model=args.model, timeout_s=120, seed=20260807)
    library = SkillLibrary()
    rows = []
    started = time.time()
    for idx, (task_type, task_text, gold) in enumerate(TASKS[args.start : args.start + args.limit]):
        t0 = time.time()
        if args.config == "static":
            result = run_static(llm, task_text)
        elif args.config == "reflexion":
            result = run_reflexion(llm, task_text)
        elif args.config == "voyager":
            result = run_voyager(llm, task_text, library)
        else:
            result = run_metagpt(llm, task_text)
        correct = (
            normalized_equal(result.get("final"), gold)
            if result.get("final") is not None
            else False
        )
        rows.append(
            {
                "index": idx,
                "task_type": task_type,
                "task": task_text,
                "gold": gold,
                "final": result.get("final"),
                "correct": correct,
                "attempts": result.get("attempt"),
                "cycles": result.get("cycle"),
                "seconds": round(time.time() - t0, 1),
                "details": result,
            }
        )
        line = (
            f"[{idx + 1}/{len(TASKS[args.start:args.start + args.limit])}] "
            f"{'OK ' if correct else 'FAIL'} gold={gold} final={result.get('final')} "
            f"{(time.time() - started):.0f}s"
        )
        print(line, flush=True)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    summary = {
        "config": args.config,
        "model": args.model,
        "n": len(rows),
        "correct": sum(1 for r in rows if r["correct"]),
        "accuracy": round(sum(1 for r in rows if r["correct"]) / len(rows), 4),
        "seconds": round(time.time() - started, 1),
        "rows": rows,
    }
    (out_dir / f"summary_{args.config}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("config", "n", "correct", "accuracy", "seconds")}))


if __name__ == "__main__":
    main()
