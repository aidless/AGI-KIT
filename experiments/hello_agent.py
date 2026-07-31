"""hello_agent.py - AGI research minimal tool-use agent"""
from __future__ import annotations
import argparse, json, re, time
from dataclasses import dataclass, field
from typing import Callable

TOOLS = {}

def tool(name, desc, params):
    def deco(fn):
        TOOLS[name] = {"desc": desc, "params": params, "fn": fn}
        return fn
    return deco

@tool("calculator", "arithmetic expression", {"expr": "str"})
def calculator(expr):
    import math
    if not re.fullmatch(r"[\d\s+\-*/().%^|&]+", expr):
        return "err: bad chars"
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return "err: " + str(e)

@tool("read_file", "read local text file", {"path": "str", "max_chars": "int=2000"})
def read_file(path, max_chars=2000):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars) or "(empty)"
    except Exception as e:
        return "err: " + str(e)

@tool("echo", "echo string back", {"msg": "str"})
def echo(msg):
    return msg

SYSTEM_PROMPT_TPL = "You are an AGI research assistant. Local ReAct Agent.\nTools:\n{tool_list}\n\nOutput format: strict JSON in ```json ... ``` block.\nCall tool: {{\"tool\": \"<name>\", \"args\": {{...}}}}\nFinal answer: {{\"final\": \"<answer>\"}}"

def render_system_prompt():
    lines = []
    for n, t in TOOLS.items():
        lines.append("- " + n + ": " + t["desc"] + "  params " + str(t["params"]))
    return SYSTEM_PROMPT_TPL.format(tool_list="\n".join(lines))

class LLM:
    def chat(self, messages, max_tokens=512, temperature=0.0):
        raise NotImplementedError

class OllamaBackend(LLM):
    def __init__(self, model="qwen3:1.7b"):
        import ollama
        self.client = ollama.Client(host="http://127.0.0.1:11434")
        self.model = model
    def chat(self, messages, max_tokens=512, temperature=0.0):
        r = self.client.chat(model=self.model, messages=messages, options={"num_predict": max_tokens, "temperature": temperature})
        return r["message"]["content"]

class TransformersBackend(LLM):
    def __init__(self, model="Qwen/Qwen3-1.7B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
    def chat(self, messages, max_tokens=512, temperature=0.0):
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)

def parse_action(text):
    m = JSON_RE.search(text)
    if m:
        cand = m.group(1)
    else:
        m2 = re.search(r"\{.*\}", text, re.S)
        if not m2:
            return None
        cand = m2.group(0)
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None

@dataclass
class Agent:
    llm: LLM
    max_steps: int = 8
    history: list = field(default_factory=list)

    def run(self, task):
        self.history = [
            {"role": "system", "content": render_system_prompt()},
            {"role": "user", "content": task},
        ]
        for step in range(1, self.max_steps + 1):
            print("\n--- step", step, "of", self.max_steps, "---")
            t0 = time.time()
            reply = self.llm.chat(self.history)
            _llm = reply[:200].encode("ascii", "replace").decode("ascii")
            print("[LLM", round(time.time()-t0, 1), "s]", _llm)
            action = parse_action(reply)
            if action is None:
                self.history += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "Output JSON please"},
                ]
                continue
            if "final" in action:
                _final = str(action["final"]).encode("ascii", "replace").decode("ascii")
                print("[FINAL]", _final)
                return action["final"]
            tname = action.get("tool")
            targs = action.get("args", {}) or {}
            if tname not in TOOLS:
                obs = "unknown tool: " + str(tname)
            else:
                try:
                    obs = TOOLS[tname]["fn"](**targs)
                except Exception as e:
                    obs = "tool error: " + str(e)
            _obs = str(obs)[:200].encode("ascii", "replace").decode("ascii")
            print("[OBS]", _obs)
            self.history += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "Observation: " + str(obs)},
            ]
        return "[agent] max steps reached"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    p.add_argument("--model", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--max-steps", type=int, default=6)
    args = p.parse_args()
    if args.backend == "ollama":
        llm = OllamaBackend(model=args.model or "qwen3:1.7b")
    else:
        llm = TransformersBackend(model=args.model or "Qwen/Qwen3-1.7B")
    agent = Agent(llm=llm, max_steps=args.max_steps)
    task = args.task or "Use calculator to compute 2 to the power of 10, then echo the answer as final"
    print("\n[TASK]", task)
    print("[RESULT]", agent.run(task))

if __name__ == "__main__":
    main()