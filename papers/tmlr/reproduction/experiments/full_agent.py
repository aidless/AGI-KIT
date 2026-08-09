"""full_agent.py - AGI research agent with browser + RAG + tools"""
from __future__ import annotations
import argparse, json, re, sys, time, os
from dataclasses import dataclass, field
from typing import Callable

TOOLS = {}

def tool(name, desc, params):
    def deco(fn):
        TOOLS[name] = {"desc": desc, "params": params, "fn": fn}
        return fn
    return deco

# ============== 鍩虹宸ュ叿 ==============
@tool("calculator", "Evaluate an arithmetic expression (supports +,-,*,/,**,%,(),^)", {"expr": "str expression"})
def calculator(expr: str) -> str:
    import math
    if not re.fullmatch(r"[\d\s+\-*/().%^|&]+", expr):
        return "err: bad chars"
    try:
        return str(eval(expr, {"__builtins__": {}}, {"math": math}))
    except Exception as e:
        return "err: " + str(e)

@tool("read_file", "Read a local text file", {"path": "str abs path", "max_chars": "int=3000"})
def read_file(path: str, max_chars: int = 3000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars) or "(empty)"
    except Exception as e:
        return "err: " + str(e)

@tool("read_pdf", "Read a PDF file (returns first N pages text)", {"path": "str", "max_pages": "int=5"})
def read_pdf(path: str, max_pages: int = 5) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(path)
        out = []
        for i, p in enumerate(r.pages[:max_pages]):
            out.append("[page " + str(i+1) + "]\n" + p.extract_text())
        return "\n".join(out) or "(no text)"
    except Exception as e:
        return "err: " + str(e)

@tool("echo", "Echo a string back (sanity test)", {"msg": "str"})
def echo(msg: str) -> str:
    return msg

@tool("list_dir", "List files in a directory", {"path": "str dir path", "pattern": "str glob optional e.g. *.pdf"})
def list_dir(path: str, pattern: str = "*") -> str:
    import glob
    try:
        if pattern and pattern != "*":
            files = glob.glob(os.path.join(path, pattern))
        else:
            files = glob.glob(os.path.join(path, "*"))
        return "\n".join(os.path.basename(f) for f in files[:50]) or "(empty)"
    except Exception as e:
        return "err: " + str(e)

@tool("shell", "Run a short shell command (pwd/ls/cat/type etc, no pipes)", {"cmd": "str short command"})
def shell(cmd: str) -> str:
    import subprocess
    if any(x in cmd for x in ["|", ">", "<", "&", ";"]):
        return "err: disallowed chars"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return (r.stdout + r.stderr).strip()[:2000] or "(no output)"
    except Exception as e:
        return "err: " + str(e)

# ============== 娴忚鍣ㄥ伐鍏?==============
_BROWSER = None
_PAGE = None

def _ensure_browser():
    global _BROWSER, _PAGE
    if _BROWSER is not None:
        return
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    _BROWSER = pw.chromium.launch(headless=True)
    _PAGE = _BROWSER.new_page()

@tool("web_search", "Search Bing, returns top N titles+urls+snippets", {"query": "str", "top_k": "int=5"})
def web_search(query: str, top_k: int = 5) -> str:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
            page.set_extra_http_headers({"User-Agent": ua})
            page.goto("https://www.bing.com/search?q=" + query.replace(" ", "+"), timeout=20000)
            page.wait_for_timeout(2500)
            results = page.locator("li.b_algo").all()
            out = []
            for r in results[:top_k]:
                try:
                    t = r.locator("h2 a").first.text_content(timeout=500).strip()
                except Exception:
                    t = ""
                try:
                    u = r.locator("h2 a").first.get_attribute("href", timeout=500)
                except Exception:
                    u = ""
                try:
                    s = r.locator(".b_caption p, p").first.text_content(timeout=500).strip()
                except Exception:
                    s = ""
                if t or u:
                    out.append(("- " + t + "\n  " + (u or "") + "\n  " + s[:200]).strip())
            browser.close()
            return "\n\n".join(out) or "(no results)"
    except Exception as e:
        return "err: " + str(e)[:200]

@tool("web_fetch", "Fetch URL content as plain text", {"url": "str http url", "max_chars": "int=3000"})
def web_fetch(url: str, max_chars: int = 3000) -> str:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(1000)
            txt = page.evaluate("() => document.body.innerText")
            browser.close()
            return (txt or "").strip()[:max_chars] or "(empty)"
    except Exception as e:
        return "err: " + str(e)[:200]

# ============== RAG 宸ュ叿 ==============
_RAG = None

def _ensure_rag():
    global _RAG
    if _RAG is not None:
        return _RAG
    from rag_store import RagStore
    _RAG = RagStore()
    return _RAG

@tool("rag_add", "Index a file (txt/md/pdf) into local RAG store", {"path": "str"})
def rag_add(path: str) -> str:
    try:
        rag = _ensure_rag()
        n = rag.add_file(path)
        return "indexed " + str(n) + " chunks"
    except Exception as e:
        return "err: " + str(e)[:200]

@tool("rag_search", "Semantic search over indexed documents", {"query": "str", "top_k": "int=3"})
def rag_search(query: str, top_k: int = 3) -> str:
    try:
        rag = _ensure_rag()
        results = rag.search(query, top_k=top_k)
        if not results:
            return "(no match)"
        return "\n\n---\n\n".join(r["text"][:800] + "\n[src: " + r["source"] + "]" for r in results)
    except Exception as e:
        return "err: " + str(e)[:200]

@tool("rag_clear", "Clear the RAG index", {})
def rag_clear() -> str:
    try:
        rag = _ensure_rag()
        rag.clear()
        return "cleared"
    except Exception as e:
        return "err: " + str(e)[:200]

# ============== Agent 鍚庣 ==============
SYSTEM_PROMPT_TPL = (
    "You are an AGI research assistant. Local ReAct Agent.\n"
    "Tools:\n{tool_list}\n\n"
    "Output strict JSON in ```json ... ```.\n"
    "Tool call: {{\"tool\": \"<name>\", \"args\": {{...}}}}\n"
    "Final answer: {{\"final\": \"<answer>\"}}\n"
    "Be concise. Use tools step by step."
)

def render_system_prompt():
    lines = []
    for n, t in TOOLS.items():
        lines.append("- " + n + ": " + t["desc"] + " | params " + str(t["params"]))
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
        r = self.client.chat(model=self.model, messages=messages,
            options={"num_predict": max_tokens, "temperature": temperature})
        return r["message"]["content"]

class TransformersBackend(LLM):
    def __init__(self, model="Qwen/Qwen3-1.7B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def chat(self, messages, max_tokens=512, temperature=0.0):
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)

def parse_action(text):
    m = JSON_RE.search(text)
    if m:
        cand = m.group(1)
    else:
        m2 = re.search(r"\{[^{}]*\"tool\"[^{}]*\}|\{.*?\"final\".*?\}", text, re.S)
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
    max_steps: int = 10
    max_tokens: int = 160
    history: list = field(default_factory=list)

    def run(self, task):
        self.history = [
            {"role": "system", "content": render_system_prompt()},
            {"role": "user", "content": task},
        ]
        for step in range(1, self.max_steps + 1):
            print("\n--- step", step, "of", self.max_steps, "---")
            t0 = time.time()
            reply = self.llm.chat(self.history, max_tokens=self.max_tokens)
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
            _obs = str(obs)[:300].encode("ascii", "replace").decode("ascii")
            print("[OBS]", _obs)
            self.history += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "Observation: " + str(obs)[:2000]},
            ]
        return "[agent] max steps"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    p.add_argument("--model", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--max-steps", type=int, default=10)
    args = p.parse_args()
    if args.backend == "ollama":
        llm = OllamaBackend(model=args.model or "qwen3:1.7b")
    else:
        llm = TransformersBackend(model=args.model or "Qwen/Qwen3-1.7B")
    agent = Agent(llm=llm, max_steps=args.max_steps)
    task = args.task or "Use calculator to compute 2**10, then echo result, then final"
    print("\n[TASK]", task.encode("ascii", "replace").decode("ascii"))
    print("[RESULT]", agent.run(task))

if __name__ == "__main__":
    main()
