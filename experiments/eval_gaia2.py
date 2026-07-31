"""eval_gaia2.py - GAIA / GAIA2 evaluation scaffold for local Agent"""
from __future__ import annotations
import argparse, json, re, sys, time, traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from full_agent import Agent, OllamaBackend, TransformersBackend, TOOLS

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def ascii_safe(s):
    if s is None:
        return ""
    return str(s).encode("ascii", "replace").decode("ascii")


def normalize_answer(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\s\u3000]+", " ", s)
    s = s.replace(",", "")
    s = re.sub(r"[^a-z0-9.+\-/ ]", "", s)
    return s.strip()


def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)


OK = "OK"
FAIL = "FAIL"
ERR = "ERR"


def run_one(agent, item):
    q = item.get("question") or item.get("query") or item.get("task") or str(item)
    gold = item.get("answer") or item.get("final_answer") or item.get("solution")
    record = {"question": ascii_safe(q), "gold": ascii_safe(gold), "ts": datetime.now().isoformat()}
    print("\n========== Q ==========")
    print(ascii_safe(q))
    print("[gold]", ascii_safe(gold))
    t0 = time.time()
    try:
        pred = agent.run(q)
        record["pred"] = ascii_safe(pred)
        record["seconds"] = round(time.time() - t0, 1)
        if gold is not None and str(gold).strip() != "":
            ok = exact_match(pred, gold)
            record["verdict"] = OK if ok else FAIL
            tag = OK if ok else FAIL
            print("[" + tag + "]")
        else:
            record["verdict"] = None
    except Exception as e:
        record["pred"] = "[ERR] " + ascii_safe(e)
        record["verdict"] = ERR
        traceback.print_exc()
    record["history_tail"] = agent.history[-20:]
    return record


def load_gaia_subset(subset, config, limit, split):
        from datasets import load_dataset
        print("[dataset] loading " + subset + " config=" + config + " split=" + split)
        ds = load_dataset(subset, config, split=split, streaming=False)
        n = min(limit, len(ds))
        print("[dataset] taking " + str(n) + " / " + str(len(ds)))
        return [ds[i] for i in range(n)]

def load_gaia(subset, limit, split):
    from datasets import load_dataset
    print("[dataset] loading " + subset + " split=" + split)
    ds = load_dataset(subset, split=split, streaming=False)
    n = min(limit, len(ds))
    print("[dataset] taking " + str(n) + " / " + str(len(ds)))
    return [ds[i] for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    p.add_argument("--model", default=None)
    p.add_argument("--subset", default="meta-agents-research-environments/gaia2")
    p.add_argument("--config", default="mini", help="dataset config (mini, adaptability, ambiguity, demo, execution, search, time)")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--split", default="test")
    p.add_argument("--single", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.backend == "ollama":
        llm = OllamaBackend(model=args.model or "qwen3:1.7b")
    else:
        llm = TransformersBackend(model=args.model or "Qwen/Qwen3-1.7B")
    agent = Agent(llm=llm, max_steps=10)

    out = args.out or str(LOG_DIR / ("eval-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl"))
    print("[log]", out)

    if args.single:
        results = [run_one(agent, {"question": args.single, "answer": None})]
    else:
        try:
            items = load_gaia_subset(args.subset, args.config, args.limit, args.split)
        except Exception as e:
            print("[ERR] dataset load failed:", ascii_safe(e))
            print("[hint] try --single 'your task' to test without dataset")
            return
        results = [run_one(agent, it) for it in items]

    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    judged = [r for r in results if r.get("verdict") in (OK, FAIL)]
    correct = sum(1 for r in judged if r["verdict"] == OK)
    total = len(judged)
    print("\n========== SUMMARY ==========")
    print("total:", len(results), " judged:", total, " correct:", correct,
          " acc:", round(correct / max(total, 1) * 100, 1), "%")


if __name__ == "__main__":
    main()