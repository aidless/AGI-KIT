# Fast arithmetic eval via Ollama
import json, urllib.request, time, sys, statistics
from pathlib import Path

ROOT = Path("experiments") if not (Path()/"a").exists() else Path()
OUT = Path("logs/seeds_arith")
OUT.mkdir(parents=True, exist_ok=True)

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3:1.7b'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

ARITH = []
import random
rng = random.Random(42)
for _ in range(N):
    a = rng.randint(10, 99)
    b = rng.randint(2, 9)
    c = rng.randint(50, 200)
    expr = str(a) + '*' + str(b) + '+' + str(c)
    ans = a * b + c
    ARITH.append((expr, ans))

def query_ollama(prompt, model):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
        return out.get("response", "").strip()

def score(pred, gold):
    import re
    m = re.search(r'-?\d+', pred)
    if not m:
        return 0
    try:
        return int(m.group()) == gold
    except Exception:
        return 0

print(f"Model={MODEL}, N={N}")
results = []
t_total = time.time()
for expr, ans in ARITH:
    t0 = time.time()
    try:
        pred = query_ollama(f"Compute {expr}. Output just the number, nothing else.", MODEL)
    except Exception as e:
        pred = f"ERROR: {e}"
    dt = time.time() - t0
    s = score(pred, ans)
    results.append({"expr": expr, "gold": ans, "pred": pred[:80], "correct": bool(s), "sec": round(dt, 2)})
    print(f"  {expr} = {ans} | pred={pred[:40]!r} | ok={s} | {dt:.1f}s")
total = time.time() - t_total
correct = sum(1 for r in results if r["correct"])
acc = correct / max(1, len(results))
print(f"\nAccuracy: {correct}/{len(results)} = {100*acc:.1f}% in {total:.1f}s")

summary = {
    "model": MODEL,
    "n": N,
    "correct": correct,
    "accuracy": acc,
    "wall_clock_sec": round(total, 2),
    "avg_sec_per_q": round(total / max(1, N), 2),
    "per_question": results,
}
out_path = OUT / f"arith_{MODEL.replace(":","_").replace(".","_")}_n{N}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("Wrote", out_path)