"""Find the 4 distinct task patterns that account for wrong cases in reeval."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
from pathlib import Path
import re
from collections import Counter

wrong_cases = []
for gen_dir in sorted(Path("logs/full_run3").glob("gen-*")):
    p = gen_dir / "samples.jsonl"
    if not p.exists():
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        msgs = r.get("messages", [])
        user_text = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        emitted = None
        for m in msgs:
            if m.get("role") == "assistant":
                content = m.get("content", "")
                for match in re.finditer(r"\`\`\`json\s*(\{.*?\})\s*\`\`\`", content, re.DOTALL):
                    try:
                        obj = json.loads(match.group(1))
                        if "final" in obj:
                            emitted = str(obj["final"])
                    except Exception:
                        continue
        if not emitted:
            continue
        exprs = re.findall(r"\d+[+\-*/]\d+(?:[+\-*/]\d+)*", user_text)
        if not exprs:
            continue
        expected = []
        for e in exprs:
            try:
                expected.append((e, eval(e)))
            except Exception:
                pass
        if not expected:
            continue
        exp_vals = [v for (_, v) in expected]
        got = [int(x) for x in re.findall(r"-?\d+", emitted)]
        matched = sum(1 for v in exp_vals if v in got)
        is_correct = (matched == len(exp_vals))
        if not is_correct:
            wrong_cases.append({
                "gen": gen_dir.name,
                "prompt": user_text[:200],
                "expected": exp_vals,
                "got": got,
                "emitted": emitted[:80],
            })

# Bucket by template (extract the first verb phrase as template key)
def template(prompt):
    # Strip numbers
    t = re.sub(r"\d+", "N", prompt)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

buckets = Counter()
for c in wrong_cases:
    buckets[template(c["prompt"])] += 1

print("=== Top templates by wrong-case count ===")
for tpl, n in buckets.most_common(10):
    print(str(n) + "x: " + tpl[:120])
