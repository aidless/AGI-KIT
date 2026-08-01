"""Round 14 audit: verify all claims."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
from pathlib import Path

content = open("papers/preprint_unified_en.md", encoding="utf-8").read()

print("=" * 60)
print("ROUND 14 AUDIT")
print("=" * 60)

print()
print("FACTUAL CHECKS:")

# llama3.2:1b data
llama_bare = json.load(open("logs/cross_model_bare_llama1b/summary.json", encoding="utf-8"))
llama_full = json.load(open("logs/cross_model_layers_llama1b/summary.json", encoding="utf-8"))
print("Llama3.2:1b bare: " + str(llama_bare["correct"]) + "/" + str(llama_bare["n_tasks"]) + " = " + str(llama_bare["accuracy"]))
print("Llama3.2:1b full: " + str(llama_full["full_l1_l4"]["correct"]) + "/" + str(llama_full["n_tasks"]) + " = " + str(llama_full["full_l1_l4"]["accuracy"]))

# Swap-out data
swap = json.load(open("logs/swap_out/summary.json", encoding="utf-8"))
print("Swap-out by_category:")
for cat, conf in swap["by_category"].items():
    print("  " + cat + ": bare=" + str(round(conf["bare"]*100, 1)) + "% full=" + str(round(conf["full_l1_l4"]*100, 1)) + "%")

# Verify paper claims
print()
print("CLAIM CROSS-CHECKS:")
for pat, desc in [
    ("Llama-3.2-1B goes from **0%** bare", "abstract llama claim"),
    ("+100 pp vs its bare", "Section 4.1 llama claim"),
    ("Format-explicit prompts do not close", "Appendix E swap-out claim"),
    ("OPRO-style", "OPRO negative result claim"),
    ("77.6% correctness", "Section 4.1.2 retroactive claim"),
]:
    cnt = content.count(pat)
    print("  " + pat + ": " + str(cnt) + "x (" + desc + ")")

# Verify no fabricated numbers
print()
print("FABRICATED NUMBERS (should be 0):")
for pat, desc in [
    ("+38 pp", "old fabricated delta"),
    ("p<0.01", "fake t-test"),
    ("t=14.6", "fake t-test stat"),
    ("30% static baseline", "fabricated"),
]:
    cnt = content.count(pat)
    marker = "BAD" if cnt > 0 else "OK"
    print("  " + marker + " " + pat + ": " + str(cnt) + "x (" + desc + ")")

# Verify all numbers are documented
print()
print("DATA INTEGRITY:")
print("  ablation 5 configs all 77.8%: " + str(all(
    round(json.load(open(f"logs/ablation/{c}/summary.json", encoding="utf-8"))["success_rate"]*100, 1) == 77.8
    for c in ["static", "l1_only", "l1_l2", "l1_l2_l3", "full"])))
