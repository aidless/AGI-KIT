"""Update cover letter to reflect the controlled finding."""
content = open("papers/COVER_LETTER.md", encoding="utf-8").read()

# Find the headline numbers block in cover letter
old = """- Layer ablation: all 5 configs score 77.8% on synthetic GAIA2 mini (eval saturates). On the harder 20-task arithmetic eval, bare scores **5.0%** and full L1-L4 scores **100.0%** - a **+95 percentage point** delta attributable to the layers (Section 4.1)."""

new = """- Layer ablation: all 5 configs score 77.8% on synthetic GAIA2 mini (eval saturates). On the harder 20-task arithmetic eval, bare (max_steps=3) = **5.0%**, full L1-L4 (max_steps=6) = **100.0%** - a **+95 percentage point** delta. We decomposed this into **+30 pp from step-budget doubling** and **+65 pp from L1-L4 at matched budget** (Section 4.1); a separate Round 14 baseline test shows Static one-shot with appropriate prompting is competitive with AGI Kit L1-L4 on multi-step tasks."""

if old in content:
    content = content.replace(old, new, 1)
    print("Cover letter updated with controlled finding")
else:
    print("Cover letter old text not matched")

open("papers/COVER_LETTER.md", "w", encoding="utf-8").write(content)
