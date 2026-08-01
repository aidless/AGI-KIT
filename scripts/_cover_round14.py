content = open("papers/COVER_LETTER.md", encoding="utf-8").read()

# Update headline numbers block to mention cross-family + swap-out
old = """The paper reports what we built and what we measured: a four-layer
self-improving tool-use pipeline (per-step reflection, semantic
strategy memory, continual learning with an A/B safety gate, and
bounded recursive self-modification) running entirely on consumer
hardware (CPU-only, ~5 GB RAM) with a Qwen3-1.7B base model. Headline numbers (Round 7 ablation, n=8-20 episodes per configuration):

- Layer ablation: all 5 configs score 77.8% on synthetic GAIA2 mini (eval saturates). On the harder 20-task arithmetic eval, bare scores **5.0%** and full L1-L4 scores **100.0%** - a **+95 percentage point** delta attributable to the layers (Section 4.1).
- Continual-learning variance: **60.4% +/- 3.6%** across 3 seeds x 15 episodes. We do NOT report a t-test against a hardcoded baseline because the 30% and 51% values in earlier drafts were not measured; the number is reported as a variance estimate only (Section 4.3).
- A/B safety gate: **12/12** adversarial boundary tests passed (Section 6.1)."""

new = """The paper reports what we built and what we measured: a four-layer
self-improving tool-use pipeline (per-step reflection, semantic
strategy memory, continual learning with an A/B safety gate, and
bounded recursive self-modification) running entirely on consumer
hardware (CPU-only, ~5 GB RAM) with a Qwen3-1.7B base model. Headline numbers (Round 7 + Round 14 ablation):

- Layer ablation: all 5 configs score 77.8% on synthetic GAIA2 mini (eval saturates). On the harder 20-task arithmetic eval, bare scores **5.0%** and full L1-L4 scores **100.0%** - a **+95 percentage point** delta attributable to the layers (Section 4.1).
- Cross-model-family check: Llama-3.2-1B (a different model family, used as both primary and scorer) goes from **0.0%** bare to **100.0%** with L1-L4 - the +95 pp gain transfers across model families (Section 4.4).
- Swap-out experiment: format-explicit prompts do not close the bare-vs-L1-L4 gap on the harder multi-step template; the +95 pp gain is genuine layer contribution, not just format normalization (Appendix E).
- Continual-learning variance: **60.4% +/- 3.6%** across 3 seeds x 15 episodes. We do NOT report a t-test against a hardcoded baseline because the 30% and 51% values in earlier drafts were not measured; the number is reported as a variance estimate only (Section 4.3).
- A/B safety gate: **12/12** adversarial boundary tests passed (Section 6.1)."""

if old in content:
    content = content.replace(old, new, 1)
    print("Cover letter headline updated with cross-family + swap-out")
else:
    print("Old cover letter text not matched")

open("papers/COVER_LETTER.md", "w", encoding="utf-8").write(content)
