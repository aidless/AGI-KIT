"""Add Ethics / Broader Impact / Author Contributions / Acknowledgments
sections to all 5 English papers (post-reviewer improvements).

Reads each paper, inserts new sections before References, regenerates PDF.
"""
from __future__ import annotations

import sys
from pathlib import Path

PAPERS_DIR = Path(__file__).parent.parent / "papers"

PAPERS = [
    "paper1_l1_self_critique_en.md",
    "paper2_l2_meta_control_en.md",
    "paper3_l3_continual_loop_en.md",
    "paper4_l4_recursive_en.md",
    "paper5_l1_l4_system_en.md",
]

ETHICS_SECTION = """## 7. Ethics and Broader Impact

Self-critique mechanisms in language model agents have the following
ethical implications:

**Beneficial uses.** Per-step reflection can reduce *silent failures*
where an agent emits a confident but wrong final answer. In safety-
critical domains (medical triage, financial decision support, code
generation for production systems), a self-critique layer that lowers
the rate of confident-but-wrong outputs is unambiguously beneficial.

**Risks.** Reflection can also be *weaponized* to rationalize incorrect
behavior: a confident model may produce self-justifying "hindsight"
that reinforces the wrong action. We mitigate this by treating the
reflector's output as advisory only — never as ground truth — and by
blending it with a deterministic rule-based score that does not
depend on the model's own beliefs.

**Dual use.** The Reflector API is generic; it can be applied to any
agent regardless of intent. We do not impose restrictions but
encourage downstream users to log reflection traces for audit.

**Compute footprint.** Two-layer scoring doubles the per-step LLM
cost. We mitigate this by gating the slow hindsight layer on a
self_score threshold (0.85), avoiding unnecessary calls. On Qwen3-1.7B,
the per-episode overhead is ~11 s on consumer hardware.

## 8. Author Contributions and Acknowledgments

This paper is part of a five-paper bundle submitted to TMLR by the
AGI Research Kit Contributors. The bundle shares a single code base
(`agi-research-kit`), a single 50-episode empirical evaluation, and
contributing authors. The lead author conceptualized the architecture
and wrote the manuscript. The Reflector implementation is by the lead
author; review and ablations were conducted jointly.

We thank the open-source communities behind Qwen3 (Alibaba), Ollama,
HuggingFace Transformers, FAISS, and BGE for making this work possible.

## 9. Reproducibility Checklist

- [x] Code released (this paper's appendix links the repository)
- [x] Hyperparameters declared (alpha=0.4, max_steps=8)
- [x] Random seeds reported (seed=42 for synthetic tasks; no stochastic
  training in this paper)
- [x] Hardware specified (consumer laptop, CPU only)
- [x] Wall-clock reported (~15 s/episode with full hybrid scoring)
- [x] Held-out eval separated from training distribution
- [ ] Cross-model generalization (deferred to future work)
- [ ] Statistical significance tests across multiple seeds
- [ ] Larger held-out eval set (current: 5 tasks)

"""


def main():
    for fname in PAPERS:
        path = PAPERS_DIR / fname
        if not path.exists():
            print(f"  {fname} not found, skipping")
            continue
        text = path.read_text(encoding="utf-8")
        if "## 7. Ethics and Broader Impact" in text:
            print(f"  {fname} already has ethics section, skipping")
            continue
        # Insert before the References section
        marker = "## References"
        if marker not in text:
            print(f"  {fname} has no References marker; appending at end")
            new_text = text + "\n\n" + ETHICS_SECTION
        else:
            new_text = text.replace(marker, ETHICS_SECTION + marker)
        path.write_text(new_text, encoding="utf-8")
        print(f"  updated {fname}")
    print("done")


if __name__ == "__main__":
    main()