"""Add Novelty vs Prior Work sections to all 5 English papers.

Each section explicitly contrasts the paper''s contribution with closest
prior work, addressing Reviewer 2''s most common concern.
"""
from __future__ import annotations

from pathlib import Path

PAPERS_DIR = Path(__file__).parent.parent / "papers"

NOVELTY = {
    "paper1_l1_self_critique_en.md": """## 1.5 Novelty vs Prior Work

The closest prior works on agent self-critique are *Self-Refine* (Madaan
et al. 2023) and *Reflexion* (Shinn et al. 2023). Both embed self-critique
in the prompt and rely on the same LLM to evaluate its own output.
Our contribution differs in three ways:

1. **First-class API.** Self-Refine and Reflexion treat critique as a
   prompt strategy; we treat it as a discrete *module* (`Reflector`)
   with three named operations. This makes the critique step
   inspectable, replaceable, and composable.

2. **Two-layer scoring.** Self-Refine and Reflexion rely on a single
   LLM judgment. Our hybrid scheme (40% rule + 60% LLM) uses a fast
   heuristic as a prior, making it cheaper and more robust to LLM
   hallucination.

3. **Persistent logging.** Both prior works discard the critique after
   one episode. Our Reflector writes every reflection to a JSONL
   trace, enabling downstream analysis (Paper 2''s StrategyMiner,
   Paper 4''s PromptMutator).

In short: we move self-critique from *prompt trick* to *programmatic
abstraction*.

""",

    "paper2_l2_meta_control_en.md": """## 1.5 Novelty vs Prior Work

Three lines of prior work come closest to ours:

*Reflexion* (Shinn et al. 2023) uses verbal reinforcement in the prompt
but does not separate a *strategy memory* from a *meta-controller* —
both are entangled in the prompt context. This saturates the context
window and dies across runs.

*Voyager* (Wang et al. 2023) maintains a skill library in Minecraft,
tightly coupled to GPT-4 and to Minecraft''s API. Our Playbook is
language-model-agnostic and task-agnostic, indexed by open-source
embeddings (BGE-small).

*MetaGPT* (Hong et al. 2024) introduces a meta-controller but operates
*across* agents, not *within* a single ReAct loop. Our MetaController
is the first *per-step, single-agent* controller.

Our contribution: a small, fast, deterministic meta-layer that
operates *within* a single agent''s loop, is fully inspectable, and
requires no LLM sampling at decision time.

""",

    "paper3_l3_continual_loop_en.md": """## 1.5 Novelty vs Prior Work

The closest prior work on safety-gated model deployment is *A/B
testing* in production ML (Barakat 2021, Huyen 2022). Our contribution
adapts this to the *self-improving agent* setting, where the A/B gate
is between consecutive generations of the *same* agent (not just
between different models).

Three distinctions from prior continual learning work:

*Experience replay* (Mnih et al. 2015) is a structural idea we
re-purpose, but our loop adds a generation-tracking state and a
safety gate.

*Catastrophic-forgetting methods* (Kirkpatrick et al. 2017;
Rolnick et al. 2019) protect a fixed model against new data. Our
agent *intentionally* retrains on its own data and uses the gate to
gate the resulting model.

*Online learning theory* (Vovk et al. 2005) provides adaptive
thresholds (conformal prediction). We use a fixed threshold and
leave conformal-style adaptation as future work.

Our contribution: a *complete* continual loop with A/B safety gate
that runs on a 1.7B model on consumer hardware.

""",

    "paper4_l4_recursive_en.md": """## 1.5 Novelty vs Prior Work

Recursive self-modification has been studied in three communities:
*meta-learning* (Thrun & Pratt 1998), *AutoML* (Zoph & Le 2017; Real
et al. 2020), and *agent self-improvement* (Reflexion; Voyager). Our
contribution differs from each:

*Meta-learning* optimizes an outer-loop objective (e.g. learning
rate schedules) but does not modify the *agent''s own control
parameters at runtime*. We mutate `MetaControllerConfig` fields
thresholds, prompts, tools in a closed loop.

*AutoML* mutates *architecture* and *training pipelines*. We mutate
*agent control* (Schema, Tool, Prompt) and use the agent itself
(LLM) to perform the mutation. We are closer to *AutoML-Zero''s*
discovery of algorithms from primitives, but our primitives are
higher-level (thresholds, prompts, function bodies).

*Agent self-improvement* prior work uses *prompt-based* or
*fine-tuning-based* self-modification. We add a third option —
*programmatic* self-modification guarded by sandbox + lineage + gate.

Our contribution: a unified framework for *bounded* recursive
self-modification with three concrete mutator classes.

""",

    "paper5_l1_l4_system_en.md": """## 1.5 Novelty vs Prior Work

Two prior systems are most comparable to AGI Kit:

*Reflexion* (Shinn et al. 2023) is a GPT-4-class cloud agent with
verbal reinforcement. AGI Kit differs in three ways: (i) we run on
a 1.7B model on CPU; (ii) we provide a programmatic L1 layer, not
prompt tricks; (iii) we integrate L1–L4 with hard safety gating.

*Voyager* (Wang et al. 2023) is a GPT-4 Minecraft agent with a
skill library. AGI Kit differs in that: (i) our strategy memory is
semantic, not lexical; (ii) our meta-controller is rule-based, not
LLM-based; (iii) our continual loop is open-weight (Qwen3-1.7B), not
closed-API.

Our unique contribution: a *consumer-hardware* self-improving agent
that runs end-to-end (L1–L4) in <6 GB RAM and finishes a 50-episode
benchmark in under 17 minutes. Prior self-improving agents require
cloud-scale compute.

""",
}


def main():
    for fname, novelty in NOVELTY.items():
        path = PAPERS_DIR / fname
        if not path.exists():
            print(f"  {fname} not found")
            continue
        text = path.read_text(encoding="utf-8")
        marker = "## 2. Related Work"
        if marker not in text:
            print(f"  {fname} no Related Work section")
            continue
        if "## 1.5 Novelty vs Prior Work" in text:
            print(f"  {fname} already has Novelty section")
            continue
        text = text.replace(marker, novelty + marker)
        path.write_text(text, encoding="utf-8")
        print(f"  added Novelty to {fname}")


if __name__ == "__main__":
    main()