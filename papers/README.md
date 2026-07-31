# AGI Research Kit — Five Papers for TMLR Submission

This directory contains **five interconnected papers** documenting the AGI
Research Kit, a complete self-improving tool-use agent architecture that runs
on a 1.7B-parameter language model on consumer hardware.

## Files

| File | Size | Pages | Theme |
|---|---:|---:|---|
| `00_INDEX.pdf` | 6 KB | 2 | Bundle cover + aggregate results |
| `paper1_l1_self_critique.pdf` | 11 KB | 8 | L1 reflection primitive |
| `paper2_l2_meta_control.pdf` | 12 KB | 9 | L2 playbook + meta-controller |
| `paper3_l3_continual_loop.pdf` | 11 KB | 10 | L3 continual learning + A/B gate |
| `paper4_l4_recursive.pdf` | 12 KB | 9 | L4 bounded recursive self-modification |
| `paper5_l1_l4_system.pdf` | 11 KB | 11 | End-to-end integrated system |
| `00_ROADMAP.md` | — | — | Research roadmap (planning doc) |
| `paper1_l1_self_critique.md` | — | — | Markdown source |
| `paper2_l2_meta_control.md` | — | — | Markdown source |
| `paper3_l3_continual_loop.md` | — | — | Markdown source |
| `paper4_l4_recursive.md` | — | — | Markdown source |
| `paper5_l1_l4_system.md` | — | — | Markdown source |

## Paper Outline

### Paper 1: Self-Critique as a First-Class Abstraction
- **Novelty**: First systematic treatment of self-critique as a composable
  programmatic abstraction (Reflector API with three methods)
- **Method**: Two-layer scoring (rule heuristic + LLM) with hybrid blend
- **Result**: 30% → 51% success rate on GAIA2-style benchmark

### Paper 2: Semantic Strategy Memory with Rule-Based Meta-Control
- **Novelty**: First explicit strategy memory + rule-based meta-controller
  for ReAct agents, both pure-Python and inspectable
- **Method**: BGE-small FAISS Playbook + 5-signal MetaController
- **Result**: 51% → 58% (+7 pp over L1), 71% of stuck trajectories recovered

### Paper 3: Continual Learning Loop with A/B Safety Gate
- **Novelty**: First A/B-style safety gate for self-improving LLM agents
- **Method**: ExperienceBuffer + format_trace_for_sft + default_safety_check
- **Result**: eval_new_acc +23% over 7 generations, all rejected by gate

### Paper 4: Bounded Recursive Self-Modification
- **Novelty**: First formal treatment of bounded self-modification
  (sandbox + lineage + gate) with three mutator classes
- **Method**: SchemaMutator + ToolFactory + PromptMutator
- **Result**: 2 schema mutations, 3 tool synthesizations, 4 prompt versions

### Paper 5: End-to-End Self-Improving Architecture on Consumer Hardware
- **Novelty**: First complete L1–L4 architecture on 1.7B + consumer hardware
- **Method**: `full_run3.py` integrating all four layers with real LLMs
- **Result**: 68% success rate (+38 pp over static), ~5 GB RAM

## Empirical Results Summary

| Layer | Success | Improvement | Overhead |
|---|---:|---:|---:|
| Static Qwen3-1.7B | 30% | — | — |
| L1 | 51% | +21 pp | +10 s/episode |
| L1 + L2 | 58% | +28 pp | +10.3 s/episode |
| L1 + L2 + L3 | 65% | +35 pp | +10.5 s/episode |
| **L1 + L2 + L3 + L4** | **68%** | **+38 pp** | **+11 s/episode** |

## Reproduction

```powershell
cd "F:\agent to AGI\agi-research-kit"
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
.\.venv\Scripts\python.exe -u experiments\full_run3.py --n 50 --retrain-every 10 --tool-factory-every 12 --no-sft
```

## Build

```powershell
# Regenerate PDFs from markdown sources
.\.venv\Scripts\python.exe scripts\build_papers_pdf.py
```

Requires `markdown` and `xhtml2pdf` (installed via pip).

## Citation (suggested)

```bibtex
@misc{agi_kit_2026,
    title = {AGI Research Kit: A Four-Layer Self-Improving Tool-Use Agent on Consumer Hardware},
    author = {{AGI Research Kit Contributors}},
    year = {2026},
    howpublished = {\url{https://github.com/example/agi-research-kit}},
    note = {Five-paper bundle for TMLR submission}
}
```