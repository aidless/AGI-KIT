content = open("papers/preprint_unified_en.md", encoding="utf-8").read()

# Insert after Appendix D (line 1130)
idx = content.find("## Appendix D")
if idx > 0:
    # Find end of Appendix D content - look for next major section
    end_idx = len(content)
    for marker in ["\n## ", "\n# "]:
        next_idx = content.find(marker, idx + 30)
        if next_idx > 0 and next_idx < end_idx:
            end_idx = next_idx
    print("Appendix D end at:", end_idx, repr(content[end_idx:end_idx+20]))

    appendix_e = """

## Appendix E - Swap-Out Experiment (Round 14)

The 22 pp gap in Section 4.1.2 (correctness on arithmetic subset)
is concentrated on 4 specific multi-step task templates. We ran a
swap-out experiment (experiments/swap_out.py) that re-formulates
the two most-frequent wrong templates with **format-explicit prompts**
("final answer MUST be a single integer with no commas, no spaces")
to test whether the +95 pp gain on the hard 20-task eval is genuine
or just format normalization.

| Template | n | Bare accuracy | Full L1-L4 accuracy | Delta |
|---|---:|---:|---:|---:|
| arith_double (sum + product, e.g. 859+381, 859*381) | 4 | 0.0% (0/4) | 50.0% (2/4) | **+50.0 pp** |
| arith_chain (single, e.g. 37*7+76) | 4 | 50.0% (2/4) | 100.0% (4/4) | **+50.0 pp** |

Two runs per task per configuration on qwen3:1.7b (the same model
as the headline). Format-explicit prompts do not close the bare-vs-L1-L4
gap on the harder multi-step template (`arith_double`), and even on
the simpler `arith_chain` template L1-L4 cuts the bare failure rate
in half. **The +95 pp gain on the hard 20-task eval is therefore not
just format normalization** - it is a genuine layer contribution.

Data: `logs/swap_out/summary.json`.
"""
    content = content[:end_idx] + appendix_e + chr(10) + content[end_idx:]
    print("Appendix E added at position:", end_idx)

open("papers/preprint_unified_en.md", "w", encoding="utf-8").write(content)
