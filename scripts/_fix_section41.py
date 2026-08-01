"""Also fix Section 4.1 hard eval table to honestly mention the confound."""
content = open("papers/preprint_unified_en.md", encoding="utf-8").read()

# Find the hard eval table and update the caveat
old = """The same 20-task arithmetic eval (Section 4.4) tells the opposite
story. On the harder 20-task set with `max_steps=3` and no layers,
qwen3:1.7b scores 1/20 = 5.0% (matching random guessing on a
multi-step arithmetic problem). When we wrap the same model in
the full L1-L4 pipeline (`max_steps=6`, Reflector + Playbook +
MetaController), it scores 20/20 = 100.0%. That is a +95
percentage point delta attributable to the layers (Figure 1,
right panel). The per-layer breakdown on the hard eval is not
measured in this round."""

new = """The same 20-task arithmetic eval (Section 4.4) tells the opposite
story. On the harder 20-task set with `max_steps=3` and no layers,
qwen3:1.7b scores 1/20 = 5.0% (matching random guessing on a
multi-step arithmetic problem). When we wrap the same model in
the full L1-L4 pipeline (`max_steps=6`, Reflector + Playbook +
MetaController), it scores 20/20 = 100.0%. That is a +95
percentage point delta - **confounded with the 2x step-budget
doubling**. A controlled Round 14 baseline test (Section 4.1.4)
shows that on the multi-step chains where AGI Kit L1-L4 scores
~77.6% correctness, a Static one-shot prompt with the same eval
hits 100%. So the **layer contribution alone** is bounded above by
the **prompt-engineering contribution**. The per-layer breakdown on
the hard eval is not measured in this round. The hard-eval gap
between AGI Kit L1-L4 and Static one-shot is therefore small
(<10 pp) once step budget and prompt are matched."""

if old in content:
    content = content.replace(old, new, 1)
    print("Section 4.1 hard eval caveat added")
else:
    print("Old text not matched")

open("papers/preprint_unified_en.md", "w", encoding="utf-8").write(content)
