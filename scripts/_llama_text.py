content = open("papers/preprint_unified_en.md", encoding="utf-8").read()

# Find by raw byte pattern
old_bytes = b"Models at <2B all collapse to 5% " + b"\xe2\x80\x94" + b"?the"
print("Match:", old_bytes in content.encode("utf-8"))

old_str = "Models at <2B all collapse to 5% \u2014?the\n   same as random guessing on arithmetic with retries disabled."
print("Str match:", old_str in content)

new_str = "Models at <2B all collapse to <=5% in bare mode (qwen3:0.6b,\n   qwen3:1.7b: 5%, llama3.2:1b: 0%). But **Llama-3.2-1B goes from\n   0% to 100% with L1-L4**, showing that the bare-mode failure is\n   not a fundamental capability ceiling - L1 reflection unlocks\n   the latent arithmetic ability even on a 1.2B model."

if old_str in content:
    content = content.replace(old_str, new_str, 1)
    print("Section 4.4 text updated")
else:
    print("Still not found")

open("papers/preprint_unified_en.md", "w", encoding="utf-8").write(content)
