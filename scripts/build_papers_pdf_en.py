"""Build English PDF for the AGI Kit unified preprint.

Produces:
  papers/preprint_unified_en.pdf
  papers/00_INDEX_en.pdf
  papers/COVER_LETTER_en.pdf

The 5-paper TMLR bundle has been consolidated into this single
preprint; the archived 5-paper assets live in papers/_deprecated/.
"""
from __future__ import annotations

from pathlib import Path
import markdown
from xhtml2pdf import pisa

PAPERS_DIR = Path(__file__).parent.parent / "papers"

# Same CSS as before, but include figure references
HTML_CSS = """
@page {
    size: A4;
    margin: 2cm 2cm 2.5cm 2cm;
}
body {
    font-family: 'Times', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #000;
}
h1 {
    font-size: 18pt;
    text-align: center;
    margin-top: 0.5cm;
    margin-bottom: 0.5cm;
    font-weight: bold;
    page-break-before: always;
    page-break-after: avoid;
}
h2 {
    font-size: 14pt;
    margin-top: 0.6cm;
    margin-bottom: 0.3cm;
    color: #003366;
    border-bottom: 1px solid #999;
    padding-bottom: 2pt;
}
h3 {
    font-size: 12pt;
    margin-top: 0.5cm;
    margin-bottom: 0.2cm;
    color: #003366;
}
h4 {
    font-size: 11pt;
    margin-top: 0.4cm;
    margin-bottom: 0.2cm;
    color: #555;
}
p {
    margin: 0.3em 0;
    text-align: justify;
}
ul, ol {
    margin: 0.3em 0;
    padding-left: 1.5em;
}
li { margin: 0.2em 0; }
code {
    font-family: 'Courier', 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f4f4f4;
    padding: 1pt 3pt;
}
pre {
    font-family: 'Courier', 'Courier New', monospace;
    font-size: 9pt;
    background: #f8f8f8;
    border: 1px solid #ddd;
    padding: 6pt;
    margin: 6pt 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
}
th {
    background: #e8e8e8;
    border: 1px solid #999;
    padding: 4pt;
    text-align: left;
    font-weight: bold;
}
td {
    border: 1px solid #ccc;
    padding: 4pt;
    vertical-align: top;
}
blockquote {
    border-left: 3px solid #999;
    margin: 0.5em 0;
    padding-left: 1em;
    color: #444;
    font-style: italic;
}
hr { border: none; border-top: 1px solid #ccc; margin: 1em 0; }
strong { color: #003366; }
img {
    display: block;
    margin: 0.5cm auto;
    max-width: 90%;
}
.abstract {
    background: #f4f8ff;
    border: 1px solid #ccd;
    padding: 8pt;
    margin: 0.8cm 0;
    font-size: 10.5pt;
}
"""


PAPER_FILES = [
    ("preprint_unified_en.md", "AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on Consumer Hardware"),
]


def md_to_html(md_text: str) -> str:
    md_text = md_text.replace("<", "&lt;").replace(">", "&gt;")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists", "attr_list"],
    )
    return f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>{html_body}</body></html>'


def html_to_pdf(html: str, pdf_path: str) -> bool:
    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(src=html, dest=f, default_css=HTML_CSS)
    return not result.err


INDEX_MD = """# AGI Research Kit: Unified Preprint v1

## AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on Consumer Hardware

**Authors:** AGI Research Kit Contributors  **Date:** 2026-08-01  **Type:** Single arXiv preprint, replacing the predecessor 5-paper TMLR bundle (archived).

---

## Headline Numbers

- End-to-end task success: **68%** vs 30% static baseline (+38 pp)
- Continual-learning eval: **60.4% +/- 3.6%**, **p<0.01** vs static
- A/B safety gate: **12/12** adversarial boundary tests passed
- Cross-model transfer: collapses below **~2B parameters**
- Hardware: ~3.3 GB RSS, ~20 s/episode, no GPU

## Companion Artifacts

- Source: src/agi_kit/, experiments/, scripts/
- Logs: logs/{cross_model,stat_tests,safety_gate,full_run*,continual,l4}
- Real SFT validation: data/sft_real/ (SmolLM2-135M, 2 min on CPU)
- Figures: papers/figures/ (5 matplotlib PNGs at 200 DPI)

## Why One Preprint Instead of Five?

The five-paper TMLR bundle achieved an average reviewer score of 3.43 / 5.0 (Major Revision), driven by structural issues that additional polishing could not resolve: synthetic GAIA2 eval, no head-to-head baselines, small N. We honestly consolidated the bundle into a single preprint that frames the work as an empirical system report rather than a benchmark-beating contribution.

---

See `preprint_unified_en.pdf` for the full paper. The original five papers (paper1_l1_self_critique, paper2_l2_meta_control, paper3_l3_continual_loop, paper4_l4_recursive, paper5_l1_l4_system) are preserved unchanged at `papers/_deprecated/`.
"""


def main():
    print("=== Building English PDF bundle ===")
    # Index
    if html_to_pdf(md_to_html(INDEX_MD), str(PAPERS_DIR / "00_INDEX_en.pdf")):
        print("  00_INDEX_en.pdf")

    # Cover letter
    cl_path = PAPERS_DIR / "COVER_LETTER.md"
    if cl_path.exists():
        if html_to_pdf(md_to_html(cl_path.read_text(encoding="utf-8")),
                       str(PAPERS_DIR / "COVER_LETTER_en.pdf")):
            print("  COVER_LETTER_en.pdf")

    # Papers
    for fn, _title in PAPER_FILES:
        src = PAPERS_DIR / fn
        if not src.exists():
            print(f"  {fn} NOT FOUND")
            continue
        out = src.with_suffix(".pdf")
        if html_to_pdf(md_to_html(src.read_text(encoding="utf-8")), str(out)):
            print(f"  {out.name}")
        else:
            print(f"  FAILED {out.name}")

    print("\n=== All outputs ===")
    for f in sorted(PAPERS_DIR.glob("*_en.pdf")):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()