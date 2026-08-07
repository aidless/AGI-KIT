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
import re
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
    page-break-after: avoid;
}
h2 {
    display: block;
    font-size: 14pt;
    margin-top: 0.6cm;
    margin-bottom: 0.3cm;
    color: #003366;
    border-bottom: 1px solid #999;
    padding-bottom: 2pt;
    page-break-after: avoid;
}
h3 {
    display: block;
    font-size: 12pt;
    margin-top: 0.5cm;
    margin-bottom: 0.2cm;
    color: #003366;
    page-break-after: avoid;
}
h4 {
    display: block;
    font-size: 11pt;
    margin-top: 0.4cm;
    margin-bottom: 0.2cm;
    color: #555;
    page-break-after: avoid;
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
    page-break-inside: avoid;
}
.codehilite {
    page-break-inside: avoid;
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
    page-break-inside: avoid;
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
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists", "attr_list"],
    )
    hard_eval_table = "<table>\n<thead>\n<tr>\n<th>Configuration on hard 20-task eval</th>"
    html_body = html_body.replace(
        hard_eval_table,
        "<pdf:nextpage />\n" + hard_eval_table,
        1,
    )
    round15_table = (
        "<table>\n<thead>\n<tr>\n<th>Configuration</th>\n"
        '<th style="text-align: right;">Perfect (3/3)</th>'
    )
    html_body = html_body.replace(
        round15_table,
        "<pdf:nextpage />\n" + round15_table,
        1,
    )
    baseline_table = (
        "<table>\n<thead>\n<tr>\n<th>Configuration</th>\n"
        '<th style="text-align: right;">Emission</th>\n'
        '<th style="text-align: right;">Correctness</th>\n<th>Note</th>'
    )
    first_baseline = html_body.find(baseline_table)
    second_baseline = html_body.find(
        baseline_table,
        first_baseline + len(baseline_table),
    )
    if second_baseline >= 0:
        html_body = (
            html_body[:second_baseline]
            + "<pdf:nextpage />\n"
            + html_body[second_baseline:]
        )
    swapout_table = (
        "<table>\n<thead>\n<tr>\n<th>Template</th>\n"
        '<th style="text-align: right;">n</th>\n'
        '<th style="text-align: right;">Bare accuracy</th>\n'
        '<th style="text-align: right;">Full L1-L4 accuracy</th>\n'
        '<th style="text-align: right;">Delta</th>'
    )
    html_body = html_body.replace(
        swapout_table,
        "<pdf:nextpage />\n" + swapout_table,
        1,
    )
    appendix_b = html_body.find('<h2 id="appendix-b-reproduction">')
    reproduction_code = html_body.find('<div class="codehilite">', appendix_b)
    if reproduction_code >= 0:
        html_body = (
            html_body[:reproduction_code]
            + "<pdf:nextpage />\n"
            + html_body[reproduction_code:]
        )
    html_body = re.sub(
        r"(<pre><code[^>]*>)(.*?)(</code></pre>)",
        lambda match: match.group(1) + match.group(2).replace("\n", "<br />") + match.group(3),
        html_body,
        flags=re.DOTALL,
    )
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<style>{HTML_CSS}</style></head><body>{html_body}</body></html>"
    )


def html_to_pdf(html: str, pdf_path: str) -> bool:
    def resolve_asset(uri: str, _rel: str) -> str:
        if uri.startswith(("data:", "http://", "https://")):
            return uri
        return str((PAPERS_DIR / uri).resolve())

    if Path(pdf_path).name in {"00_INDEX_en.pdf", "COVER_LETTER_en.pdf"}:
        html = html.replace(
            "</style>",
            "body { font-size: 10pt; line-height: 1.3; } "
            "p { margin: 0.22em 0; } li { margin: 0.12em 0; }"
            "</style>",
            1,
        )
    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(
            src=html,
            dest=f,
            path=str(PAPERS_DIR),
            link_callback=resolve_asset,
        )
    return not result.err


def main():
    print("=== Building English PDF bundle ===")
    # Index: generated from the same source delivered to readers.
    index_md = (PAPERS_DIR / "00_INDEX_en.md").read_text(encoding="utf-8")
    if html_to_pdf(md_to_html(index_md), str(PAPERS_DIR / "00_INDEX_en.pdf")):
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
