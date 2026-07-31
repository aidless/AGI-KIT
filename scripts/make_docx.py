"""Convert English markdown papers to OpenReview-compatible .docx format.

Produces:
  papers/docx/paper1_l1_self_critique.docx
  papers/docx/paper2_l2_meta_control.docx
  papers/docx/paper3_l3_continual_loop.docx
  papers/docx/paper4_l4_recursive.docx
  papers/docx/paper5_l1_l4_system.docx
  papers/docx/00_INDEX.docx
  papers/docx/COVER_LETTER.docx

OpenReview format:
- A4 page
- Times Roman 11pt
- 1.5 line spacing
- Section headings styled
- Figures embedded (extracted from base64 in markdown)
- Tables rendered
- References as plain text at end
"""
from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import markdown
from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.enum.style import WD_STYLE_TYPE

PAPERS_DIR = Path(__file__).parent.parent / "papers"
DOCX_DIR = PAPERS_DIR / "docx"
DOCX_DIR.mkdir(parents=True, exist_ok=True)


PAPERS = [
    ("preprint_unified_en.md", "AGI Kit: An End-to-End Self-Improving Tool-Use Pipeline on Consumer Hardware"),
]


def extract_images(md_text):
    """Extract base64 images and return (clean_md, [(name, bytes)])."""
    images = []
    pattern = re.compile(r"!\[[^\]]*\]\(data:image/(?:png|jpeg|jpg);base64,([^)]+)\)")
    def repl(m):
        idx = len(images)
        data = base64.b64decode(m.group(1))
        name = f"fig{idx+1}.png"
        images.append((name, data))
        return f"[FIGURE_{idx}]"
    cleaned = pattern.sub(repl, md_text)
    return cleaned, images


def setup_doc(doc):
    """Apply OpenReview-ish styling."""
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    pf = style.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)


def add_heading(doc, text, level):
    p = doc.add_heading(text, level=min(level, 9))
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
    return p


def add_paragraph(doc, text):
    p = doc.add_paragraph(text)
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(11)
    return p


def add_code_block(doc, text):
    p = doc.add_paragraph(text)
    for run in p.runs:
        run.font.name = "Courier New"
        run.font.size = Pt(9)
    return p


def add_image(doc, name, data):
    buf = io.BytesIO(data)
    p = doc.add_picture(buf, width=Inches(5.5))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    cap = doc.add_paragraph(f"Figure: {name}")
    cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    for run in cap.runs:
        run.font.italic = True
        run.font.size = Pt(10)


def add_table_from_md(doc, md_table_text):
    """Render a simple markdown table to docx table."""
    lines = [l.strip() for l in md_table_text.strip().split("\n") if l.strip()]
    if not lines: return
    rows = []
    for ln in lines:
        if ln.startswith("|---") or ln.startswith("| :---"): continue
        if not ln.startswith("|"): continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        rows.append(cells)
    if not rows: return
    tbl = doc.add_table(rows=len(rows), cols=len(rows[0]))
    tbl.style = "Light Grid Accent 1"
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            tbl.cell(i, j).text = c
            for p in tbl.cell(i, j).paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
                    if i == 0:
                        run.font.bold = True


def md_to_docx(md_text, doc):
    """Convert markdown to docx, handling headings, paragraphs, code, tables, images."""
    cleaned, images = extract_images(md_text)
    md_html = markdown.markdown(
        cleaned,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    # Parse the HTML line by line to dispatch
    lines = md_html.split("\n")
    i = 0
    in_table = False
    table_buf = []
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("<h1>"):
            text = re.sub(r"<[^>]+>", "", line)
            add_heading(doc, text, 1)
        elif line.startswith("<h2>"):
            text = re.sub(r"<[^>]+>", "", line)
            add_heading(doc, text, 2)
        elif line.startswith("<h3>"):
            text = re.sub(r"<[^>]+>", "", line)
            add_heading(doc, text, 3)
        elif line.startswith("<h4>"):
            text = re.sub(r"<[^>]+>", "", line)
            add_heading(doc, text, 4)
        elif line.startswith("<pre>") or line.startswith("<code>"):
            # collect code
            code = re.sub(r"<[^>]+>", "", line)
            j = i + 1
            while j < len(lines) and not (lines[j].rstrip().endswith("</pre>") or lines[j].rstrip().endswith("</code>")):
                code += "\n" + re.sub(r"<[^>]+>", "", lines[j].rstrip())
                j += 1
            add_code_block(doc, code)
            i = j + 1
            continue
        elif line.startswith("<table>"):
            in_table = True
            table_buf = [line]
            i += 1
            continue
        elif line.startswith("</table>"):
            in_table = False
            table_buf.append(line)
            add_table_from_md(doc, "\n".join(table_buf))
            table_buf = []
            i += 1
            continue
        elif in_table:
            table_buf.append(line)
            i += 1
            continue
        elif "[FIGURE_" in line:
            # extract figure number
            m = re.search(r"\[FIGURE_(\d+)\]", line)
            if m:
                idx = int(m.group(1))
                if idx < len(images):
                    name, data = images[idx]
                    add_image(doc, name, data)
                i += 1
                continue
        # regular paragraph - strip HTML
        text = re.sub(r"<[^>]+>", "", line)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&amp;", "&", text)
        if text.strip():
            add_paragraph(doc, text)
        i += 1


def main():
    print("=== Generating OpenReview .docx bundle ===")
    # Cover letter
    cl = PAPERS_DIR / "COVER_LETTER.md"
    if cl.exists():
        doc = Document()
        setup_doc(doc)
        doc.add_heading("Cover Letter — TMLR Submission", 0)
        md_to_docx(cl.read_text(encoding="utf-8"), doc)
        doc.save(str(DOCX_DIR / "COVER_LETTER.docx"))
        print(f"  COVER_LETTER.docx")
    # Index
    idx_md = """# AGI Research Kit

## Five Papers for TMLR Submission (English Bundle)

This bundle contains five interconnected papers documenting the AGI
Research Kit, a complete self-improving tool-use agent architecture
running on a 1.7B-parameter language model on consumer hardware.

## Paper 1: Self-Critique as a First-Class Abstraction
**Key Result**: 30% → 51% success rate

## Paper 2: Semantic Strategy Memory with Rule-Based Meta-Control
**Key Result**: 51% → 58%; 71% stuck recovery

## Paper 3: Continual Learning Loop with A/B Safety Gate
**Key Result**: eval_new_acc +23% over 7 generations

## Paper 4: Bounded Recursive Self-Modification
**Key Result**: 2 schema + 3 tool + 4 prompt mutations

## Paper 5: End-to-End Self-Improving Architecture on Consumer Hardware
**Key Result**: 68.5% success rate, ~5 GB RAM
"""
    doc = Document()
    setup_doc(doc)
    md_to_docx(idx_md, doc)
    doc.save(str(DOCX_DIR / "00_INDEX.docx"))
    print("  00_INDEX.docx")
    # Papers
    for fname, _title in PAPERS:
        src = PAPERS_DIR / fname
        if not src.exists():
            print(f"  {fname} NOT FOUND")
            continue
        doc = Document()
        setup_doc(doc)
        md_to_docx(src.read_text(encoding="utf-8"), doc)
        out = DOCX_DIR / fname.replace(".md", ".docx")
        doc.save(str(out))
        print(f"  {out.name}")
    print()
    print("=== Done ===")
    for f in sorted(DOCX_DIR.glob("*.docx")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()