"""Reviewer simulator: generate 3 reviewer critiques per paper.

Uses heuristic-based pattern matching on the paper text to surface
likely reviewer concerns. No external LLM required.

Outputs:
  papers/reviews/paper1_review.txt
  papers/reviews/paper2_review.txt
  ... (one per paper)
  papers/reviews/summary.txt
"""
from __future__ import annotations

import re
from pathlib import Path

PAPERS_DIR = Path(__file__).parent.parent / "papers"
REVIEWS_DIR = PAPERS_DIR / "reviews"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


PAPERS = [
    ("paper1_l1_self_critique_en.md", "Paper 1: Self-Critique as a First-Class Abstraction"),
    ("paper2_l2_meta_control_en.md", "Paper 2: Semantic Strategy Memory with Rule-Based Meta-Control"),
    ("paper3_l3_continual_loop_en.md", "Paper 3: Continual Learning Loop with A/B Safety Gate"),
    ("paper4_l4_recursive_en.md", "Paper 4: Bounded Recursive Self-Modification"),
    ("paper5_l1_l4_system_en.md", "Paper 5: End-to-End Self-Improving Architecture"),
]


REVIEWERS = [
    {
        "id": "R1-Methods",
        "name": "Reviewer 1 (Methods rigor)",
        "focus": "experimental design, ablations, statistical significance",
    },
    {
        "id": "R2-Novelty",
        "name": "Reviewer 2 (Novelty & positioning)",
        "focus": "contribution clarity, related work, claims",
    },
    {
        "id": "R3-Practice",
        "name": "Reviewer 3 (Practical impact)",
        "focus": "real-world applicability, deployment, costs",
    },
]


def score_to_severity(score: float) -> str:
    if score >= 4.0: return "MAJOR"
    if score >= 3.0: return "MINOR"
    return "WEAK"


def review_paper(text: str, fname: str, title: str) -> dict:
    """Generate 3 reviewer critiques based on text analysis."""
    text_lower = text.lower()
    n_words = len(text.split())
    has_figure = "!["
    has_table = "|---" in text or text.count("|---|") >= 1
    has_ethics = "ethics" in text_lower or "broader impact" in text_lower
    has_limitations = "limitations" in text_lower
    has_ablation = "ablation" in text_lower
    has_reproducibility = "reproducib" in text_lower or "appendix a" in text_lower
    has_real_sft = "real sft" in text_lower or "fine-tun" in text_lower
    has_no_gpu = "no gpu" in text_lower or "cpu" in text_lower
    has_qwen_only = "qwen3" in text_lower and "llama" not in text_lower
    has_open_source = "github" in text_lower or "released" in text_lower
    n_citations = text.count("et al.")

    reviews = []

    # === Reviewer 1: Methods ===
    r1_score = 3.5
    r1_concerns = []
    if not has_ablation:
        r1_concerns.append("Missing ablation: how do we know each design choice contributes?")
        r1_score -= 0.5
    if n_words < 6000:
        r1_concerns.append(f"Paper is short ({n_words} words); methodology section may be terse")
        r1_score -= 0.2
    if n_citations < 5:
        r1_concerns.append(f"Only {n_citations} citation clusters; lacks grounding")
        r1_score -= 0.3
    if "statistical" not in text_lower and "significance" not in text_lower:
        r1_concerns.append("No statistical significance tests across runs; single-seed results")
        r1_score -= 0.4
    if not has_real_sft:
        r1_concerns.append("SFT/retrain is mocked (gen_meta only); no real training loss reported")
        r1_score -= 0.3
    r1_concerns.append("Tiny held-out eval (5 arithmetic tasks); variance likely high")
    reviews.append({
        "reviewer": REVIEWERS[0],
        "score": max(1.0, min(5.0, r1_score)),
        "summary": "Methods section is reasonable but lacks depth in a few places.",
        "concerns": r1_concerns,
        "questions": [
            "How sensitive is performance to the alpha=0.4 hybrid weight? Have you tried alpha=0.2 and 0.6?",
            "What is the variance in eval_new_acc across multiple ContinualLoop runs with different seeds?",
            "For Paper 3's gate: how is the held-out eval set chosen to be independent of the training distribution?",
        ],
    })

    # === Reviewer 2: Novelty ===
    r2_score = 3.3
    r2_concerns = []
    if "Reflexion" not in text:
        r2_concerns.append("Doesn't discuss Reflexion (Shinn et al. 2023) — closest prior work")
    if "Voyager" not in text:
        r2_concerns.append("No comparison with Voyager (Wang et al. 2023) lifelong-learning skill library")
    if "MetaGPT" not in text:
        r2_concerns.append("MetaGPT not cited — closest analog of a meta-controller")
    if has_qwen_only:
        r2_concerns.append("Only Qwen3 evaluated; claims about generality are weak")
        r2_score -= 0.5
    r2_concerns.append("Hard to tell from the writeup what is the marginal contribution beyond composing existing ideas")
    r2_concerns.append("Title says 'first-class abstraction' — but is this just a Python module?")
    reviews.append({
        "reviewer": REVIEWERS[1],
        "score": max(1.0, min(5.0, r2_score)),
        "summary": "Novelty is incremental; many components are well-known in isolation.",
        "concerns": r2_concerns,
        "questions": [
            "What is the smallest experiment that demonstrates the Reflector cannot be replaced by a prompt?",
            "How does AGI Kit differ from LangChain ReAct + Reflexion wrapper classes?",
            "Is the SchemaMutator actually doing anything useful, or just renaming hyperparameters?",
        ],
    })

    # === Reviewer 3: Practice ===
    r3_score = 3.7
    r3_concerns = []
    if not has_no_gpu:
        r3_concerns.append("Hardware constraints unclear; does this require a GPU?")
    if not has_open_source:
        r3_concerns.append("Code release unclear; is the implementation available?")
    if not has_reproducibility:
        r3_concerns.append("Reproduction appendix missing or thin")
    r3_concerns.append("Synthetic GAIA2 tasks are not real GAIA2 — the benchmark is harder than shown")
    r3_concerns.append("ToolFactory only tested on 3 trigger tasks — what about cross-domain tools?")
    if not has_ethics:
        r3_concerns.append("No ethics / broader impact section; self-modification has alignment implications")
        r3_score -= 0.4
    reviews.append({
        "reviewer": REVIEWERS[2],
        "score": max(1.0, min(5.0, r3_score)),
        "summary": "Practical appeal is strong but real-world applicability needs more work.",
        "concerns": r3_concerns,
        "questions": [
            "What happens when the agent is deployed on a user-facing task with no safety guardrails?",
            "How does ToolFactory handle attempted prompt injection in the LLM-generated tool code?",
            "Can the A/B safety gate be calibrated per-deployment (e.g., stricter in medical, looser in games)?",
        ],
    })

    # Common reviewer recommendation
    avg_score = sum(r["score"] for r in reviews) / len(reviews)
    if avg_score >= 3.5:
        recommendation = "Weak Accept (revisions needed before publication)"
    elif avg_score >= 2.5:
        recommendation = "Major Revision"
    else:
        recommendation = "Reject"
    return {
        "title": title,
        "filename": fname,
        "word_count": n_words,
        "has_figure": has_figure in text,
        "has_table": has_table,
        "has_ethics": has_ethics,
        "has_ablation": has_ablation,
        "has_reproducibility": has_reproducibility,
        "reviews": reviews,
        "avg_score": avg_score,
        "recommendation": recommendation,
    }


def write_review(report: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"REVIEWER REPORT — {report['title']}")
    lines.append(f"File: {report['filename']}")
    lines.append(f"Word count: {report['word_count']}")
    lines.append(f"Has figure: {report['has_figure']}, Has table: {report['has_table']}")
    lines.append(f"Has ethics: {report['has_ethics']}, Has ablation: {report['has_ablation']}")
    lines.append(f"Has reproducibility: {report['has_reproducibility']}")
    lines.append("")
    lines.append(f"OVERALL: avg score = {report['avg_score']:.2f} / 5.0")
    lines.append(f"RECOMMENDATION: {report['recommendation']}")
    lines.append("=" * 70)
    lines.append("")
    for r in report["reviews"]:
        lines.append(f"\n{r['reviewer']['name']} (focus: {r['reviewer']['focus']})")
        lines.append(f"  Score: {r['score']:.2f}/5.0  Severity: {score_to_severity(r['score'])}")
        lines.append(f"  Summary: {r['summary']}")
        lines.append("  Concerns:")
        for c in r["concerns"]:
            lines.append(f"    - {c}")
        lines.append("  Questions for authors:")
        for q in r["questions"]:
            lines.append(f"    ? {q}")
        lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REVIEW")
    return "\n".join(lines)


def main():
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("REVIEWER SIMULATION SUMMARY (5 papers × 3 reviewers)")
    summary_lines.append("=" * 70)
    summary_lines.append("")

    for fname, title in PAPERS:
        src = PAPERS_DIR / fname
        if not src.exists():
            print(f"  {fname} not found, skipping")
            continue
        text = src.read_text(encoding="utf-8")
        report = review_paper(text, fname, title)
        out = REVIEWS_DIR / fname.replace(".md", "_review.txt")
        out.write_text(write_review(report), encoding="utf-8")
        print(f"  {out.name}  (avg score {report['avg_score']:.2f})")
        summary_lines.append(f"{title}")
        summary_lines.append(f"  avg score {report['avg_score']:.2f} → {report['recommendation']}")
        for r in report["reviews"]:
            summary_lines.append(f"    {r['reviewer']['id']}: {r['score']:.2f}/5  ({score_to_severity(r['score'])})")
        summary_lines.append("")

    summary_file = REVIEWS_DIR / "summary.txt"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")
    print()
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()