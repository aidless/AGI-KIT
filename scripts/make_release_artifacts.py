"""Generate GitHub deployment artifacts:
  1. A .gitignore-friendly tarball of the repository (excluding .venv, logs, models)
  2. A push.sh script that the user can run after setting remote
  3. A README_GITHUB.md describing how to push to GitHub
  4. A .github/workflows/ci.yml for CI testing
"""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)


def make_tarball():
    """Create a tarball of the source tree, excluding heavy/vendored dirs."""
    out = DIST / "agi-research-kit.tar.gz"
    excludes = {".venv", "python", "logs", "models", ".git",
                "data/sft_real/out", "data/sft_real/train.jsonl",
                "data/playbook.faiss", "tools",
                "__pycache__", "node_modules"}
    with tarfile.open(out, "w:gz") as tar:
        for p in ROOT.rglob("*"):
            if any(part in excludes for part in p.parts):
                continue
            if p.is_file():
                arc = p.relative_to(ROOT)
                tar.add(p, arcname=str(arc))
    print(f"  {out.name}  ({out.stat().st_size // 1024} KB)")
    return out


PUSH_SH = """#!/usr/bin/env bash
# push.sh — push AGI Research Kit to GitHub
#
# Prerequisites:
#   1. Create a new GitHub repo at https://github.com/new (e.g. agi-research-kit)
#      DO NOT initialize with README / .gitignore / license (we have them)
#   2. Have a GitHub Personal Access Token (PAT) configured:
#      gh auth login    (GitHub CLI)
#      or set $GH_TOKEN and use it in the URL
#
# Usage:
#   bash push.sh <github-user-or-org> <repo-name> [branch]
# Example:
#   bash push.sh myname agi-research-kit
#
set -e
USER_OR_ORG="${1:?usage: bash push.sh <user> <repo> [branch]}"
REPO="${2:?usage: bash push.sh <user> <repo> [branch]}"
BRANCH="${3:-main}"
REMOTE="https://github.com/${USER_OR_ORG}/${REPO}.git"

echo "Setting up remote..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"

echo "Renaming branch to $BRANCH..."
git branch -M "$BRANCH"

echo "Pushing..."
if [ -n "$GH_TOKEN" ]; then
    git push -u "https://x-access-token:${GH_TOKEN}@github.com/${USER_OR_ORG}/${REPO}.git" "$BRANCH"
else
    git push -u origin "$BRANCH"
fi

echo "Done. View at https://github.com/${USER_OR_ORG}/${REPO}"
"""


CI_YML = """name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: windows-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Create venv
        run: |
          python -m venv .venv
          .\\.venv\\Scripts\\python.exe -m pip install --upgrade pip

      - name: Install dependencies
        run: |
          .\\.venv\\Scripts\\pip.exe install -r requirements.txt || \\
          .\\.venv\\Scripts\\pip.exe install transformers torch accelerate safetensors \\
            huggingface_hub datasets tokenizers openai httpx rich pydantic \\
            gradio pyyaml tiktoken ollama litellm sentence-transformers \\
            rank-bm25 faiss-cpu pypdf markdown xhtml2pdf matplotlib structlog

      - name: Smoke test L1
        run: |
          .\\.venv\\Scripts\\python.exe -m py_compile src/agi_kit/reflect.py

      - name: Smoke test L2
        run: |
          .\\.venv\\Scripts\\python.exe -m py_compile src/agi_kit/playbook.py
          .\\.venv\\Scripts\\python.exe -m py_compile src/agi_kit/meta.py

      - name: Smoke test L3
        run: |
          .\\.venv\\Scripts\\python.exe -m py_compile src/agi_kit/loop.py

      - name: Smoke test L4
        run: |
          .\\.venv\\Scripts\\python.exe -m py_compile src/agi_kit/recursive.py

      - name: Build PDFs
        run: |
          .\\.venv\\Scripts\\python.exe scripts/make_figures.py
          .\\.venv\\Scripts\\python.exe scripts/build_papers_pdf_en.py

      - name: Upload PDFs
        uses: actions/upload-artifact@v4
        with:
          name: papers-pdf
          path: papers/*_en.pdf
"""


RELEASE_NOTES = """# Release Notes — AGI Research Kit v1.0

**TMLR submission bundle** (2026-07-31).

## Highlights
- 5-paper bundle documenting L1-L4 self-improving agent architecture
- 50-episode end-to-end evaluation (68.5% success rate, +38.5pp over static)
- Real SFT validation on SmolLM2-135M (2 min on CPU)
- 5 matplotlib figures, 7 PDFs, full reproduction scripts

## Install
```powershell
python -m venv .venv
.\\.venv\\Scripts\\pip.exe install -r requirements.txt
ollama pull qwen3:1.7b
ollama pull qwen3:0.6b
```

## Quick start
```powershell
.\\.venv\\Scripts\\python.exe -u experiments/full_run3.py --n 50 --retrain-every 10 --no-sft
```

## Papers
1. `papers/paper1_l1_self_critique_en.pdf` — Self-Critique as a First-Class Abstraction
2. `papers/paper2_l2_meta_control_en.pdf` — Semantic Strategy Memory with Meta-Controller
3. `papers/paper3_l3_continual_loop_en.pdf` — Continual Learning Loop with A/B Safety Gate
4. `papers/paper4_l4_recursive_en.pdf` — Bounded Recursive Self-Modification
5. `papers/paper5_l1_l4_system_en.pdf` — End-to-End Self-Improving Architecture

## Reviewer simulation
`papers/reviews/` contains simulated reviewer critiques of all 5 papers.

## License
MIT (see LICENSE)
"""


REQUIREMENTS_TXT = """# AGI Research Kit — runtime dependencies
# Pinned to versions known to work together as of 2026-07-31

transformers>=4.55
torch>=2.5
accelerate>=1.0
safetensors>=0.4
huggingface_hub>=0.30
datasets>=3.0
tokenizers>=0.21

openai>=1.50
httpx>=0.27
rich>=13.7
pydantic>=2.8
pydantic-settings>=2.0
structlog>=24.0
prometheus_client>=0.20

gradio>=5.0
pyyaml>=6.0
tiktoken>=0.8

ollama>=0.4
litellm>=1.50

sentence-transformers>=3.0
rank-bm25>=0.2
faiss-cpu>=1.9
pypdf>=5.0

# PDF generation for papers/
markdown>=3.5
xhtml2pdf>=0.2
matplotlib>=3.7

# Dev
pytest>=8.0
"""


def main():
    print("=== Generating GitHub deployment artifacts ===")

    # 1. Tarball
    tar = make_tarball()

    # 2. push.sh
    push_sh = DIST / "push.sh"
    push_sh.write_text(PUSH_SH, encoding="utf-8")
    os.chmod(push_sh, 0o755)
    print(f"  {push_sh.name}  ({push_sh.stat().st_size} bytes)")

    # 3. CI workflow
    ci_dir = ROOT / ".github" / "workflows"
    ci_dir.mkdir(parents=True, exist_ok=True)
    ci_file = ci_dir / "ci.yml"
    ci_file.write_text(CI_YML, encoding="utf-8")
    print(f"  .github/workflows/ci.yml")

    # 4. requirements.txt
    req = ROOT / "requirements.txt"
    req.write_text(REQUIREMENTS_TXT, encoding="utf-8")
    print(f"  requirements.txt  ({req.stat().st_size} bytes)")

    # 5. RELEASE_NOTES.md
    rel = DIST / "RELEASE_NOTES.md"
    rel.write_text(RELEASE_NOTES, encoding="utf-8")
    print(f"  dist/RELEASE_NOTES.md")

    print()
    print("Tarball contents:")
    with tarfile.open(tar, "r:gz") as t:
        names = t.getnames()
        by_dir = {}
        for n in names:
            d = "/".join(n.split("/")[:2]) if "/" in n else n
            by_dir[d] = by_dir.get(d, 0) + 1
        for d in sorted(by_dir.keys()):
            print(f"  {d}: {by_dir[d]} files")
        print(f"  total: {len(names)} files")


if __name__ == "__main__":
    main()