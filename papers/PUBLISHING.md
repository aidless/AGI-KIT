# Publishing to GitHub

This document walks you through publishing the AGI Research Kit to
GitHub and submitting the paper bundle to TMLR.

## 1. Push to GitHub (5 minutes)

### Option A: One-shot script (recommended)

```bash
cd /path/to/agi-research-kit
export GH_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx    # https://github.com/settings/tokens
bash dist/push.sh my-github-username agi-research-kit
```

### Option B: Manual

```bash
# 1. Create empty repo at https://github.com/new (DO NOT add README/.gitignore)
# 2. Then locally:
cd /path/to/agi-research-kit
git remote add origin https://github.com/YOUR_USERNAME/agi-research-kit.git
git branch -M main
git push -u origin main
```

### What gets pushed

- `src/`, `experiments/`, `scripts/`, `papers/`, `logs/` (small JSONs only), `configs/`, `requirements.txt`
- `dist/agi-research-kit.tar.gz` (500 MB — remove if too large: `git rm dist/agi-research-kit.tar.gz`)

### What is NOT pushed (per .gitignore)

- `.venv/` (Python virtual env)
- `python/` (Python installer)
- `tools/OllamaSetup.exe` (480 MB installer)
- `data/sft_real/out/` (538 MB trained weights)
- `models/`, `*.safetensors`, `*.gguf`, `*.bin`

To push the trained SFT model separately, use Git LFS:

```bash
git lfs install
git lfs track "*.safetensors"
git add data/sft_real/out/model.safetensors
git commit -m "Add trained SmolLM2-135M SFT model"
git push
```

## 2. Verify CI passes

After pushing, GitHub Actions (`.github/workflows/ci.yml`) will:

1. Set up Python 3.12 on Windows
2. Install all dependencies
3. Smoke-test all four layers (L1, L2, L3, L4)
4. Build the matplotlib figures
5. Build the PDFs
6. Upload the PDFs as artifacts

If any step fails, fix and push again. The CI is the canonical
"is the codebase working" check.

## 3. Submit to TMLR (via OpenReview)

1. Create an OpenReview account at https://openreview.net
2. Wait for the TMLR submission window to open (rolling submissions)
3. Go to https://openreview.net/group?id=TMLR
4. Click "Submit" and follow the form
5. Upload each paper as a separate PDF or as a single bundle

### What TMLR wants (per submission template)

- `paper.pdf` — the paper (we have `papers/paper*_en.pdf`)
- `paper.zip` — the supplementary bundle (we have `dist/agi-research-kit.tar.gz`)
- `code.zip` — code release (we have `dist/agi-research-kit.tar.gz` minus paper files)
- `rebuttal.pdf` — for revision rounds (we don't have this yet)

### Suggested cover-letter text

See `papers/COVER_LETTER.md` and `papers/COVER_LETTER_en.pdf`.

## 4. Recommended GitHub repo settings

After pushing:

- **Repository name**: `agi-research-kit`
- **Description**: "Self-improving tool-use agents on consumer hardware (TMLR submission)"
- **Topics**: `agent`, `self-improvement`, `qwen3`, `reflection`, `meta-learning`, `agi`
- **Website**: link to your lab
- **Releases**: tag `v1.0` after first commit

### Suggested social-media announcement

```text
AGI Research Kit v1.0 released!
A complete 4-layer architecture for self-improving tool-use agents:
- L1 reflection primitive
- L2 semantic strategy memory + rule-based meta-controller
- L3 continual learning loop with A/B safety gate
- L4 bounded recursive self-modification
Runs on consumer hardware (~5 GB RAM, 1.7B model).
5-paper TMLR submission bundle included.

GitHub: https://github.com/YOUR_USERNAME/agi-research-kit
```

## 5. Optional: Host a static site

Each paper PDF can be hosted on GitHub Pages for free. Create a
`gh-pages` branch with the PDFs and link from your README.

```bash
git checkout --orphan gh-pages
git checkout gh-pages -- papers/ README.md
git commit -m "Deploy papers site"
git push origin gh-pages
```

Your papers will be available at
`https://YOUR_USERNAME.github.io/agi-research-kit/`.

## 6. CI to add later (optional enhancements)

- Add `pytest` test suite (currently we rely on smoke tests)
- Add benchmark harness for paper claims
- Add Docker image for one-command reproduction
- Add Discord / Slack community links

## 7. License

This bundle is released under MIT (see `LICENSE`). The 5 papers are
copyright their authors; the code under MIT. TMLR submissions
typically retain paper copyright.

## 8. Timeline expectations

- **TMLR review cycle**: 2-4 months from submission
- **Major revision** is the most likely outcome (per our simulated reviewer)
- **Accept** requires addressing all reviewer concerns + ethics + statistics
- **Reject-and-resubmit** is also possible; TMLR is friendly to this

## 9. Contact

- **Authors**: AGI Research Kit Contributors
- **Email**: agi-research@example.com
- **GitHub Issues**: open a ticket on the repo