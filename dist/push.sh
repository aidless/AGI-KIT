#!/usr/bin/env bash
# push.sh - push AGI Research Kit to GitHub, with self-checks
#
# Prerequisites:
#   1. Create a new GitHub repo at https://github.com/new (e.g. agi-research-kit)
#      DO NOT initialize with README / .gitignore / license (we have them)
#   2. Have a GitHub Personal Access Token (PAT) configured:
#      gh auth login    (GitHub CLI)
#      or set $GH_TOKEN and use it in the URL
#
# Usage:
#   bash dist/push.sh [--check-only] <github-user-or-org> <repo-name> [branch]
# Example:
#   bash dist/push.sh myname agi-research-kit
#   bash dist/push.sh --check-only                      # run guards only, do not push
#
# Self-checks (run before push):
#   1. Working tree clean (no staged/unstaged/modified files)
#   2. dist/agi-research-kit.tar.gz is NOT tracked by git
#   3. No file >50 MB in the working tree (GitHub hard cap is 100 MB, soft 50 MB)
#   4. LICENSE, README.md, papers/README.md present
#   5. .env does NOT exist (use .env.example)
#   6. dist/push.sh is executable
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

CHECK_ONLY=0
if [ "${1:-}" = "--check-only" ]; then
    CHECK_ONLY=1
    shift
fi

USER_OR_ORG="${1:?usage: bash push.sh [--check-only] <user> <repo> [branch]}"
REPO="${2:?usage: bash push.sh [--check-only] <user> <repo> [branch]}"
BRANCH="${3:-main}"
REMOTE="https://github.com/${USER_OR_ORG}/${REPO}.git"

FAIL=0
pass() { printf '  \033[32mOK\033[0m %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$1"; FAIL=1; }

echo "[1/6] Working tree clean..."
if [ -z "$(git status --porcelain)" ]; then
    pass "no uncommitted or untracked files"
else
    warn "working tree dirty: $(git status --short | tr '\n' ' ')"
fi

echo "[2/6] dist/agi-research-kit.tar.gz must be git-ignored..."
if git ls-files --error-unmatch dist/agi-research-kit.tar.gz >/dev/null 2>&1; then
    warn "dist/agi-research-kit.tar.gz is tracked! remove it before push"
else
    pass "dist/agi-research-kit.tar.gz not tracked"
fi

echo "[3/6] No file >50 MB in working tree..."
big_files=$(find . -type f -size +50M \
    -not -path "./.git/*" \
    -not -path "./.venv/*" \
    -not -path "./data/sft_real/*" \
    -not -path "./logs/*/gen-*" \
    -not -path "./logs/*/samples.jsonl" \
    -not -path "./dist/agi-research-kit.tar.gz" 2>/dev/null)
if [ -n "$big_files" ]; then
    warn "files >50 MB found (excluding tarball):"
    echo "$big_files" | sed 's/^/    /'
else
    pass "no oversize files"
fi

echo "[4/6] Required files present..."
for f in LICENSE README.md papers/README.md; do
    if [ -f "$f" ]; then pass "$f"; else warn "$f missing"; fi
done

echo "[5/6] .env must NOT be committed..."
if [ -f ".env" ]; then
    warn ".env exists locally - move secrets out before push"
else
    pass ".env absent"
fi

echo "[6/6] dist/push.sh is executable..."
if [ -x "dist/push.sh" ]; then
    pass "dist/push.sh executable"
else
    chmod +x dist/push.sh
    warn "dist/push.sh was not executable - fixed locally, please re-stage"
fi

if [ "$FAIL" -ne 0 ]; then
    echo
    echo "\033[31mSelf-checks FAILED. Aborting push. Fix the items above and retry.\033[0m"
    exit 2
fi
echo
echo "All self-checks passed."

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "(check-only mode: not pushing)"
    exit 0
fi

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

echo "Done. View at https://github.com/${USER_OR_GITHUB:-}${USER_OR_ORG}/${REPO}"