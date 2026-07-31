#!/usr/bin/env bash
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
