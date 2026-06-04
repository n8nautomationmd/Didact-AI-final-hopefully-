#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/commit_and_push.sh "commit message"
MSG=${1:-"Enable neural model UI + fallback"}

git add src/neural_student_state_model.py app.py
git status --porcelain
git add src/neural_student_state_model.py app.py
if git diff --staged --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "$MSG"
  git push origin main
fi

echo "Done."