#!/usr/bin/env bash
# Archive an existing completed experimental run to the SAME GitHub branch.
# Requires explicit execution on AIScalar because GitHub cannot read cluster files.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="results/graft_independent_qwen7b_20261105_123254"
BRANCH="graft-time25-qwen7b-budget40k-results"
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  echo "Wrong branch; expected $BRANCH. Aborting." >&2
  exit 2
fi
if [[ ! -f "$ROOT/control/results.json" || ! -f "$ROOT/anchored/results.json" ]]; then
  echo "Both control and anchored results required; archive not started" >&2
  exit 2
fi
if [[ -n "$(git diff --cached --name-only)" ]]; then
  echo "Unrelated staged changes exist. Review and commit separately." >&2
  exit 2
fi
# Do not upload lock files. Avoid silently committing large binary artifacts.
find "$ROOT" -type f -size +95M -print > "$ROOT/oversize_paths.txt"
if [[ -s "$ROOT/oversize_paths.txt" ]]; then
  echo "Files exceed 95 MB; review before uploading:" >&2
  cat "$ROOT/oversize_paths.txt" >&2
  exit 2
fi
rm "$ROOT/oversize_paths.txt"
# These are research logs and can include the full prompts and external source.
# Fail closed for obvious credential literals. Manually review as needed.
if grep -rIlE --include='*.json' --include='*.jsonl' --include='*.log' \
  '(^|["[:space:]])(sk-[a-zA-Z0-9]{24,}|ghp_[a-zA-Z0-9]{30,}|github_pat_[a-zA-Z0-9_]{30,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)' \
  "$ROOT" | head -1 | grep -q .; then
  echo "Possible credential text in experiment files; do not upload before review" >&2
  exit 2
fi
git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"
git add -f -- "$ROOT"
# Remove only ephemeral lock files from index, preserving local files.
while IFS= read -r lock; do
  git rm --cached -q --ignore-unmatch -- "$lock"
done < <(find "$ROOT" -type f -name '*.lock')
echo "Archiving staged files:"
git diff --cached --stat
if git diff --cached --quiet; then
  echo "No new files to commit."
  exit 0
fi
git commit -m "experiment(graft): preserve complete three-bug Qwen7B paired logs and results"
git push origin "$BRANCH"
echo "Pushed archive to $BRANCH"
