#!/usr/bin/env bash
# Upload a previously packaged, locally verified research snapshot as GitHub
# Release assets. Requires gh authentication; never modifies the git tree.
set -euo pipefail

DEST="${1:-$HOME/RAVE_FL_GitHub_Archives/rave-fl-artifacts-20261027}"
TAG="rave-fl-artifacts-2026-10-27"
REPO="ParisaArbab/AdaptiveContextOpt"

[[ -d "$DEST" ]] || { echo "Archive directory not found: $DEST" >&2; exit 2; }
command -v gh >/dev/null || {
  echo "Install gh in Conda first: conda install -n rave-fl -c conda-forge gh" >&2
  exit 2
}
gh auth status >/dev/null || {
  echo "Sign in to your own account first: gh auth login --web" >&2
  exit 2
}
(
  cd "$DEST"
  sha256sum -c SHA256SUMS.txt
)
shopt -s nullglob
parts=("$DEST"/RAVE_FL_ALL_RESULTS_LOGS.tar.gz.part-*)
if (( ${#parts[@]} == 0 )); then
  echo "Archive has no parts; aborting" >&2
  exit 2
fi
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "Release tag already exists: $TAG" >&2
  echo "Inspect existing upload; do not silently replace research artifacts." >&2
  exit 2
fi

echo "PUBLIC REPOSITORY: https://github.com/$REPO"
echo "Release assets: ${#parts[@]} archive parts + checksums + provenance"
echo "These archives contain model prompts, raw logs and local paths."
read -r -p "Have you reviewed for secrets/private content? Type PUBLISH to continue: " confirmation
[[ "$confirmation" == "PUBLISH" ]] || { echo "Upload cancelled."; exit 0; }

gh release create "$TAG" \
  "${parts[@]}" \
  "$DEST/SHA256SUMS.txt" \
  "$DEST/ARCHIVE_INFO.txt" \
  --repo "$REPO" \
  --target rave-fl-adaptive-pilot \
  --title "RAVE-FL Defects4J research artifacts (2026-10-27)" \
  --notes-file "$DEST/ARCHIVE_INFO.txt" \
  --latest=false

echo "=== Published GitHub Release ==="
gh release view "$TAG" --repo "$REPO" --json url,assets \
  --jq '{url: .url, assets: [.assets[].name]}'
