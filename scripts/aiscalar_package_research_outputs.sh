#!/usr/bin/env bash
# Package all locally stored RAVE-FL research outputs for GitHub Release assets.
# This script does NOT publish or change the source repository. Review the
# output before uploading it: AdaptiveContextOpt is a public repository.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
DEST="${1:-$HOME/RAVE_FL_GitHub_Archives/rave-fl-artifacts-20261027}"

# Prevent recursively archiving the output into itself.
case "$DEST" in
  "$ROOT"|"$ROOT"/*)
    echo "ERROR: archive destination must be outside the source repository" >&2
    exit 2
    ;;
esac

if [[ -e "$DEST" ]]; then
  echo "ERROR: destination already exists: $DEST" >&2
  echo "Choose a new output folder to preserve the previous archive." >&2
  exit 2
fi

for path in results logs configs; do
  if [[ ! -d "$path" ]]; then
    echo "ERROR: expected directory is missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$DEST"
INPUTS=(results logs configs)
if [[ -d data/rave_fl_observed_failures ]]; then
  INPUTS+=(data/rave_fl_observed_failures)
fi

{
  echo "RAVE-FL research output snapshot (GitHub release archive)"
  echo "Captured UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "Repository: https://github.com/ParisaArbab/AdaptiveContextOpt"
  echo "Branch: $(git branch --show-current)"
  echo "Commit: $(git rev-parse HEAD)"
  echo "Working tree dirty: $(if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then echo yes; else echo no; fi)"
  echo "Host: $(hostname)"
  echo "Included paths:"
  printf '  %s\n' "${INPUTS[@]}"
  echo
  echo "Key run IDs:"
  echo "  d4j97_promptfix_20261027_122406 (original 8K, 97 selected)"
  echo "  d4j_recovery4_16k_20261027_165751 (separate 16K, 4 selected)"
  echo "  chart17_promptfix_20261027_114655 (prompt-fix pilot)"
  echo "  pilot2_promptfix_20261027_121624 (prompt-fix pilot)"
  echo
  echo "Source sizes:"
  du -sh "${INPUTS[@]}"
  echo
  echo "Restore: sha256sum -c SHA256SUMS.txt && cat RAVE_FL_ALL_RESULTS_LOGS.tar.gz.part-* | tar -xzf -"
  echo "NOTE: archives include raw logs, local paths, model prompts, and recorded evidence."
  echo "NOTE: public GitHub Release assets are readable by anyone once published."
} > "$DEST/ARCHIVE_INFO.txt"

# Divide the compressed stream into files comfortably below the 2 GiB
# per-asset release ceiling. This also avoids staging an extra giant tarball.
tar -czf - "${INPUTS[@]}" \
  | split -b 900m -d -a 3 - "$DEST/RAVE_FL_ALL_RESULTS_LOGS.tar.gz.part-"

(
  cd "$DEST"
  sha256sum RAVE_FL_ALL_RESULTS_LOGS.tar.gz.part-* > SHA256SUMS.txt
  sha256sum -c SHA256SUMS.txt
  cat RAVE_FL_ALL_RESULTS_LOGS.tar.gz.part-* | gzip -t
)
echo "Archive created and validated: $DEST"
du -sh "$DEST"
echo
echo "Upload ONLY after reviewing sensitive content and confirming GitHub authentication:"
echo "  gh release create rave-fl-artifacts-2026-10-27 \\"
echo "    \"$DEST\"/RAVE_FL_ALL_RESULTS_LOGS.tar.gz.part-* \\"
echo "    \"$DEST\"/SHA256SUMS.txt \"$DEST\"/ARCHIVE_INFO.txt \\"
echo "    --repo ParisaArbab/AdaptiveContextOpt --target rave-fl-adaptive-pilot \\"
echo "    --title 'RAVE-FL Defects4J research artifacts (2026-10-27)' \\"
echo "    --notes 'Full research output archive including original 8K 97-bug benchmark, 16K four-bug recovery, pilots, logs, and configs.' \\"
echo "    --latest=false"
