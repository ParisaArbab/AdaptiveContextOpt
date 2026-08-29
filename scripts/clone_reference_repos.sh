#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-references}"
mkdir -p "$ROOT"

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [[ -d "$dir/.git" ]]; then
    echo "Updating $dir"
    git -C "$dir" pull --ff-only
  else
    echo "Cloning $url -> $dir"
    git clone "$url" "$dir"
  fi
}

# Graphify is cloned for source inspection. Runtime uses the graphify CLI.
clone_or_update https://github.com/Graphify-Labs/graphify.git "$ROOT/graphify"

# This replication provides original FlexFL data, traditional FL rankings,
# trigger tests, bug reports and ground truth used by the benchmark.
clone_or_update https://github.com/ParisaArbab/FlexFL_OriginalReplication.git "$ROOT/FlexFL_OriginalReplication"

# LeanCTX is cloned for installation/source inspection. Runtime uses lean-ctx call.
clone_or_update https://github.com/yvgude/lean-ctx.git "$ROOT/lean-ctx"
