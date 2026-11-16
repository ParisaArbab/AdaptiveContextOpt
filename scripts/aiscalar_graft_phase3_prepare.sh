#!/usr/bin/env bash
# Prepare official buggy versions with complete, authentic failing tests.
# INSPECT default: no checkout, tests, network or model calls.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(git branch --show-current)" == "graft-fl-aiscalar-20261113" ]] || {
  echo "Wrong research branch" >&2
  exit 2
}
MODE="$(echo "$@" | awk '{print $1}')"
if [[ -z "$MODE" ]]; then MODE="inspect"; fi
if [[ "$MODE" != "inspect" && "$MODE" != "collect" ]]; then
  echo "Usage: bash scripts/aiscalar_graft_phase3_prepare.sh [inspect|collect]" >&2
  exit 2
fi
python -m pytest -q tests/test_graft_phase3_real_bugs.py
ROOT="data/graft_phase3_official_$(date +%Y%m%d_%H%M%S)"
if [[ -n "$(printenv GRAFT_PHASE3_INPUT_ROOT || true)" ]]; then
  ROOT="$(printenv GRAFT_PHASE3_INPUT_ROOT)"
fi
echo "Preparation root: $ROOT"
if [[ "$MODE" == "inspect" ]]; then
  python -m scripts.prepare_graft_phase3_real_bugs \
    --manifest configs/rave_independent_smoke.json --output "$ROOT"
  exit 0
fi
if ! command -v defects4j >/dev/null 2>&1; then
  echo "Defects4J not on PATH. Official installation: https://github.com/rjust/defects4j" >&2
  echo "Then add /path/to/defects4j/framework/bin to PATH." >&2
  echo "No checkout or test started." >&2
  exit 3
fi
python -m scripts.prepare_graft_phase3_real_bugs \
  --manifest configs/rave_independent_smoke.json \
  --output "$ROOT" \
  --collect --allow-checkout --allow-real-tests
echo "Local input preparation at $ROOT"
echo "Review $ROOT/preparation_report.json for authenticity and missing cases."
echo "If all three are ready, run:"
echo "bash scripts/aiscalar_graft_phase3_multibug.sh $ROOT/manifest.json"
