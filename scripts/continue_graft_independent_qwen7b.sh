#!/usr/bin/env bash
# Continue the prepared independent comparison without replaying control.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash $0 results/graft_independent_qwen7b_YYYYMMDD_HHMMSS" >&2
  exit 2
fi
ROOT="$1"
if [[ ! -f "$ROOT/control/results.json" || ! -f "$ROOT/config_anchored.json" || ! -f "$ROOT/instances.json" ]]; then
  echo "Missing control results or prepared inputs in $ROOT" >&2
  exit 2
fi
if [[ -e "$ROOT/anchored" ]]; then
  echo "Anchored output already exists; refusing to overwrite: $ROOT/anchored" >&2
  exit 2
fi
echo "Preserving existing control results and 51 paid model requests"
python -m rave_fl.graft.cli report --run-dir "$ROOT/control" > "$ROOT/control_report.log"
set +e
python -m rave_fl.graft.cli run \
  --config "$ROOT/config_anchored.json" \
  --output "$ROOT/anchored" --allow-inference \
  2>&1 | tee "$ROOT/anchored.log"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 && "$status" -ne 3 ]]; then
  echo "Anchored run failed with exit code $status" >&2
  exit "$status"
fi
python -m rave_fl.graft.cli report --run-dir "$ROOT/anchored" > "$ROOT/anchored_report.log"
python scripts/diagnose_graft_independent.py "$ROOT"
echo "Preserved run directory: $ROOT"
