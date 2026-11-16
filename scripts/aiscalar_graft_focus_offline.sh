#!/usr/bin/env bash
# Phase-2 focused retrieval offline validation: NO QWEN CALLS.
set -euo pipefail
cd "$(dirname "$0")/.."
BRANCH="graft-fl-aiscalar-20261113"
[[ "$(git branch --show-current)" == "$BRANCH" ]] || { echo "Wrong branch" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "Tracked changes present; preserve them before running" >&2; exit 2; }
git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"
python -m pytest -q \
  tests/test_graft_focused.py \
  tests/test_graft_selection.py \
  tests/test_graft_controller.py \
  tests/test_graft_graph_suspicion.py \
  tests/test_graft_pipeline.py
REPORT="results/graft_focused_offline_$(date +%Y%m%d_%H%M%S).json"
python -m scripts.audit_graft_focused_offline \
  results/graft_phase1_live_20261113_174425/run \
  --budget 4096 \
  --output "$REPORT"
python - "$REPORT" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data["new_provider_calls"] == 0
assert data["results"]
assert all(r["focused_status"] == "ready" and r["focused_input_units"] <= 4096 for r in data["results"])
print("Offline focused packages verified; no Qwen calls.")
PY
git add -f -- "$REPORT"
git commit -m "result(graft): verify focused source retrieval and budget offline"
git push origin "$BRANCH"
echo "Saved and pushed: $REPORT"
