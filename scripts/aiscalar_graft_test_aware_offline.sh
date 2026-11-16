#!/usr/bin/env bash
# Phase-3 offline test-aware prompt audit. NEVER makes Qwen calls.
set -euo pipefail
cd "$(dirname "$0")/.."
BRANCH="graft-fl-aiscalar-20261113"
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  echo "Wrong branch; no work performed." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Uncommitted tracked changes found. Preserve them first." >&2
  exit 2
fi
git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"

python -m pytest -q \
  tests/test_graft_test_aware.py \
  tests/test_graft_test_aware_live_config.py \
  tests/test_graft_focused.py \
  tests/test_graft_selection.py \
  tests/test_graft_controller.py \
  tests/test_graft_graph_suspicion.py \
  tests/test_graft_pipeline.py

REPORT="results/graft_test_aware_offline_$(date +%Y%m%d_%H%M%S).json"
python -m scripts.audit_graft_test_aware_offline \
  results/graft_phase2_time4_live_20261113_202949/run \
  --budget 4096 \
  --output "$REPORT"

python - "$REPORT" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
assert data["new_provider_calls"] == 0
assert not data["uses_gold_during_routing"]
assert data["results"]
for result in data["results"]:
    selection=result["new_test_aware"]
    assert selection["status"]=="ready", selection["reason"]
    assert selection["input_units"]<=4096
    signal=result["test_signal"]
    assert signal["kind"]=="possible_expected_exception"
    assert signal["certainty"]=="incomplete_test_excerpt"
    assert result["stop_decision"] is None
    diag=selection["diagnostics"]
    assert diag["gold_used"] is False
    assert diag["runtime_reachability_proven"] is False
    print("Validated:", result["instance"], selection["input_units"], "/4096")
    print("Selected target:", result["selected_target"])
    print("Target line groups:", selection["target_source_lines"])
    print("Remaining missing source:",diag["omitted_target_code_ids"])
PY

git add -f -- "$REPORT"
git commit -m "result(graft): audit test-aware source paths with zero Qwen calls"
git push origin "$BRANCH"
echo "Offline report saved: $REPORT"
