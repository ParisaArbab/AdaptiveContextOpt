#!/usr/bin/env bash
# Fresh isolated graph-specific ablation on Time-4; preserve earlier runs.
set -euo pipefail
export GIT_PAGER=cat
cd "$(dirname "$0")/.."
python -m pytest -q tests/test_graft_controller.py tests/test_graft_pipeline.py || {
  echo "Tests failed; no model calls made" >&2
  exit 2
}
ROOT="results/graft_time4_graph_ablation_$(date +%Y%m%d_%H%M%S)"
SOURCE="results/graft_time4_qwen7b/config_anchored.json"
python scripts/prepare_graft_graph_ablation.py --source "$SOURCE" --output "$ROOT"
python -m rave_fl.graft.cli preflight --config "$ROOT/config.json" > "$ROOT/preflight.json"
python - "$ROOT/preflight.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for r in d.get("candidate_preparations",[]):
    print("PREPARED",r)
if not d.get("ready") or len(d.get("candidate_preparations",[])) != 1:
    raise SystemExit("Preflight invalid. Inference not started")
PY
set +e
python -m rave_fl.graft.cli run \
  --config "$ROOT/config.json" \
  --output "$ROOT/run" \
  --allow-inference 2>&1 | tee "$ROOT/run.log"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 && "$status" -ne 3 ]]; then
  echo "GRAFT runner failed with status $status" >&2
  exit "$status"
fi
python -m rave_fl.graft.cli report --run-dir "$ROOT/run" > "$ROOT/report.log"
python scripts/report_graft_graph_ablation.py "$ROOT/run" | tee "$ROOT/ablation_summary.txt"
echo "Run saved: $ROOT"
if [[ "$status" -eq 3 ]]; then
  echo "One or more policies incomplete. Do not treat as a completed comparison."
fi
