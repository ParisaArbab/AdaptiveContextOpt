#!/usr/bin/env bash
# Phase-3 test-aware retrieval and stopping ablation (real Qwen).
# Run ONLY after aiscalar_graft_test_aware_offline.sh succeeds.
# Time-4 is development data, not held-out.
set -euo pipefail
cd "$(dirname "$0")/.."
BRANCH="graft-fl-aiscalar-20261113"
[[ "$(git branch --show-current)" == "$BRANCH" ]] || {
  echo "Wrong branch, no Qwen calls." >&2; exit 2;
}
git diff --quiet && git diff --cached --quiet || {
  echo "Tracked worktree changes exist. Preserve them before inference." >&2; exit 2;
}
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

OFFLINE=$(find results -maxdepth 1 -type f -name 'graft_test_aware_offline_*.json' | sort | tail -n 1)
if [[ -z "$OFFLINE" ]]; then
  echo "Missing verified Phase-3 offline audit. Run scripts/aiscalar_graft_test_aware_offline.sh first." >&2
  exit 2
fi
python - "$OFFLINE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get("new_provider_calls")==0 and d.get("uses_gold_during_routing") is False
assert d.get("results")
assert all(r["new_test_aware"]["status"]=="ready" and
           r["new_test_aware"]["input_units"]<=4096
           for r in d["results"])
print("Offline audit verified:",sys.argv[1])
PY

ROOT="results/graft_phase3_test_aware_time4_$(date +%Y%m%d_%H%M%S)"
SOURCE="results/graft_phase2_time4_live_20261113_202949/config.json"
python -m scripts.prepare_graft_test_aware_live --source "$SOURCE" --output "$ROOT"

if ! python -m rave_fl.graft.cli preflight \
  --config "$ROOT/config.json" > "$ROOT/preflight.json"; then
  python - "$ROOT/preflight.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("dataset_error:",d.get("dataset_error"))
print("preparation_error:",d.get("preparation_error"))
print("candidates:",d.get("candidate_preparations"))
PY
  echo "Preflight failed. No new Qwen calls." >&2
  exit 3
fi
python - "$ROOT/preflight.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
items=d.get("candidate_preparations",[])
assert d.get("ready") and d.get("inference_performed") is False
assert len(items)==1 and items[0].get("instance_id")=="Time-4"
assert items[0].get("status")=="ready"
print("Preflight ready for Time-4, zero inference so far.")
PY

echo "Starting bounded Qwen comparison. Caps: 45 physical new calls and 60,000 provider tokens."
set +e
python -m rave_fl.graft.cli run \
  --config "$ROOT/config.json" \
  --output "$ROOT/run" \
  --allow-inference 2>&1 | tee "$ROOT/run.log"
run_status=$?
set -e
echo "Inference exit code: $run_status"

if [[ -f "$ROOT/run/results.json" && -f "$ROOT/run/accounting.json" ]]; then
  python scripts/report_graft_graph_ablation.py "$ROOT/run" > "$ROOT/ablation_summary.txt" 2>&1 || true
  cat "$ROOT/ablation_summary.txt"
fi
if [[ -f "$ROOT/run/manifest.json" ]]; then
  python -m rave_fl.graft.cli report --run-dir "$ROOT/run" > "$ROOT/report.log" 2>&1 || true
fi

large=$(find "$ROOT" -type f -size +90M -print -quit)
[[ -z "$large" ]] || {
  echo "Oversize result file; review before publishing: $large" >&2
  exit 2
}
if grep -RIlE --include="*.json" --include="*.jsonl" --include="*.log" \
  "(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)" \
  "$ROOT" | grep -q .; then
  echo "Possible secrets found. Results kept locally, not pushed." >&2
  exit 2
fi

git add -f -- "$ROOT"
while IFS= read -r lock; do
  git rm --cached -q --ignore-unmatch -- "$lock"
done < <(find "$ROOT" -type f -name "*.lock")
git diff --cached --stat
if ! git diff --cached --quiet; then
  git commit -m "experiment(graft): archive Phase-3 test-aware and stopping ablations"
  git push origin "$BRANCH"
fi
echo "Saved: $ROOT"
if [[ "$run_status" -ne 0 ]]; then
  echo "Run incomplete/failed, logs preserved." >&2
  exit "$run_status"
fi
