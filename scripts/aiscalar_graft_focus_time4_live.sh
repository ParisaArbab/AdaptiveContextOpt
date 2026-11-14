#!/usr/bin/env bash
# Bounded Phase-2 Time-4 real-Qwen run. Requires the previously validated
# source/manifest. Development case only, NOT held-out research evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
BRANCH="graft-fl-aiscalar-20261113"
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  echo "Wrong research branch; no inference." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked changes exist; preserve them before running." >&2
  exit 2
fi
git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"

python -m pytest -q \
  tests/test_graft_focused.py \
  tests/test_graft_focused_time4_live_config.py \
  tests/test_graft_selection.py \
  tests/test_graft_controller.py \
  tests/test_graft_graph_suspicion.py \
  tests/test_graft_pipeline.py

SOURCE="results/graft_phase1_live_20261113_174425/config.json"
ROOT="results/graft_phase2_time4_live_$(date +%Y%m%d_%H%M%S)"
python -m scripts.prepare_graft_focused_time4_live \
  --source "$SOURCE" \
  --output "$ROOT"

if ! python -m rave_fl.graft.cli preflight \
  --config "$ROOT/config.json" > "$ROOT/preflight.json"; then
  echo "Time-4 preflight failed. No Qwen calls made. Saved: $ROOT/preflight.json" >&2
  python - "$ROOT/preflight.json" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
print("dataset_error:", data.get("dataset_error"))
print("preparation_error:", data.get("preparation_error"))
for item in data.get("candidate_preparations", []):
    print("candidate:", item.get("instance_id"), item.get("status"), item.get("error"))
PY
  exit 3
fi
python - "$ROOT/preflight.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
items=data.get("candidate_preparations",[])
print("Preflight ready:",data.get("ready"))
print("Candidates:", [(x.get("instance_id"),x.get("status")) for x in items])
assert data.get("ready") and len(items)==1 and items[0].get("instance_id")=="Time-4"
assert data.get("inference_performed") is False
PY

echo "Starting Qwen Phase-2 development comparison. Hard caps: 40 new calls / 55,000 provider tokens."
set +e
python -m rave_fl.graft.cli run \
  --config "$ROOT/config.json" \
  --output "$ROOT/run" \
  --allow-inference 2>&1 | tee "$ROOT/run.log"
run_status=$?
set -e
echo "Experiment exit code: $run_status"

if [[ -f "$ROOT/run/results.json" && -f "$ROOT/run/accounting.json" ]]; then
  python scripts/report_graft_graph_ablation.py "$ROOT/run" > "$ROOT/ablation_summary.txt" 2>&1 || true
  cat "$ROOT/ablation_summary.txt"
fi
if [[ -f "$ROOT/run/manifest.json" ]]; then
  python -m rave_fl.graft.cli report --run-dir "$ROOT/run" > "$ROOT/report.log" 2>&1 || true
fi

large=$(find "$ROOT" -type f -size +90M -print -quit)
if [[ -n "$large" ]]; then
  echo "Oversize artifact, review locally before public upload: $large" >&2
  exit 2
fi
if grep -RIlE --include="*.json" --include="*.jsonl" --include="*.log" \
   "(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)" \
   "$ROOT" | grep -q .; then
  echo "Possible credentials found. Keep artifacts local until reviewed." >&2
  exit 2
fi

git add -f -- "$ROOT"
while IFS= read -r lock; do
  git rm --cached -q --ignore-unmatch -- "$lock"
done < <(find "$ROOT" -type f -name "*.lock")
git diff --cached --stat
if ! git diff --cached --quiet; then
  git commit -m "experiment(graft): archive Phase-2 Time-4 focused graph comparison"
  git push origin "$BRANCH"
fi
echo "Results saved: $ROOT"
if [[ "$run_status" -ne 0 ]]; then
  echo "Run failed or incomplete; retained diagnostics." >&2
  exit "$run_status"
fi
