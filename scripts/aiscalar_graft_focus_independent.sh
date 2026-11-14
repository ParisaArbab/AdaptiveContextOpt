#!/usr/bin/env bash
# Exploratory comparison on Time-1, Lang-1, Math-1. These are NOT fresh held-out bugs.
# Run only AFTER scripts/aiscalar_graft_focus_offline.sh passed.
set -euo pipefail
cd "$(dirname "$0")/.."
BRANCH="graft-fl-aiscalar-20261113"
[[ "$(git branch --show-current)" == "$BRANCH" ]] || { echo "Wrong branch" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "Tracked changes present; preserve before running" >&2; exit 2; }
git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"
python -m pytest -q \
  tests/test_graft_focused.py \
  tests/test_graft_selection.py \
  tests/test_graft_graph_suspicion.py \
  tests/test_graft_controller.py \
  tests/test_graft_pipeline.py
ROOT="results/graft_focused_independent_$(date +%Y%m%d_%H%M%S)"
python -m scripts.prepare_graft_focused_independent \
  --source configs/graft_independent_qwen7b_control.json \
  --output "$ROOT"
python -m rave_fl.graft.cli preflight --config "$ROOT/config.json" > "$ROOT/preflight.json"
python - "$ROOT/preflight.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
items=d.get("candidate_preparations", [])
print("Preflight ready:", d.get("ready"))
print("Bug readiness:", [(x.get("instance_id"), x.get("status")) for x in items])
wanted={"Time-1", "Lang-1", "Math-1"}
actual={x.get("instance_id") for x in items if x.get("status") == "ready"}
if not d.get("ready") or actual != wanted or len(items) != 3:
    raise SystemExit("Preflight failed, do not send any inference calls")
PY
echo "Preflight passed for three predeclared exploratory bugs. Starting bounded Qwen experiment."
set +e
python -m rave_fl.graft.cli run \
  --config "$ROOT/config.json" \
  --output "$ROOT/run" \
  --allow-inference 2>&1 | tee "$ROOT/run.log"
run_status=${PIPESTATUS[0]}
set -e
if [[ -f "$ROOT/run/results.json" ]]; then
  python scripts/report_graft_graph_ablation.py "$ROOT/run" > "$ROOT/ablation_summary.txt" 2>&1 || true
  cat "$ROOT/ablation_summary.txt"
fi
if [[ -f "$ROOT/run/manifest.json" ]]; then
  python -m rave_fl.graft.cli report --run-dir "$ROOT/run" > "$ROOT/report.log" 2>&1 || true
fi
oversize=$(find "$ROOT" -type f -size +90M -print -quit)
[[ -z "$oversize" ]] || { echo "Oversize artifact, review before public upload: $oversize" >&2; exit 2; }
if grep -RIlE --include="*.json" --include="*.jsonl" --include="*.log" \
  "(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)" \
  "$ROOT" | grep -q .; then
  echo "Possible credentials in result files; kept locally for review, not uploaded." >&2
  exit 2
fi
git add -f -- "$ROOT"
git diff --cached --stat | tail -50
if ! git diff --cached --quiet; then
  git commit -m "experiment(graft): archive focused retrieval comparison on three exploratory bugs"
  git push origin "$BRANCH"
fi
echo "Results directory: $ROOT"
echo "Run exit status: $run_status"
if [[ "$run_status" -ne 0 ]]; then exit "$run_status"; fi
