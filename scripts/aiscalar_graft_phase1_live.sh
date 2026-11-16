#!/usr/bin/env bash
# Controlled real Qwen Time-4 Phase-1 ablation. Run on AIScalar after explicit review.
set -euo pipefail
export GIT_PAGER=cat
cd "$(dirname "$0")/.."
BRANCH="graft-fl-aiscalar-20261113"
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  echo "Please run on named research branch $BRANCH (no detached HEAD)." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree has tracked changes; preserve them before experiment." >&2
  exit 2
fi
git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"

# Fail before inference if regression tests do not pass.
python -m pytest -q \
  tests/test_graft_fusion_ablation.py \
  tests/test_graft_graph_suspicion.py \
  tests/test_graft_controller.py \
  tests/test_graft_pipeline.py \
  tests/test_graft_phase1_live_config.py

# Preserve the independently verified zero-inference result in its own commit.
OFFLINE="results/graft_time4_fusion_replay_local_20261113.json"
if [[ -f "$OFFLINE" ]]; then
  python - "$OFFLINE" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
if data.get("new_model_calls") != 0 or data.get("gold_used_for_scoring") is not False:
    raise SystemExit("Offline replay metadata failed verification")
PY
  git add -f -- "$OFFLINE"
  if ! git diff --cached --quiet; then
    git commit -m "result(graft): archive verified AIScalar zero-call fusion replay"
    git push origin "$BRANCH"
  fi
fi

ROOT="results/graft_phase1_live_$(date +%Y%m%d_%H%M%S)"
SOURCE="results/graft_time4_graph_ablation_20261113_143016/config.json"
python -m scripts.prepare_graft_phase1_live --source "$SOURCE" --output "$ROOT"

python -m rave_fl.graft.cli preflight --config "$ROOT/config.json" > "$ROOT/preflight.json"
python - "$ROOT/preflight.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("Preflight ready:", d.get("ready"))
print("Candidates:", d.get("candidate_preparations"))
if not d.get("ready") or len(d.get("candidate_preparations", [])) != 1:
    raise SystemExit("Preflight invalid; no Qwen calls were made")
PY

# Explicit real inference; 45 new calls and 60,000 provider-token hard caps.
set +e
python -m rave_fl.graft.cli run \
  --config "$ROOT/config.json" \
  --output "$ROOT/run" \
  --allow-inference 2>&1 | tee "$ROOT/run.log"
run_status=${PIPESTATUS[0]}
set -e
echo "Qwen runner status: $run_status"

# Preserve a failed or incomplete experiment as evidence too.
if [[ -f "$ROOT/run/manifest.json" ]]; then
  python -m rave_fl.graft.cli report --run-dir "$ROOT/run" > "$ROOT/report.log" 2>&1 || true
fi
if [[ -f "$ROOT/run/results.json" ]]; then
  python scripts/report_graft_graph_ablation.py "$ROOT/run" > "$ROOT/ablation_summary.txt" 2>&1 || true
  cat "$ROOT/ablation_summary.txt"
fi

# Refuse to publish oversize archives or obvious credentials to public GitHub.
oversize=$(find "$ROOT" -type f -size +90M -print -quit)
if [[ -n "$oversize" ]]; then
  echo "Review file >90 MB before publishing: $oversize" >&2
  exit 2
fi
if grep -RIlE --include="*.json" --include="*.jsonl" --include="*.log" \
  "(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)" \
  "$ROOT" | grep -q .; then
  echo "Possible secrets in outputs. Results retained locally; publish after redaction." >&2
  exit 2
fi

git add -f -- "$ROOT"
while IFS= read -r lock; do
  git rm --cached -q --ignore-unmatch -- "$lock"
done < <(find "$ROOT" -type f -name "*.lock")
git diff --cached --stat
if ! git diff --cached --quiet; then
  git commit -m "experiment(graft): archive Phase-1 fusion and selection Qwen comparison"
  git push origin "$BRANCH"
else
  echo "No new experiment artifacts were staged."
fi
echo "Experiment output: $ROOT"
echo "Git branch: $BRANCH"
if [[ "$run_status" -ne 0 ]]; then
  echo "Experiment ended incomplete/failed (status $run_status); artifacts archived if safe." >&2
  exit "$run_status"
fi
