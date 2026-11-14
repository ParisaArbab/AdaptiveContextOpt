#!/usr/bin/env bash
# Bounded Phase-3 exploratory 3-bug REAL-QWEN evaluation.
# Requires fully prepared authentic official Defects4J inputs.
# Never substitutes synthetic logs or reduces missing cohort from 3 to 1.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(git branch --show-current)" != "graft-fl-aiscalar-20261113" ]]; then
  echo "Wrong research branch, no inference" >&2
  exit 2
fi
git diff --quiet && git diff --cached --quiet || {
  echo "Uncommitted tracked changes found. Preserve them first." >&2
  exit 2
}
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/aiscalar_graft_phase3_multibug.sh data/<new-official-prep>/manifest.json" >&2
  exit 2
fi
DATASET="$1"
if [[ ! -f "$DATASET" ]]; then
  echo "Missing verified new dataset: $DATASET" >&2
  exit 2
fi

python -m pytest -q \
  tests/test_graft_phase3_real_bugs.py \
  tests/test_graft_phase3_bm25_java_spans.py \
  tests/test_graft_test_aware.py \
  tests/test_graft_test_aware_live_config.py \
  tests/test_graft_focused.py \
  tests/test_graft_selection.py \
  tests/test_graft_controller.py \
  tests/test_graft_pipeline.py

ROOT="results/graft_phase3_real_bugs_$(date +%Y%m%d_%H%M%S)"
python -m scripts.prepare_graft_phase3_multibug \
  --source configs/graft_independent_qwen7b_control.json \
  --dataset "$DATASET" \
  --output "$ROOT"

if ! python -m rave_fl.graft.cli preflight \
  --config "$ROOT/config.json" > "$ROOT/preflight.json"; then
  python - "$ROOT/preflight.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("dataset_error:",d.get("dataset_error"))
print("preparation_error:",d.get("preparation_error"))
print("candidates:",d.get("candidate_preparations"))
PY
  echo "Preflight failed. No Qwen calls, no incomplete case substitutions." >&2
  exit 3
fi

python - "$ROOT/preflight.json" "$ROOT/config.json" <<'PY'
import json,sys
from rave_fl.eval.adapters import load_instances
d=json.load(open(sys.argv[1]))
rows=d.get("candidate_preparations",[])
expected={"Time-1","Lang-1","Math-1"}
ready={r.get("instance_id") for r in rows if r.get("status")=="ready"}
print("Preflight prepared:",rows)
assert d.get("ready") and d.get("inference_performed") is False
assert ready==expected and len(rows)==3
config=json.load(open(sys.argv[2]))
instances, _ = load_instances(config["benchmark"])
assert {i.instance_id for i in instances}==expected
assert all(i.evidence.kind=="observed_failure" for i in instances)
print("All three buggy revisions, candidate pools and observed evidence validated.")
PY

echo "Starting bounded Qwen run, up to 135 physical calls and 240,000 provider tokens."
set +e
python -m rave_fl.graft.cli run \
  --config "$ROOT/config.json" \
  --output "$ROOT/run" \
  --allow-inference 2>&1 | tee "$ROOT/run.log"
run_status=$?
set -e
echo "Qwen run exit code: $run_status"

if [[ -f "$ROOT/run/results.json" && -f "$ROOT/run/accounting.json" ]]; then
  python -m scripts.audit_graft_phase3_multibug \
    "$ROOT/run" \
    --output "$ROOT/paired_audit.json" \
    > "$ROOT/paired_audit.txt" 2>&1 || true
  cat "$ROOT/paired_audit.txt"
fi
if [[ -f "$ROOT/run/manifest.json" ]]; then
  python -m rave_fl.graft.cli report \
    --run-dir "$ROOT/run" > "$ROOT/report.log" 2>&1 || true
fi

# Keep public git commits scoped only to run outputs; never publish separate
# official source checkouts, reproduction test trees or large dataset mirrors.
big=$(find "$ROOT" -type f -size +90M -print -quit)
if [[ -n "$big" ]]; then
  echo "Oversize file: $big. Results retained locally for review." >&2
  exit 2
fi
if grep -RIlE --include="*.json" --include="*.jsonl" --include="*.log" \
  "(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)" \
  "$ROOT" | grep -q .; then
  echo "Possible secrets in output. Retained locally, not pushed." >&2
  exit 2
fi

git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"
git add -f -- "$ROOT"
while IFS= read -r lock; do
  git rm --cached -q --ignore-unmatch -- "$lock"
done < <(find "$ROOT" -type f -name "*.lock")
git diff --cached --stat
if ! git diff --cached --quiet; then
  git commit -m "experiment(graft): archive Phase-3 real three-bug paired BM25 and test-aware comparison"
  git push origin graft-fl-aiscalar-20261113
fi
echo "Experiment archived locally: $ROOT"
if [[ "$run_status" -ne 0 ]]; then
  echo "Incomplete experiment, archived evidence if safe. Never infer missing results." >&2
  exit "$run_status"
fi
