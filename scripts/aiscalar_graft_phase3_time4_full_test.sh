#!/usr/bin/env bash
# Bounded Time-4 real-Qwen follow-up with AUTHENTIC complete buggy test source.
# Development data only, separate from the new independent-bug evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(git branch --show-current)" == "graft-fl-aiscalar-20261113" ]] || {
  echo "Wrong branch" >&2; exit 2;
}
git diff --quiet && git diff --cached --quiet || {
  echo "Preserve tracked changes before running" >&2; exit 2;
}
python -m pytest -q \
  tests/test_graft_time4_official_trigger.py \
  tests/test_graft_phase3_real_bugs.py \
  tests/test_graft_test_aware.py \
  tests/test_graft_test_aware_live_config.py \
  tests/test_graft_controller.py

ROOT="results/graft_phase3_time4_full_test_$(date +%Y%m%d_%H%M%S)"
python -m scripts.prepare_graft_time4_authentic_test --output "$ROOT"
python - "$ROOT/provenance.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
source=p["official_trigger_provenance"]
assert source["test_source_role"]=="official_defects4j_trigger_test_only"
assert source["raw_buggy_revision_contains_test_method"] is False
assert source["fixed_production_source_read"] is False
assert source["buggy_checkout_modified"] is False
assert source["new_test_execution_performed"] is False
assert p["trigger_is_original_and_not_observed"] is True
print("Verified: authentic Defects4J trigger test only, production checkout remains buggy.")
PY
if ! python -m rave_fl.graft.cli preflight \
   --config "$ROOT/config.json" > "$ROOT/preflight.json"; then
  python - "$ROOT/preflight.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("dataset_error:", d.get("dataset_error"))
print("preparation_error:", d.get("preparation_error"))
print("candidates:", d.get("candidate_preparations"))
PY
  echo "Preflight failed, no Qwen calls." >&2
  exit 3
fi
python - "$ROOT/preflight.json" "$ROOT/config.json" <<'PY'
import json,sys
from rave_fl.eval.adapters import load_instances
d=json.load(open(sys.argv[1]))
x=d.get("candidate_preparations",[])
assert d.get("ready") and d.get("inference_performed") is False
assert len(x)==1 and x[0].get("instance_id")=="Time-4" and x[0].get("status")=="ready"
config=json.load(open(sys.argv[2]))
instances, _ = load_instances(config["benchmark"])
assert len(instances)==1 and instances[0].instance_id=="Time-4"
assert instances[0].evidence.kind=="trigger_description"
print("Full-test Time-4 preflight passed, evidence kind checked against loaded manifest.")
PY
set +e
python -m rave_fl.graft.cli run \
  --config "$ROOT/config.json" --output "$ROOT/run" \
  --allow-inference 2>&1 | tee "$ROOT/run.log"
status=$?
set -e
if [[ -f "$ROOT/run/results.json" && -f "$ROOT/run/accounting.json" ]]; then
  python scripts/report_graft_graph_ablation.py "$ROOT/run" \
    > "$ROOT/ablation_summary.txt" 2>&1 || true
  cat "$ROOT/ablation_summary.txt"
fi
if [[ -f "$ROOT/run/manifest.json" ]]; then
  python -m rave_fl.graft.cli report --run-dir "$ROOT/run" \
    > "$ROOT/report.log" 2>&1 || true
fi
big=$(find "$ROOT" -type f -size +90M -print -quit)
if [[ -n "$big" ]]; then echo "Oversize archive: $big" >&2; exit 2; fi
if grep -RIlE --include="*.json" --include="*.jsonl" --include="*.log" \
  "(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)" \
  "$ROOT" | grep -q .; then
  echo "Possible secret, review before pushing." >&2; exit 2
fi
git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"
git add -f -- "$ROOT"
while IFS= read -r lock; do
  git rm --cached -q --ignore-unmatch -- "$lock"
done < <(find "$ROOT" -type f -name "*.lock")
if ! git diff --cached --quiet; then
  git commit -m "experiment(graft): archive authentic full test Time-4 comparison"
  git push origin graft-fl-aiscalar-20261113
fi
echo "Saved: $ROOT"
exit "$status"
