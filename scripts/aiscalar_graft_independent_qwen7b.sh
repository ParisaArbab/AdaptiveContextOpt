#!/usr/bin/env bash
# Exploratory independent-bug comparison; not a held-out study.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="results/graft_independent_qwen7b_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROOT"
echo "Results directory: $ROOT"
python scripts/prepare_graft_independent_checkouts.py --fetch-pinned
python scripts/prepare_graft_independent_inputs.py --output "$ROOT" --expand-flexfl

for arm in control anchored; do
  config="$ROOT/config_$arm.json"
  python -m rave_fl.graft.cli preflight --config "$config" > "$ROOT/preflight_$arm.json"
done

python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
want={"Time-1","Lang-1","Math-1"}
for arm in ("control","anchored"):
    p=json.loads((root/f"preflight_{arm}.json").read_text())
    actual={row["instance_id"] for row in p.get("candidate_preparations", [])}
    print(f"{arm} preflight: ready={p.get('ready')} selected={sorted(actual)}")
    for row in p.get("candidate_preparations", []):
        print(" ",row["instance_id"],row["status"],"candidates",row.get("candidates"),"error",row.get("error"))
    if not p.get("ready") or actual != want:
        print("ERROR:", p.get("dataset_error"), p.get("preparation_error"))
        raise SystemExit("Input preparation failed. No inference launched; inspect preflight JSON.")
PY

for arm in control anchored; do
  config="$ROOT/config_$arm.json"
  output="$ROOT/$arm"
  echo "Starting $arm..."
  set +e
  python -m rave_fl.graft.cli run \
    --config "$config" --output "$output" --allow-inference \
    2>&1 | tee "$ROOT/$arm.log"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 && "$status" -ne 3 ]]; then
    echo "Fatal GRAFT runner exit $status for $arm" >&2
    exit "$status"
  fi
  if [[ "$status" -eq 3 ]]; then
    echo "INCOMPLETE $arm, retained for operational evaluation"
  fi
  python -m rave_fl.graft.cli report --run-dir "$output" \
    2>&1 | tee "$ROOT/${arm}_report.log"
done

python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for arm in ("control","anchored"):
    rows=json.loads((root/arm/"results.json").read_text())
    print("\n",arm.upper())
    for row in rows:
        if row.get("policy") not in ("prior","fixed_l1","highest_suspicion","l1_graph","graft_adaptive"):
            continue
        m=row.get("metrics",{}).get("java_strict_v2",{}).get("method",{})
        print(row.get("instance_id"),row.get("policy"),row.get("status"),
              "rank",m.get("first_relevant_rank"),
              "Top5",m.get("Top-5"),"MRR",m.get("MRR"),
              "unlabeled",row.get("gold_unavailable_reason"))
    a=json.loads((root/arm/"accounting.json").read_text())
    print("physical calls",a["new_calls"],"tokens",a["known_provider_tokens"])
print("\nSAVE THIS DIRECTORY:",root)
PY
