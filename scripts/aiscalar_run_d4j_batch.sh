#!/usr/bin/env bash
# Run independent real Defects4J RAVE-FL comparisons sequentially on AIScalar.
# Each bug gets its own durable request journal, budget and output directory.
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST="${1:-configs/aiscalar_multi_d4j.json}"
ENDPOINT="${RAVE_OLLAMA_ENDPOINT:-http://127.0.0.1:11437}"
MODEL="${RAVE_MODEL:-qwen2.5:7b}"
# Keep the original two-method batch as the default for reproducibility.
# Set RAVE_BATCH_SIZE=1 to test robust singleton screening as a separate arm.
BATCH_SIZE="${RAVE_BATCH_SIZE:-2}"
OUTPUT_TOKENS="${RAVE_OUTPUT_TOKENS:-160}"
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ && "$OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: RAVE_BATCH_SIZE and RAVE_OUTPUT_TOKENS must be positive integers" >&2
  exit 2
fi
STAMP="${RAVE_BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="results/d4j_multibug/${STAMP}"
LOGROOT="logs/d4j_multibug/${STAMP}"

if [[ ! -s "$MANIFEST" ]]; then
  echo "ERROR: missing manifest: $MANIFEST" >&2
  exit 2
fi
if ! curl --max-time 5 --fail --silent "$ENDPOINT/api/version" >/dev/null; then
  echo "ERROR: dedicated Ollama endpoint is not responsive: $ENDPOINT" >&2
  exit 2
fi
mkdir -p "$LOGROOT"
echo "Batch configuration: model=$MODEL, endpoint=$ENDPOINT, L1 batch size=$BATCH_SIZE, output tokens per candidate=$OUTPUT_TOKENS" | tee -a "$LOGROOT/progress.log"
printf 'bug\texit_code\trun_dir\n' > "$LOGROOT/run_status.tsv"
mapfile -t BUGS < <(python3 - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text())["instances"]
if not rows:
    raise SystemExit("ERROR: no prepared Defects4J instances")
for row in rows:
    print(row["instance_id"])
PY
)

for BUG in "${BUGS[@]}"; do
  ID="d4j_${BUG}_v2_${STAMP}"
  OUTPUT="$ROOT/$BUG"
  echo "=== Starting $BUG: Full / Fixed L1 / V2 Utility ===" | tee -a "$LOGROOT/progress.log"
  # The runner invokes the same real inference provider for every selected bug.
  # Do not resume a failed request with unknown usage or delete past ledgers.
  if bash scripts/rave_fl_run.sh run \
      --dataset defects4j \
      --dataset-file "$MANIFEST" \
      --bugs "$BUG" \
      --candidate-provider flexfl \
      --candidate-count 24 \
      --top-per-method 10 \
      --backend ollama \
      --model "$MODEL" \
      --endpoint "$ENDPOINT" \
      --device a100 \
      --policies full,fixed_l1,v2_utility \
      --fusions shared_prior_evidence_fusion_v1 \
      --context-window 8192 \
      --batch-size "$BATCH_SIZE" \
      --output-tokens "$OUTPUT_TOKENS" \
      --timeout 900 \
      --admission strict \
      --budget 120000 \
      --physical-budget 100000 \
      --per-bug-physical-budget 80000 \
      --max-new-calls 80 \
      --max-full 2 \
      --workers 1 \
      --require-observed-failure \
      --run-id "$ID" \
      --output-dir "$OUTPUT" \
      --allow-inference > "$LOGROOT/$BUG.log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\t%s\t%s\n' "$BUG" "$rc" "$OUTPUT" >> "$LOGROOT/run_status.tsv"
  echo "=== Finished $BUG (exit $rc), output $OUTPUT ===" | tee -a "$LOGROOT/progress.log"
done

echo "=== Batch completed. Check per-bug statuses and reports ==="
cat "$LOGROOT/run_status.tsv"
echo "Log folder: $LOGROOT"
echo "Result folder: $ROOT"
