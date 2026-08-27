#!/usr/bin/env bash
#
# run_pipeline.sh — the whole study in one command.
#
#   fetch instances -> run EVERY ablation arm (no optimization, full
#   pipeline, and each element removed) -> evaluate -> plot.
#
# The LLM and the benchmark are both arguments, so re-running the study on a
# different model or a different dataset is a flag change, not an edit:
#
#   ./scripts/run_pipeline.sh --llm ollama --model qwen2.5-coder:32b
#   ./scripts/run_pipeline.sh --llm deepseek --dataset swe-bench-verified
#   ./scripts/run_pipeline.sh --llm vllm --model Qwen/Qwen2.5-Coder-32B-Instruct \
#       --dataset swe-bench-java --language java --n 20
#
# Every run writes to its own directory under results/, keyed by llm +
# dataset + timestamp, so runs never overwrite each other and two models can
# be compared afterwards from the artifacts alone.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WP1="$ROOT/wp1"
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------- defaults
LLM="heuristic"
MODEL=""
BASE_URL=""
API_KEY_ENV=""
TOKENIZER_MODEL=""
TEMPERATURE="0.0"
MAX_TOKENS="1024"

DATASET="swe-bench-lite"
LANGUAGE=""
SPLIT=""
N="15"
SEED="42"
INSTANCE_IDS=""
FIELD_MAP=()

ARMS="default"
TARGET_DENSITY="0.4"
LIMIT="0"
LOCAL_FALLBACK=0
METRIC="top1"

RUN_DIR=""
INSTANCES_FILE=""
SKIP_FETCH=0
SKIP_PLOTS=0
PREFLIGHT=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
run_pipeline.sh — fetch, run all ablation arms, evaluate, and plot.

MODEL
  --llm NAME             provider or alias (default: heuristic)
                         hosted:  openai gpt anthropic claude gemini deepseek
                                  qwen openrouter together groq mistral
                                  ollama-cloud
                         local:   ollama vllm lmstudio tgi llamacpp
                         other:   custom (with --base-url), heuristic
  --model ID             model id; required for ollama/vllm/openrouter/...
  --base-url URL         override the provider endpoint
  --api-key-env VAR      env var holding the key
  --tokenizer-model ID   HF repo id for exact token counts (e.g. with ollama tags)
  --temperature F        default 0.0, keeps arms comparable
  --max-tokens N         default 1024
  --preflight            one cheap test call before the run
  --list-providers       print the provider table and exit

BENCHMARK
  --dataset NAME         swe-bench-lite | swe-bench-verified | swe-bench |
                         swe-bench-java | swe-bench-multimodal | any HF id
  --language NAME        python | java (override the registry default)
  --split NAME           dataset split (default: test)
  --n N                  instances to sample (default: 15)
  --seed N               sampling seed (default: 42)
  --instance-ids A,B     pin exact instance ids instead of sampling
  --field-map k=v        schema override, repeatable
  --instances FILE       reuse an existing instances file (implies --skip-fetch)

RUN
  --arms SPEC            default | all | comma-separated names (default: default)
                         'default' = full, no_graphify, no_leanctx,
                                     no_feedback, pure_flexfl
  --local-fallback       run trigger tests on the host instead of Docker images
  --target-density F     lean-ctx density target (default: 0.4)
  --limit N              cap instances processed (0 = all)
  --metric NAME          accuracy axis for the trade-off plot (default: top1)
  --out-dir DIR          output directory (default: results/<llm>_<dataset>_<ts>)
  --skip-fetch           reuse the instances file without re-fetching
  --skip-plots           stop after the evaluation report
  --dry-run              print the commands without running them
  -h, --help             this text
USAGE
}

# ------------------------------------------------------------------ parse
while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm|--backend)    LLM="$2"; shift 2 ;;
    --model)            MODEL="$2"; shift 2 ;;
    --base-url)         BASE_URL="$2"; shift 2 ;;
    --api-key-env)      API_KEY_ENV="$2"; shift 2 ;;
    --tokenizer-model)  TOKENIZER_MODEL="$2"; shift 2 ;;
    --temperature)      TEMPERATURE="$2"; shift 2 ;;
    --max-tokens)       MAX_TOKENS="$2"; shift 2 ;;
    --preflight)        PREFLIGHT=1; shift ;;
    --list-providers)   "$PYTHON" "$WP1/llm_backends.py"; exit 0 ;;
    --dataset)          DATASET="$2"; shift 2 ;;
    --language)         LANGUAGE="$2"; shift 2 ;;
    --split)            SPLIT="$2"; shift 2 ;;
    --n)                N="$2"; shift 2 ;;
    --seed)             SEED="$2"; shift 2 ;;
    --instance-ids)     INSTANCE_IDS="$2"; shift 2 ;;
    --field-map)        FIELD_MAP+=("--field-map" "$2"); shift 2 ;;
    --instances)        INSTANCES_FILE="$2"; SKIP_FETCH=1; shift 2 ;;
    --arms)             ARMS="$2"; shift 2 ;;
    --local-fallback)   LOCAL_FALLBACK=1; shift ;;
    --target-density)   TARGET_DENSITY="$2"; shift 2 ;;
    --limit)            LIMIT="$2"; shift 2 ;;
    --metric)           METRIC="$2"; shift 2 ;;
    --out-dir)          RUN_DIR="$2"; shift 2 ;;
    --skip-fetch)       SKIP_FETCH=1; shift ;;
    --skip-plots)       SKIP_PLOTS=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; echo "try --help" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ paths
slug() { echo "$1" | tr '/:. ' '____' | tr -cd '[:alnum:]_-'; }
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="$(slug "$LLM")$([[ -n "$MODEL" ]] && echo "_$(slug "$MODEL")")_$(slug "$DATASET")_$STAMP"
RUN_DIR="${RUN_DIR:-$ROOT/results/$RUN_TAG}"
INSTANCES_FILE="${INSTANCES_FILE:-$RUN_DIR/instances.json}"
RESULTS_JSON="$RUN_DIR/wp1_results.json"
REPORT_JSON="$RUN_DIR/compression_tax_report.json"
PLOTS_DIR="$RUN_DIR/plots"
LOG="$RUN_DIR/run.log"

run() {
  echo "+ $*" | tee -a "$LOG"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
  fi
}

mkdir -p "$RUN_DIR"
: > "$LOG"

cat <<BANNER | tee -a "$LOG"
==========================================================================
 AdaptiveContextOpt — full ablation study
   llm        : $LLM${MODEL:+ ($MODEL)}
   benchmark  : $DATASET${LANGUAGE:+ / $LANGUAGE}   n=$N seed=$SEED
   arms       : $ARMS
   capture    : $([[ $LOCAL_FALLBACK -eq 1 ]] && echo "local pytest fallback" || echo "SWE-bench Docker images")
   output     : $RUN_DIR
==========================================================================
BANNER

# Fail fast on a missing dependency rather than after the (slow) capture step.
if [[ $DRY_RUN -eq 0 ]]; then
  MISSING="$("$PYTHON" - <<'PY'
missing = []
for mod, why in (("datasets", "--dataset fetching"), ("matplotlib", "plotting")):
    try:
        __import__(mod)
    except ImportError:
        missing.append(f"{mod} ({why})")
print("; ".join(missing))
PY
)"
  if [[ -n "$MISSING" ]]; then
    echo "warning: missing optional dependency: $MISSING" | tee -a "$LOG"
    echo "         pip install -r $ROOT/requirements.txt" | tee -a "$LOG"
  fi
fi

if [[ $PREFLIGHT -eq 1 && "$LLM" != "heuristic" ]]; then
  PRE=("$PYTHON" "$WP1/llm_backends.py" --check "$LLM")
  [[ -n "$MODEL" ]]       && PRE+=(--model "$MODEL")
  [[ -n "$BASE_URL" ]]    && PRE+=(--base-url "$BASE_URL")
  [[ -n "$API_KEY_ENV" ]] && PRE+=(--api-key-env "$API_KEY_ENV")
  run "${PRE[@]}"
fi

# ------------------------------------------------------- 1. fetch instances
if [[ $SKIP_FETCH -eq 0 ]]; then
  FETCH=("$PYTHON" "$WP1/fetch_instances.py"
         --dataset "$DATASET" --n "$N" --seed "$SEED" --out "$INSTANCES_FILE")
  [[ -n "$LANGUAGE" ]]      && FETCH+=(--language "$LANGUAGE")
  [[ -n "$SPLIT" ]]         && FETCH+=(--split "$SPLIT")
  [[ -n "$INSTANCE_IDS" ]]  && FETCH+=(--instance-ids "$INSTANCE_IDS")
  [[ ${#FIELD_MAP[@]} -gt 0 ]] && FETCH+=("${FIELD_MAP[@]}")
  run "${FETCH[@]}"
else
  echo "reusing instances: $INSTANCES_FILE" | tee -a "$LOG"
  [[ $DRY_RUN -eq 1 || -f "$INSTANCES_FILE" ]] || { echo "missing: $INSTANCES_FILE" >&2; exit 1; }
fi

# ------------------------------------- 2. every arm, in a single invocation
BENCH=("$PYTHON" "$WP1/run_wp1_benchmark.py"
       --instances "$INSTANCES_FILE"
       --arms "$ARMS"
       --llm "$LLM"
       --temperature "$TEMPERATURE"
       --max-tokens "$MAX_TOKENS"
       --target-density "$TARGET_DENSITY"
       --repos-dir "$ROOT/data/repos"
       --out "$RESULTS_JSON")
[[ -n "$MODEL" ]]            && BENCH+=(--model "$MODEL")
[[ -n "$BASE_URL" ]]         && BENCH+=(--base-url "$BASE_URL")
[[ -n "$API_KEY_ENV" ]]      && BENCH+=(--api-key-env "$API_KEY_ENV")
[[ -n "$TOKENIZER_MODEL" ]]  && BENCH+=(--tokenizer-model "$TOKENIZER_MODEL")
[[ "$LIMIT" != "0" ]]        && BENCH+=(--limit "$LIMIT")
[[ $LOCAL_FALLBACK -eq 1 ]]  && BENCH+=(--local-fallback)
run "${BENCH[@]}"

# A run where every instance was skipped (no Docker images, clone failures,
# captures with no failure evidence) would otherwise fail deep inside the
# plotting code. Fail here instead, where the cause is still visible.
if [[ $DRY_RUN -eq 0 ]]; then
  N_OUTCOMES="$("$PYTHON" - "$RESULTS_JSON" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
outcomes = payload["outcomes"] if isinstance(payload, dict) else payload
skipped = payload.get("skipped", []) if isinstance(payload, dict) else []
print(len(outcomes))
if not outcomes and skipped:
    print("skip reasons:", file=sys.stderr)
    seen = set()
    for s in skipped:
        r = s.get("reason", "?")
        if r not in seen:
            seen.add(r)
            print(f"  - {s.get('instance_id','?')}: {r}", file=sys.stderr)
PY
)"
  if [[ "$N_OUTCOMES" == "0" ]]; then
    echo "no outcomes produced — nothing to evaluate or plot (see reasons above and $LOG)" >&2
    exit 1
  fi
  echo "outcomes: $N_OUTCOMES" | tee -a "$LOG"
fi

# ------------------------------------------------- 3. evaluation framework
run "$PYTHON" "$WP1/compression_tax_analyzer.py" \
    --results "$RESULTS_JSON" --out "$REPORT_JSON"

# ------------------------------------------------------------- 4. figures
if [[ $SKIP_PLOTS -eq 0 ]]; then
  run "$PYTHON" "$WP1/plot_results.py" \
      --report "$REPORT_JSON" --results "$RESULTS_JSON" \
      --out-dir "$PLOTS_DIR" --metric "$METRIC"
fi

cat <<DONE | tee -a "$LOG"

--------------------------------------------------------------------------
 done — $RUN_DIR
   instances : $INSTANCES_FILE
   outcomes  : $RESULTS_JSON
   report    : $REPORT_JSON
$([[ $SKIP_PLOTS -eq 0 ]] && echo "   plots     : $PLOTS_DIR")
   log       : $LOG
--------------------------------------------------------------------------
DONE
