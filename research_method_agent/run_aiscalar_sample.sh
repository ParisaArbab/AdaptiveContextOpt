#!/usr/bin/env bash
# Run from root AdaptiveContextOpt. No Qwen calls in prepare/test modes.
set -euo pipefail
cd "$(dirname "$0")/.."
# Make the parent repository importable when executing the agent by file path.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
MODE="${1:-prepare}"
CONFIG="${GRAFT_CONFIG:-data/graft_100_cohort_20261115/batches/batch_00/experiment/config.json}"
OUT="${METHOD_AGENT_OUTPUT:-results/method_discovery_batch00}"
PYTHON="${RAVE_PYTHON:-python}"
case "$MODE" in
  test) "$PYTHON" -m unittest discover -s research_method_agent/tests -v ;;
  prepare)
    "$PYTHON" research_method_agent/research_method_agent.py study --config "$CONFIG" --output-dir "$OUT" --candidate-limit 24
    echo "No inference occurred. Review $OUT/manifest.json."
    ;;
  run)
    "$PYTHON" research_method_agent/research_method_agent.py study --config "$CONFIG" --output-dir "$OUT" --candidate-limit 24 --ollama-url "${OLLAMA_URL:-http://127.0.0.1:11434}" --l1-model "${L1_MODEL:-qwen2.5:7b}" --full-model "${FULL_MODEL:-qwen2.5:7b}" --max-calls "${MAX_CALLS:-450}" --physical-token-cap "${PHYSICAL_TOKEN_CAP:-800000}" --budget "${POLICY_TOKEN_BUDGET:-60000}" --allow-inference
    ;;
  sweep)
    "$PYTHON" research_method_agent/research_method_agent.py sweep --manifest "$OUT/manifest.json" --observations "$OUT/observations.json" --gold "$OUT/gold_offline.json" --budgets "${BUDGETS:-16000,24000,40000,60000}" --max-verifies "${FULL_CAPS:-1,2,3}" --buffers "${BUFFERS:-0,1}" --output "$OUT/sweep"
    ;;
  *) echo "Usage: bash research_method_agent/run_aiscalar_sample.sh {test|prepare|run|sweep}" >&2; exit 2;;
esac
