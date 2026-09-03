#!/usr/bin/env bash
set -euo pipefail

# Example: MODELS='llama3:8b,qwen2:7b,mistral:7b' BUGS='Time-25' ./scripts/run_defects4j_benchmark.sh
MODELS="${MODELS:-llama3:8b,qwen2:7b,mistral:7b}"
BUGS="${BUGS:-Time-25}"
BACKEND="${BACKEND:-ollama}"

python3 wp1/run_wp1_benchmark.py \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --backend "$BACKEND" \
  --models "$MODELS" \
  --bugs "$BUGS" \
  "$@"
