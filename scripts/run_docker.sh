#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
compose=(docker compose -f compose.yaml)
mode="${PIPELINE_DEVICE:-auto}"
if [[ "$mode" == auto ]]; then
  if [[ "$(uname -s)" == Linux ]] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    mode=nvidia
  else
    mode=cpu
  fi
fi
case "$mode" in
  nvidia) compose+=(-f compose.nvidia.yaml) ;;
  cpu) ;;
  *) echo 'PIPELINE_DEVICE must be auto, nvidia, or cpu' >&2; exit 2 ;;
esac
mkdir -p "${RESULTS_DIR:-results}"
"${compose[@]}" config --quiet
docker info >/dev/null
echo "Docker pipeline mode: $mode"
"${compose[@]}" build pipeline
"${compose[@]}" up -d --wait ollama
if [[ "$mode" == nvidia ]]; then
  "${compose[@]}" exec -T ollama nvidia-smi
fi
"${compose[@]}" run --rm -T pipeline "$@"
