#!/usr/bin/env bash
# Run the frozen Top-8 experiment with the repository importable.
# IMPORTANT: Do not edit mixed_top8_validation.py here. Its SHA256 is part of
# the experiment's frozen_protocol.json created before model inference.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Direct invocation of python research_method_agent/mixed_top8_validation.py
# puts research_method_agent/ (not the repository root) on sys.path. That
# hides the local rave_fl package and breaks preflight before any LLM calls.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$ROOT/rave_fl" ]]; then
  echo "ERROR: local rave_fl/ package not found at $ROOT/rave_fl" >&2
  echo "Expected the full AdaptiveContextOpt checkout (not a sparse research_method_agent checkout)." >&2
  exit 2
fi

exec "${RAVE_PYTHON:-python}" research_method_agent/mixed_top8_validation.py "$@"
