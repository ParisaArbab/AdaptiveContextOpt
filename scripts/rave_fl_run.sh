#!/usr/bin/env bash
set -euo pipefail
RAVE_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RAVE_REPO_ROOT="$(CDPATH= cd -- "$RAVE_SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$RAVE_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${RAVE_PYTHON:-python3}" -m rave_fl.eval.cli "$@"
