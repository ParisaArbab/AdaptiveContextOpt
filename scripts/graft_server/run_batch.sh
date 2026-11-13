#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec "${GRAFT_PYTHON:-python3}" -B -m scripts.graft_server.batch run-batch "$@"
