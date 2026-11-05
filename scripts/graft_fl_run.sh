#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${GRAFT_PYTHON:-python3}" -m rave_fl.graft.cli "$@"
