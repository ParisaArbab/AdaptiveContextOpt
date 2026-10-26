#!/usr/bin/env bash
set -euo pipefail
RAVE_SETUP_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
printf '%s\n' 'Read-only NVIDIA preflight. CUDA_VISIBLE_DEVICES controls a local model server, not a remote endpoint.'
printf '%s\n' 'Keep Ollama bound to loopback; configure an authenticated tunnel for remote clients.'
exec bash "$RAVE_SETUP_DIR/rave_fl_run.sh" preflight --device a100 "$@"
