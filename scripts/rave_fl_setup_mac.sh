#!/usr/bin/env bash
set -euo pipefail
RAVE_SETUP_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "$(uname -s)" in Darwin) ;; *) printf '%s\n' 'This setup check targets macOS; no packages will be installed.' >&2 ;; esac
printf '%s\n' 'Read-only macOS dependency/resource checks. No downloads, services, tests or inference.'
exec bash "$RAVE_SETUP_DIR/rave_fl_run.sh" preflight --device auto "$@"
