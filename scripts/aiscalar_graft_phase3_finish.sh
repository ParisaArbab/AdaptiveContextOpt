#!/usr/bin/env bash
# Detached one-command launcher for the *real*, fixed 3-bug GRAFT-FL graph
# comparison. Never fakes results, increases provider caps or uses a new cohort.
# The worker archives the full result even if the comparison remains incomplete.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="data/graft_phase3_official_20261113_215102/manifest.json"
ARCHIVE="results/graft_phase3_real_bugs_20261115_132745/run"

if [[ "$(git branch --show-current)" != "graft-fl-aiscalar-20261113" ]]; then
  echo "Wrong branch; no Qwen inference started." >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1 || ! command -v nohup >/dev/null 2>&1; then
  echo "flock and nohup are required to avoid duplicate runs and SSH hangups." >&2
  exit 2
fi
if [[ ! -f "$DATASET" || ! -f "$ARCHIVE/manifest.json" || \
      ( ! -d "$ARCHIVE/cache" && ! -d "$ARCHIVE/../cache" ) ]]; then
  echo "Verified prepared data or original Qwen cache is missing." >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit or preserve modified tracked files before starting inference." >&2
  exit 2
fi
mkdir -p results
stamp="$(date +%Y%m%d_%H%M%S)"
log="results/graft_phase3_detached_$stamp.log"
lock="results/.graft_phase3_real_3bug_inference.lock"

# Atomic advisory lock is held for the full worker process. A second launcher
# cannot start a duplicate job while one is executing.
nohup flock -n -E 12 "$lock" \
  bash scripts/aiscalar_graft_phase3_multibug.sh \
    "$DATASET" --seed-run "$ARCHIVE" \
  > "$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "results/graft_phase3_detached_$stamp.pid"

echo "Started detached experiment process: $pid"
echo "Live log: $log"
echo "Follow: tail -f '$log'"
echo "Check process: ps -p $pid -o pid,etime,stat,command"
echo "Completion requires all paired arms and actual graph calls; do not assume success from job launch."
