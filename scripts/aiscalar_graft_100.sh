#!/usr/bin/env bash
# AIScalar 100-bug study. Authentic official failures, strictly bounded
# Qwen calls, 20 five-bug batches, and explicit incomplete/missing accounting.
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="graft-fl-aiscalar-20261113"
ROOT="data/graft_100_cohort_20261115"
MODE="${1:-}"
BATCH="${2:-}"

if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  echo "Wrong research branch, no inference started." >&2
  exit 2
fi
if [[ "$MODE" != "status" && "$MODE" != "aggregate" ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Uncommitted tracked files, preserve first." >&2
    exit 2
  fi
fi
if [[ "$MODE" == "collect" || "$MODE" == "run" || "$MODE" == "all" || "$MODE" == "launch-all" ]]; then
  export PERL5LIB="$HOME/perl5/lib/perl5"
  export PATH="$HOME/defects4j/framework/bin:$PATH"
  export TZ="America/Los_Angeles"
fi

archive() {
  local message="$1"
  shift
  local path
  for path in "$@"; do
    if [[ ! -e "$path" ]]; then continue; fi
    if [[ -d "$path" ]]; then
      if find "$path" -type f -size +90M -print -quit | grep -q .; then
        echo "Oversize artifact, not pushed: $path" >&2
        return 2
      fi
      if grep -RIlE --include="*.json" --include="*.jsonl" --include="*.log" \
        "(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)" \
        "$path" | grep -q .; then
        echo "Possible credentials in artifact, not pushed: $path" >&2
        return 2
      fi
    fi
  done
  git config user.name "ParisaArbab"
  git config user.email "pa.arbab@gmail.com"
  for path in "$@"; do
    [[ ! -e "$path" ]] || git add -f -- "$path"
  done
  if ! git diff --cached --quiet; then
    git diff --cached --check
    git commit -m "$message" | tail -n 6
    git push origin "$BRANCH"
  fi
}

archive_prepared() {
  local idx="$1"
  local base="$ROOT/batches/batch_$idx/prepared"
  local files=()
  local file
  for file in "$base"/manifest.json "$base"/preparation_report.json \
      "$base"/records/*.json "$base"/audits/*.json \
      "$base"/evidence/*.txt "$base"/test_runs/*.stdout.txt \
      "$base"/test_runs/*.stderr.txt; do
    [[ ! -f "$file" ]] || files+=("$file")
  done
  archive "dataset(graft): archive authentic 100-bug batch $idx evidence and preparation" "$base/preparation_report.json" "$base/manifest.json" "$base/records" "$base/audits" "$base/evidence" "$base/test_runs"
}

archive_experiment() {
  local idx="$1"
  local path="$ROOT/batches/batch_$idx/experiment"
  if [[ -d "$path" ]]; then
    # Preserve all results, cache, ledgers and configs, but never lock files.
    find "$path" -type f -name '*.lock' -delete
    archive "experiment(graft): archive 100-bug Qwen batch $idx with full accounting" "$path"
  fi
}

case "$MODE" in
  freeze)
    [[ ! -e "$ROOT/cohort.json" ]] || { echo "Cohort already frozen." >&2; exit 2; }
    python -m scripts.graft_100_dataset freeze --root "$ROOT" --seed 42
    archive "research(graft): freeze 100 Defects4J bug IDs before any observations" \
      "$ROOT/cohort.json" "$ROOT/splits.json"
    ;;
  inspect)
    python -m scripts.graft_100_dataset inspect --root "$ROOT" --batch "$BATCH"
    ;;
  collect)
    [[ "$BATCH" =~ ^(0?[0-9]|1[0-9])$ ]] || { echo "Batch 0..19 required" >&2; exit 2; }
    python -m scripts.graft_100_dataset collect --root "$ROOT" \
      --batch "$((10#$BATCH))" --allow-real-tests
    ;;
  run)
    [[ "$BATCH" =~ ^(0?[0-9]|1[0-9])$ ]] || { echo "Batch 0..19 required" >&2; exit 2; }
    python -m scripts.graft_100_run run --root "$ROOT" \
      --batch "$((10#$BATCH))" --allow-inference
    ;;
  aggregate)
    python -m scripts.graft_100_run aggregate --root "$ROOT"
    archive "research(graft): aggregate frozen 100-bug real experiment with attrition" "$ROOT/aggregate"
    ;;
  all)
    [[ -f "$ROOT/cohort.json" ]] || { echo "Run freeze first." >&2; exit 2; }
    failures=0
    for index in $(seq 0 19); do
      printf -v idx "%02d" "$index"
      echo "======= BUGS BATCH $idx OF 19 ======="
      status=0
      python -m scripts.graft_100_dataset collect --root "$ROOT" \
        --batch "$index" --allow-real-tests || status=$?
      archive_prepared "$idx"
      if [[ "$status" -eq 0 ]]; then
        python -m scripts.graft_100_run run --root "$ROOT" \
          --batch "$index" --allow-inference || status=$?
        archive_experiment "$idx"
      else
        echo "Batch $idx has unavailable real tests. No bug substitutions."
      fi
      if [[ "$status" -ne 0 ]]; then
        failures=$((failures + 1))
        echo "Batch $idx marked incomplete, preserved for audit."
      else
        echo "Batch $idx real graph comparison complete."
      fi
    done
    python -m scripts.graft_100_run aggregate --root "$ROOT"
    archive "research(graft): report real 100-bug cohort outcomes and failures" "$ROOT/aggregate"
    echo "Attempted all 20 frozen batches; incomplete batches: $failures"
    [[ "$failures" -eq 0 ]] || exit 3
    ;;
  launch-all)
    [[ -f "$ROOT/cohort.json" ]] || { echo "Freeze bug IDs before launch." >&2; exit 2; }
    command -v flock >/dev/null && command -v nohup >/dev/null || {
      echo "flock and nohup are required for safe detached execution" >&2
      exit 2
    }
    mkdir -p "$ROOT/logs"
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="$ROOT/logs/graft_100_$stamp.log"
    lock="$ROOT/.graft100_process.lock"
    nohup flock -n -E 12 "$lock" bash scripts/aiscalar_graft_100.sh all \
      > "$log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$ROOT/logs/graft_100_$stamp.pid"
    echo "Detached AIScalar process: $pid"
    echo "Log: $log"
    echo "Follow: tail -f '$log'"
    echo "Physical limits per batch: 500k provider tokens and 220 new Qwen calls."
    ;;
  status)
    python -m scripts.graft_100_run aggregate --root "$ROOT"
    echo "See $ROOT/aggregate/aggregate.md for completed denominators"
    ;;
  *)
    echo "Usage: bash scripts/aiscalar_graft_100.sh freeze | inspect BATCH | collect BATCH | run BATCH | launch-all | status | aggregate" >&2
    exit 2
    ;;
esac
