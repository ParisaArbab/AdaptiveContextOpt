#!/usr/bin/env bash
# Checkout Defects4J Time-25b and validate the Phase 3b source extraction.
# Uses a pre-existing Defects4J command. Does not install system packages.
set -euo pipefail

if [[ ! -f "rave_fl/compare_expanded.py" ]]; then
  echo "ERROR: run in the AdaptiveContextOpt repository root" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  for startup in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                 "$HOME/anaconda3/etc/profile.d/conda.sh" \
                 "$HOME/miniforge3/etc/profile.d/conda.sh"; do
    if [[ -r "$startup" ]]; then source "$startup"; break; fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: Conda not found. Run the bootstrap instructions first." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${RAVE_CONDA_ENV:-rave-fl}"

D4J_HOME="${D4J_HOME:-$HOME/defects4j}"
if [[ -d "$D4J_HOME/framework/bin" ]]; then
  export PATH="$D4J_HOME/framework/bin:$PATH"
fi
if ! command -v defects4j >/dev/null 2>&1; then
  echo "ERROR: Defects4J missing. Install it in your account or set D4J_HOME." >&2
  echo "Defects4J setup: https://github.com/rjust/defects4j" >&2
  exit 2
fi

echo "Defects4J path: $(command -v defects4j)"
defects4j info -p Time -b 25 >/dev/null
checkout="$PWD/data/rave_fl_checkouts/Time-25"
if [[ -e "$checkout/.defects4j.config" ]]; then
  echo "Reusing existing Defects4J checkout: $checkout"
elif [[ -e "$checkout" ]]; then
  echo "ERROR: $checkout exists but is not a valid Defects4J checkout." >&2
  exit 2
else
  mkdir -p "$(dirname "$checkout")"
  echo "Checking out actual buggy Time-25b version..."
  defects4j checkout -p Time -v 25b -w "$checkout"
fi

echo "Running source validation without making LLM calls..."
python -m rave_fl.compare_expanded \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --checkout "$checkout" \
  --bug Time-25 \
  --top-per-method 10 \
  --dry-run \
  --output results/rave_fl_phase3/expanded_dry_run.json

echo "Success. Real source bodies resolved. No LLM was called."
echo "Saved preflight result: results/rave_fl_phase3/expanded_dry_run.json"
