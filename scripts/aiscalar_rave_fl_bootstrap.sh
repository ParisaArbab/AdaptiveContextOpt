#!/usr/bin/env bash
# Bootstrap an isolated RAVE-FL Conda environment on AIScalar.
# Run from the root of AdaptiveContextOpt:
#   bash scripts/aiscalar_rave_fl_bootstrap.sh
# Does NOT use sudo, install Ollama, reserve a GPU, or modify other projects.
set -euo pipefail

if [[ ! -f "rave_fl/expanded_pool.py" ]]; then
  echo "ERROR: run this script from the AdaptiveContextOpt repository root." >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  for startup in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                 "$HOME/anaconda3/etc/profile.d/conda.sh" \
                 "$HOME/miniforge3/etc/profile.d/conda.sh"; do
    if [[ -r "$startup" ]]; then
      # shellcheck disable=SC1090
      source "$startup"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: Conda not found. Check 'module avail' or your server's Conda path." >&2
  exit 2
fi

# Activate Conda in this non-interactive shell.
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
NAME="${RAVE_CONDA_ENV:-rave-fl}"
echo "Conda base: $(conda info --base)"
if conda env list | awk -v n="$NAME" '$1 == n {found=1} END {exit !found}'; then
  echo "Using existing Conda environment: $NAME"
else
  echo "Creating Conda environment $NAME with Python 3.11 and Java 11..."
  conda create -y --name "$NAME" --channel conda-forge python=3.11 openjdk=11
fi
conda activate "$NAME"

echo "Environment: $CONDA_DEFAULT_ENV"
python --version
java -version
git config user.name "ParisaArbab"
git config user.email "pa.arbab@gmail.com"

echo "Running RAVE-FL unit tests..."
python -m unittest discover -s tests -p 'test_rave_fl*.py'

echo "Fetching only the FlexFL benchmark data needed for Phase 3..."
mkdir -p references
REF="references/FlexFL_OriginalReplication"
if [[ -d "$REF/.git" ]]; then
  git -C "$REF" pull --ff-only
elif [[ -e "$REF" ]]; then
  echo "ERROR: $REF exists but is not a Git checkout" >&2
  exit 2
else
  git clone https://github.com/ParisaArbab/FlexFL_OriginalReplication.git "$REF"
fi

echo "Checking the expanded candidate pool (gold labels used ONLY for this diagnostic)..."
python -m rave_fl.expanded_pool --flexfl-repo "$REF" --bug Time-25 --top-per-method 10 --diagnose

echo
echo "==== AIScalar prerequisite report ===="
echo "Machine: $(hostname)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(command -v python)"
for tool in defects4j git svn perl cpanm ollama nvidia-smi; do
  if command -v "$tool" >/dev/null 2>&1; then
    printf '%-12s %s\n' "$tool" "$(command -v "$tool")"
  else
    printf '%-12s %s\n' "$tool" "NOT FOUND"
  fi
done
echo "Java:"
java -version 2>&1 | head -n 2
echo "Free disk space:"
df -h . | tail -n 1
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU summary (if your server allows nvidia-smi):"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
fi
echo "Git email: $(git config user.email)"
echo "==== Bootstrap finished ===="
echo "Next: examine this report, then prepare Defects4J and an allowed GPU Ollama session."
