#!/usr/bin/env bash
# Fetch only real Time-25 FlexFL files with a Git sparse checkout.
# Keeps any existing references/FlexFL_OriginalReplication checkout untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$ROOT/references/FlexFL_Time25"
URL="https://github.com/ParisaArbab/FlexFL_OriginalReplication.git"

if [[ ! -d "$REF/.git" ]]; then
  if [[ -e "$REF" ]]; then
    echo "ERROR: $REF already exists but is not a Git checkout; inspect it manually." >&2
    exit 2
  fi
  mkdir -p "$(dirname "$REF")"
  echo "Creating small sparse FlexFL reference clone: $REF"
  git clone --depth 1 --filter=blob:none --sparse "$URL" "$REF"
fi

# All of these paths exist in the upstream repository. Bug report for Time-25
# is not present there; the trigger-test file supplies real failure context.
git -C "$REF" sparse-checkout set --no-cone \
  'FlexFL/data/FL_results/SBIR/Defects4J/Time-25_method-susps.csv' \
  'FlexFL/data/FL_results/Ochiai/Defects4J/Time-25_method-susps.csv' \
  'FlexFL/data/FL_results/BoostN/Defects4J/Time-25_method-susps.csv' \
  'FlexFL/data/input/ground_truth/Defects4J/gt.json' \
  'FlexFL/data/input/trigger_tests/Defects4J/Time-25.txt'

for method in SBIR Ochiai BoostN; do
  file="$REF/FlexFL/data/FL_results/$method/Defects4J/Time-25_method-susps.csv"
  if [[ ! -s "$file" ]]; then
    echo "ERROR: missing or empty $file" >&2
    exit 2
  fi
done
for file in \
  "$REF/FlexFL/data/input/ground_truth/Defects4J/gt.json" \
  "$REF/FlexFL/data/input/trigger_tests/Defects4J/Time-25.txt"; do
  if [[ ! -s "$file" ]]; then
    echo "ERROR: missing or empty $file" >&2
    exit 2
  fi
done

echo "Verified minimal FlexFL Time-25 reference data in $REF"
du -sh "$REF" 2>/dev/null || true
