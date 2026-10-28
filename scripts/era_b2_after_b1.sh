#!/usr/bin/env bash
# After the three narrative banks are complete (>= 15,000 rows each): build the local release package and
# regenerate the paper's tables. Public upload is a separate, manual step (era_push_release.py --public).
#
#   bash scripts/era_b2_after_b1.sh [paper tables dir]
set -euo pipefail
cd "$(dirname "$0")/.."
TABLES_DIR=${1:-paper/tables}
MIN=15000
for w in 2020 2022 2024; do
  f=data/synthetic/era_${w}_narrative/personas.jsonl
  n=$(wc -l < "$f" 2>/dev/null || echo 0)
  echo "wave=$w n=$n"
  if (( n < MIN )); then echo "wave $w is incomplete; stopping"; exit 1; fi
done
uv run python scripts/era_package_release.py --version v0.1-era --min-n "$MIN"
uv run python paper/make_tables_era.py --tables-dir "$TABLES_DIR"
