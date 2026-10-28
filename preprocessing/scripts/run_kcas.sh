#!/usr/bin/env bash
# Reproduce the KCAS track. Kept separate from the official track.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

echo "=== kcas_01_build ==="
"$PY" scripts/kcas_01_build.py
echo "=== validate_kcas ==="
"$PY" scripts/validate_kcas.py
