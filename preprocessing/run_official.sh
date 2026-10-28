#!/usr/bin/env bash
# Reproduce the whole official raw data track. data/official_raw_data/ must be present
# (it is gitignored). The last step enforces the contracts and exits non-zero on a
# violation.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export PYTHONWARNINGS=ignore
# The numbers are the running order. 04 (the schema) comes before 05 (the reproduction
# check) because the early-wave published comparison reads the person x item table.
for s in 01_dictionary 02_instrument 03_samples 04_schema 05_reproduce 06_series 07_impute \
         09_wave_metadata 10_domains 11_living_area 12_cohorts 13_exports 14_weight_provenance 15_domain_items 16_domain_missingness 17_registry 18_manifest 08_report; do
  echo "=== official_$s ==="
  "$PY" "scripts/official_$s.py" 2>/dev/null
done
echo "=== validate_official ==="
"$PY" scripts/validate_official.py
