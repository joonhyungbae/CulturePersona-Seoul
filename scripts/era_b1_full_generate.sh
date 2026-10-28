#!/usr/bin/env bash
# Generate the full narrative bank (15,000 personas per weighted wave), one wave at a time.
# Generation resumes by quant_id, so a run that stops (e.g. a server timeout) is retried until each wave
# reaches its target row count.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
LOG=outputs/logs/full_narrative_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -Iseconds) log=$LOG"
TARGET=15000
MAX_TRIES=50
for w in 2024 2022 2020; do
  f="data/synthetic/era_${w}_narrative/personas.jsonl"
  for try in $(seq 1 $MAX_TRIES); do
    n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
    if (( n >= TARGET )); then break; fi
    echo "=== wave $w try $try n=$n $(date -Iseconds) ==="
    uv run python scripts/era_generate_narrative.py --wave "$w" --n 0 --concurrency 6 \
      || { echo "wave $w stopped (try $try); retrying in 60 s"; sleep 60; }
  done
  n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
  echo "wave $w done n=$n $(date -Iseconds)"
  if (( n < TARGET )); then echo "FAILED wave $w n=$n after $MAX_TRIES tries"; exit 1; fi
done
echo "all waves done $(date -Iseconds)"
