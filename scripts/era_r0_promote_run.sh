#!/usr/bin/env bash
# Regenerate the evaluation layer after switching each wave's root to that wave's resident-registration
# population (ADR-013): evaluation narratives (300 per wave), held-out items, then the remaining analyses.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
EMBED_PY=.venv-embed/bin/python; [ -x "$EMBED_PY" ] || EMBED_PY="uv run python"

echo "=== evaluation narratives, n=300 per wave $(date -Iseconds) ==="
for w in 2024 2022 2020; do
  uv run python scripts/era_generate_narrative.py --wave "$w" --n 300 --concurrency 6
  echo "  wave $w narratives: $(wc -l < data/synthetic/era_${w}_narrative/personas.jsonl) $(date -Iseconds)"
done
echo "=== held-out items $(date -Iseconds) ==="
for w in 2024 2022 2020; do
  uv run python scripts/era_holdout_run.py --wave "$w"
  echo "  wave $w held-out done $(date -Iseconds)"
done
echo "=== remaining analyses $(date -Iseconds) ==="
uv run python scripts/era_eval_suite.py
uv run python scripts/era_holdout_score.py
uv run python scripts/era_holdout_metrics.py
uv run python scripts/era_answer_model.py --waves 2024 2022 2020
uv run python scripts/era_holdout_ppi.py
uv run python scripts/era_holdout_crosswave.py
uv run python scripts/era_holdout_sample_smoke.py || true
$EMBED_PY scripts/era_ssr_validate.py --wave 2024
uv run python scripts/decade_error_atlas.py
uv run python scripts/era_r0_wave_study.py || true
echo "=== done $(date -Iseconds) ==="
