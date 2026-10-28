#!/usr/bin/env bash
# Analyses that need all three full narrative banks (ADR-018/019/021): narrative probes, bank diversity and the
# two downstream tasks, then the paper's number macros and tables.
#
#   bash scripts/run_wave_analyses.sh [paper tables dir]
#
# Log: outputs/logs/wave_analyses_<timestamp>.log
set -uo pipefail
cd "$(dirname "$0")/.."

TABLES_DIR=${1:-paper/tables}
TS=$(date +%Y%m%d_%H%M%S)
LOG="outputs/logs/wave_analyses_${TS}.log"
mkdir -p outputs/logs

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Stop if a bank is incomplete.
for w in 2020 2022 2024; do
  f="data/synthetic/era_${w}_narrative/personas.jsonl"
  n=$(wc -l < "$f" 2>/dev/null || echo 0)
  say "wave ${w}: ${n} rows"
  if [ "$n" -lt 15000 ]; then
    say "wave ${w} has fewer than 15,000 personas; analyses skipped"
    exit 1
  fi
done

run() {
  local name="$1"; shift
  say "start ${name}"
  if "$@" >>"$LOG" 2>&1; then say "done ${name}"; else say "FAILED ${name} (see log)"; fi
}

run "narrative probes"       uv run python scripts/era_probes.py --wave 2020 --wave 2022 --wave 2024
run "duplicates, diversity"  uv run python scripts/era_bank_diversity.py --wave 2020 --wave 2022 --wave 2024
run "genre portfolio"        uv run python scripts/era_genre_portfolio.py
run "imputation (TSTR)"      uv run python scripts/era_tstr.py
run "design pretest"         uv run python scripts/era_design_pretest.py
run "paper tables"           uv run python paper/make_tables_era.py --tables-dir "$TABLES_DIR"
say "finished"
