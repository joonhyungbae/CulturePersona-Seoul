#!/usr/bin/env bash
# Open-ended item experiment (2024 only): code the survey's answers into themes, build the theme table, run a
# pilot that must pass before the full bank, generate the field for the bank, and evaluate.
# Run: nohup bash scripts/era_theme_v2.sh &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
LOG=outputs/logs/theme_v2.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run() { say "stage $1"; uv run python scripts/era_theme_v2.py "$1" >> "$LOG" 2>&1; local rc=$?; say "stage $1 rc=$rc"; return $rc; }

run code-real && run table && run pilot
rc=$?
if [ $rc -eq 0 ]; then
  run bank && run bank-eval; rc=$?
else
  say "pilot did not pass (rc=$rc); bank stage skipped"
fi
say "exit $rc"
