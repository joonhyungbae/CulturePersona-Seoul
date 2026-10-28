#!/usr/bin/env bash
# Positive control for the held-out-item test. Re-running resumes from checkpoints: rows already present in the
# generated personas, answer and cosine files are skipped.
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs/logs
export SFP_LLM_NUM_CTX=4096
export PYTHONUNBUFFERED=1
exec uv run python scripts/era_holdout_positive_control.py all
