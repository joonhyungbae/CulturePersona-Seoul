#!/usr/bin/env bash
# Repeat the narrative-layer measurements with two further generators on the 2024 personas (300 per generator):
# generate, answer the held-out items (answering model = default generator), embed the answers for SSR, score,
# then the open-ended item and the held-out-genre task with the new narratives as examples.
set -u
cd "$(dirname "$0")/.."
# IQ4_XS quantisations so that each 31-32B model fits on one 24 GB GPU without CPU offload.
EXAONE="hf.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF:IQ4_XS"
GEMMA="hf.co/bartowski/google_gemma-4-31b-it-GGUF:IQ4_XS"
mkdir -p outputs/logs
LOG=outputs/logs/second_generator.log
WAVE="--wave 2024"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

for pair in "exaone:$EXAONE" "gemma31:$GEMMA"; do
  tag=${pair%%:*}; model=${pair#*:}
  say "generate $tag ($model)"
  uv run python scripts/era_second_generator.py generate --tag "$tag" --model "$model" --concurrency 2 $WAVE >> "$LOG" 2>&1
done
say "held-out answers"
for tag in exaone gemma31; do
  uv run python scripts/era_second_generator.py holdout --tag "$tag" $WAVE >> "$LOG" 2>&1
done
say "SSR embeddings (CPU)"
EMBED_PY=.venv-embed/bin/python; [ -x "$EMBED_PY" ] || EMBED_PY="uv run python"
for tag in exaone gemma31; do
  CUDA_VISIBLE_DEVICES= $EMBED_PY scripts/ssr_embed.py --dir "data/eval/era_holdout/w2024" --pattern "answers_release_${tag}_ssr.jsonl" >> "$LOG" 2>&1
done
say "evaluate"
for tag in exaone gemma31; do uv run python scripts/era_second_generator.py evaluate --tag "$tag" $WAVE >> "$LOG" 2>&1; done
uv run python scripts/era_second_generator.py report --tags exaone,gemma31 >> "$LOG" 2>&1
say "open-ended item"
for tag in exaone gemma31; do
  uv run python scripts/era_open_text.py --narrative-path "data/synthetic/era_2024_narrative_${tag}/personas.jsonl" --tag "$tag" >> "$LOG" 2>&1
done
say "held-out-genre task with the new narratives as examples"
for tag in exaone gemma31; do
  SFP_BANK_PATH="data/synthetic/era_2024_narrative_${tag}/personas.jsonl" SFP_OUT_TAG="$tag" uv run python scripts/era_llm_genre.py --conditions zero,shot_anchor,shot_narr --limit 300 >> "$LOG" 2>&1
done
say "done"
