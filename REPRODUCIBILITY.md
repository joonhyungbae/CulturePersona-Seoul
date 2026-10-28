# Reproducibility

## Environment

```bash
uv sync --extra dev      # core dependencies and pytest
uv sync --extra embed    # sentence-embedding models for SSR and probes (torch, sentence-transformers)
```

Python 3.11+. Narrative generation and SSR answering use a local [Ollama](https://ollama.com) server
(Qwen3.8 27B as generator, EXAONE 4.0 32B and Gemma 4 31B as further generators and readers, gemma4:12b as
model judge). The server address is read from `SFP_LLM_BASE_URL` (default `http://localhost:11434`;
see `.env.example`). Embedding steps use `.venv-embed/bin/python` when that environment exists (we kept
the embedding stack apart to avoid NumPy version clashes) and otherwise the current interpreter after
`uv sync --extra embed`.

## What runs from this repository alone

| Step | Command | Output |
| --- | --- | --- |
| Tests (release surface, cell floor, sampler invariants) | `uv run pytest -q` | — |
| Draw quantitative personas from the sampler specification in `data/grounding/era_*/` | `uv run python scripts/era_sample_quant.py --wave 2024 --n 1000 --out-suffix _check` | `data/synthetic/era_2024_quant/` |
| Paper number macros and tables from `outputs/reports/` | `uv run python paper/make_tables_era.py --tables-dir <dir>` | `<dir>/*.tex` |

Every number in the paper is read from `outputs/reports/`, which this repository includes. Run from an empty
directory with all inputs present, the table script writes every number macro the paper uses and every table,
identical to the submitted version. In a fresh clone it skips the dataset-statistics and example-persona tables
and the macros they set (they read the persona banks, which are in the review data supplement) and leaves out
the cell counts of the unreleased internal tables T1–T5 in the grounding-table summary; it lists what it skipped.

## What needs the individual response records

The reconstruction ladder, held-out fit, downstream tasks, held-out survey items, disclosure checks and the
reproduction gate all compare against the individual response records. These were provided by the publisher
for this work and cannot be redistributed. The scripts are included so that the procedure can be inspected
and rerun by anyone with access to the same records:

| Step | Command |
| --- | --- |
| Microdata preparation (publisher files → harmonised person-level layout; KCAS) | `uv sync --extra preprocess`, then `uv run bash preprocessing/run_official.sh` and `uv run bash preprocessing/scripts/run_kcas.sh` |
| Reproduction gate (published rates within tolerance) | `scripts/official_reproduce_gate.py` |
| Grounding tables (internal and `*_release.parquet`) | `scripts/era_grounding.py` |
| Reconstruction ladder | `decade_error_atlas.py` → `kcas_atlas.py` → `era_atlas_uncertainty.py --boot 200` → `era_ladder_crossfit.py` → `era_info_ladder.py` |
| Sampler fit and held-out fit | `era_sample_quant.py`, `era_chain_crossfit.py` |
| Feasibility and unseen-combination audit (compares drawn personas with the records) | `era_sampler_audit.py`, `era_unseen_combos.py` |
| Narrative bank (GPU, Ollama) | `bash scripts/era_b1_full_generate.sh` |
| Held-out items, SSR, baselines, PPI | `era_holdout_run.py` → `era_holdout_score.py`, `era_holdout_baselines.py`, `era_holdout_ppi.py` |
| Positive control, further generators | `bash scripts/era_holdout_positive_control.sh`, `bash scripts/era_second_generator.sh` |
| Downstream tasks | `era_genre_portfolio.py`, `era_tstr.py`, `era_design_pretest.py`, `era_downstream_baselines.py`, `era_llm_genre.py` |
| Disclosure checks | `era_sdc_differencing.py`, `era_attribute_disclosure.py`, `era_probes.py` |
| Registered human audit | `anno_build_sample.py` → `anno_make_traps.py` → `anno_reduce_sample.py` → (rating in `annotation/`) → `anno_pull.py` → `anno_analysis.py` |

Paths to the records are set in `src/sfac_persona/config.py`.

## Release contract

* Only `data/grounding/*/**_release.parquet` are released tables (unweighted cells ≥ 30).
* `provenance.json` records the SHA-256 of the microdata file, respondent count, weight variable, cell floor
  and the reference month of the population root.
* No individual-level file is in this repository.
