# CulturePersona-Seoul

Code, released grounding tables and evaluation layer for **CulturePersona-Seoul**, a dataset of
cultural-participation personas built only from aggregate tables of the 2020, 2022 and 2024 waves of the *Seoul
Citizens' Cultural Participation Survey* (Seoul Foundation for Arts and Culture) and evaluated against the survey's
individual response records.

This repository accompanies a paper under anonymous review. It holds all of the study's code, from microdata
preparation to the script that writes every number in the paper, so that each step can be inspected. The persona banks themselves are in the review supplement (see
[Data](#data)); the individual response records are not, and never will be, part of any release.

## What is here

| Path | Contents |
| --- | --- |
| `src/sfac_persona/` | Library: survey reading, anchor codebook, grounding tables, reconstruction ladder, disclosure checks, sampler chain, narrative validators, SSR answering |
| `src/sfac_persona/grounding/` | `official.py` (anchor frame), `era_tables.py` (grounding tables), `chain.py` (sampler), `error_study.py` (reconstruction ladder), `sdc.py` (differencing checks) |
| `scripts/` | Pipeline scripts. `era_*` produce the paper's results, `anno_*` the registered human audit and model judge |
| `tests/` | Unit and release-surface tests; tests that need the microdata are skipped when it is absent |
| `preprocessing/` | Microdata preparation: the publisher's person-level files to one harmonised layout, plus the national survey (KCAS) used in §4; see its README |
| `annotation/` | The web tool used for the registered human audit (interface, backend, protocol tests); see its README |
| `paper/make_tables_era.py` | Writes the paper's number macros and tables from `outputs/reports/` |
| `registration/` | The audit analysis script as registered, and its diff against the current one |
| `data/grounding/era_*/` | Released grounding tables (`*_release.parquet`, cells with at least 30 unweighted respondents), population root, sampler specification |
| `data/eval/` | Held-out item registry, SSR reference statements, frozen temperature |
| `outputs/reports/` | Aggregate results that the paper's tables are generated from |
| Prompts | The narrative prompt is built in `src/sfac_persona/narrative.py`; the model-judge rubric is in `scripts/anno_llm_judge.py`; the paper's appendix prints the rendered prompt |

## Where each result in the paper comes from

| Paper | Script(s) | Report(s) in `outputs/reports/` |
| --- | --- | --- |
| §3 Reproduction of published statistics | `official_reproduce_gate.py` | `official_reproduce_gate*` |
| §3 Grounding tables per wave (Table 7), population root, sampler specification | `era_grounding.py` | writes `data/grounding/era_*/` |
| §4 Reconstruction ladder (Table 3), error by anchor, genre correlations, residual pairs | `decade_error_atlas.py`, `era_atlas_uncertainty.py`, `era_ladder_crossfit.py` | `decade_error_atlas_*`, `era_atlas_uncertainty.csv`, `era_ladder_crossfit.csv` |
| §4 KL robustness | `era_info_ladder.py` | `era_info_ladder_pairs.csv` |
| §4 National-survey replication (in sample and held out) | `kcas_atlas.py`, `kcas_ladder_crossfit.py` | `kcas_atlas_*`, `kcas_weight_provenance.csv`, `kcas_ladder_crossfit.csv` |
| §5 Sampler chain, ablations, held-out fit; measured vs shipped configuration | `era_sample_quant.py`, `era_eval_suite.py`, `era_ablation_seeds.py`, `era_chain_crossfit.py`, `era_shipped_fit.py` | `era_eval_suite.csv`, `era_sampler_ablation.csv`, `era_chain_crossfit*`, `era_shipped_fit.csv` |
| §5 Feasibility, unseen combinations, segment holdout, variance | `era_sampler_audit.py`, `era_unseen_combos.py`, `era_appendix_audit.py`, `era_r0_wave_study.py`, `era_bank_diversity.py` | `era_sampler_*`, `era_unseen_*`, `era_segment_holdout.csv`, `era_variance.csv` |
| §5 Downstream tasks | `era_genre_portfolio.py`, `era_tstr.py`, `era_tstr_paired.py`, `era_design_pretest.py`, `era_design_deep.py`, `era_design_oos_pairs.py`, `era_downstream_baselines.py`, `era_tables_only_bank.py` | `era_genre_portfolio.csv`, `era_tstr.csv`, `era_tstr_paired.csv`, `era_design_pretest.csv`, `era_design_deep*.csv`, `era_design_oos_pairs*.csv`, `era_downstream_baselines.csv` |
| §6 Held-out survey items, twin arm, baselines, all-fields arm | `era_holdout_run.py`, `era_holdout_score.py`, `era_holdout_baselines.py`, `era_holdout_crosswave.py`, `era_holdout_fullnarr.py` | `era_holdout_*` |
| §6 Positive control, further generators, further readers, template arm | `era_holdout_positive_control.py`, `era_second_generator.py`, `era_holdout_readers.py`, `era_template_prose.py` | `era_holdout_positive_control.csv`, `era_holdout_generators.csv`, `era_holdout_readers.csv` |
| §6 Second elicitation (option log-probabilities), reader prior, recalibration | `era_holdout_logprob.py`, `era_holdout_prior_ssr.py`, `era_holdout_recal.py`, `era_holdout_corr_bc.py`, `era_holdout_marginal.py`, `era_family_mentions.py` | `era_holdout_logprob.csv`, `era_holdout_prior_ssr.csv`, `era_holdout_recal.csv` |
| §6 Narratives as examples, open-ended item, coding real answers from synthetic text, second coder | `era_llm_genre.py`, `era_theme_v2.py`, `era_theme_coding.py`, `era_theme_coder2.py` | `era_llm_genre*`, `era_theme_v2_*`, `era_theme_coding.csv`, `era_theme_coder2.csv` |
| §6 Prediction-powered inference | `era_holdout_ppi.py`, `era_ppi_wave_concentration.py` | `era_holdout_ppi.csv`, `era_ppi_wave_concentration.csv` |
| Ethics: differencing, attribute disclosure, probes | `era_sdc_differencing.py`, `era_attribute_disclosure.py`, `era_probes.py` | `era_sdc_*`, `era_attribute_disclosure.csv`, `era_probes.csv` |
| Appendix: model-judge reproducibility | `anno_judge_repro.py` | `anno_judge_repro.*` |

Results of the registered human audit are added to `outputs/reports/` once the registered analysis
(`scripts/anno_analysis.py`) has been run on the completed ratings.

Other scripts in `scripts/` are steps or checks that the scripts above depend on:

| Purpose | Script(s) |
| --- | --- |
| Narrative generation and release packaging | `era_generate_narrative.py`, `era_package_release.py`, `era_push_release.py` (public release only) |
| Held-out items: answer model, embeddings, metrics, small smoke runs | `era_answer_model.py`, `ssr_embed.py`, `ssr_validate.py`, `era_ssr_validate.py`, `era_holdout_metrics.py`, `era_holdout_sample_smoke.py`, `era_holdout_ssr_smoke.py` |
| Open-ended item: theme table and narrative-layer comparison | `era_theme_table.py`, `era_open_text.py` |
| Sampler inputs and checks | `structure_priors.py`, `era_r0_promotion_effect.py`, `fetch_population_prior_history.py`, `era_consistency_checker.py` |
| Human audit: design, sample, checks, exploratory analysis | `human_audit_sample_size.py`, `human_validation_power.py`, `anno_reduce_sample.py`, `anno_trap_style_check.py`, `anno_dryrun.py`, `anno_exploratory.py` |
| Privacy step on local interim files (not needed to reproduce) | `redact_interim_text.py` |
| Run orders: analyses after the full banks, packaging, evaluation-layer rerun | `run_wave_analyses.sh`, `era_b2_after_b1.sh`, `era_r0_promote_run.sh` |

## Data

* **Released grounding tables** are in `data/grounding/`. Every released cell has at least 30 unweighted
  respondents; complementary suppression is applied to the pair-joint table; geography stops at the nine
  living areas.
* **Persona banks**: all 45,000 personas (2020, 2022 and 2024 waves, 15,000 each), with their narratives,
  and the frozen evaluation subset of 300 personas per wave are in the review data supplement submitted with
  the paper. The public release of the narratives follows the registered human audit.
* **Individual response records** were used, under the publisher's approval for derivative works, only to
  compute tables and to score the dataset. They are not in this repository, and no synthetic persona
  derives from a record.

## Running

```bash
uv sync --extra dev
uv run pytest -q
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for which steps run from the released tables alone and which
need the microdata, a GPU or a local LLM server.

## Notes on the code

Comments and some JSON outputs carry tags such as `ADR-007`. They point to entries in the authors' internal
design log, which is not part of this repository. The tags have no effect on the code.

## Licence

CC-BY-4.0 for code, grounding tables and personas, subject to the publisher's derivative-release terms
(CC-BY-NC-4.0 if required); see `LICENSE`. Please cite the survey as
*Seoul Citizens' Cultural Participation Survey* (Seoul Foundation for Arts and Culture; 2020, 2022 and 2024 waves, latest report 2025).
