# data/

| Path | Contents | In this repository |
| --- | --- | --- |
| `grounding/era_*/` | Released grounding tables (`*_release.parquet`, cells with at least 30 unweighted respondents), population root, sampler specifications, `provenance.json`, reproduction-gate results | Yes |
| `grounding/structure_priors.json` | Structure prior used by the sampler | Yes |
| `eval/` | Held-out item registry, SSR reference statements, frozen SSR temperature | Yes |
| `reference/` | Published report tables, resident-registration population counts, district-to-living-area crosswalk | Yes |
| `processed/microdata/microdata_value_labels.csv` | Questionnaire value labels (codebook) | Yes |
| `interim/` | Harmonised person-level survey files | No — individual-level, never distributed |
| `synthetic/`, `release/` | Generated personas and release bundles | No — provided as review supplement |

Paths are set in `src/sfac_persona/config.py`.
