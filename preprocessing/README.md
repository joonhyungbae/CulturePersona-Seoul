# preprocessing — microdata preparation

This directory turns the survey publisher's person-level delivery into the files that the persona pipeline
(`src/sfac_persona/`) reads. It is the code behind the paper's appendix on microdata preparation. It needs the
individual response records, which are not distributed (see the main README); everything it writes stays
local (`preprocessing/data/` and `preprocessing/reports/` are git-ignored).

## Inputs (not in the repository)

| Path | What |
| --- | --- |
| `preprocessing/data/official_raw_data/` | The publisher's delivery of the Seoul Citizens' Cultural Participation Survey, 2014–2024 (data files and codebooks) |
| `preprocessing/data/raw/` | The National Survey of Cultural and Arts Activities (KCAS) files, obtained through MDIS (second case in §4) |
| `data/reference/` (tracked) | Values printed in the survey reports and the district-to-living-area crosswalk |

## Running

```bash
uv sync --extra preprocess
uv run bash preprocessing/run_official.sh       # Seoul survey: steps 01–18, then validate_official.py
uv run bash preprocessing/scripts/run_kcas.sh   # KCAS: kcas_01_build.py, then validate_kcas.py
```

Each track ends with a validator that checks the declared contracts (`config/*_contracts.json`) and exits
non-zero on any violation. The Seoul track checks 212 contracts and the KCAS track 270.

## Steps (Seoul survey)

| Script | What it does |
| --- | --- |
| `official_00_common.py` | Paths, readers and shared vocabulary |
| `official_01_dictionary.py` | Variable dictionary and value labels from the codebooks |
| `official_02_instrument.py` | Attendance-battery fingerprint per wave and the questionnaire regimes derived from it |
| `official_03_samples.py` | Sample definitions, frames, weights and person-level keys |
| `official_04_schema.py` | One person × item table across waves |
| `official_05_reproduce.py` | Reproduction of the published tables from the delivered files |
| `official_06_series.py` | Attendance series from the microdata, with comparability grades |
| `official_07_impute.py` | Missingness classification; only item non-response is filled, routed-out items are not |
| `official_09_wave_metadata.py` | Fielding period, recall window and sample design, read from the report text |
| `official_10_domains.py`, `official_15_domain_items.py`, `official_16_domain_missingness.py` | Items outside the attendance battery: concordance, harmonisation and missingness |
| `official_11_living_area.py` | Sampling error of living-area estimates |
| `official_12_cohorts.py` | Birth-cohort coverage |
| `official_13_exports.py` | Value labels and per-wave frames in the format `src/sfac_persona` reads |
| `official_14_weight_provenance.py` | What the final weights are calibrated to (demographics or attendance) |
| `official_17_registry.py` | Status of every variable in every wave |
| `official_18_manifest.py` | Output manifest and data dictionary |
| `official_08_report.py` | Summary report (`reports/official_overview.md`) |
| `validate_official.py` | Contract checks |

The persona pipeline then copies the person-level outputs it needs into `data/interim/` (also git-ignored) and
falls back to `preprocessing/data/` when a copy is missing (`preprocessing_root` in
`src/sfac_persona/config.py`). Dong-level identifiers are kept only in local person-level files; no table or
persona uses geography below the nine living areas.

## Published values (`data/reference/published_tables_2018_2024.csv`)

The reproduction check (§3) compares against values printed in the survey reports. These were transcribed from
text extracted from the report PDFs, not computed by code; each of the 965 rows records its source in the
`provenance` column (report text file, line number and table title) and its `extraction_method`, so every value
can be checked against the public reports. The table is frozen and is not edited by hand.
