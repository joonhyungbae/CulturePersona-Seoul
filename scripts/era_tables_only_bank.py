"""A bank drawn from the printed tables alone, scored on the same downstream tasks as the released bank.

The ladder (Section 4) scores what a builder holding only the printed report can reconstruct: the skip-logic rung
(age-band conditionals plus the questionnaire's routing). The released bank instead draws from conditionals fitted
to the records. This script draws a bank from the former and puts it through the imputation and genre tasks, so
that the cost of invention is measured on the tasks and not only on anchor pairs. The design crossings are run by
era_design_oos_pairs.py --variant tables_only.

Design fixed before running (2026-10-10):
  chain       era_sampler_audit.variant_nodes("tables_only"): root = the 2024 registration population (the shipped
              bank's root); every anchor conditional on age band; the attendance block (ten genres, count band,
              spending band, companion) conditional on attendance status and age band, which the printed age-band
              conditionals and the skip logic determine; genre count derived from the genres; no genre tree;
              the same cell floor and backoff as every other chain
  draw        N = 15,000 (the shipped 2024 bank's size), one seed
  checks      in-sample pair TVD over the ladder's non-structural pairs, read against the skip rung (0.032)
              and the shipped specification
  imputation  era_tstr.run with this bank in place of the shipped one (all settings and feature sets);
              primary contrast: cross-wave, all anchors, tables-only bank minus shipped bank, respondent-paired
              bootstrap over the same rows (both banks' classes kept, 2,000 draws, 95% percentile interval);
              secondary: tables-only minus pair-table naive Bayes, same draws
  genre       era_genre_portfolio.run with this bank in place of the shipped one
  root check  (added 2026-10-11, before running) the same tables-only chain drawn with the survey-weighted root in
              place of the registration root, pair TVD only, to attribute the gap between the drawn bank and the
              analytic skip rung (root versus floor, backoff and the finite draw); stage `rootcheck`
Every result is reported, whatever its sign.

Run: uv run python scripts/era_tables_only_bank.py
Outputs: outputs/reports/era_tables_only_bank.csv (aggregates only);
         data/synthetic/era_2024_quant/personas_tables_only.parquet (the bank, local)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_genre_portfolio as G
import era_tstr as T
from era_downstream_baselines import _feature_list, predict_labels
from era_sampler_audit import _non_root, build_variant, pair_tvd
from era_tstr import MIN_CLASS, real_frame

from sfac_persona.config import settings
from sfac_persona.grounding import chain as C
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import ERA_CORE_ANCHORS, ROOT_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
for _n in ("sfac_persona.grounding.chain", "sfac_persona.grounding.official"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("era_tables_only_bank")

WAVE = 2024
N_DRAW = 15_000
SEED = 20261010
N_BOOT = 2000
REPORTS = settings.outputs_dir / "reports"
BANK_PATH = settings.synthetic_dir / f"era_{WAVE}_quant" / "personas_tables_only.parquet"


def draw_bank(frame: pd.DataFrame) -> pd.DataFrame:
    r0 = settings.era_grounding_dir(WAVE) / "R0_population_root.parquet"
    spec = build_variant(frame, "tables_only", root_source="population", r0_path=r0)
    bank = C.ChainSampler(spec).sample(N_DRAW, seed=SEED)
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(BANK_PATH, index=False)
    log.info("tables-only bank: %d personas -> %s", len(bank), BANK_PATH)
    return bank


def paired_hits(
    banks: dict[str, pd.DataFrame],
    real: pd.DataFrame,
    tables: G.PairTables,
    target: str,
    features: list[str],
) -> dict[str, np.ndarray] | None:
    """Per-respondent correctness of a classifier trained on each bank, on the same real rows.
    Classes are those with at least MIN_CLASS rows in every bank and present in the real data."""
    cols = [*features, target]
    r = real.dropna(subset=cols)
    keep = set(r[target].unique())
    subs = {}
    for name, syn in banks.items():
        s = syn.dropna(subset=cols)
        counts = s[target].value_counts()
        keep &= set(counts[counts >= MIN_CLASS].index)
        subs[name] = s
    if len(keep) < 2 or r.empty:
        return None
    r = r[r[target].isin(keep)]
    yr = r[target].astype(str).to_numpy()
    out: dict[str, np.ndarray] = {"w": r["w"].to_numpy(dtype=float)}
    for name, s in subs.items():
        s = s[s[target].isin(keep)]
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        enc.fit(pd.concat([s[features].astype(str), r[features].astype(str)]))
        xs, xr = enc.transform(s[features].astype(str)), enc.transform(r[features].astype(str))
        pred = (
            LogisticRegression(max_iter=1000).fit(xs, s[target].astype(str).to_numpy()).predict(xr)
        )
        out[name] = (pred == yr).astype(float)
    out["pair_tables"] = (predict_labels(tables, target, r, features).astype(str) == yr).astype(
        float
    )
    return out


def _summary(rows: list[dict[str, np.ndarray]], method: str, idx: list[np.ndarray] | None) -> float:
    accs = []
    for k, h in enumerate(rows):
        sel = idx[k] if idx is not None else slice(None)
        accs.append(float(np.average(h[method][sel], weights=h["w"][sel])))
    return float(np.mean(accs))


def paired_imputation(bank: pd.DataFrame, shipped: pd.DataFrame) -> list[dict]:
    banks = {"tables_only": bank, "shipped": shipped}
    tables = G.PairTables(WAVE)
    targets = [a for a in ERA_CORE_ANCHORS if a in bank.columns and a not in ROOT_ANCHORS]
    rng = np.random.default_rng(SEED)
    out = []
    for setting, waves in (("cross-wave", [2022, 2020]), ("in-wave", [WAVE])):
        rows: list[dict[str, np.ndarray]] = []
        for wave in waves:
            real = real_frame(wave)
            for target in targets:
                features = _feature_list("all", target, shipped, real)
                h = paired_hits(banks, real, tables, target, features)
                if h is None:
                    log.warning("skip %s wave %d", target, wave)
                    continue
                rows.append(h)
        point = {m: _summary(rows, m, None) for m in ("tables_only", "shipped", "pair_tables")}
        draws = {"shipped": [], "pair_tables": []}
        for _ in range(N_BOOT):
            idx = [rng.integers(0, len(h["w"]), len(h["w"])) for h in rows]
            s_ = _summary(rows, "tables_only", idx)
            for ref, d in draws.items():
                d.append(s_ - _summary(rows, ref, idx))
        for ref, d in draws.items():
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append(
                {
                    "record": "imputation_paired",
                    "setting": setting,
                    "features": "all",
                    "contrast": f"tables_only-{ref}",
                    "value": 100 * (point["tables_only"] - point[ref]),
                    "ci_lo": 100 * float(lo),
                    "ci_hi": 100 * float(hi),
                    "acc_tables_only": point["tables_only"],
                    "acc_ref": point[ref],
                    "n": len(rows),
                }
            )
    return out


def root_check() -> int:
    """Append the survey-root tables-only bank's pair TVD to the report (aggregates only)."""
    frame = build_official_anchor_frame(WAVE)
    frame = frame.dropna(subset=["sex", "age_band", "living_area"]).reset_index(drop=True)
    frame = C.add_derived_columns(frame)
    spec = build_variant(frame, "tables_only", root_source="survey")
    bank = C.ChainSampler(spec).sample(N_DRAW, seed=SEED)
    ns, st, n_ns, _n_st = pair_tvd(frame, bank, _non_root(frame))
    log.info(
        "tables_only, survey root: non-structural %.4f (%d pairs), structural %.4f", ns, n_ns, st
    )
    path = REPORTS / "era_tables_only_bank.csv"
    res = pd.read_csv(path)
    res = res[~((res.record == "pair_tvd") & (res.bank == "tables_only_survey_root"))]
    row = {
        "record": "pair_tvd",
        "bank": "tables_only_survey_root",
        "value": ns,
        "structural": st,
        "n": n_ns,
    }
    res = pd.concat([res, pd.DataFrame([row])], ignore_index=True)
    res.to_csv(path, index=False)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "rootcheck":
        return root_check()
    frame = build_official_anchor_frame(WAVE)
    frame = frame.dropna(subset=["sex", "age_band", "living_area"]).reset_index(drop=True)
    frame = C.add_derived_columns(frame)
    bank = draw_bank(frame)
    shipped = T.synthetic_bank(WAVE)
    rows: list[dict] = []
    # In-sample pair TVD, same pair set as the ladder and the sampler audit.
    anchors = _non_root(frame)
    for name, syn in (("tables_only", bank), ("shipped", shipped)):
        ns, st, n_ns, n_st = pair_tvd(frame, syn, anchors)
        rows.append({"record": "pair_tvd", "bank": name, "value": ns, "structural": st, "n": n_ns})
        log.info(
            "%s: non-structural %.4f (%d pairs), structural %.4f (%d)", name, ns, n_ns, st, n_st
        )
    # Imputation and genre, each script's own run with the bank swapped in.
    T.synthetic_bank = lambda w: bank  # type: ignore[assignment]
    tstr = T.run(WAVE, [2024, 2022, 2020])
    tstr.to_csv(REPORTS / "era_tstr_tables_only.csv", index=False)
    for (setting, fset), g in tstr.groupby(["setting", "features"]):
        rows.append(
            {
                "record": "imputation",
                "setting": setting,
                "features": fset,
                "acc_marginal": g.acc_marginal.mean(),
                "acc_age": g.acc_age.mean(),
                "acc_tables_only": g.acc_tstr.mean(),
                "acc_microdata": g.acc_trtr.mean(),
                "n": len(g),
            }
        )
    rows += paired_imputation(bank, shipped)
    G.synthetic_bank = lambda w: bank  # type: ignore[assignment]
    genre = G.run(WAVE, [2024, 2022, 2020])
    genre.to_csv(REPORTS / "era_genre_portfolio_tables_only.csv", index=False)
    for (setting, fset), g in genre.groupby(["setting", "features"]):
        rows.append(
            {
                "record": "genre",
                "setting": setting,
                "features": fset,
                "auc_age_table": g.auc_age_table.mean(),
                "auc_pair_tables": g.auc_pair_tables.mean(),
                "auc_tables_only": g.auc_synthetic.mean(),
                "auc_microdata": g.auc_microdata.mean(),
                "n": len(g),
            }
        )
    res = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS / "era_tables_only_bank.csv", index=False)
    print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
