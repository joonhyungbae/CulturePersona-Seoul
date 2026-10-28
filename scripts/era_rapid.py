"""RAPID audit of the public persona dump (Templ, Thees and Mueller 2026, arXiv:2602.09235).

Threat model (RAPID Section 2.1): an attacker knows a real respondent's quasi-identifiers
(sex, age band, living area, education, occupation), trains a model on the released data and
predicts a sensitive attribute. For record i with true class y_i the model gives
g_i = P(y_i | x_i); the baseline b_i is the share of y_i among the original respondents; the
normalised gain is r_i = (g_i - b_i) / (1 - b_i), and RAPID(tau) is the share of records with
r_i > tau (default tau = 0.3, RAPID Section 2.2).

Attackers (Section 2.5 recommends strong learners and the maximum):
  rf        random forest, 500 trees (the paper's default attacker)
  logit     one-hot logistic regression
  gbm       histogram gradient boosting on ordinal-coded keys

Sources the attacker trains on:
  bank        the released personas of the wave (data/release/<version>/personas_<wave>.parquet)
  tables      no training: the released tables' posterior (era_attribute_disclosure.ReleasedTables)
  microdata   reference only, never released: 5-fold out-of-fold models on the respondents
              themselves, i.e. what releasing the records would expose

Targets are the respondents of each weighted wave. Individual rows stay in memory; only
aggregates are written. Intervals are Wilson 95% intervals (RAPID Section 2.5).

Run: uv run python scripts/era_rapid.py
Outputs: outputs/reports/era_rapid.csv · era_rapid.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from era_attribute_disclosure import (
    KEYS,
    RELEASE_VERSION,
    ReleasedTables,
    _bank_path,
    _complete,
    _table_scores,
)

from sfac_persona.config import settings
from sfac_persona.grounding.chain import add_derived_columns
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import WEIGHTED_WAVES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("era_rapid")

SEED = 20261016
TAU = 0.3
TAU_GRID = (0.1, 0.2, 0.3, 0.5)
# The three targets of era_attribute_disclosure (tables posterior available for these).
TARGETS = ["household_income_band", "attend_any", "n_genres_band"]
ATTACKERS = ("rf", "logit", "gbm")
N_FOLDS = 5


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def normalised_gain(g: np.ndarray, y: np.ndarray) -> np.ndarray:
    """r_i = (g_i - b_i) / (1 - b_i) with b_i the original (unweighted) share of y_i."""
    share = pd.Series(y).value_counts(normalize=True)
    b = pd.Series(y).map(share).to_numpy(dtype=float)
    return (g - b) / (1.0 - b)


def _model(name: str):
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED)
    if name == "logit":
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if name == "gbm":
        return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED)
    raise ValueError(name)


def _encode(name: str, train: pd.DataFrame, test: pd.DataFrame):
    if name == "gbm":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr = pd.DataFrame(enc.fit_transform(train[KEYS]), columns=KEYS).astype("category")
        te = pd.DataFrame(enc.transform(test[KEYS]), columns=KEYS)
        for c in KEYS:
            cats = tr[c].cat.categories
            te[c] = pd.Categorical(te[c].where(te[c].isin(cats)), categories=cats)
        return tr, te
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=(name == "logit"))
    return enc.fit_transform(train[KEYS]), enc.transform(test[KEYS])


def prob_true(name: str, train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    """Probability the attacker assigns to each test record's true class (0 if never seen)."""
    xtr, xte = _encode(name, train, test)
    clf = _model(name)
    clf.fit(xtr, train[target].to_numpy())
    proba = clf.predict_proba(xte)
    col = {c: j for j, c in enumerate(clf.classes_)}
    y = test[target].to_numpy()
    idx = np.array([col.get(v, -1) for v in y])
    out = np.zeros(len(y))
    seen = idx >= 0
    out[seen] = proba[np.flatnonzero(seen), idx[seen]]
    return out


def prob_true_oof(name: str, real: pd.DataFrame, target: str) -> np.ndarray:
    """Out-of-fold version on the respondents themselves (reference, never released)."""
    out = np.zeros(len(real))
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in folds.split(real, real[target]):
        out[te] = prob_true(name, real.iloc[tr], real.iloc[te], target)
    return out


def summarise(
    wave: int, target: str, source: str, attacker: str, r: np.ndarray, w: np.ndarray
) -> dict:
    n = len(r)
    row = {"wave": wave, "target": target, "source": source, "attacker": attacker, "n": n}
    for tau in TAU_GRID:
        hit = r > tau
        row[f"rapid_{tau}"] = float(hit.mean())
        row[f"rapid_w_{tau}"] = float(np.dot(hit, w) / w.sum())
    lo, hi = wilson(int((r > TAU).sum()), n)
    row["ci_lo"], row["ci_hi"] = lo, hi
    row["mean_gain"] = float(r.mean())
    return row


def run_wave(wave: int) -> list[dict]:
    path = _bank_path(wave)
    if not path.exists():
        raise SystemExit(f"released bank missing: {path}")
    real_all = add_derived_columns(build_official_anchor_frame(wave))
    bank_all = pd.read_parquet(path, columns=[*KEYS, *TARGETS])
    tables = ReleasedTables(wave)
    rows: list[dict] = []
    for target in TARGETS:
        real = _complete(real_all, target)
        bank = bank_all.dropna(subset=[*KEYS, target]).astype(str)
        y, w = real[target].to_numpy(), real["w"].to_numpy(dtype=float)
        log.info("wave %s %s: %d respondents, %d bank rows", wave, target, len(real), len(bank))
        _pred, p_tab, _hit = _table_scores(tables, real, target)
        rows.append(summarise(wave, target, "tables", "posterior", normalised_gain(p_tab, y), w))
        for name in ATTACKERS:
            r_bank = normalised_gain(prob_true(name, bank, real, target), y)
            rows.append(summarise(wave, target, "bank", name, r_bank, w))
            r_micro = normalised_gain(prob_true_oof(name, real, target), y)
            rows.append(summarise(wave, target, "microdata", name, r_micro, w))
    return rows


def envelope(res: pd.DataFrame) -> pd.DataFrame:
    """Mean and maximum over attackers per (wave, target, source) at the default tau."""
    col = f"rapid_{TAU}"
    g = res.groupby(["wave", "target", "source"], sort=False)[col]
    return pd.DataFrame({"mean": g.mean(), "max": g.max()}).reset_index()


def write_report(res: pd.DataFrame, path_md: Path) -> None:
    env = envelope(res)
    piv = env.pivot_table(index=["wave", "target"], columns="source", values="max", sort=False)
    piv = piv[["tables", "bank", "microdata"]]
    lines = [
        "# RAPID audit of the public dump",
        "",
        f"Release `{RELEASE_VERSION}`. RAPID (Templ, Thees and Mueller 2026, arXiv:2602.09235): share of real respondents",
        f"whose normalised gain over the class base rate exceeds tau = {TAU} when an attacker who knows their",
        "sex, age band, living area, education and occupation predicts the target. Maximum over the attackers",
        "(random forest 500 trees, logistic regression, gradient boosting); `tables` is the released tables'",
        "posterior; `microdata` is the reference of releasing the records (5-fold out-of-fold, not released).",
        "Unweighted shares over respondents, as RAPID defines; survey-weighted shares are in the CSV.",
        "",
        "| wave | target | tables | bank | microdata (reference) |",
        "|---|---|---|---|---|",
    ]
    for (wave, target), r in piv.iterrows():
        lines.append(
            f"| {wave} | {target} | {r['tables']:.3f} | {r['bank']:.3f} | {r['microdata']:.3f} |"
        )
    worst = res[res["source"] == "bank"].sort_values(f"rapid_{TAU}").iloc[-1]
    lines += [
        "",
        (
            f"Highest bank value: {worst[f'rapid_{TAU}']:.3f} (wave {worst['wave']}, {worst['target']}, {worst['attacker']}; "
            f"Wilson 95% {worst['ci_lo']:.3f}-{worst['ci_hi']:.3f})."
        ),
        "",
        "Threshold curve (bank, maximum over attackers):",
        "",
        "| wave | target | " + " | ".join(f"tau={t}" for t in TAU_GRID) + " |",
        "|---|---|" + "---|" * len(TAU_GRID),
    ]
    bank = res[res["source"] == "bank"].groupby(["wave", "target"], sort=False)
    for (wave, target), part in bank:
        vals = " | ".join(f"{part[f'rapid_{t}'].max():.3f}" for t in TAU_GRID)
        lines.append(f"| {wave} | {target} | {vals} |")
    lines += [
        "",
        "The narrative columns are not read: the anchors in the same row already carry every structured value the",
        "prose states, so the structured attack on the dump is the attack on its anchors.",
        "",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, action="append", default=None)
    args = ap.parse_args()
    waves = args.wave or list(WEIGHTED_WAVES)
    res = pd.DataFrame([row for wave in waves for row in run_wave(wave)])
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    res.to_csv(reports / "era_rapid.csv", index=False)
    write_report(res, reports / "era_rapid.md")
    log.info("wrote %s", reports / "era_rapid.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
