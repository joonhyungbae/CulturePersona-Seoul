"""Paired interval for the imputation comparison: the bank's classifier against pair-table naive Bayes.

era_tstr.py reports the train-on-synthetic accuracy and era_downstream_baselines.py the accuracy of naive Bayes
over the released pair tables on the same respondents, but the paper quotes the two means without an interval.
This script recomputes both predictions on the same rows and bootstraps the difference over respondents, paired.

Design fixed before running (2026-10-10):
  setting     cross-wave (2024 bank and tables, 2022 and 2020 respondents), feature set "all", the summary the
              paper quotes (mean weighted accuracy over target anchors and test waves)
  methods     synthetic = multinomial logistic trained on the 2024 bank (era_tstr.evaluate_target); pair tables =
              naive Bayes over the released 2024 tables (era_downstream_baselines.predict_labels); age table =
              mode of the released age-band conditional
  primary     synthetic minus pair tables, in accuracy points: respondents resampled with replacement within
              each (wave, target) row, the same draw for both methods, row accuracies averaged as in the
              summary, 2,000 draws, 95% percentile interval
  secondary   synthetic minus age table; in-wave setting
Every result is reported, whatever its sign.

Run: uv run python scripts/era_tstr_paired.py
Output: outputs/reports/era_tstr_paired.csv (aggregates only)
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

from era_downstream_baselines import _feature_list, predict_labels
from era_genre_portfolio import PairTables
from era_tstr import MIN_CLASS, published_modes, real_frame, synthetic_bank

from sfac_persona.config import settings
from sfac_persona.schema import ERA_CORE_ANCHORS, ROOT_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("tstr_paired")

TRAIN_WAVE = 2024
TEST_WAVES = (2024, 2022, 2020)
N_BOOT = 2000
SEED = 20261010


def hits_for_target(
    syn: pd.DataFrame, real: pd.DataFrame, tables: PairTables, target: str, features: list[str]
) -> dict[str, np.ndarray] | None:
    """Per-respondent correctness of each method on the rows era_tstr.evaluate_target scores."""
    cols = [*features, target]
    s = syn.dropna(subset=cols)
    r = real.dropna(subset=cols)
    if s.empty or r.empty:
        return None
    counts = s[target].value_counts()
    keep = set(counts[counts >= MIN_CLASS].index) & set(r[target].unique())
    if len(keep) < 2:
        return None
    s = s[s[target].isin(keep)]
    r = r[r[target].isin(keep)]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    enc.fit(pd.concat([s[features].astype(str), r[features].astype(str)]))
    xs, xr = enc.transform(s[features].astype(str)), enc.transform(r[features].astype(str))
    ys, yr = s[target].astype(str).to_numpy(), r[target].astype(str).to_numpy()
    pred_syn = LogisticRegression(max_iter=1000).fit(xs, ys).predict(xr)
    pred_pair = predict_labels(tables, target, r, features).astype(str)
    pub_marg, pub_by_age = published_modes(TRAIN_WAVE, target, keep)
    marginal_mode = pub_marg if pub_marg is not None else s[target].value_counts().idxmax()
    syn_by_age = s.groupby("age_band")[target].agg(lambda x: x.value_counts().idxmax())
    by_age = {a: pub_by_age.get(str(a), v) for a, v in syn_by_age.items()}
    pred_age = r["age_band"].map(by_age).fillna(marginal_mode).astype(str).to_numpy()
    return {
        "w": r["w"].to_numpy(dtype=float),
        "synthetic": (pred_syn == yr).astype(float),
        "pair_tables": (pred_pair == yr).astype(float),
        "age_table": (pred_age == yr).astype(float),
    }


def summary(rows: list[dict[str, np.ndarray]], method: str, idx: list[np.ndarray] | None) -> float:
    accs = []
    for k, h in enumerate(rows):
        sel = idx[k] if idx is not None else slice(None)
        accs.append(float(np.average(h[method][sel], weights=h["w"][sel])))
    return float(np.mean(accs))


def main() -> int:
    syn = synthetic_bank(TRAIN_WAVE)
    tables = PairTables(TRAIN_WAVE)
    targets = [a for a in ERA_CORE_ANCHORS if a in syn.columns and a not in ROOT_ANCHORS]
    rng = np.random.default_rng(SEED)
    out = []
    for setting, waves in (
        ("cross-wave", [w for w in TEST_WAVES if w != TRAIN_WAVE]),
        ("in-wave", [TRAIN_WAVE]),
    ):
        rows: list[dict[str, np.ndarray]] = []
        for wave in waves:
            real = real_frame(wave)
            for target in targets:
                features = _feature_list("all", target, syn, real)
                h = hits_for_target(syn, real, tables, target, features)
                if h is None:
                    log.warning("skip %s wave %d", target, wave)
                    continue
                rows.append(h)
                log.info("%s %d %-24s n=%d", setting, wave, target, len(h["w"]))
        point = {m: summary(rows, m, None) for m in ("synthetic", "pair_tables", "age_table")}
        draws = {"pair_tables": [], "age_table": []}
        for _ in range(N_BOOT):
            idx = [rng.integers(0, len(h["w"]), len(h["w"])) for h in rows]
            s_ = summary(rows, "synthetic", idx)
            for ref, d in draws.items():
                d.append(s_ - summary(rows, ref, idx))
        for ref, d in draws.items():
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append(
                {
                    "setting": setting,
                    "features": "all",
                    "contrast": f"synthetic-{ref}",
                    "acc_synthetic": point["synthetic"],
                    "acc_ref": point[ref],
                    "diff_pp": 100 * (point["synthetic"] - point[ref]),
                    "ci_lo_pp": 100 * float(lo),
                    "ci_hi_pp": 100 * float(hi),
                    "n_rows": len(rows),
                    "n_boot": N_BOOT,
                }
            )
    res = pd.DataFrame(out)
    path = settings.outputs_dir / "reports" / "era_tstr_paired.csv"
    res.to_csv(path, index=False)
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
