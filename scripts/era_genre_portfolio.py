"""Downstream 3 — filling in a genre portfolio (ADR-021).

This resource's distinctive task. Demographic modelling has better data elsewhere, but
**which genres an attender sees together** exists only in this survey's microdata and not
in the published tables. The report prints per-genre attendance rates broken down by sex,
age, and living area, but **the joint among genres** is never printed.

Task: given the rest of one attender's cultural profile, predict attendance of **one
held-out genre**. Same shape as hiding one item and predicting it in a recommender, and a
question a cultural institution actually asks (may one recommend 무용 to this person).

  **Attenders only.** Non-attenders have all 10 genres as '아니오', so a correct prediction
  is free — that is survey routing, not taste structure (the same trap as ADR-016·019).

Two conditions.

  cultural     the other 9 genres plus count, spend, and companion. **Information absent
               from the published tables.**
  demographic  sex, age, living area, education, income, and so on. Information the
               published tables provide.

Comparison.

  age table    the training wave's published age-band genre attendance rates (T2) as the
               score (when the user has only the tables).
  pair tables  naive Bayes from the published pair joints (T6) and conditional tables
               (T2–T4) — the strongest thing a user with only the tables can do (mock
               review 2026-09-23, DA's "direct baseline from the published pair joints").
  synthetic    trained on the synthetic bank (our resource).
  microdata    trained and cross-validated on real respondents (upper bound).

The metric is AUC. Genre flags are sparse, so accuracy is masked by the majority class.
Differences (synthetic − age table, synthetic − pair tables, microdata − synthetic) are
percentile intervals from a paired bootstrap of test respondents (B=500).

Run: uv run python scripts/era_genre_portfolio.py
Outputs: outputs/reports/era_genre_portfolio.csv · .md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import GENRE_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("genre_portfolio")

SEED = 20260921
DEMOGRAPHIC = [
    "sex",
    "age_band",
    "living_area",
    "education",
    "occupation",
    "household_income_band",
    "household_size_band",
    "seoul_years_band",
    "leisure_sufficiency",
    "leisure_companion",
]
CULTURAL_EXTRA = ["attend_count_band", "paid_spend_band", "viewing_companion"]
MIN_POS = 30
N_BOOT = 500
ROOT_COND = {"sex": "T3", "age_band": "T2", "living_area": "T4"}


class PairTables:
    """Naive Bayes scores from the published grounding tables (the release) only. Uses the
    training wave's tables.
    """

    def __init__(self, wave: int):
        era = settings.era_grounding_dir(wave)
        t1 = pd.read_parquet(era / "T1_release.parquet")
        self.marg = {(r.anchor, str(r.value)): float(r.share) for r in t1.itertuples()}
        t6 = pd.read_parquet(era / "T6_pair_joints_release.parquet")
        self.joint: dict[tuple, float] = {}
        for r in t6.itertuples():
            self.joint[(r.anchor_a, str(r.value_a), r.anchor_b, str(r.value_b))] = float(r.joint)
            self.joint[(r.anchor_b, str(r.value_b), r.anchor_a, str(r.value_a))] = float(r.joint)
        self.cond: dict[tuple, float] = {}
        for root, t in ROOT_COND.items():
            d = pd.read_parquet(era / f"{t}_release.parquet")
            for r in d.itertuples():
                self.cond[(r.anchor, str(r.value), root, str(getattr(r, root)))] = float(r.share)

    def age_rate(self, target: str, df: pd.DataFrame) -> np.ndarray:
        """Published age-band genre attendance rate (training wave's public T2) — a baseline
        that does not use test labels.
        """
        col = df["age_band"].astype(str).to_numpy()
        fallback = self.marg.get((target, "예"), 0.5)
        return np.array([self.cond.get((target, "예", "age_band", v), fallback) for v in col])

    def score(self, target: str, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Attender-conditional naive Bayes. The test set is attenders only, so every
        probability is renormalized to attenders (attend_any=예) — possible from the T1
        and T6 pair joints (f × attend_any) and (f × target) alone (technical review
        2026-09-24):
          P(g=예 | att) = P(g) / P(att)                       (g=예 ⊂ attenders)
          P(f=v | g=예) is unchanged                           (g=예 ⊂ attenders)
          P(f=v | g=아니오, att) = [P(f=v, att) − P(f=v, g=예)] / [P(att) − P(g)]
        Suppressed cells and values get likelihood ratio 0 (independence).
        """
        p_att = self.marg.get(("attend_any", "예"), 1.0)
        pg = self.marg.get((target, "예"), 0.5 * p_att)
        p1 = min(max(pg / p_att, 1e-6), 1 - 1e-6)
        p0 = 1.0 - p1
        out = np.full(len(df), np.log(p1 / p0))
        for f in features:
            col = df[f].astype(str).to_numpy()
            contrib = np.zeros(len(df))
            for v in np.unique(col):
                if f in ROOT_COND:
                    q = self.cond.get((target, "예", f, v))
                    qa = self.cond.get(("attend_any", "예", f, v))
                    if q is None or qa is None or qa <= 0:
                        lr = 0.0
                    else:
                        qq = min(max(q / qa, 1e-6), 1 - 1e-6)
                        lr = np.log(qq / (1 - qq)) - np.log(p1 / p0)
                else:
                    j1 = self.joint.get((target, "예", f, v))
                    ja = self.joint.get(("attend_any", "예", f, v)) if f != "attend_any" else None
                    if j1 is None or ja is None or ja <= j1:
                        lr = 0.0  # suppressed cell or value: treat as independent
                    else:
                        l1 = j1 / pg
                        l0 = (ja - j1) / max(p_att - pg, 1e-9)
                        lr = np.log(max(l1, 1e-9) / max(l0, 1e-9))
                contrib[col == v] = lr
            out += contrib
        return out


def synthetic_bank(wave: int) -> pd.DataFrame:
    p = settings.data_root / "synthetic" / f"era_{wave}_quant" / "personas_structure_T.parquet"
    return pd.read_parquet(p)


def attenders(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["attend_any"].astype(str) == "예"].copy()


def _auc(y: np.ndarray, score: np.ndarray, w: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score, sample_weight=w))


def evaluate(
    target: str,
    syn: pd.DataFrame,
    real: pd.DataFrame,
    features: list[str],
    tables: PairTables | None = None,
) -> dict | None:
    s, r = syn.dropna(subset=[*features, target]), real.dropna(subset=[*features, target])
    ys = (s[target].astype(str) == "예").to_numpy(dtype=int)
    yr = (r[target].astype(str) == "예").to_numpy(dtype=int)
    if ys.sum() < MIN_POS or yr.sum() < MIN_POS or len(np.unique(yr)) < 2:
        return None
    w = r["w"].to_numpy(dtype=float)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    enc.fit(pd.concat([s[features].astype(str), r[features].astype(str)]))
    Xs, Xr = enc.transform(s[features].astype(str)), enc.transform(r[features].astype(str))

    syn_clf = LogisticRegression(max_iter=1000).fit(Xs, ys)
    sc_syn = syn_clf.predict_proba(Xr)[:, 1]

    # Published-table baseline: score with the training wave's public age-band genre
    # attendance rate (T2) — test labels are not used.
    sc_age = (
        tables.age_rate(target, r) if tables is not None else np.full(len(yr), float(yr.mean()))
    )
    # Published pair-joint naive Bayes (the best a user with only the tables can do).
    sc_nb = tables.score(target, r, features) if tables is not None else np.full(len(yr), 0.0)

    # Upper bound: train and cross-validate on real respondents.
    oof = np.zeros(len(yr), dtype=float)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in folds.split(Xr, yr):
        clf = LogisticRegression(max_iter=1000).fit(Xr[tr], yr[tr])
        oof[te] = clf.predict_proba(Xr[te])[:, 1]

    scores = {"age_table": sc_age, "pair_tables": sc_nb, "synthetic": sc_syn, "microdata": oof}
    auc = {k: _auc(yr, v, w) for k, v in scores.items()}
    # Paired bootstrap: recompute all four AUCs on the same resample and interval the
    # differences.
    rng = np.random.default_rng(SEED)
    diffs = {"syn_age": [], "syn_nb": [], "real_syn": []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(yr), len(yr))
        if len(np.unique(yr[idx])) < 2:
            continue
        b = {k: _auc(yr[idx], v[idx], w[idx]) for k, v in scores.items()}
        diffs["syn_age"].append(b["synthetic"] - b["age_table"])
        diffs["syn_nb"].append(b["synthetic"] - b["pair_tables"])
        diffs["real_syn"].append(b["microdata"] - b["synthetic"])
    ci = {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in diffs.items()
    }
    return {
        "target": target,
        "n_train_syn": len(s),
        "n_test_real": len(r),
        "positives_real": int(yr.sum()),
        "auc_age_table": auc["age_table"],
        "auc_pair_tables": auc["pair_tables"],
        "auc_synthetic": auc["synthetic"],
        "auc_microdata": auc["microdata"],
        "d_syn_age_lo": ci["syn_age"][0],
        "d_syn_age_hi": ci["syn_age"][1],
        "d_syn_nb_lo": ci["syn_nb"][0],
        "d_syn_nb_hi": ci["syn_nb"][1],
        "d_real_syn_lo": ci["real_syn"][0],
        "d_real_syn_hi": ci["real_syn"][1],
    }


def run(train_wave: int, test_waves: list[int]) -> pd.DataFrame:
    syn = attenders(synthetic_bank(train_wave))
    tables = PairTables(train_wave)
    rows = []
    for tw in test_waves:
        real = attenders(build_official_anchor_frame(tw))
        for target in GENRE_ANCHORS:
            if target not in syn.columns or target not in real.columns:
                continue
            others = [g for g in GENRE_ANCHORS if g != target]
            for cond, feats in (
                ("cultural", [*others, *CULTURAL_EXTRA]),
                ("demographic", DEMOGRAPHIC),
                ("both", [*others, *CULTURAL_EXTRA, *DEMOGRAPHIC]),
            ):
                feats = [f for f in feats if f in syn.columns and f in real.columns]
                m = evaluate(target, syn, real, feats, tables)
                if m is None:
                    continue
                m.update(
                    train_wave=train_wave,
                    test_wave=tw,
                    setting="in-wave" if tw == train_wave else "cross-wave",
                    features=cond,
                )
                rows.append(m)
                log.info(
                    "%-26s %-10s %-12s age=%.3f nb=%.3f syn=%.3f real=%.3f",
                    target,
                    m["setting"],
                    cond,
                    m["auc_age_table"],
                    m["auc_pair_tables"],
                    m["auc_synthetic"],
                    m["auc_microdata"],
                )
    cols = [
        "train_wave",
        "test_wave",
        "setting",
        "features",
        "target",
        "positives_real",
        "n_test_real",
        "auc_age_table",
        "auc_pair_tables",
        "auc_synthetic",
        "auc_microdata",
        "d_syn_age_lo",
        "d_syn_age_hi",
        "d_syn_nb_lo",
        "d_syn_nb_hi",
        "d_real_syn_lo",
        "d_real_syn_hi",
    ]
    return pd.DataFrame(rows)[cols]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-wave", type=int, default=2024)
    ap.add_argument("--test-wave", type=int, action="append", default=None)
    args = ap.parse_args()
    test_waves = args.test_wave or [2024, 2022, 2020]

    res = run(args.train_wave, test_waves)
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    res.to_csv(reports / "era_genre_portfolio.csv", index=False)

    agg = res.groupby(["setting", "features"])[
        ["auc_age_table", "auc_pair_tables", "auc_synthetic", "auc_microdata"]
    ].mean()
    lines = [
        "# Downstream 3 — 장르 포트폴리오 채우기 (ADR-021)",
        "",
        "관람자 한 명의 나머지 문화 프로필로 가려진 장르 하나를 맞힌다. 지표는 AUC.",
        "비관람자는 제외한다 — 장르가 전부 '아니오' 라 맞히기가 공짜이고, 그것은 라우팅이다.",
        "",
        agg.round(4).to_markdown(),
        "",
        "## 장르별 (cross-wave, cultural)",
        res[(res.setting == "cross-wave") & (res.features == "cultural")]
        .groupby("target")[
            [
                "auc_age_table",
                "auc_pair_tables",
                "auc_synthetic",
                "auc_microdata",
                "d_syn_nb_lo",
                "d_syn_nb_hi",
            ]
        ]
        .mean()
        .round(3)
        .to_markdown(),
        "",
        "`cultural` 은 나머지 9 장르와 횟수·지출·동반자 — **공표 표에 없는 정보**다.",
        "`demographic` 은 공표 표가 주는 정보. 둘의 차이가 이 자원이 더하는 몫이다.",
    ]
    (reports / "era_genre_portfolio.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", reports / "era_genre_portfolio.csv")
    print(agg.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
