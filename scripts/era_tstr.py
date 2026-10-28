"""Downstream 1 — train on synthetic, test on real (TSTR, ADR-019).

Question: if a **user without microdata** trains on a synthetic population built
from published tables, how well do they fill in a real respondent's missing
item. This attaches the standard protocol of the synthetic-data literature
(TSTR) to the real task of survey imputation.

Comparisons are **things that same user can actually use**.

  marginal    Mode of the published margin — anyone with the tables can do this.
  age         Mode of the published age-band conditional — what reports actually print.
  TSTR        Multinomial logistic trained on synthetic personas, applied to real respondents (this resource).
  TRTR        Train and cross-validate on real microdata (upper bound — a reference for a user who cannot have it).

Two settings.

  in-wave     2024 synthetic → 2024 real. Not independent, because the synthetic came from that wave's tables.
              This measures the upper bound of "what a user who has only the tables can do that year".
  cross-wave  2024 synthetic → 2022 and 2020 real. True extrapolation. Do they predict another year's respondents?

Weights: accuracy on the real test set is weighted by the survey weight (the target is a weighted estimate).

Run: uv run python scripts/era_tstr.py
Outputs: outputs/reports/era_tstr.csv · era_tstr.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import ERA_CORE_ANCHORS, GENRE_ANCHORS, ROOT_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("era_tstr")

SEED = 20260920
MIN_CLASS = 30

# The attendance block determines itself via survey skip logic (non-attenders are '아니오' on all 10 genres).
# Predicting within this block yields accuracy 1.0, and that is routing, not a model.
# The feature sets are therefore split in two.
ATTEND_BLOCK = frozenset(
    {*GENRE_ANCHORS, "attend_any", "attend_count_band", "paid_spend_band", "viewing_companion"}
)
FEATURE_SETS = ("all", "demographic")


def synthetic_bank(wave: int) -> pd.DataFrame:
    p = settings.data_root / "synthetic" / f"era_{wave}_quant" / "personas_structure_T.parquet"
    return pd.read_parquet(p)


def real_frame(wave: int) -> pd.DataFrame:
    return build_official_anchor_frame(wave)


def _wacc(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    return float(np.average((y_true == y_pred).astype(float), weights=w))


def published_modes(wave: int, target: str, allowed: set) -> tuple[str | None, dict[str, str]]:
    """Mode of the released marginal (T1) and of the released age-band conditional (T2) for one anchor,
    among the values both waves share (the same class set the models are scored on)."""
    g = settings.era_grounding_dir(wave)
    t1 = pd.read_parquet(g / "T1_release.parquet")
    t1 = t1[(t1.anchor == target) & t1.value.isin(allowed)]
    marg = str(t1.loc[t1.share.idxmax(), "value"]) if len(t1) else None
    t2 = pd.read_parquet(g / "T2_release.parquet")
    t2 = t2[(t2.anchor == target) & t2.value.isin(allowed)]
    by_age = {
        str(a): str(d.loc[d.share.idxmax(), "value"])
        for a, d in t2.groupby("age_band", observed=True)
    }
    return marg, by_age


def evaluate_target(
    target: str, syn: pd.DataFrame, real: pd.DataFrame, features: list[str], table_wave: int
) -> dict[str, float] | None:
    cols = [*features, target]
    s = syn.dropna(subset=cols)
    r = real.dropna(subset=cols)
    if s.empty or r.empty:
        return None
    # Values absent from the synthetic data cannot be learned; rare classes are dropped on both sides.
    counts = s[target].value_counts()
    keep = set(counts[counts >= MIN_CLASS].index) & set(r[target].unique())
    if len(keep) < 2:
        return None
    s = s[s[target].isin(keep)]
    r = r[r[target].isin(keep)]
    w = r["w"].to_numpy(dtype=float)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    enc.fit(pd.concat([s[features].astype(str), r[features].astype(str)]))
    Xs, Xr = enc.transform(s[features].astype(str)), enc.transform(r[features].astype(str))
    ys, yr = s[target].astype(str).to_numpy(), r[target].astype(str).to_numpy()

    tstr = LogisticRegression(max_iter=1000).fit(Xs, ys)
    acc_tstr = _wacc(yr, tstr.predict(Xr), w)

    # Published-table baselines — mode of the released margin (T1) and of the released age-band conditional
    # (T2) of the training wave, i.e. what the tables give directly. An anchor or age band with no released
    # cell falls back to the synthetic bank's mode.
    pub_marg, pub_by_age = published_modes(table_wave, target, keep)
    marginal_mode = pub_marg if pub_marg is not None else s[target].value_counts().idxmax()
    acc_marg = _wacc(yr, np.full(len(yr), marginal_mode), w)
    syn_by_age = s.groupby("age_band")[target].agg(lambda x: x.value_counts().idxmax())
    by_age = {a: pub_by_age.get(str(a), v) for a, v in syn_by_age.items()}
    pred_age = r["age_band"].map(by_age).fillna(marginal_mode).to_numpy()
    acc_age = _wacc(yr, pred_age, w)

    # Upper bound — train and validate on the real data (a user who has the microdata).
    acc_trtr = np.nan
    vc = pd.Series(yr).value_counts()
    if (vc >= 5).all() and len(r) > 100:
        folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
        hits = np.zeros(len(yr), dtype=float)
        for tr, te in folds.split(Xr, yr):
            clf = LogisticRegression(max_iter=1000).fit(Xr[tr], yr[tr])
            hits[te] = (clf.predict(Xr[te]) == yr[te]).astype(float)
        acc_trtr = float(np.average(hits, weights=w))

    return {
        "target": target,
        "n_classes": len(keep),
        "n_train_syn": len(s),
        "n_test_real": len(r),
        "acc_marginal": acc_marg,
        "acc_age": acc_age,
        "acc_tstr": acc_tstr,
        "acc_trtr": acc_trtr,
    }


def run(train_wave: int, test_waves: list[int]) -> pd.DataFrame:
    syn = synthetic_bank(train_wave)
    targets = [a for a in ERA_CORE_ANCHORS if a in syn.columns]
    rows = []
    for tw in test_waves:
        real = real_frame(tw)
        for target in targets:
            if target in ROOT_ANCHORS:
                continue
            for fset in FEATURE_SETS:
                features = [a for a in targets if a != target and a in real.columns]
                if fset == "demographic":
                    features = [a for a in features if a not in ATTEND_BLOCK]
                elif target in ATTEND_BLOCK:
                    pass  # 'all' also uses the attendance block on itself — a control showing how easy the routing is
                m = evaluate_target(target, syn, real, features, train_wave)
                if m is None:
                    log.warning("skip %s (wave %d, %s)", target, tw, fset)
                    continue
                m.update(
                    train_wave=train_wave,
                    test_wave=tw,
                    setting="in-wave" if tw == train_wave else "cross-wave",
                    features=fset,
                    n_features=len(features),
                )
                rows.append(m)
                log.info(
                    "%-24s %-10s %-11s marg=%.3f age=%.3f TSTR=%.3f TRTR=%.3f",
                    target,
                    m["setting"],
                    fset,
                    m["acc_marginal"],
                    m["acc_age"],
                    m["acc_tstr"],
                    m["acc_trtr"],
                )
    cols = [
        "train_wave",
        "test_wave",
        "setting",
        "features",
        "n_features",
        "target",
        "n_classes",
        "n_train_syn",
        "n_test_real",
        "acc_marginal",
        "acc_age",
        "acc_tstr",
        "acc_trtr",
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
    res.to_csv(reports / "era_tstr.csv", index=False)

    agg = res.groupby(["setting", "features"])[
        ["acc_marginal", "acc_age", "acc_tstr", "acc_trtr"]
    ].mean()
    gap = res.assign(
        gain_over_age=lambda d: d.acc_tstr - d.acc_age,
        share_of_upper=lambda d: (d.acc_tstr - d.acc_age) / (d.acc_trtr - d.acc_age),
    )
    lines = [
        "# Downstream 1 — TSTR (ADR-019)",
        "",
        "합성으로 학습하고 실제 응답자로 시험한다. 기준선은 공표 표로 할 수 있는 것,",
        "상한선은 마이크로데이터로 학습한 것.",
        "",
        "## 설정·특징집합별 평균 (가중 정확도)",
        agg.round(4).to_markdown(),
        "",
        "## 앵커별",
        gap.round(4).to_markdown(index=False),
        "",
        "`features=all` 은 다른 앵커 전부, `demographic` 은 관람 블록을 뺀 것이다 — 관람 블록끼리는",
        "설문 건너뛰기가 서로를 결정하므로 `all` 의 관람 관련 값은 모형이 아니라 라우팅이다.",
        "`share_of_upper` 는 연령대 조건부 기준선과 마이크로데이터 상한 사이에서",
        "합성 학습이 메운 몫이다. in-wave 는 합성이 그 파동 표에서 나왔으므로 독립이 아니다 —",
        "cross-wave 가 진짜 외삽이다.",
    ]
    (reports / "era_tstr.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", reports / "era_tstr.csv")
    print(agg.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
