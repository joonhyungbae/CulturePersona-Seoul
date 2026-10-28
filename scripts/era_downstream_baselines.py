"""Collect published-table baselines for the three downstream tasks in one table.

The genre task (`era_genre_portfolio.py`) already reports an attender-conditional naive
Bayes from the published T6 and T2–T4. Those numbers are copied, not recomputed. What was
missing is imputation (Table 26, `era_tstr`) and design (Table 27, `era_design_pretest`).

Imputation does not reuse `PairTables.score` as-is. That score is the attender-conditional
log-odds of a binary genre, whereas imputation predicts a multinomial anchor over all
respondents and scores weighted accuracy. From the same published tables, take the MAP
class of a multinomial naive Bayes. The prior is T1, the likelihood of non-root features
is the T6 pair joint, and the roots (sex, age, living area) are T2–T4 inverted by Bayes.
A cell absent from the release is treated as independent for that class only (the
feature's marginal probability). Binary conditionals are the exception. If only one
class's share is published, the rest is one minus that share. That subtraction does not
revive a suppressed count; it is the complement of a published share. Empty cells of a
pair joint are not filled this way. Differences from the marginal are not used.

Design's output is a cell attendance rate, not a class. Apply the same likelihood ratio
to the two attendance classes and convert to P(attendance | cell). No attender-conditional
adjustment, because the quantity being estimated is the attendance rate itself. When the
axis is one and that cell is published, this expression equals the published conditional
probability. The age×sex table (T5) is unused by the genre baseline, so it is unused here
too; age and sex each enter the likelihood.

It is not applied to design's out-of-sample split-half (`era_chain_crossfit_design`). The
published tables are built on all respondents and already include the scoring half.

Individual rows are read only inside `build_official_anchor_frame` (`era_tstr.real_frame`).
Design ground truth uses the cell rates in the existing `era_design_pretest.csv` as-is.

Run: uv run python scripts/era_downstream_baselines.py
Outputs: outputs/reports/era_downstream_baselines.csv · .md
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)

from era_design_pretest import required_n
from era_genre_portfolio import ROOT_COND, PairTables
from era_tstr import (
    ATTEND_BLOCK,
    MIN_CLASS,
    real_frame,
    synthetic_bank,
)

from sfac_persona.config import settings
from sfac_persona.schema import ERA_CORE_ANCHORS

log = logging.getLogger("downstream_baselines")

_LOG_FLOOR = 1e-12
TRAIN_WAVE = 2024

METHODS = {
    "genre": ["age_table", "pair_tables", "synthetic", "microdata"],
    "imputation": ["marginal", "age_table", "pair_tables", "synthetic", "microdata"],
    "design": ["marginal", "age_table", "pair_tables", "synthetic"],
}
SLICE_ORDER = {
    "genre": ["cultural", "demographic", "both"],
    "imputation": ["all", "demographic"],
    "design": ["unpublished", "published"],
}
SETTING_ORDER = ["in-wave", "cross-wave", "pooled"]
GENRE_COLS = {
    "age_table": "auc_age_table",
    "pair_tables": "auc_pair_tables",
    "synthetic": "auc_synthetic",
    "microdata": "auc_microdata",
}
IMPUTE_COLS = {
    "marginal": "acc_marginal",
    "age_table": "acc_age",
    "synthetic": "acc_tstr",
    "microdata": "acc_trtr",
}


class TableFactors(Protocol):
    marg: dict[tuple[str, str], float]
    joint: dict[tuple[str, str, str, str], float]
    cond: dict[tuple[str, str, str, str], float]


def _log(x: float) -> float:
    return math.log(max(float(x), _LOG_FLOOR))


def _expit(log_odds: float) -> float:
    if log_odds >= 0:
        z = math.exp(-log_odds)
        return 1.0 / (1.0 + z)
    z = math.exp(log_odds)
    return z / (1.0 + z)


def _conditional_share(
    tables: TableFactors, target: str, class_value: str, feature: str, value: str
) -> float | None:
    """Published conditional P(target=class | feature). If binary and only one side is
    published, the complement.

    The release drops suppressed rows, but shares on the remaining rows are the pre-drop
    values. For a binary anchor, the complement of that share is the missing class. Do not
    use this for an empty multinomial cell or an empty pair-joint cell. The latter would
    recover a suppressed cell from the difference against the marginal.
    """
    q = tables.cond.get((target, class_value, feature, value))
    if q is not None and q > 0:
        return float(q)
    classes = [
        val for (anchor, val), share in tables.marg.items() if anchor == target and share > 0
    ]
    if len(classes) != 2 or class_value not in classes:
        return None
    other = classes[0] if classes[1] == class_value else classes[1]
    q_other = tables.cond.get((target, other, feature, value))
    if q_other is None:
        return None
    comp = 1.0 - float(q_other)
    if comp <= 0:
        return None
    return comp


def factor_loglik(
    tables: TableFactors, target: str, class_value: str, feature: str, value: str
) -> float:
    """log P(feature=value | target=class).

    Published cells only. If a cell is absent, use log P(feature=value), so that class is
    independent. A root feature inverts the published conditional P(target | feature) by
    Bayes.
    """
    pf = tables.marg.get((feature, value))
    pc = tables.marg.get((target, class_value))
    if feature in ROOT_COND:
        q = _conditional_share(tables, target, class_value, feature, value)
        if q is not None and pc is not None and pc > 0 and pf is not None and pf > 0:
            return _log(q) + _log(pf) - _log(pc)
    else:
        joint = tables.joint.get((target, class_value, feature, value))
        if joint is not None and joint > 0 and pc is not None and pc > 0:
            return _log(joint) - _log(pc)
    if pf is not None and pf > 0:
        return _log(pf)
    return 0.0


def classes_from_marginal(tables: TableFactors, target: str) -> list[str]:
    found = [val for (anchor, val), share in tables.marg.items() if anchor == target and share > 0]
    return sorted(found)


def predict_labels(
    tables: TableFactors,
    target: str,
    frame: pd.DataFrame,
    features: list[str],
    classes: list[str] | None = None,
) -> np.ndarray:
    """MAP class of multinomial naive Bayes on the published tables. Row order matches frame."""
    if classes is None:
        classes = classes_from_marginal(tables, target)
    if not classes:
        raise ValueError(f"{target}: no classes in the published marginal")
    prior = np.array([_log(tables.marg.get((target, c), _LOG_FLOOR)) for c in classes])
    scores = np.broadcast_to(prior, (len(frame), len(classes))).copy()
    for feature in features:
        col = frame[feature].astype(str).to_numpy()
        for value in pd.unique(col):
            delta = np.array(
                [factor_loglik(tables, target, c, feature, str(value)) for c in classes]
            )
            scores[col == value] += delta
    return np.asarray(classes, dtype=object)[scores.argmax(axis=1)]


def attendance_probability(tables: TableFactors, keys: list[str], values: list[str]) -> float:
    """P(attend_any=예 | cell). With one axis and a published cell, this equals that
    conditional probability.
    """
    p1 = tables.marg[("attend_any", "예")]
    p0 = tables.marg[("attend_any", "아니오")]
    log_odds = _log(p1) - _log(p0)
    for key, value in zip(keys, values, strict=True):
        ll1 = factor_loglik(tables, "attend_any", "예", key, value)
        ll0 = factor_loglik(tables, "attend_any", "아니오", key, value)
        log_odds += ll1 - ll0
    return _expit(log_odds)


def _record(
    task: str,
    grain: str,
    setting: str,
    slice_: str,
    wave: str,
    unit: str,
    method: str,
    metric: str,
    value: float,
    n: int,
) -> dict[str, object]:
    return {
        "task": task,
        "grain": grain,
        "setting": setting,
        "slice": slice_,
        "wave": wave,
        "unit": unit,
        "method": method,
        "metric": metric,
        "value": float(value),
        "n": int(n),
    }


def _wacc(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(y.astype(str) == pred.astype(str), weights=w))


def genre_records(reports: Path) -> list[dict[str, object]]:
    """Genre AUC is the mean of the existing CSV. Pair-joint columns are not recomputed."""
    g = pd.read_csv(reports / "era_genre_portfolio.csv")
    rows: list[dict[str, object]] = []
    grouped = g.groupby(["setting", "features"], sort=False)
    for (setting, features), part in grouped:
        for method, col in GENRE_COLS.items():
            rows.append(
                _record(
                    "genre",
                    "summary",
                    str(setting),
                    str(features),
                    "",
                    "*",
                    method,
                    "auc",
                    float(part[col].mean()),
                    int(part[col].notna().sum()),
                )
            )
    return rows


def _feature_list(fset: str, target: str, syn: pd.DataFrame, real: pd.DataFrame) -> list[str]:
    targets = [a for a in ERA_CORE_ANCHORS if a in syn.columns]
    features = [a for a in targets if a != target and a in real.columns]
    if fset == "demographic":
        features = [a for a in features if a not in ATTEND_BLOCK]
    return features


def _impute_one(
    tables: PairTables,
    syn: pd.DataFrame,
    real: pd.DataFrame,
    target: str,
    features: list[str],
) -> tuple[float, int]:
    """Published-table naive Bayes weighted accuracy, on the same rows as
    era_tstr.evaluate_target.
    """
    cols = [*features, target]
    s = syn.dropna(subset=cols)
    r = real.dropna(subset=cols)
    counts = s[target].value_counts()
    keep = set(counts[counts >= MIN_CLASS].index) & set(r[target].unique())
    if len(keep) < 2:
        raise ValueError(f"{target}: fewer than two classes")
    r = r[r[target].isin(keep)]
    pred = predict_labels(tables, target, r, features)
    y = r[target].to_numpy()
    w = r["w"].to_numpy(dtype=float)
    return _wacc(y, pred, w), len(r)


def imputation_records(reports: Path) -> list[dict[str, object]]:
    tstr = pd.read_csv(reports / "era_tstr.csv")
    train_wave = int(tstr.train_wave.iloc[0])
    if train_wave != TRAIN_WAVE:
        raise ValueError(f"TSTR training wave is not {TRAIN_WAVE}: {train_wave}")
    syn = synthetic_bank(train_wave)
    tables = PairTables(train_wave)
    reals = {int(w): real_frame(int(w)) for w in sorted(tstr.test_wave.unique())}
    detail: list[dict[str, object]] = []
    for row in tstr.itertuples(index=False):
        real = reals[int(row.test_wave)]
        features = _feature_list(str(row.features), str(row.target), syn, real)
        if len(features) != int(row.n_features):
            raise RuntimeError(
                f"{row.target} {row.features} wave {row.test_wave}: "
                f"{len(features)} features, CSV has {int(row.n_features)}"
            )
        acc, n = _impute_one(tables, syn, real, str(row.target), features)
        if n != int(row.n_test_real):
            raise RuntimeError(
                f"{row.target} {row.features} wave {row.test_wave}: "
                f"{n} test rows, CSV has {int(row.n_test_real)}"
            )
        detail.append(
            {
                "setting": str(row.setting),
                "slice": str(row.features),
                "wave": str(int(row.test_wave)),
                "unit": str(row.target),
                "n": n,
                "marginal": float(row.acc_marginal),
                "age_table": float(row.acc_age),
                "pair_tables": acc,
                "synthetic": float(row.acc_tstr),
                "microdata": float(row.acc_trtr),
            }
        )
        log.info(
            "%-24s %-11s wave %s pair=%.3f bank=%.3f",
            row.target,
            row.features,
            int(row.test_wave),
            acc,
            float(row.acc_tstr),
        )
    detail_df = pd.DataFrame(detail)
    rows: list[dict[str, object]] = []
    for rec in detail:
        for method in METHODS["imputation"]:
            rows.append(
                _record(
                    "imputation",
                    "target",
                    str(rec["setting"]),
                    str(rec["slice"]),
                    str(rec["wave"]),
                    str(rec["unit"]),
                    method,
                    "weighted_accuracy",
                    float(rec[method]),
                    int(rec["n"]),
                )
            )
    for (setting, slice_), part in detail_df.groupby(["setting", "slice"], sort=False):
        for method in METHODS["imputation"]:
            rows.append(
                _record(
                    "imputation",
                    "summary",
                    str(setting),
                    str(slice_),
                    "",
                    "*",
                    method,
                    "weighted_accuracy",
                    float(part[method].mean()),
                    len(part),
                )
            )
        _append_win_counts(rows, part, str(setting), str(slice_), "")
    for (wave, slice_), part in detail_df.groupby(["wave", "slice"], sort=False):
        setting = str(part.setting.iloc[0])
        for method in METHODS["imputation"]:
            rows.append(
                _record(
                    "imputation",
                    "wave",
                    setting,
                    str(slice_),
                    str(wave),
                    "*",
                    method,
                    "weighted_accuracy",
                    float(part[method].mean()),
                    len(part),
                )
            )
    return rows


def _append_win_counts(
    rows: list[dict[str, object]], part: pd.DataFrame, setting: str, slice_: str, wave: str
) -> None:
    n = len(part)
    bank_beats = int((part.synthetic > part.pair_tables).sum())
    pair_beats = int((part.pair_tables > part.synthetic).sum())
    rows.append(
        _record(
            "imputation",
            "summary",
            setting,
            slice_,
            wave,
            "*",
            "synthetic",
            "n_rows_beats_pair_tables",
            bank_beats,
            n,
        )
    )
    rows.append(
        _record(
            "imputation",
            "summary",
            setting,
            slice_,
            wave,
            "*",
            "pair_tables",
            "n_rows_beats_synthetic",
            pair_beats,
            n,
        )
    )


def _design_metric_rows(
    part: pd.DataFrame,
    grain: str,
    setting: str,
    slice_: str,
    wave: str,
    unit: str,
) -> list[dict[str, object]]:
    n = len(part)
    metrics = {
        "mean_abs_err_pp": float(part.abs_err_pp.mean()),
        "median_n_ratio": float(part.n_ratio.median()),
        "under_planned_pct": 100.0 * float((part.n_ratio < 0.9).mean()),
    }
    return [
        _record("design", grain, setting, slice_, wave, unit, str(part.method.iloc[0]), m, v, n)
        for m, v in metrics.items()
    ]


def design_records(reports: Path) -> list[dict[str, object]]:
    """Attach published-table attendance rates beside existing cells' p_true. Microdata is
    not reread.
    """
    src = pd.read_csv(reports / "era_design_pretest.csv")
    base = src[src.method == "marginal"][
        ["wave", "cell_key", "published_axis", "cell", "p_true", "n_required"]
    ].copy()
    tables = {int(w): PairTables(int(w)) for w in sorted(base.wave.unique())}
    hat: list[float] = []
    for row in base.itertuples(index=False):
        keys = str(row.cell_key).split("+")
        values = str(row.cell).split(" / ")
        if len(keys) != len(values):
            raise RuntimeError(f"failed to split cell: {row.cell_key} = {row.cell}")
        hat.append(attendance_probability(tables[int(row.wave)], keys, values))
    pair = base.copy()
    pair["p_hat"] = hat
    pair["method"] = "pair_tables"
    pair["abs_err_pp"] = 100 * (pair.p_hat - pair.p_true).abs()
    pair["n_ratio"] = [
        required_n(float(p)) / float(n_req)
        for p, n_req in zip(pair.p_hat, pair.n_required, strict=True)
    ]
    age = pair[pair.cell_key == "age_band"]
    age_err = float(age.abs_err_pp.mean())
    log.info("age_band cells: pair-table mean abs err %.4f pp (n=%d)", age_err, len(age))
    if age_err > 0.05:
        raise RuntimeError(
            f"published-table error on the age cells is {age_err:.3f} pp. "
            "With a single axis, naive Bayes must equal T2."
        )

    existing = src.rename(columns={})[
        ["wave", "cell_key", "published_axis", "cell", "method", "abs_err_pp", "n_ratio"]
    ].copy()
    existing["method"] = existing.method.replace({"age-conditional": "age_table"})
    both = pd.concat(
        [
            existing,
            pair[["wave", "cell_key", "published_axis", "cell", "method", "abs_err_pp", "n_ratio"]],
        ],
        ignore_index=True,
    )
    both["slice"] = np.where(both.published_axis.astype(bool), "published", "unpublished")
    both["wave"] = both.wave.astype(int).astype(str)

    rows: list[dict[str, object]] = []
    for (slice_, method), part in both.groupby(["slice", "method"], sort=False):
        rows.extend(_design_metric_rows(part, "summary", "pooled", str(slice_), "", "*"))
    for slice_ in both["slice"].unique():
        closer, n_compared = _cells_closer(both, str(slice_))
        rows.append(
            _record(
                "design",
                "summary",
                "pooled",
                str(slice_),
                "",
                "*",
                "pair_tables",
                "n_cells_closer_than_synthetic",
                closer,
                n_compared,
            )
        )
    for (wave, slice_, method), part in both.groupby(["wave", "slice", "method"], sort=False):
        rows.extend(_design_metric_rows(part, "wave", "in-wave", str(slice_), str(wave), "*"))
    for (slice_, unit, method), part in both.groupby(["slice", "cell_key", "method"], sort=False):
        rows.extend(_design_metric_rows(part, "cell_key", "pooled", str(slice_), "", str(unit)))
    return rows


def _cells_closer(both: pd.DataFrame, slice_: str) -> tuple[int, int]:
    sub = both[both["slice"] == slice_]
    pair = sub[sub.method == "pair_tables"][["wave", "cell_key", "cell", "abs_err_pp"]]
    bank = sub[sub.method == "synthetic"][["wave", "cell_key", "cell", "abs_err_pp"]]
    m = pair.merge(bank, on=["wave", "cell_key", "cell"], suffixes=("_pair", "_bank"))
    return int((m.abs_err_pp_pair < m.abs_err_pp_bank).sum()), len(m)


def _lookup(df: pd.DataFrame, **kw: object) -> float:
    m = df
    for key, val in kw.items():
        m = m[m[key] == val]
    if len(m) != 1:
        raise RuntimeError(f"lookup {kw} → {len(m)} rows")
    return float(m.value.iloc[0])


def _fmt(metric: str, value: float) -> str:
    if metric == "under_planned_pct":
        return f"{value:.1f}"
    if metric.startswith("n_"):
        return f"{value:.0f}"
    return f"{value:.3f}"


def _md_table(headers: list[str], body: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    lines = ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join([head, sep, *lines])


def _ordered(df: pd.DataFrame, task: str, grain: str) -> pd.DataFrame:
    sub = df[(df.task == task) & (df.grain == grain)].copy()
    sub["setting"] = pd.Categorical(sub.setting, SETTING_ORDER, ordered=True)
    sub["slice"] = pd.Categorical(sub["slice"], SLICE_ORDER[task], ordered=True)
    sub["method"] = pd.Categorical(sub.method, METHODS[task], ordered=True)
    return sub.sort_values(["setting", "slice", "wave", "unit", "method", "metric"])


def render_md(df: pd.DataFrame) -> str:
    lines = [
        "# 하류 과제 기준선 — 공개표 나이브 베이즈",
        "",
        "> 자동 생성: `scripts/era_downstream_baselines.py`.",
        "> 기존 `era_genre_portfolio.csv` · `era_tstr.csv` · `era_design_pretest.csv` 의 값은 그대로 두고, 공개표 기준선만 더했다.",
        "",
        "## 각 과제가 재는 것",
        "",
        "장르(표 25)는 관람자의 가려진 장르 하나를 맞힌다. 지표는 AUC. 비관람자는 빼는데, 장르가 전부 '아니오'인 것은 설문 라우팅이기 때문이다.",
        "",
        "대치(표 26)는 실제 응답자의 빠진 앵커를 나머지 앵커로 맞힌다. 지표는 조사가중 정확도. `synthetic` 은 2024 뱅크로 학습한 다항 로지스틱, `microdata` 는 실제 응답자 교차검증이다.",
        "",
        "설계(표 27)는 칸 관람률로 필요 표본을 미리 잡을 때의 오차다. `mean_abs_err_pp` 는 |p̂ − p| 를 퍼센트포인트로 평균한 것이고, `n_ratio` 는 계획한 표본 / 필요한 표본, `under_planned_pct` 는 그 배율이 0.9 미만인 칸의 비율이다.",
        "",
        "## 기준선 정의",
        "",
        "장르는 기존 `PairTables.score` 다. 학습 파동(2024)의 공개 T1·T6·T2–T4 로 만든 관람자 조건부 나이브 베이즈이고, 이번 스크립트는 그 AUC 평균을 옮기기만 한다.",
        "",
        "대치는 같은 표를 쓰는 다항 나이브 베이즈다. `PairTables.score` 는 이진 장르를 관람자 조건으로 채점하므로, 다항 앵커와 응답자 전원에는 다른 질문이다. 사전은 T1, 비뿌리 특징의 우도는 T6, 성·연령·생활권은 T2–T4 를 뒤집은 우도다. 공개판에 없는 셀은 그 계급만 독립으로 둔다. 이진 조건부에서 한 계급만 공개된 경우에는 여사건을 1−공개비율로 둔다. 쌍결합의 빈 셀은 주변과의 차분으로 채우지 않는다. 표는 2024 공개판이고, 채점 행은 기존 TSTR 과 같다(합성 뱅크에서 30 명 미만인 계급은 뺀다).",
        "",
        "설계는 그 우도비를 관람 여부 두 계급에 적용한 P(관람 | 칸) 이다. 칸 비율이 필요해서 로그오즈 점수를 확률로 바꿨고, 관람자 조건 보정은 하지 않는다. 축이 하나이고 셀이 공개돼 있으면 공개 조건부 확률과 같다. 각 파동의 공개판을 그 파동의 칸에 쓴다. 연령×성 직접표(T5)는 장르 기준선과 표 집합을 맞추려고 쓰지 않는다.",
        "",
        "서사에서 앵커를 복원하는 프로브는 여기 넣지 않는다. 입력이 문장이라 쌍결합 우도를 붙일 앵커가 없다. 표본 밖 설계(`era_chain_crossfit_design`)에도 넣지 않는다. 공개표는 응답자 전원으로 만들어져 채점 절반을 포함한다.",
        "",
        "## 장르 — AUC",
        "",
        _metric_table(df, "genre", "summary", "auc"),
        "",
        "## 대치 — 가중 정확도",
        "",
        _metric_table(df, "imputation", "summary", "weighted_accuracy"),
        "",
        _impute_prose(df),
        "",
        "### 시험 파동별",
        "",
        _wave_table(df, "imputation", "weighted_accuracy"),
        "",
        "## 설계 — 칸 관람률",
        "",
        _design_summary_table(df),
        "",
        _design_prose(df),
        "",
        "### 파동별 평균 절대오차 (pp)",
        "",
        _wave_table(df, "design", "mean_abs_err_pp"),
        "",
        "### 칸 정의별 평균 절대오차 (pp, 세 파동 합산)",
        "",
        _cell_key_table(df),
        "",
        "연령대만인 칸에서 이 기준선은 공개 T2 와 같고, 평균 절대오차는 연령 조건부 기준선과 같이 0 이다.",
        "",
    ]
    return "\n".join(lines)


def _metric_table(df: pd.DataFrame, task: str, grain: str, metric: str) -> str:
    sub = _ordered(df, task, grain)
    sub = sub[sub.metric == metric]
    headers = ["setting", "slice", *METHODS[task]]
    body: list[list[str]] = []
    for (setting, slice_), part in sub.groupby(["setting", "slice"], sort=False, observed=True):
        vals = {str(r.method): _fmt(metric, float(r.value)) for r in part.itertuples()}
        body.append([str(setting), str(slice_), *[vals.get(m, "") for m in METHODS[task]]])
    return _md_table(headers, body)


def _wave_table(df: pd.DataFrame, task: str, metric: str) -> str:
    sub = df[(df.task == task) & (df.grain == "wave") & (df.metric == metric)].copy()
    sub["slice"] = pd.Categorical(sub["slice"], SLICE_ORDER[task], ordered=True)
    sub["method"] = pd.Categorical(sub.method, METHODS[task], ordered=True)
    sub = sub.sort_values(["slice", "wave", "method"])
    headers = ["slice", "wave", *METHODS[task]]
    body: list[list[str]] = []
    for (slice_, wave), part in sub.groupby(["slice", "wave"], sort=False, observed=True):
        vals = {str(r.method): _fmt(metric, float(r.value)) for r in part.itertuples()}
        body.append([str(slice_), str(wave), *[vals.get(m, "") for m in METHODS[task]]])
    return _md_table(headers, body)


def _design_summary_table(df: pd.DataFrame) -> str:
    metrics = ["mean_abs_err_pp", "median_n_ratio", "under_planned_pct"]
    sub = _ordered(df, "design", "summary")
    sub = sub[sub.metric.isin(metrics)]
    headers = ["slice", "method", *metrics]
    body: list[list[str]] = []
    for (slice_, method), part in sub.groupby(["slice", "method"], sort=False, observed=True):
        vals = {str(r.metric): _fmt(str(r.metric), float(r.value)) for r in part.itertuples()}
        body.append([str(slice_), str(method), *[vals[m] for m in metrics]])
    return _md_table(headers, body)


def _cell_key_table(df: pd.DataFrame) -> str:
    sub = df[(df.task == "design") & (df.grain == "cell_key") & (df.metric == "mean_abs_err_pp")]
    sub = sub.copy()
    sub["slice"] = pd.Categorical(sub["slice"], SLICE_ORDER["design"], ordered=True)
    sub["method"] = pd.Categorical(sub.method, METHODS["design"], ordered=True)
    sub = sub.sort_values(["slice", "unit", "method"])
    headers = ["slice", "cell_key", *METHODS["design"]]
    body: list[list[str]] = []
    for (slice_, unit), part in sub.groupby(["slice", "unit"], sort=False, observed=True):
        vals = {str(r.method): f"{float(r.value):.3f}" for r in part.itertuples()}
        body.append([str(slice_), str(unit), *[vals.get(m, "") for m in METHODS["design"]]])
    return _md_table(headers, body)


def _impute_prose(df: pd.DataFrame) -> str:
    bits = []
    for setting, slice_ in (
        ("cross-wave", "all"),
        ("cross-wave", "demographic"),
        ("in-wave", "all"),
        ("in-wave", "demographic"),
    ):
        n = int(
            _lookup(
                df,
                task="imputation",
                grain="summary",
                setting=setting,
                slice=slice_,
                method="synthetic",
                metric="n_rows_beats_pair_tables",
            )
        )
        p = int(
            _lookup(
                df,
                task="imputation",
                grain="summary",
                setting=setting,
                slice=slice_,
                method="pair_tables",
                metric="n_rows_beats_synthetic",
            )
        )
        total = int(
            df[
                (df.task == "imputation")
                & (df.grain == "summary")
                & (df.setting == setting)
                & (df["slice"] == slice_)
                & (df.method == "synthetic")
                & (df.metric == "n_rows_beats_pair_tables")
            ].n.iloc[0]
        )
        pair = _lookup(
            df,
            task="imputation",
            grain="summary",
            setting=setting,
            slice=slice_,
            method="pair_tables",
            metric="weighted_accuracy",
        )
        bank = _lookup(
            df,
            task="imputation",
            grain="summary",
            setting=setting,
            slice=slice_,
            method="synthetic",
            metric="weighted_accuracy",
        )
        bits.append(
            f"{setting} · {slice_}: 공개표 {pair:.3f}, 뱅크 {bank:.3f} "
            f"(행 {total} 중 뱅크가 높은 행 {n}, 공개표가 높은 행 {p}, 동점 {total - n - p})."
        )
    return " ".join(bits)


def _design_prose(df: pd.DataFrame) -> str:
    bits = []
    for slice_ in ("unpublished", "published"):
        pair = _lookup(
            df,
            task="design",
            grain="summary",
            setting="pooled",
            slice=slice_,
            method="pair_tables",
            metric="mean_abs_err_pp",
        )
        bank = _lookup(
            df,
            task="design",
            grain="summary",
            setting="pooled",
            slice=slice_,
            method="synthetic",
            metric="mean_abs_err_pp",
        )
        closer = int(
            _lookup(
                df,
                task="design",
                grain="summary",
                setting="pooled",
                slice=slice_,
                method="pair_tables",
                metric="n_cells_closer_than_synthetic",
            )
        )
        n = int(
            df[
                (df.task == "design")
                & (df.grain == "summary")
                & (df.setting == "pooled")
                & (df["slice"] == slice_)
                & (df.method == "pair_tables")
                & (df.metric == "n_cells_closer_than_synthetic")
            ].n.iloc[0]
        )
        bits.append(
            f"{slice_}: 공개표 평균오차 {pair:.3f}pp, 뱅크 {bank:.3f}pp "
            f"(칸 {n} 중 공개표가 더 가까운 칸 {closer})."
        )
    return " ".join(bits)


def main() -> int:
    reports = settings.outputs_dir / "reports"
    rows = [
        *genre_records(reports),
        *imputation_records(reports),
        *design_records(reports),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(reports / "era_downstream_baselines.csv", index=False)
    text = render_md(df)
    (reports / "era_downstream_baselines.md").write_text(text, encoding="utf-8")
    log.info("wrote %s", reports / "era_downstream_baselines.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
