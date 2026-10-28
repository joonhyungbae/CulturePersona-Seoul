"""Explain the error ladder with conditional mutual information.

If reconstruction R is conditionally independent given the condition set C_R,
and margins are taken as truth, measure KL(P(X,Y) || P_R) alongside
I(X;Y|C_R) and compare the Pinsker bound sqrt(KL/2) with TVD. Person-level
rows are read only inside the frame loader; outputs are aggregate tables.

Run: uv run python scripts/era_info_ladder.py
Outputs: outputs/reports/era_info_ladder_pairs.csv
         outputs/reports/era_info_ladder.csv
         outputs/reports/era_info_ladder.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.error_study import (
    NON_ATTENDER,
    RECON,
    _align,
    _cond_indep,
    _cond_route,
    _joint,
    ladder_mask,
    pair_errors,
    routed_values,
)
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import ERA_CORE_ANCHORS, ROOT_ANCHORS, WEIGHTED_WAVES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decade_error_atlas import _pair_class
from kcas_atlas import anchor_frame, kcas_path, non_root_anchors

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _name in ("sfac_persona.grounding.error_study", "sfac_persona.grounding.official"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("era_info_ladder")

# Add eps only to cells where P_R is 0 and P_true > 0. Do not renormalize.
KL_EPS = 1e-12
# Condition variables of the conditional-independence reconstruction.
# route uses the two subgroups of _cond_route.
COND_BY: dict[str, list[str]] = {
    "indep": [],
    "cond_age": ["age_band"],
    "cond_age_sex": ["age_band", "sex"],
    "cond_age_attend": ["age_band", "attend_any"],
}
# Amount a rung removes: (name, upper rung, lower rung). age→route skips the sex rung.
DELTAS: tuple[tuple[str, str, str], ...] = (
    ("indep_to_age", "indep", "cond_age"),
    ("age_to_sex", "cond_age", "cond_age_sex"),
    ("age_to_route", "cond_age", "cond_age_route"),
    ("route_to_attend", "cond_age_route", "cond_age_attend"),
)
PAIR_CLASSES = ("both", "one", "neither")


def kl_tvd(p: np.ndarray, q: np.ndarray, eps: float = KL_EPS) -> tuple[float, float, int]:
    """TVD, KL(p||q) (nats), and the number of cells with q=0 and p>0.

    Add eps only on zero cells. The sum is no longer 1, so KL is close to a lower bound.
    """
    tvd = float(0.5 * np.abs(p - q).sum())
    zero = (q <= 0.0) & (p > 0.0)
    n_zero = int(zero.sum())
    q_use = np.where(zero, eps, q)
    mask = p > 0.0
    kl = float(np.sum(p[mask] * np.log(p[mask] / q_use[mask])))
    return tvd, kl, n_zero


def mutual_info(joint: np.ndarray) -> float:
    """Mutual information I(X;Y) of a weighted joint (already sums to 1), in nats. Zero cells drop out."""
    total = float(joint.sum())
    if total <= 0.0:
        return 0.0
    p = joint / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / (px * py)[mask])))


def _weighted_joint(g: pd.DataFrame, a: str, b: str) -> np.ndarray:
    t = g.pivot_table(index=a, columns=b, values="w", aggfunc="sum", fill_value=0.0, observed=True)
    return t.to_numpy(dtype=float)


def cmi_by(df: pd.DataFrame, a: str, b: str, by: list[str]) -> float:
    """I(X;Y|C) = Σ_c P(c) I(X;Y|C=c). If C=∅, this is I(X;Y).

    As in _cond_indep, the sample uses only rows where (a, b) are present.
    """
    sub = df.dropna(subset=[a, b])
    total = float(sub.w.sum())
    if total <= 0.0:
        return 0.0
    if not by:
        return mutual_info(_weighted_joint(sub, a, b))
    acc = 0.0
    for _, g in sub.groupby(by, observed=True, dropna=True):
        w = float(g.w.sum())
        if w <= 0.0:
            continue
        acc += (w / total) * mutual_info(_weighted_joint(g, a, b))
    return float(acc)


def cmi_route(df: pd.DataFrame, a: str, b: str) -> float:
    """Same split as _cond_route: I from the true joint of the two non-attender/attender subgroups within an age band.

    This is the observed joint of those subgroups, not the margins the reconstruction uses (routing point mass and clip).
    """
    sub = df.dropna(subset=[a, b])
    total = float(sub.w.sum())
    if total <= 0.0 or "attend_any" not in sub.columns:
        return 0.0
    acc = 0.0
    for _, g in sub.groupby("age_band", observed=True, dropna=True):
        wg = float(g.w.sum())
        if wg <= 0.0:
            continue
        pg = wg / total
        non = g[g["attend_any"] == NON_ATTENDER]
        att = g[g["attend_any"] != NON_ATTENDER]
        for part, pi in ((non, float(non.w.sum()) / wg), (att, float(att.w.sum()) / wg)):
            if pi <= 0.0 or float(part.w.sum()) <= 0.0:
                continue
            acc += pg * pi * mutual_info(_weighted_joint(part, a, b))
    return float(acc)


def pinsker_bound(kl: float) -> float:
    return float(np.sqrt(max(kl, 0.0) / 2.0))


def _core_anchors(frame: pd.DataFrame) -> list[str]:
    return [a for a in ERA_CORE_ANCHORS if a in frame.columns and a not in ROOT_ANCHORS]


def _reconstruct(
    df: pd.DataFrame, a: str, b: str, rung: str, routed: dict[str, str], obs: pd.DataFrame
) -> pd.DataFrame:
    if rung == "indep":
        return pd.DataFrame(
            np.outer(obs.sum(axis=1).to_numpy(), obs.sum(axis=0).to_numpy()),
            index=obs.index,
            columns=obs.columns,
        )
    if rung == "cond_age_route":
        return _cond_route(df, a, b, routed)
    return _cond_indep(df, a, b, COND_BY[rung])


def pair_info(df: pd.DataFrame, anchors: list[str]) -> pd.DataFrame:
    """ladder_mask pairs × five rungs. Person rows are read only here; no row-level result is returned."""
    routed = routed_values(df, anchors)
    rows: list[dict[str, object]] = []
    for _, pair in pair_errors(df, anchors=anchors).iterrows():
        if not bool(ladder_mask(pd.DataFrame([pair])).iloc[0]):
            continue
        a, b = str(pair["anchor_a"]), str(pair["anchor_b"])
        obs = _joint(df, a, b)
        klass = _pair_class(a, b)
        for rung in RECON:
            q = _reconstruct(df, a, b, rung, routed, obs)
            p_arr, q_arr = _align(obs, q)
            tvd, kl, n_zero = kl_tvd(p_arr, q_arr)
            cmi = (
                cmi_route(df, a, b) if rung == "cond_age_route" else cmi_by(df, a, b, COND_BY[rung])
            )
            bound = pinsker_bound(kl)
            rows.append(
                {
                    "anchor_a": a,
                    "anchor_b": b,
                    "rung": rung,
                    "tvd": tvd,
                    "kl": kl,
                    "cmi": cmi,
                    "bound": bound,
                    "ratio": (tvd / bound) if bound > 0.0 else np.nan,
                    "pair_class": klass,
                    "n_zero": n_zero,
                    "tvd_atlas": float(pair[f"tvd_{rung}"])
                    if pd.notna(pair.get(f"tvd_{rung}"))
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan"), float("nan"), n
    pr = stats.pearsonr(x[mask], y[mask])
    sr = stats.spearmanr(x[mask], y[mask])
    return float(sr.statistic), float(pr.statistic), n


def _mean(s: pd.Series) -> float:
    return float(s.mean())


def aggregate(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-rung means, rung gaps, route/beyond shares, Pinsker correlation, route prediction."""
    rows: list[dict[str, object]] = []
    for (source, wave), g in pairs.groupby(["source", "wave"], sort=False):
        wide_tvd = g.pivot_table(index=["anchor_a", "anchor_b"], columns="rung", values="tvd")
        wide_cmi = g.pivot_table(index=["anchor_a", "anchor_b"], columns="rung", values="cmi")
        for rung in RECON:
            sub = g[g.rung == rung]
            rows.append(
                {
                    "source": source,
                    "wave": wave,
                    "kind": "rung",
                    "rung": rung,
                    "pair_class": "all",
                    "n": len(sub),
                    "mean_tvd": _mean(sub.tvd),
                    "mean_kl": _mean(sub.kl),
                    "mean_cmi": _mean(sub.cmi),
                    "mean_bound": _mean(sub.bound),
                    "max_abs_kl_cmi": float((sub.kl - sub.cmi).abs().max()),
                    "n_zero_cells": int(sub.n_zero.sum()),
                }
            )
        for name, hi, lo in DELTAS:
            rows.append(
                {
                    "source": source,
                    "wave": wave,
                    "kind": "delta",
                    "rung": name,
                    "pair_class": "all",
                    "n": len(wide_tvd),
                    "delta_tvd": float((wide_tvd[hi] - wide_tvd[lo]).mean()),
                    "delta_cmi": float((wide_cmi[hi] - wide_cmi[lo]).mean()),
                }
            )
        age_t, route_t = _mean(wide_tvd["cond_age"]), _mean(wide_tvd["cond_age_route"])
        att_t = _mean(wide_tvd["cond_age_attend"])
        age_i, route_i = _mean(wide_cmi["cond_age"]), _mean(wide_cmi["cond_age_route"])
        att_i = _mean(wide_cmi["cond_age_attend"])
        rows.append(
            {
                "source": source,
                "wave": wave,
                "kind": "share",
                "rung": "route_beyond",
                "pair_class": "all",
                "n": len(wide_tvd),
                "route_cut_pct_tvd": 100.0 * (1.0 - route_t / age_t) if age_t else np.nan,
                "beyond_pct_tvd": 100.0 * (1.0 - att_t / route_t) if route_t else np.nan,
                "route_cut_pct_cmi": 100.0 * (1.0 - route_i / age_i) if age_i else np.nan,
                "beyond_pct_cmi": 100.0 * (1.0 - att_i / route_i) if route_i else np.nan,
            }
        )
        for klass in ("all", *PAIR_CLASSES):
            sub = g if klass == "all" else g[g.pair_class == klass]
            for rung in RECON:
                r = sub[sub.rung == rung]
                if r.empty:
                    continue
                # The bound is sqrt(I/2). How loose Pinsker is: TVD/sqrt(KL/2).
                rho_s, rho_p, n = _corr(r.tvd.to_numpy(), np.sqrt(r.cmi.clip(lower=0.0) / 2.0))
                ratio = r.ratio.to_numpy(dtype=float)
                rows.append(
                    {
                        "source": source,
                        "wave": wave,
                        "kind": "corr",
                        "rung": rung,
                        "pair_class": klass,
                        "n": n,
                        "spearman_tvd_sqrt_cmi": rho_s,
                        "pearson_tvd_sqrt_cmi": rho_p,
                        "ratio_p25": float(np.nanpercentile(ratio, 25)),
                        "ratio_p50": float(np.nanpercentile(ratio, 50)),
                        "ratio_p75": float(np.nanpercentile(ratio, 75)),
                    }
                )
        d_cmi = (wide_cmi["cond_age"] - wide_cmi["cond_age_route"]).to_numpy()
        d_tvd = (wide_tvd["cond_age"] - wide_tvd["cond_age_route"]).to_numpy()
        rho_s, rho_p, n = _corr(d_cmi, d_tvd)
        rows.append(
            {
                "source": source,
                "wave": wave,
                "kind": "predict",
                "rung": "age_minus_route",
                "pair_class": "all",
                "n": n,
                "spearman_route_cut": rho_s,
                "pearson_route_cut": rho_p,
            }
        )
    return pd.DataFrame(rows)


def _fmt(x: object, nd: int = 4) -> str:
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return "" if not np.isfinite(v) else f"{v:.{nd}f}"
    if x is None:
        return ""
    return str(x)


def _table_from(df: pd.DataFrame, cols: list[str], nd: int = 4) -> str:
    show = df.loc[:, cols]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for rec in show.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, nd) for v in rec) + " |")
    return "\n".join(lines)


def write_markdown(summary: pd.DataFrame, verify_note: str) -> str:
    """Table and a short reading. Numbers are taken only from summary."""
    seoul = summary[summary.source == "seoul"]
    rungs = seoul[seoul.kind == "rung"][
        ["wave", "rung", "n", "mean_tvd", "mean_cmi", "mean_kl", "max_abs_kl_cmi"]
    ]
    deltas = seoul[seoul.kind == "delta"][["wave", "rung", "delta_tvd", "delta_cmi"]]
    shares = seoul[seoul.kind == "share"][
        [
            "wave",
            "route_cut_pct_tvd",
            "route_cut_pct_cmi",
            "beyond_pct_tvd",
            "beyond_pct_cmi",
        ]
    ]
    corr = seoul[(seoul.kind == "corr") & (seoul.pair_class == "all")][
        [
            "wave",
            "rung",
            "spearman_tvd_sqrt_cmi",
            "pearson_tvd_sqrt_cmi",
            "ratio_p50",
            "ratio_p25",
            "ratio_p75",
        ]
    ]
    corr_cls = seoul[
        (seoul.kind == "corr") & (seoul.rung == "cond_age") & (seoul.pair_class != "all")
    ][["wave", "pair_class", "spearman_tvd_sqrt_cmi", "pearson_tvd_sqrt_cmi", "ratio_p50"]]
    pred = seoul[seoul.kind == "predict"][["wave", "spearman_route_cut", "pearson_route_cut", "n"]]
    kcas_share = summary[(summary.source == "kcas") & (summary.kind == "share")][
        ["wave", "route_cut_pct_tvd", "route_cut_pct_cmi", "beyond_pct_tvd", "beyond_pct_cmi"]
    ]
    indep_gap = rungs[rungs.rung == "indep"]["max_abs_kl_cmi"].max()
    other = rungs[rungs.rung != "indep"]
    other_gap = other["max_abs_kl_cmi"].max()
    route_rho = pred["spearman_route_cut"].median()
    text = f"""# 오차 사다리와 조건부 정보량

가중 서울 파동(2020·2022·2024)과 KCAS 가중 프레임. 쌍은 `ladder_mask` (구조적 쌍·관람 여부 쌍 제외).
`pair_class`: both = 관람 블록 안, one = 블록–나머지, neither = 나머지–나머지 (`decade_error_atlas._pair_class`).

## 검증

{verify_note}

독립 단에서 max |KL − I| = {_fmt(indep_gap, 6)}. 그 외 단의 max |KL − CMI| = {_fmt(other_gap, 4)}.
KL(P(X,Y) || P_R) 는 (X,Y) 로 주변화한 거리이고, I(X;Y|C) 는 (X,Y,C) 결합에서의 조건부 독립 KL 이다.
C 가 비어 있을 때만 둘이 같다. C 가 있으면 체인 법칙으로 KL(주변) ≤ I 이고, 차이는 C 의 사후분포 KL 이다.
route 단은 여기에 더해 `_split_by_routing` 이 참 하위집단 주변분포를 쓰지 않는 칸(점질량·clip)이 있다.
CMI 는 그 하위집단의 **참** 결합으로 재므로, route 의 CMI 는 연령×관람 CMI 와 같고 KL 과는 어긋난다.

## 단별 평균 (서울)

{_table_from(rungs, ["wave", "rung", "n", "mean_tvd", "mean_kl", "mean_cmi", "max_abs_kl_cmi"])}

## 단이 깎은 오차와 정보량 (서울)

{_table_from(deltas, ["wave", "rung", "delta_tvd", "delta_cmi"])}

## route cut · beyond (TVD 와 정보량)

{_table_from(shares, ["wave", "route_cut_pct_tvd", "route_cut_pct_cmi", "beyond_pct_tvd", "beyond_pct_cmi"], 2)}

KCAS (파동별, 같은 정의):

{_table_from(kcas_share, ["wave", "route_cut_pct_tvd", "route_cut_pct_cmi", "beyond_pct_tvd", "beyond_pct_cmi"], 2)}

## TVD 와 sqrt(I/2) (서울, 전체 쌍)

{_table_from(corr, ["wave", "rung", "spearman_tvd_sqrt_cmi", "pearson_tvd_sqrt_cmi", "ratio_p25", "ratio_p50", "ratio_p75"], 3)}

연령 단, 쌍 부류별:

{_table_from(corr_cls, ["wave", "pair_class", "spearman_tvd_sqrt_cmi", "pearson_tvd_sqrt_cmi", "ratio_p50"], 3)}

## 예측: I(X;Y|age) − I(X;Y|route) 대 TVD_age − TVD_route

{_table_from(pred, ["wave", "spearman_route_cut", "pearson_route_cut", "n"], 3)}

## 해석

평균 KL 은 TVD 사다리 순서(독립 > 연령 > 연령×성 > route > 관람)를 따른다. 평균 CMI 는 그렇지 않다.
성별을 더하면 TVD·KL 은 거의 그대로인데 CMI 는 올라간다(ΔCMI 가 음수). 조건화가 상호정보량을 줄인다는 보장이 없기 때문이다.
route 의 CMI 는 연령×관람 분할의 참 결합이라 관람 단 CMI 와 같고, beyond 는 정보량으로 0% 다. TVD beyond 는 서울 13–22% 다.
route cut 방향은 같다(서울 TVD 26–32%, CMI 40–46%). 크기와 beyond 가 어긋나는 곳은 route 단이다.
Pinsker 비율 TVD/sqrt(KL/2) 중앙값은 약 0.53–0.65 로 상한은 느슨하다. P_R 의 0 칸은 없었다.
I(X;Y|age)−I(X;Y|route) 와 TVD 감소의 서울 Spearman 은 {_fmt(route_rho, 3)} (파동 0.52–0.75). 쌍 순서는 따라가지만, 그 정보량 차이는 관람 단이 깎는 양이다.
"""
    return text


def _check_tvd_csv(pairs: pd.DataFrame) -> str:
    path = settings.outputs_dir / "reports" / "decade_error_atlas_pairs.csv"
    if not path.exists():
        log.warning("comparison csv missing: %s", path)
        return "decade_error_atlas_pairs.csv 가 없어 TVD 대조를 건너뛰었다."
    atlas = pd.read_csv(path)
    atlas = atlas[atlas.wave.isin(WEIGHTED_WAVES)]
    atlas = atlas[ladder_mask(atlas)]
    long = atlas.melt(
        id_vars=["wave", "anchor_a", "anchor_b"],
        value_vars=[f"tvd_{k}" for k in RECON],
        var_name="rung",
        value_name="tvd_csv",
    )
    long["rung"] = long["rung"].str.removeprefix("tvd_")
    seoul = pairs[pairs.source == "seoul"][
        ["wave", "anchor_a", "anchor_b", "rung", "tvd", "tvd_atlas"]
    ]
    m = seoul.merge(long, on=["wave", "anchor_a", "anchor_b", "rung"], how="inner")
    diff_csv = (m.tvd - m.tvd_csv).abs()
    diff_mem = (m.tvd - m.tvd_atlas).abs()
    log.info(
        "TVD check n=%d max|vs csv|=%.3e max|vs pair_errors|=%.3e",
        len(m),
        float(diff_csv.max()),
        float(diff_mem.max()),
    )
    return (
        f"서울 가중 파동 TVD: csv 대조 {len(m)} 칸, 최대 절대차 {float(diff_csv.max()):.3e}; "
        f"같은 실행의 `pair_errors` 와 최대 절대차 {float(diff_mem.max()):.3e}."
    )


def _run_source(source: str, wave: int, frame: pd.DataFrame, anchors: list[str]) -> pd.DataFrame:
    log.info("%s wave=%s n=%d anchors=%d", source, wave, len(frame), len(anchors))
    out = pair_info(frame, anchors)
    out.insert(0, "wave", wave)
    out.insert(0, "source", source)
    log.info(
        "%s wave=%s pairs=%d max|kl-cmi|=%.4f zero_cells=%d",
        source,
        wave,
        out.groupby(["anchor_a", "anchor_b"]).ngroups,
        float((out.kl - out.cmi).abs().max()),
        int(out.n_zero.sum()),
    )
    return out


def main() -> int:
    out_dir = settings.outputs_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for wave in WEIGHTED_WAVES:
        frame = build_official_anchor_frame(wave)
        frames.append(_run_source("seoul", wave, frame, _core_anchors(frame)))

    raw_all = pd.read_parquet(kcas_path())
    for wave, raw in raw_all.groupby("wave"):
        raw = raw.reset_index(drop=True)
        frame = anchor_frame(raw)
        frames.append(_run_source("kcas", int(wave), frame, non_root_anchors(frame)))

    pairs = pd.concat(frames, ignore_index=True)
    note = _check_tvd_csv(pairs)
    summary = aggregate(pairs)
    keep = [
        "source",
        "wave",
        "anchor_a",
        "anchor_b",
        "rung",
        "tvd",
        "kl",
        "cmi",
        "bound",
        "ratio",
        "pair_class",
        "n_zero",
    ]
    pairs[keep].to_csv(out_dir / "era_info_ladder_pairs.csv", index=False)
    summary.to_csv(out_dir / "era_info_ladder.csv", index=False)
    (out_dir / "era_info_ladder.md").write_text(write_markdown(summary, note), encoding="utf-8")
    log.info("wrote %s", out_dir / "era_info_ladder.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
