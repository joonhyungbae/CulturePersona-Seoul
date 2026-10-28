"""Measured grounding error — the paper's second contribution.

Question: if the joint is reconstructed from published aggregates alone (marginals
and age-band conditionals), how far is it from the observed joint?

Three reconstruction assumptions are compared with the observed joint P(i, j).
  indep      P(i) P(j)                          — full independence (marginals only)
  cond_age   Σ_a P(a) P(i|a) P(j|a)             — conditional independence given age
                                                  band (what a per-band 2D IPF gives)
  cond_age_sex Σ P(a,s) P(i|a,s) P(j|a,s)       — conditional independence given age × sex
  cond_age_route age table + survey skip structure — no new table; only the values
                                                  fixed for non-attenders are known
  cond_age_attend Σ P(a,t) P(i|a,t) P(j|a,t)     — conditional independence given age × attendance

`cond_age_route` removes the part of
the attendance-table saving that survey routing already provides,
so that what remains is 「what you gain by publishing one more table」. Read
P(non-attendance | age) from the age table (the attendance-rate table is the 0-count
cell of the count-band table); in the non-attender stratum, pin routed anchors to
that fixed value, and take the attender-stratum conditional as the age table with
that mass removed. Anchors outside the routing use the same age table in both
strata — attendance is assumed to carry no information about them. That information
is the gap versus `cond_age_attend`.

Distance: total variation TVD = ½ Σ |P̂ − P|, and maximum cell deviation (%p).
Phi correlations among the 10 genres are reported separately — attendance-battery
correlation is not captured by any conditional independence.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from sfac_persona.config import settings
from sfac_persona.schema import ANCHORS, GENRE_ANCHORS

logger = logging.getLogger(__name__)

RECON = ("indep", "cond_age", "cond_age_sex", "cond_age_route", "cond_age_attend")

NON_ATTENDER = "아니오"  # non-attendance value of attend_any

# Anchors structurally tied to attendance. Non-attenders have companion and
# satisfaction '해당 없음', and count and spend of 0.
# Pairs inside this block are deterministic dependence, a conditioning-design
# issue rather than 'restoration error' → flagged separately.
ATTEND_BLOCK: frozenset[str] = frozenset(
    {
        "attend_any",
        "attend_count_band",
        "paid_spend_band",
        "viewing_companion",
        "viewing_satisfaction",
    }
)


def _joint(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    sub = df.dropna(subset=[a, b])
    t = sub.pivot_table(
        index=a, columns=b, values="w", aggfunc="sum", fill_value=0.0, observed=True
    )
    return t / t.values.sum()


def _cond_indep(df: pd.DataFrame, a: str, b: str, by: list[str]) -> pd.DataFrame:
    """Σ_g P(g) P(a|g) P(b|g), aligned to the same axes as the observed joint."""
    sub = df.dropna(subset=[a, b])
    total = sub.w.sum()
    acc: pd.DataFrame | None = None
    for _, g in sub.groupby(by, observed=True):
        pg = g.w.sum() / total
        pa = g.groupby(a, observed=True).w.sum() / g.w.sum()
        pb = g.groupby(b, observed=True).w.sum() / g.w.sum()
        block = pd.DataFrame(np.outer(pa.values, pb.values), index=pa.index, columns=pb.index) * pg
        acc = block if acc is None else acc.add(block, fill_value=0.0)
    assert acc is not None
    # An (a, b) pair absent from every group stays NaN even after add(fill_value) → probability 0
    return acc.fillna(0.0)


def routed_values(df: pd.DataFrame, anchors: list[str]) -> dict[str, str]:
    """Values survey routing pins for non-attenders — anchors with only one value in that stratum.

    Confirms, on the frame, facts readable from the questionnaire (genre flag '아니오',
    count 0회, spend 0원, companion '해당 없음'). The same rule catches them when
    names differ across waves and surveys.
    """
    if "attend_any" not in df.columns:
        return {}
    non = df[df["attend_any"] == NON_ATTENDER]
    out: dict[str, str] = {}
    for c in [*anchors, "attend_any"]:
        if c not in df.columns:
            continue
        vals = non[c].dropna().unique()
        if len(vals) == 1 and df[c].dropna().nunique() > 1:
            out[c] = str(vals[0])
    return out


def _split_by_routing(
    g: pd.DataFrame, x: str, pi: float, routed: dict[str, str]
) -> tuple[pd.Series, pd.Series]:
    """Within one age band, (non-attender, attender) distributions of anchor x —
    from the age table and the routing rules only.
    """
    wg = g.w.sum()
    p_all = g.groupby(x, observed=True).w.sum() / wg
    if x not in routed:
        return p_all, p_all
    v0 = routed[x]
    if v0 not in p_all.index:
        p_all = pd.concat([p_all, pd.Series({v0: 0.0})])
    p_non = pd.Series(0.0, index=p_all.index)
    p_non[v0] = 1.0
    if pi >= 1.0:
        return p_non, p_all
    p_att = (p_all - pi * p_non).clip(lower=0.0)
    tot = p_att.sum()
    return p_non, (p_att / tot if tot > 0 else p_all)


def _cond_route(df: pd.DataFrame, a: str, b: str, routed: dict[str, str]) -> pd.DataFrame:
    """Age table + skip structure: Σ_g P(g) [π_g P(a|g,non) P(b|g,non) + (1−π_g) P(a|g,att) P(b|g,att)].

    π_g is the non-attender share on the age table. A routed anchor is a fixed value
    in the non-attender stratum and, in the attender stratum, the age-table
    distribution with that mass removed. An anchor outside the routing keeps the age
    table in both strata (no attendance information).
    """
    sub = df.dropna(subset=[a, b])
    total = sub.w.sum()
    acc: pd.DataFrame | None = None
    for _, g in sub.groupby("age_band", observed=True):
        wg = g.w.sum()
        pg = wg / total
        pi = float(g.w[g["attend_any"] == NON_ATTENDER].sum() / wg)
        an, aa = _split_by_routing(g, a, pi, routed)
        bn, ba = _split_by_routing(g, b, pi, routed)
        block = pi * pd.DataFrame(
            np.outer(an.values, bn.values), index=an.index, columns=bn.index
        ) + (1.0 - pi) * pd.DataFrame(
            np.outer(aa.values, ba.values), index=aa.index, columns=ba.index
        )
        block = block.fillna(0.0) * pg
        acc = block if acc is None else acc.add(block, fill_value=0.0)
    assert acc is not None
    return acc.fillna(0.0)


def _align(p: pd.DataFrame, q: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    idx = p.index.union(q.index)
    col = p.columns.union(q.columns)
    return (
        p.reindex(index=idx, columns=col, fill_value=0.0).values,
        q.reindex(index=idx, columns=col, fill_value=0.0).values,
    )


def pair_errors(df: pd.DataFrame, anchors: list[str] | None = None) -> pd.DataFrame:
    anchors = anchors or [a.name for a in ANCHORS if not a.root]
    routed = routed_values(df, anchors)
    rows = []
    for a, b in itertools.combinations(anchors, 2):
        obs = _joint(df, a, b)
        recon = {
            "indep": pd.DataFrame(
                np.outer(obs.sum(axis=1).values, obs.sum(axis=0).values),
                index=obs.index,
                columns=obs.columns,
            ),
            "cond_age": _cond_indep(df, a, b, ["age_band"]),
            "cond_age_sex": _cond_indep(df, a, b, ["age_band", "sex"]),
        }
        if routed:
            recon["cond_age_route"] = _cond_route(df, a, b, routed)
        # Conditioning on attendance: does the attendance block's structural dependence go away?
        if "attend_any" not in (a, b):
            recon["cond_age_attend"] = _cond_indep(df, a, b, ["age_band", "attend_any"])
        structural = a in ATTEND_BLOCK and b in ATTEND_BLOCK
        # Pairs that include the conditioning variable itself are trivially exact on the
        # attendance rung. The ladder average drops these pairs on every rung (the same
        # rule that drops pairs containing a root variable from the start) — `ladder_mask`.
        conditioning = "attend_any" in (a, b)
        row = {
            "anchor_a": a,
            "anchor_b": b,
            "cells": int(obs.size),
            "structural": structural,
            "conditioning": conditioning,
        }
        for k, q in recon.items():
            o, r = _align(obs, q)
            row[f"tvd_{k}"] = float(0.5 * np.abs(o - r).sum())
            row[f"maxdev_pp_{k}"] = float(np.abs(o - r).max() * 100)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["structural", "tvd_cond_age"], ascending=[True, False]).reset_index(
        drop=True
    )


def ladder_mask(pairs: pd.DataFrame) -> pd.Series:
    """Pairs that enter the ladder (mean TVD by reconstruction assumption): not a
    structural pair, and not a pair that includes the conditioning variable (attendance).

    Before 2026-09-23 only the attendance rung dropped attendance pairs as NaN and
    the other rungs kept them, so each rung averaged a different pair set. Every rung
    is now averaged over the same pairs.
    """
    cond = (
        pairs["conditioning"]
        if "conditioning" in pairs.columns
        else pairs["anchor_a"].eq("attend_any") | pairs["anchor_b"].eq("attend_any")
    )
    return ~pairs["structural"].fillna(False).astype(bool) & ~cond.fillna(False).astype(bool)


def genre_phi(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted phi correlation among binary genre anchors."""
    x = (df[GENRE_ANCHORS] == "예").astype(float)
    w = df.w.values
    mean = np.average(x.values, axis=0, weights=w)
    xc = x.values - mean
    cov = (xc * w[:, None]).T @ xc / w.sum()
    sd = np.sqrt(np.diag(cov))
    phi = cov / np.outer(sd, sd)
    names = [g.replace("attend_", "") for g in GENRE_ANCHORS]
    return pd.DataFrame(phi, index=names, columns=names)


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    """Mean TVD per anchor (mean over pairs with every other anchor)."""
    long = pd.concat(
        [
            pairs.rename(columns={"anchor_a": "anchor", "anchor_b": "other"}),
            pairs.rename(columns={"anchor_b": "anchor", "anchor_a": "other"}),
        ]
    )
    cols = [f"tvd_{k}" for k in RECON]
    return long.groupby("anchor")[cols].mean().sort_values("tvd_cond_age", ascending=False)


def write_report(pairs: pd.DataFrame, phi: pd.DataFrame, n: int) -> None:
    out = settings.outputs_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "phase01_error_study_pairs.csv", index=False)
    phi.round(3).to_csv(out / "phase01_genre_phi.csv")
    summ = summarize(pairs[ladder_mask(pairs)])
    summ.round(4).to_csv(out / "phase01_error_study_by_anchor.csv")

    def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
        cols = list(df.columns)
        lines = [
            "| "
            + " | ".join(
                ["", *cols] if df.index.name or not isinstance(df.index, pd.RangeIndex) else cols
            )
            + " |"
        ]
        lines.append("| " + " | ".join("---" for _ in lines[0].split("|")[1:-1]) + " |")
        for idx, r in df.iterrows():
            cells = [] if isinstance(df.index, pd.RangeIndex) else [str(idx)]
            cells += [fmt.format(v) if isinstance(v, float) else str(v) for v in r.values]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    nonstruct = pairs[ladder_mask(pairs)]
    top = nonstruct.head(20)[
        [
            "anchor_a",
            "anchor_b",
            "cells",
            "tvd_indep",
            "tvd_cond_age",
            "tvd_cond_age_sex",
            "tvd_cond_age_route",
            "tvd_cond_age_attend",
            "maxdev_pp_cond_age",
        ]
    ]
    overall = nonstruct[[f"tvd_{k}" for k in RECON]].agg(["mean", "median", "max"]).T
    struct_tbl = pairs[pairs.structural][["anchor_a", "anchor_b", "tvd_indep", "tvd_cond_age"]]
    lines = [
        "# 근거 오차 실측 (2024 일반시민, 가중)",
        "",
        "> 자동 생성 파일. 손으로 고치지 마라.",
        "",
        (
            f"응답자 {n:,}명, 앵커 {len(set(pairs.anchor_a) | set(pairs.anchor_b))}개, 앵커 쌍 {len(pairs)}개 "
            f"(관람 블록 내부의 구조적 쌍 {int(pairs.structural.sum())}개는 아래 요약에서 제외)."
        ),
        "",
        "## 복원 가정별 TVD 요약 (구조적 쌍 제외)",
        "",
        md(overall.round(4)),
        "",
        (
            "`indep` 는 주변분포만 있을 때, `cond_age` 는 연령대별 조건부 표까지 있을 때, "
            "`cond_age_sex` 는 연령대×성별 표까지 있을 때, `cond_age_route` 는 새 표 없이 설문 건너뛰기 구조만 "
            "더 알 때, `cond_age_attend` 는 연령대×관람여부 표까지 있을 때의 오차다. "
            "값의 차이가 '표를 더 공개하면 얼마나 좋아지는가' 다."
        ),
        "",
        "## 관람 블록의 구조적 쌍 (결정적 종속 — 샘플러에서 관람 여부를 부모로 둔다)",
        "",
        md(struct_tbl.reset_index(drop=True)),
        "",
        "## 연령대 조건부로도 못 잡는 쌍 — 상위 20",
        "",
        md(top.reset_index(drop=True)),
        "",
        "## 앵커별 평균 TVD",
        "",
        md(summ.round(4)),
        "",
        "## 장르 간 phi 상관 (관측)",
        "",
        "조건부 독립 복원은 이 행렬을 전부 0 으로 놓는다. 값이 클수록 관람 배터리를 독립으로 뽑으면 틀린다.",
        "",
        md(phi.round(2), "{:.2f}"),
        "",
    ]
    (out / "phase01_error_study.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("report: %s", out / "phase01_error_study.md")
