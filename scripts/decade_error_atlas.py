"""Decade grounding-error atlas (ADR-007 C3).

Run: uv run python scripts/decade_error_atlas.py
Outputs: outputs/reports/decade_error_atlas_*.{csv,md}
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.error_study import genre_phi, ladder_mask, pair_errors
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import (
    ERA_CORE_ANCHORS,
    ERA_WAVES,
    GENRE_ANCHORS,
    ROOT_ANCHORS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("decade_atlas")


def _core_anchors(frame: pd.DataFrame) -> list[str]:
    return [a for a in ERA_CORE_ANCHORS if a in frame.columns and a not in ROOT_ANCHORS]


# ---------------------------------------------------------------------------
# Split pairs by attendance-block membership and report reduction rates separately
# (2026-09-20, ADR-016).
#
# One reason the attendance conditional cuts error is the survey skip structure:
# non-attendees have every genre flag fixed to '아니오', so independence holds
# automatically in that stratum. Structural pairs (count · spend · companion) are already
# dropped, but **genre×genre and genre×attendance** have the same property. Reduction
# rates are therefore reported in three classes: both attendance-block anchors, one, or
# neither. `neither` is the reduction for purely statistical dependence, unrelated to the
# skip structure.
# ---------------------------------------------------------------------------
ATTEND_BLOCK = frozenset(
    {
        *GENRE_ANCHORS,
        "attend_any",
        "attend_count_band",
        "paid_spend_band",
        "viewing_companion",
    }
)


def _pair_class(anchor_a: str, anchor_b: str) -> str:
    n = int(anchor_a in ATTEND_BLOCK) + int(anchor_b in ATTEND_BLOCK)
    return ("neither", "one", "both")[n]


def attend_cut_by_class(pairs_all: pd.DataFrame) -> pd.DataFrame:
    """Age- and attendance-conditional reduction rates for non-structural pairs,
    split by attendance-block membership.
    """
    d = pairs_all[ladder_mask(pairs_all)].copy()
    d["pair_class"] = [_pair_class(a, b) for a, b in zip(d.anchor_a, d.anchor_b, strict=True)]
    rows = []
    for (wave, klass), g in d.groupby(["wave", "pair_class"]):
        age = float(g.tvd_cond_age.mean())
        route = float(g.tvd_cond_age_route.mean())
        att = float(g.tvd_cond_age_attend.mean())
        rows.append(
            {
                "wave": wave,
                "pair_class": klass,
                "n_pairs": len(g),
                "tvd_indep": float(g.tvd_indep.mean()),
                "tvd_cond_age": age,
                "tvd_cond_age_sex": float(g.tvd_cond_age_sex.mean()),
                "tvd_cond_age_route": route,
                "tvd_cond_age_attend": att,
                "attend_cut_pct": 100.0 * (1.0 - att / age) if age else float("nan"),
                "sex_cut_pct": 100.0 * (1.0 - float(g.tvd_cond_age_sex.mean()) / age)
                if age
                else float("nan"),
                # Share from routing (no new table), and the further share the attendance
                # table adds on top of it.
                "route_cut_pct": 100.0 * (1.0 - route / age) if age else float("nan"),
                "attend_beyond_route_pct": 100.0 * (1.0 - att / route) if route else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["wave", "pair_class"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weighting sensitivity (2026-09-18).
#
# What the weight-provenance check in preprocessing established: this survey's final weights
# are raked not only to demographics but to the **cultural-activity (attendance)
# distribution** — the very outcome we measure. Residual correlations after demographic
# controls are −0.94 for 2020 exhibitions, −0.88 for 2022 theatre, and −0.81 for 2024
# theatre, and the target genre changes by wave. Therefore report
# **weighted and unweighted side by side**. For 2020 also report `WTB`, which looks like a
# design weight (it exists only in the raw pkl).
# ---------------------------------------------------------------------------


def _mode_summary(frame: pd.DataFrame, wave: int, mode: str) -> dict:
    anchors = _core_anchors(frame)
    pairs = pair_errors(frame, anchors=anchors)
    ns = pairs[ladder_mask(pairs)]
    w = frame["w"].astype(float)
    share = float(np.average((frame["attend_any"] == "예").astype(float), weights=w))
    ent = (
        0.0
        if share in (0.0, 1.0)
        else float(-(share * np.log2(share) + (1 - share) * np.log2(1 - share)))
    )
    age = float(ns["tvd_cond_age"].mean())
    att = float(ns["tvd_cond_age_attend"].mean())
    route = float(ns["tvd_cond_age_route"].mean())
    # The sex conditional is computed in the same mode: weights are raked to sex, age, and
    # living area as well as attendance, so "sex changes almost nothing" must also be
    # checked unweighted.
    sex = float(ns["tvd_cond_age_sex"].dropna().mean())
    indep = float(ns["tvd_indep"].mean())
    return {
        "wave": wave,
        "mode": mode,
        "n": len(frame),
        "mean_tvd_indep": indep,
        "mean_tvd_cond_age": age,
        "mean_tvd_cond_age_sex": sex,
        "mean_tvd_cond_age_route": route,
        "mean_tvd_cond_age_attend": att,
        "age_cut": 1.0 - age / indep,
        "sex_cut": 1.0 - sex / age,
        "attend_cut": 1.0 - att / age,
        "route_cut": 1.0 - route / age,
        "attend_beyond_route": 1.0 - att / route,
        "attend_any_share": share,
        "attend_any_entropy_bits": ent,
    }


def _design_weight_2020(frame: pd.DataFrame) -> pd.Series | None:
    """2020 `WTB`. Row alignment is checked by whether the file's `WT` matches frame `w`."""
    pkl = settings.data_root / "interim" / "microdata" / "2020_general.pkl"
    if not pkl.exists():
        return None
    raw = pd.read_pickle(pkl)
    if "WTB" not in raw.columns or "WT" not in raw.columns or len(raw) != len(frame):
        return None
    if not np.allclose(
        raw["WT"].to_numpy(dtype=float), frame["w"].to_numpy(dtype=float), atol=1e-9
    ):
        log.warning("2020: pkl WT differs from frame w row by row — skipping WTB sensitivity")
        return None
    return pd.Series(raw["WTB"].to_numpy(dtype=float), index=frame.index)


def weighting_sensitivity() -> pd.DataFrame:
    rows = []
    for wave in ERA_WAVES:
        frame = build_official_anchor_frame(wave)
        if wave in (2016, 2018):
            rows.append(_mode_summary(frame, wave, "unweighted"))
            continue
        rows.append(_mode_summary(frame, wave, "weighted"))
        rows.append(_mode_summary(frame.assign(w=1.0), wave, "unweighted"))
        if wave == 2020:
            wtb = _design_weight_2020(frame)
            if wtb is not None:
                rows.append(_mode_summary(frame.assign(w=wtb), wave, "design_weight_WTB"))
        log.info("wave %s weighting sensitivity done", wave)
    return pd.DataFrame(rows)


def _anchor_mean_floor(base: pd.DataFrame, other: pd.DataFrame) -> float:
    """Rank correlation with no pair-specific information — the **floor** of stability ρ.

    Lesson from an earlier recoverability check: a correlation metric comes out high even for a
    predictor with no information. Pair TVD is mostly fixed by which anchors are involved,
    so learning only per-anchor means on one wave and predicting pair (a, b) as
    mean(a)+mean(b) still yields a high ρ against another wave. Read reported stability
    against this value, not against 0.
    """
    long = pd.concat(
        [
            base[["anchor_a", "tvd_cond_age"]].rename(columns={"anchor_a": "anchor"}),
            base[["anchor_b", "tvd_cond_age"]].rename(columns={"anchor_b": "anchor"}),
        ]
    )
    m = long.groupby("anchor")["tvd_cond_age"].mean()
    pred = other["anchor_a"].map(m) + other["anchor_b"].map(m)
    return float(
        pd.Series(pred.to_numpy()).corr(
            pd.Series(other["tvd_cond_age"].to_numpy()), method="spearman"
        )
    )


def main() -> int:
    out_dir = settings.outputs_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = []
    phi_rows = []
    summaries = []

    for wave in ERA_WAVES:
        frame = build_official_anchor_frame(wave)
        anchors = _core_anchors(frame)
        pairs = pair_errors(frame, anchors=anchors)
        pairs.insert(0, "wave", wave)
        pair_rows.append(pairs)
        phi = genre_phi(frame)
        phi_long = phi.stack().rename("phi").reset_index()
        phi_long.columns = ["genre_a", "genre_b", "phi"]
        phi_long = phi_long[phi_long["genre_a"] < phi_long["genre_b"]]
        phi_long.insert(0, "wave", wave)
        phi_rows.append(phi_long)
        ns = pairs[ladder_mask(pairs)]
        summaries.append(
            {
                "wave": wave,
                "n": len(frame),
                "n_pairs": len(ns),
                "mean_tvd_indep": float(ns["tvd_indep"].mean()),
                "mean_tvd_cond_age": float(ns["tvd_cond_age"].mean()),
                "mean_tvd_cond_age_sex": float(ns["tvd_cond_age_sex"].mean()),
                "mean_tvd_cond_age_route": float(ns["tvd_cond_age_route"].mean()),
                "mean_tvd_cond_age_attend": float(ns["tvd_cond_age_attend"].mean()),
                "max_phi": float(phi_long["phi"].max()) if len(phi_long) else np.nan,
            }
        )
        log.info("wave %s atlas pairs=%d", wave, len(pairs))

    pairs_all = pd.concat(pair_rows, ignore_index=True)
    phi_all = pd.concat(phi_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)

    # pair stability: rank correlation of tvd_cond_age across weighted waves
    weighted = [2020, 2022, 2024]
    stab_rows = []
    base = pairs_all[(pairs_all.wave == 2024) & ladder_mask(pairs_all)][
        ["anchor_a", "anchor_b", "tvd_cond_age"]
    ]
    for w in weighted:
        if w == 2024:
            continue
        other = pairs_all[pairs_all.wave == w][["anchor_a", "anchor_b", "tvd_cond_age"]].rename(
            columns={"tvd_cond_age": f"tvd_{w}"}
        )
        m = base.merge(other, on=["anchor_a", "anchor_b"])
        rho = m["tvd_cond_age"].corr(m[f"tvd_{w}"], method="spearman")
        floor = _anchor_mean_floor(
            m[["anchor_a", "anchor_b", "tvd_cond_age"]],
            m[["anchor_a", "anchor_b", f"tvd_{w}"]].rename(columns={f"tvd_{w}": "tvd_cond_age"}),
        )
        stab_rows.append(
            {
                "wave_a": 2024,
                "wave_b": w,
                "spearman_tvd_cond_age": rho,
                "anchor_mean_floor": floor,
                "excess_over_floor": rho - floor,
                "n_pairs": len(m),
            }
        )

    stability = pd.DataFrame(stab_rows)
    pairs_all.to_csv(out_dir / "decade_error_atlas_pairs.csv", index=False)
    phi_all.to_csv(out_dir / "decade_error_atlas_genre_phi.csv", index=False)
    summary.to_csv(out_dir / "decade_error_atlas_summary.csv", index=False)
    stability.to_csv(out_dir / "decade_error_atlas_stability.csv", index=False)
    weighting = weighting_sensitivity()
    weighting.to_csv(out_dir / "decade_error_atlas_weighting.csv", index=False)
    by_class = attend_cut_by_class(pairs_all)
    by_class.to_csv(out_dir / "decade_error_atlas_pair_class.csv", index=False)

    def _md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for r in df.itertuples(index=False):
            lines.append("| " + " | ".join(str(x) for x in r) + " |")
        return "\n".join(lines)

    lines = [
        "# Decade grounding-error atlas",
        "",
        "ADR-007 C3. 시대별 주변/조건부 복원 TVD · 장르 φ · 쌍 안정성.",
        "",
        "## 요약",
        _md_table(summary.round(4)),
        "",
        "## TVD 순위 안정성 (vs 2024, 사다리 쌍)",
        _md_table(stability.round(4)) if len(stability) else "_n/a_",
        "",
        "`anchor_mean_floor` 는 쌍 고유 정보 없이(앵커별 평균만으로) 나오는 ρ 다. 안정성은 0 이 아니라",
        "이 바닥에 견주어 읽는다 — `excess_over_floor` 가 쌍 단위 안정성의 실제 증거다.",
        "",
        "## 가중치 민감도 · 관람 여부 조건부 절감률 (`attend_cut`)",
        _md_table(weighting.round(4)),
        "",
        "`route_cut` 은 새 표 없이 설문 건너뛰기 구조(비관람자에게 고정되는 값)만 알 때의 절감,",
        "`attend_beyond_route` 는 관람 여부 표가 그 위에 더 주는 몫이다.",
        "사다리 평균은 구조적 쌍과 **관람 여부 자신이 낀 쌍**을 모든 단에서 뺀 같은 쌍 집합 위에서 낸다.",
        "",
        "가중치가 관람 분포에 레이킹돼 있으므로 가중·비가중을",
        "나란히 낸다. `attend_cut` 은 **비관람자 비율**을 따라간다 — 비관람자는 관람 여부를 아는 순간",
        "장르 플래그가 전부 정해지므로 그 층에서는 독립 가정이 정확해진다. 2016·2018(관람자 94–95%)에서",
        "작은 것은 표본 크기 탓이 아니다 (2020–24 를 n=2,000 으로 줄여도 35–36%). 비가중 절감률과",
        "비관람자 비율의 상관은 서울 5파동(2016·2018 은 비가중 진단 파동) 과 KCAS 9파동에서 따로 본다 —",
        "수치는 논문 매크로(`nonattCorrSeoul*`, `nonattCorrKcas`)가 이 CSV 에서 다시 계산한다. 엔트로피보다 맞다",
        "(KCAS 에서 엔트로피는 +0.75 — 2021 처럼 비관람자가 다수인 파동을 설명하지 못한다).",
        "",
        "## 관람 블록 소속별 절감률 (ADR-016)",
        _md_table(by_class.round(4)),
        "",
        "`neither` 는 양쪽 다 관람 블록 밖인 쌍이다 — 건너뛰기 구조가 개입하지 않는 순수 의존.",
        "여기서 절감이 작다면, 관람 여부 조건부 표의 값은 관람 블록과 그 경계에 몰려 있는 것이다.",
        "",
        "수준(관람률 %p) 횡비교 금지. 구조 지표만.",
    ]
    (out_dir / "decade_error_atlas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote decade_error_atlas.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
