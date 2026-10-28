"""KCAS as a second case — the same grounding-error atlas on another survey (ADR-015).

The National Survey of Cultural and Arts Activities (KCAS, MDIS) is **read
directly** from the cleaned copy kept in the preprocessing repository. MDIS terms forbid
redistributing person-level data, so it is not copied into persona; only
aggregates are computed in memory and written to `outputs/reports/kcas_atlas_*`.
It is **not pooled** with the Seoul survey (a different population) — the two
are only placed side by side.

Measured with the same rules as the Seoul atlas.
  - Anchors: education, income, 9 genres, attendance, attendance-count band, after dropping the root (sex, age band, province)
  - Reconstruction: independent / +age band / +age band×sex / +age band×attendance
  - Weighted and unweighted side by side, attendance entropy, stability ρ, and the anchor-mean floor
  - Whether the weight is entangled with attendance (the same diagnostic as for the Seoul survey)

Run: uv run python scripts/kcas_atlas.py
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.error_study import ladder_mask, pair_errors
from sfac_persona.schema import ATTEND_COUNT_EDGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("kcas_atlas")

REPORTS = settings.outputs_dir / "reports"
GENRES = [
    "literature_event",
    "exhibition",
    "western_classical",
    "traditional",
    "theater",
    "musical",
    "dance_performance",
    "movie",
    "popular_performance",
]
ROOT = ["sex", "age_band", "region"]
AGE_EDGES = [
    (0, 19, "10대"),
    (20, 29, "20대"),
    (30, 39, "30대"),
    (40, 49, "40대"),
    (50, 59, "50대"),
    (60, 69, "60대"),
    (70, 200, "70대 이상"),
]


def _load_atlas_module():
    path = Path(__file__).resolve().parent / "decade_error_atlas.py"
    spec = importlib.util.spec_from_file_location("decade_error_atlas", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def kcas_path() -> Path:
    return settings.preprocessing_root / "data" / "interim" / "kcas" / "kcas_person.parquet"


def _band(x: float, edges: list[tuple[int, int | None, str]]) -> str | None:
    if pd.isna(x):
        return None
    for lo, hi, lab in edges:
        if x >= lo and (hi is None or x <= hi):
            return lab
    return None


def anchor_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """KCAS person → an anchor frame in the same shape as the Seoul atlas (memory only)."""
    out = pd.DataFrame(index=raw.index)
    out["w"] = raw["weight"].astype(float)
    out["sex"] = raw["female"].map({1.0: "여성", 0.0: "남성"})
    out["age_band"] = raw["age"].map(lambda a: _band(a, AGE_EDGES))
    out["region"] = raw["sido"].astype(int).astype(str)
    out["education"] = raw["edu_ord"].astype(int).astype(str)
    out["household_income_band"] = raw["inc_ord"].map(lambda v: None if pd.isna(v) else str(int(v)))
    flags = []
    for g in GENRES:
        col = f"attend_{g}"
        out[col] = raw[g].map({1.0: "예", 0.0: "아니오"})
        flags.append(col)
    any_att = (raw[GENRES] == 1.0).any(axis=1)
    out["attend_any"] = np.where(any_att, "예", "아니오")
    total = raw[[f"{g}_count" for g in GENRES]].fillna(0).sum(axis=1)
    out["attend_count_band"] = total.map(lambda x: _band(float(x), ATTEND_COUNT_EDGES))
    out.loc[out["attend_any"] == "아니오", "attend_count_band"] = "0회"
    return out


def non_root_anchors(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("w", *ROOT)]


def weight_provenance(raw: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Whether the residual of log(weight) after demographics (plus education and income) is entangled with attendance."""
    X = pd.get_dummies(
        frame[["sex", "age_band", "region", "education"]].astype(str), drop_first=True
    )
    X = X.astype(float)
    X.insert(0, "const", 1.0)
    y = np.log(frame["w"].astype(float).clip(lower=1e-9))
    beta, *_ = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)
    resid = y.to_numpy() - X.to_numpy() @ beta
    r2 = 1 - resid.var() / y.to_numpy().var()
    corrs = {g: float(np.corrcoef(resid, raw[g].fillna(0).to_numpy())[0, 1]) for g in GENRES}
    top = min(corrs, key=lambda k: corrs[k])
    any_corr = float(np.corrcoef(resid, (frame["attend_any"] == "예").astype(float))[0, 1])
    return {
        "r2_log_weight_demo": float(r2),
        "residual_corr_any_attend": any_corr,
        "top_item": top,
        "top_item_corr": corrs[top],
        "max_abs_item_corr": max(abs(v) for v in corrs.values()),
        "verdict": "outcome_entangled" if abs(corrs[top]) > 0.3 else "demographic_like",
    }


def main() -> int:
    atlas = _load_atlas_module()
    raw_all = pd.read_parquet(kcas_path())
    REPORTS.mkdir(parents=True, exist_ok=True)

    summary, prov, pair_rows = [], [], []
    for wave, raw in raw_all.groupby("wave"):
        raw = raw.reset_index(drop=True)
        frame = anchor_frame(raw)
        anchors = non_root_anchors(frame)
        for mode in ("weighted", "unweighted"):
            f = frame if mode == "weighted" else frame.assign(w=1.0)
            p = pair_errors(f, anchors=anchors)
            ns = p[ladder_mask(p)]
            share = float(np.average((f["attend_any"] == "예").astype(float), weights=f["w"]))
            ent = (
                0.0
                if share in (0, 1)
                else float(-(share * np.log2(share) + (1 - share) * np.log2(1 - share)))
            )
            age = float(ns["tvd_cond_age"].mean())
            att = float(ns["tvd_cond_age_attend"].mean())
            route = float(ns["tvd_cond_age_route"].mean())
            summary.append(
                {
                    "wave": int(wave),
                    "mode": mode,
                    "n": len(f),
                    "n_pairs": len(ns),
                    "mean_tvd_indep": float(ns["tvd_indep"].mean()),
                    "mean_tvd_cond_age": age,
                    "mean_tvd_cond_age_sex": float(ns["tvd_cond_age_sex"].mean()),
                    "mean_tvd_cond_age_route": route,
                    "mean_tvd_cond_age_attend": att,
                    "attend_cut": 1 - att / age,
                    "route_cut": 1 - route / age,
                    "attend_beyond_route": 1 - att / route,
                    "attend_any_share": share,
                    "attend_any_entropy_bits": ent,
                }
            )
            if mode == "weighted":
                pair_rows.append(p.assign(wave=int(wave)))
        prov.append({"wave": int(wave), **weight_provenance(raw, frame)})
        log.info("KCAS wave %s done (pairs=%d)", wave, len(anchors) * (len(anchors) - 1) // 2)

    summ = pd.DataFrame(summary)
    prov_df = pd.DataFrame(prov)
    pairs = pd.concat(pair_rows, ignore_index=True)
    pairs = pairs[ladder_mask(pairs)]

    stab = []
    base_wave = int(pairs.wave.max())
    base = pairs[pairs.wave == base_wave][["anchor_a", "anchor_b", "tvd_cond_age"]]
    for w in sorted(pairs.wave.unique()):
        if w == base_wave:
            continue
        other = pairs[pairs.wave == w][["anchor_a", "anchor_b", "tvd_cond_age"]]
        m = base.merge(other, on=["anchor_a", "anchor_b"], suffixes=("", "_o"))
        rho = m["tvd_cond_age"].corr(m["tvd_cond_age_o"], method="spearman")
        floor = atlas._anchor_mean_floor(
            m[["anchor_a", "anchor_b", "tvd_cond_age"]],
            m[["anchor_a", "anchor_b", "tvd_cond_age_o"]].rename(
                columns={"tvd_cond_age_o": "tvd_cond_age"}
            ),
        )
        stab.append(
            {
                "wave_a": base_wave,
                "wave_b": int(w),
                "spearman_tvd_cond_age": float(rho),
                "anchor_mean_floor": floor,
                "excess_over_floor": float(rho) - floor,
                "n_pairs": len(m),
            }
        )
    stab_df = pd.DataFrame(stab)

    # Relation of the reduction rate to conditioner information — separately per survey (do not pool)
    unw = summ[summ["mode"] == "unweighted"]
    rel_kcas = float(unw["attend_cut"].corr(unw["attend_any_entropy_bits"]))

    summ.to_csv(REPORTS / "kcas_atlas_summary.csv", index=False)
    prov_df.to_csv(REPORTS / "kcas_weight_provenance.csv", index=False)
    stab_df.to_csv(REPORTS / "kcas_atlas_stability.csv", index=False)
    pairs.groupby("wave")[["tvd_indep", "tvd_cond_age", "tvd_cond_age_attend"]].describe().round(
        4
    ).to_csv(REPORTS / "kcas_atlas_pair_distribution.csv")

    def md(df: pd.DataFrame) -> str:
        return df.round(4).to_markdown(index=False)

    lines = [
        "# KCAS 두 번째 사례 — grounding-error atlas (ADR-015)",
        "",
        (
            f"국민문화예술활동조사 {raw_all.wave.nunique()}파동 · {len(raw_all):,}명. "
            "정제 사본을 메모리로만 읽고 집계만 남긴다. 서울과 합치지 않는다."
        ),
        "",
        f"앵커(뿌리 제외): {', '.join(non_root_anchors(anchor_frame(raw_all.head(50))))}",
        "",
        "## 복원 오차 요약 (비구조 쌍 평균 TVD)",
        "",
        md(summ),
        "",
        "## 가중치가 관람에 얽혀 있는가",
        "",
        md(prov_df),
        "",
        f"## 파동 간 쌍 오차 순위 안정성 (vs {base_wave})",
        "",
        md(stab_df),
        "",
        f"- 비가중 절감률과 관람 여부 엔트로피의 상관 (KCAS {len(unw)}파동): **{rel_kcas:+.2f}**",
        "",
        "CSV: `kcas_atlas_summary.csv` · `kcas_weight_provenance.csv` · `kcas_atlas_stability.csv`",
        "",
    ]
    (REPORTS / "kcas_atlas.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote kcas_atlas.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
