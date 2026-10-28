"""Build grounding tables.

R0   resident-registration population (living area × sex × age band) — sampler root.
     Survey weights are not used.
T1   overall distribution of each anchor (survey weighted)
T2   anchor | age band
T3   anchor | sex
T4   anchor | living area
T5   anchor | age band × sex
T6   anchor-pair joints (for the error study; the public copy applies the cell-size rule)
T_pub  2024 published cells extracted from the report PDF (comparison copy)

Every table is long-form and carries n_unweighted and suppressed (cell < cell_floor).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sfac_persona.config import settings
from sfac_persona.grounding.microdata import build_anchor_frame
from sfac_persona.schema import (
    AGE_BANDS,
    ANCHOR_BY_NAME,
    ANCHORS,
    LIVING_AREAS,
    ROOT_ANCHORS,
    SEXES,
)

logger = logging.getLogger(__name__)

_PRIOR_AGE_MAP = {
    "15-19": "10대",
    "20-29": "20대",
    "30-39": "30대",
    "40-49": "40대",
    "50-59": "50대",
    "60-69": "60대",
    "70+": "70대 이상",
}
_PRIOR_SEX_MAP = {"male": "남성", "female": "여성"}
SEOUL_TOTAL_CODE = 1100000000


# ----------------------------------------------------------------------------
# R0 — population root
# ----------------------------------------------------------------------------


def build_root_population(*, month: str | None = None, wave: int | None = None) -> pd.DataFrame:
    """Resident-registration population root R0.

    Each wave is rooted in its own December table (ADR-013 supersedes ADR-011): the
    source CSV holds December 2016--2025, so `wave=2020` resolves to 2020-12. If no
    month of that year exists, the latest month is used, a warning is logged here and
    provenance records `wave_matched=False`.
    """
    prior = pd.read_csv(settings.population_prior_csv)
    available = sorted(prior.reference_month.unique())
    if month is None and wave is not None:
        want = [m for m in available if m.startswith(str(wave))]
        if want:
            month = want[-1]
        else:
            logger.warning(
                "wave %s: resident-registration population has no month in %s "
                "(have %s~%s). R0 uses the latest month — the era bank's population "
                "mix differs from the wave (ADR-011).",
                wave,
                wave,
                available[0],
                available[-1],
            )
    month = month or available[-1]
    if month not in available:
        raise ValueError(
            f"reference_month {month} not in population prior ({available[0]}~{available[-1]})"
        )
    p = prior[(prior.reference_month == month) & (prior.district_code != SEOUL_TOTAL_CODE)].copy()
    xw = pd.read_csv(settings.living_area_crosswalk_csv)[["district_code", "living_area_ko"]]
    p = p.merge(xw, on="district_code", how="left")
    if p.living_area_ko.isna().any():
        bad = p[p.living_area_ko.isna()].district_name.unique().tolist()
        raise ValueError(f"districts without living area: {bad}")
    p["age_band"] = p.age_band.map(_PRIOR_AGE_MAP)
    p["sex"] = p.sex.map(_PRIOR_SEX_MAP)
    r0 = (
        p.groupby(["living_area_ko", "sex", "age_band"], as_index=False)["population_count"]
        .sum()
        .rename(columns={"living_area_ko": "living_area", "population_count": "population"})
    )
    r0["share"] = r0.population / r0.population.sum()
    r0["reference_month"] = month
    r0["wave_matched"] = bool(wave is not None and month.startswith(str(wave)))
    r0["living_area"] = pd.Categorical(r0.living_area, LIVING_AREAS, ordered=True)
    r0["age_band"] = pd.Categorical(r0.age_band, AGE_BANDS, ordered=True)
    r0["sex"] = pd.Categorical(r0.sex, SEXES, ordered=True)
    return r0.sort_values(["living_area", "sex", "age_band"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Survey-based tables
# ----------------------------------------------------------------------------


def _shares(df: pd.DataFrame, anchor: str, by: list[str]) -> pd.DataFrame:
    """Weighted share + unweighted n. If by is empty, the overall distribution."""
    sub = df.dropna(subset=[anchor])
    keys = by + [anchor]
    g = (
        sub.groupby(keys, observed=True)
        .agg(w=("w", "sum"), n_unweighted=("w", "size"))
        .reset_index()
    )
    denom = g.groupby(by, observed=True)["w"].transform("sum") if by else g["w"].sum()
    g["share"] = g["w"] / denom
    base_n = (
        g.groupby(by, observed=True)["n_unweighted"].transform("sum")
        if by
        else g["n_unweighted"].sum()
    )
    g["n_base"] = base_n
    g["suppressed"] = g["n_unweighted"] < settings.cell_floor
    g = g.rename(columns={anchor: "value"})
    g.insert(0, "anchor", anchor)
    return g.drop(columns=["w"])


def build_survey_tables(frame: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    frame = build_anchor_frame() if frame is None else frame
    non_root = [a.name for a in ANCHORS if not a.root]
    tables: dict[str, list[pd.DataFrame]] = {"T1": [], "T2": [], "T3": [], "T4": [], "T5": []}
    for a in [*ROOT_ANCHORS, *non_root]:
        tables["T1"].append(_shares(frame, a, []))
        if a in ROOT_ANCHORS:
            continue
        tables["T2"].append(_shares(frame, a, ["age_band"]))
        tables["T3"].append(_shares(frame, a, ["sex"]))
        tables["T4"].append(_shares(frame, a, ["living_area"]))
        tables["T5"].append(_shares(frame, a, ["age_band", "sex"]))
    out = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    # Also keep the survey joint of the root variables (to contrast with R0)
    out["T1_root_joint"] = _shares(
        frame.assign(root=frame[ROOT_ANCHORS].agg("|".join, axis=1)), "root", []
    )
    return out


def build_pair_joints(frame: pd.DataFrame, anchors: list[str] | None = None) -> pd.DataFrame:
    """Anchor-pair joints (weighted). The observed values for the error study."""
    anchors = anchors or [a.name for a in ANCHORS if not a.root]
    rows = []
    for a, b in itertools.combinations(anchors, 2):
        g = _shares(frame, b, [a]).rename(
            columns={"anchor": "anchor_b", "value": "value_b", a: "value_a"}
        )
        g.insert(0, "anchor_a", a)
        # share is P(b | a), so multiply by P(a) to get the joint
        pa = _shares(frame, a, []).set_index("value")["share"]
        g["joint"] = g["share"] * g["value_a"].map(pa)
        rows.append(
            g[["anchor_a", "value_a", "anchor_b", "value_b", "joint", "n_unweighted", "suppressed"]]
        )
    return pd.concat(rows, ignore_index=True)


def load_published_2024() -> pd.DataFrame:
    h = pd.read_csv(settings.harmonized_long)
    h = h[(h.year == settings.survey_year) & (h.is_imputed != True)]
    return h[
        [
            "canonical_var",
            "segment_dimension",
            "segment",
            "value",
            "n_base",
            "provenance",
            "extraction_method",
        ]
    ].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_all(frame: pd.DataFrame | None = None) -> dict[str, Path]:
    settings.grounding_dir.mkdir(parents=True, exist_ok=True)
    frame = build_anchor_frame() if frame is None else frame
    paths: dict[str, Path] = {}

    r0 = build_root_population()
    paths["R0"] = settings.grounding_dir / "R0_population_root.parquet"
    r0.to_parquet(paths["R0"], index=False)

    for name, df in build_survey_tables(frame).items():
        p = settings.grounding_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p

    pairs = build_pair_joints(frame)
    paths["T6"] = settings.grounding_dir / "T6_pair_joints.parquet"
    pairs.to_parquet(paths["T6"], index=False)
    # Public copy: drop rows below the cell-size floor
    pub = pairs[~pairs.suppressed]
    paths["T6_release"] = settings.grounding_dir / "T6_pair_joints_release.parquet"
    pub.to_parquet(paths["T6_release"], index=False)

    tp = load_published_2024()
    paths["T_pub"] = settings.grounding_dir / "T_pub_2024_report_cells.parquet"
    tp.to_parquet(paths["T_pub"], index=False)

    src_files = sorted(settings.microdata_dir.glob("*일반시민_정보공개청구.xlsx"))
    prov = {
        "built_at": datetime.now(UTC).isoformat(),
        "survey": f"{settings.survey_year} 서울시민 문화향유 실태조사 일반시민 (서울문화재단)",
        "microdata_source": [{"file": f.name, "sha256": _sha256(f)} for f in src_files],
        "n_respondents": len(frame),
        "weight_var": settings.weight_var,
        "weight_sum": float(frame.w.sum()),
        "cell_floor": settings.cell_floor,
        "geo_max_level": settings.geo_max_level,
        "population_root": {
            "source": str(settings.population_prior_csv.name),
            "reference_month": str(r0.reference_month.iloc[0]),
            "population_total": int(r0.population.sum()),
        },
        "anchors": [
            {
                "name": a.name,
                "kind": a.kind,
                "source": list(a.source),
                "label_ko": a.label_ko,
                "root": a.root,
            }
            for a in ANCHORS
        ],
        "policy": (
            "개별 응답 기록은 리샘플하지 않는다. 이 표들은 재단이 승인한 파생물 범위 안의 집계이며, "
            "공개본은 비가중 셀 < cell_floor 를 제외한다. 지역은 생활권까지."
        ),
    }
    paths["provenance"] = settings.grounding_dir / "provenance.json"
    paths["provenance"].write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("grounding tables written: %s", {k: v.name for k, v in paths.items()})
    return paths


def anchor_values_in_order(anchor: str, values: list[str]) -> list[str]:
    spec = ANCHOR_BY_NAME[anchor]
    if spec.order:
        return [v for v in spec.order if v in values]
    return sorted(values)
