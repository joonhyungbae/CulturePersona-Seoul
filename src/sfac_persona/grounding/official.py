"""official cleaned file → analysis universe and Tier 0–2 anchor frame.

Person-level data stays in memory. Only the gate and grounding aggregates go to disk.
Contract: ADR-007.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sfac_persona.config import settings
from sfac_persona.schema import (
    ATTEND_COUNT_BAND_MIDPOINTS,
    ATTEND_COUNT_EDGES,
    ERA_WAVES,
    GENRE_LIST,
    HOUSEHOLD_INCOME_EDGES_MANWON,
    HOUSEHOLD_SIZE_EDGES,
    LIKERT7_TO_3,
    SEOUL_YEARS_EDGES,
    SPEND_EDGES_MANWON,
    UNWEIGHTED_WAVES,
    WAVE_REGIME,
)

logger = logging.getLogger(__name__)

GENRES = GENRE_LIST
NA_LABEL = "해당 없음"
SEX_KO = {"male": "남성", "female": "여성"}
AGE_BAND_KO = {
    "15-19": "10대",
    "20-29": "20대",
    "30-39": "30대",
    "40-49": "40대",
    "50-59": "50대",
    "60-69": "60대",
    "70+": "70대 이상",
}
_LIVING_RE = re.compile(r"^([^\s(]+)")

_CANON_LABEL_VARS = {
    "education": "DQ2",
    "occupation": "DQ3",
    "leisure_companion": "A4",
    "attendance_companion": "B2",
    "leisure_time_sufficiency": "A2",
    "household_income": "DQ8",
}


def _official_path(name: str) -> Path:
    p = settings.official_person_dir / name
    if not p.exists():
        raise FileNotFoundError(
            f"{name}: individual response records not found. This step needs the survey "
            "microdata, which is not distributed (see REPRODUCIBILITY.md)."
        )
    return p


def _band(x: float, edges: list[tuple[int, int | None, str]]) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    for lo, hi, label in edges:
        if x >= lo and (hi is None or x <= hi):
            return label
    return None


def _clean_label(x: str | None) -> str | None:
    if not isinstance(x, str):
        return None
    return x.lstrip("①②③④⑤⑥⑦ ").strip()


def _likert3_from_num(s: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    rename = {"낮음": labels[0], "보통": labels[1], "높음": labels[2]}

    def one(v: float) -> str | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        iv = int(v)
        if iv in LIKERT7_TO_3:
            return rename[LIKERT7_TO_3[iv]]
        if iv in (1, 2):
            return labels[0]
        if iv == 3:
            return labels[1]
        if iv in (4, 5):
            return labels[2]
        return None

    return pd.to_numeric(s, errors="coerce").map(one)


def load_official_value_labels(wave: int) -> dict[str, dict[str, str]]:
    path = settings.official_value_labels_csv
    if not path.exists():
        logger.warning("official_value_labels missing: %s", path)
        return {}
    v = pd.read_csv(path)
    v = v[v["wave"] == wave]
    # 2022 and others arrive only as group=mixed
    prefer = v[v["group"] == "general"]
    if prefer.empty:
        prefer = v[v["group"] == "mixed"]
    if prefer.empty:
        prefer = v
    out: dict[str, dict[str, str]] = {}
    for r in prefer.itertuples():
        key = str(r.value)
        if key.endswith(".0"):
            key = str(int(float(key)))
        out.setdefault(str(r.variable), {})[key] = str(r.label)
    return out


def _district_code_to_living_area() -> dict[int, str]:
    """2024 SQ1 (districts 1–25) → living area. The same coding on gu_code in every wave."""
    labels = load_official_value_labels(2024)
    sq1 = labels.get("SQ1", {})
    xw = pd.read_csv(settings.living_area_crosswalk_csv)
    name2la = dict(zip(xw["district_name"].astype(str), xw["living_area_ko"].astype(str)))
    out: dict[int, str] = {}
    for k, name in sq1.items():
        try:
            code = int(float(k))
        except ValueError:
            continue
        la = name2la.get(name)
        if la:
            out[code] = la
    return out


def load_general_keys(wave: int) -> pd.DataFrame:
    keys = pd.read_parquet(_official_path("official_person_keys.parquet"))
    out = keys[(keys["frame_group"] == "general") & (keys["wave"] == wave)].copy()
    if "dong_code" in out.columns:
        out = out.drop(columns=["dong_code"])
    logger.info("official keys wave=%s general n=%d", wave, len(out))
    return out.reset_index(drop=True)


def load_general_master(
    wave: int, *, battery_asked: bool = True, allow_unweighted: bool = False
) -> pd.DataFrame:
    master = pd.read_parquet(_official_path("official_person_master.parquet"))
    out = master[(master["frame_group"] == "general") & (master["wave"] == wave)].copy()
    if battery_asked and "battery_asked" in out.columns:
        out = out[out["battery_asked"]].copy()
    if "dong_code" in out.columns:
        out = out.drop(columns=["dong_code"])
    if out["weight"].isna().all():
        if not allow_unweighted and wave not in UNWEIGHTED_WAVES:
            raise ValueError(f"wave {wave}: weight missing")
        out["weight"] = 1.0
        logger.warning("wave %s: using unit weights (unweighted diagnostic)", wave)
    elif out["weight"].isna().any():
        raise ValueError(f"wave {wave}: partial weight missing")
    logger.info("official master wave=%s general n=%d", wave, len(out))
    return out.reset_index(drop=True)


def load_item_for_persons(row_uids: pd.Series, wave: int) -> pd.DataFrame:
    item = pd.read_parquet(_official_path("official_person_item.parquet"))
    return item[(item["wave"] == wave) & (item["row_uid"].isin(set(row_uids)))].reset_index(
        drop=True
    )


def _wavelocal_var(wave: int, row_uids: pd.Series, variable: str) -> pd.Series:
    wl = pd.read_parquet(_official_path("official_person_wavelocal.parquet"))
    sub = wl[
        (wl["wave"] == wave) & (wl["variable"] == variable) & (wl["row_uid"].isin(set(row_uids)))
    ]
    return sub.set_index("row_uid")["value"]


def _canonical_wide(wave: int, row_uids: pd.Series, vars_: list[str]) -> pd.DataFrame:
    canon = pd.read_parquet(_official_path("official_person_canonical.parquet"))
    sub = canon[
        (canon["wave"] == wave)
        & (canon["row_uid"].isin(set(row_uids)))
        & (canon["canonical_var"].isin(vars_))
    ]
    if sub.empty:
        return pd.DataFrame(index=pd.Index(row_uids, name="row_uid"))
    wide = sub.pivot_table(
        index="row_uid", columns="canonical_var", values="value", aggfunc="first"
    )
    return wide.reindex(row_uids)


def _map_label(series: pd.Series, labels: dict[str, str]) -> pd.Series:
    def one(v: object) -> str | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        candidates = [str(v)]
        try:
            candidates.append(str(int(float(v))))
        except (TypeError, ValueError):
            pass
        for k in candidates:
            if k in labels:
                return _clean_label(labels[k])
        return None

    return series.map(one)


def build_official_anchor_frame(wave: int) -> pd.DataFrame:
    """Tier 0–2 anchors + w. Person rows are returned only (not written to disk)."""
    allow_uw = wave in UNWEIGHTED_WAVES
    master = load_general_master(wave, allow_unweighted=allow_uw)
    labels = load_official_value_labels(wave)
    uids = master["row_uid"]

    out = pd.DataFrame(index=range(len(master)))
    out["w"] = master["weight"].to_numpy(dtype=float)
    out["sex"] = master["sex"].map(SEX_KO).to_numpy()
    out["age_band"] = master["age_band"].map(AGE_BAND_KO).to_numpy()

    gu2la = _district_code_to_living_area()
    out["living_area"] = (
        master["gu_code"]
        .map(lambda v: gu2la.get(int(float(v))) if pd.notna(v) else None)
        .to_numpy()
    )

    # If SQ1R is present, overwrite with the short living-area name (2024)
    lab_sq = labels.get("SQ1R", {}) or labels.get("SQ1_R1", {})
    sq1r = _wavelocal_var(wave, uids, "SQ1R")
    if sq1r.notna().sum() == 0:
        sq1r = _wavelocal_var(wave, uids, "SQ1_R1")

    def living_from_sq1r(v: object) -> str | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            iv = str(int(float(v)))
        except (TypeError, ValueError):
            return None
        raw_lab = lab_sq.get(iv) or lab_sq.get(str(float(v)))
        if not raw_lab:
            return None
        m = _LIVING_RE.match(_clean_label(raw_lab) or "")
        return m.group(1) if m else None

    if lab_sq and sq1r.notna().any():
        alt = sq1r.reindex(uids).map(living_from_sq1r).to_numpy()
        mask = pd.notna(alt)
        out.loc[mask, "living_area"] = alt[mask]

    canon_vars = [
        "education",
        "occupation",
        "household_income",
        "years_in_seoul",
        "household_size",
        "leisure_time_sufficiency",
        "leisure_companion",
        "attendance_companion",
    ]
    wide = _canonical_wide(wave, uids, canon_vars)

    edu_lab = labels.get(_CANON_LABEL_VARS["education"], {}) or labels.get("DQ2", {})
    occ_lab = labels.get(_CANON_LABEL_VARS["occupation"], {}) or labels.get("DQ3", {})
    out["education"] = (
        _map_label(wide["education"], edu_lab).to_numpy() if "education" in wide.columns else None
    )
    out["occupation"] = (
        _map_label(wide["occupation"], occ_lab).to_numpy() if "occupation" in wide.columns else None
    )

    if "household_income" in wide.columns:
        code = pd.to_numeric(wide["household_income"], errors="coerce")
        lower = (code - 1) * 50
        out["household_income_band"] = lower.map(
            lambda x: _band(x, HOUSEHOLD_INCOME_EDGES_MANWON)
        ).to_numpy()
    else:
        out["household_income_band"] = None

    if "years_in_seoul" in wide.columns:
        out["seoul_years_band"] = (
            pd.to_numeric(wide["years_in_seoul"], errors="coerce")
            .map(lambda x: _band(x, SEOUL_YEARS_EDGES))
            .to_numpy()
        )
    else:
        out["seoul_years_band"] = None

    if "household_size" in wide.columns:
        out["household_size_band"] = (
            pd.to_numeric(wide["household_size"], errors="coerce")
            .map(lambda x: _band(x, HOUSEHOLD_SIZE_EDGES))
            .to_numpy()
        )
    else:
        out["household_size_band"] = None

    if "leisure_time_sufficiency" in wide.columns:
        out["leisure_sufficiency"] = _likert3_from_num(
            wide["leisure_time_sufficiency"], ("부족", "보통", "충분")
        ).to_numpy()
    else:
        out["leisure_sufficiency"] = None

    lc_lab = (
        labels.get(_CANON_LABEL_VARS["leisure_companion"], {})
        or labels.get("A4", {})
        or labels.get("A4RR", {})
    )
    out["leisure_companion"] = (
        _map_label(wide["leisure_companion"], lc_lab).to_numpy()
        if "leisure_companion" in wide.columns
        else None
    )

    vc_lab = (
        labels.get(_CANON_LABEL_VARS["attendance_companion"], {})
        or labels.get("B2", {})
        or labels.get("B3", {})
        or labels.get("B3RR", {})
    )
    out["viewing_companion"] = (
        _map_label(wide["attendance_companion"], vc_lab).to_numpy()
        if "attendance_companion" in wide.columns
        else None
    )

    item = load_item_for_persons(uids, wave)
    piv = item.pivot_table(
        index="row_uid", columns="canonical_item", values="attended", aggfunc="first"
    )
    for g in GENRES:
        col = f"attend_{g}"
        if g in piv.columns:
            out[col] = piv[g].reindex(uids).map(lambda x: "예" if bool(x) else "아니오").to_numpy()
        else:
            out[col] = "아니오"

    out["attend_any"] = (
        master["attended_any"].map(lambda x: "예" if bool(x) else "아니오").to_numpy()
    )

    counts = item.pivot_table(
        index="row_uid", columns="canonical_item", values="total_count", aggfunc="sum", fill_value=0
    )
    total_count = counts.reindex(uids).fillna(0).sum(axis=1)
    out["attend_count_band"] = total_count.map(
        lambda x: _band(float(x), ATTEND_COUNT_EDGES)
    ).to_numpy()

    costs = item.pivot_table(
        index="row_uid",
        columns="canonical_item",
        values="annual_cost_krw",
        aggfunc="sum",
        fill_value=0,
    )
    total_manwon = costs.reindex(uids).fillna(0).sum(axis=1) / 10000.0
    out["paid_spend_band"] = total_manwon.map(
        lambda x: _band(float(x), SPEND_EDGES_MANWON)
    ).to_numpy()

    out.loc[out["attend_any"] == "아니오", "viewing_companion"] = NA_LABEL
    out.loc[out["attend_any"] == "아니오", "attend_count_band"] = "0회"
    out.loc[out["attend_any"] == "아니오", "paid_spend_band"] = "0원"
    out["paid_spend_band"] = out["paid_spend_band"].fillna("0원")
    out["viewing_companion"] = out["viewing_companion"].fillna(NA_LABEL)

    genre_cols = [f"attend_{g}" for g in GENRES]
    out["width"] = (out[genre_cols] == "예").sum(axis=1).astype(int)
    out["depth"] = pd.to_numeric(master["total_visits"], errors="coerce").fillna(0).to_numpy()

    out["survey_year"] = wave
    out["survey_era"] = WAVE_REGIME.get(wave, "unknown")
    out["weighting"] = "none" if wave in UNWEIGHTED_WAVES else "survey"
    out["instrument_regime"] = WAVE_REGIME.get(wave, "unknown")

    missing = {c: int(out[c].isna().sum()) for c in out.columns if out[c].isna().any()}
    if missing:
        logger.info("wave %s anchor missing: %s", wave, missing)
    return out


def weighted_attendance_overall(wave: int = 2024) -> pd.DataFrame:
    allow_uw = wave in UNWEIGHTED_WAVES
    master = load_general_master(wave, allow_unweighted=allow_uw)
    wsum = float(master["weight"].sum())
    rows: list[dict] = []
    any_rate = float((master["attended_any"].astype(float) * master["weight"]).sum() / wsum * 100)
    rows.append(
        {"canonical_var": "offline_total", "estimate_pct": round(any_rate, 4), "n": len(master)}
    )
    item = load_item_for_persons(master["row_uid"], wave)
    item = item.merge(master[["row_uid", "weight"]], on="row_uid", how="inner")
    for g in GENRES:
        sub = item[item["canonical_item"] == g]
        if sub.empty:
            raise ValueError(f"missing canonical_item={g} in wave {wave}")
        rate = float(
            (sub["attended"].astype(float) * sub["weight"]).sum() / float(sub["weight"].sum()) * 100
        )
        rows.append({"canonical_var": g, "estimate_pct": round(rate, 4), "n": len(master)})
    return pd.DataFrame(rows)


def load_published_overall(year: int = 2024) -> pd.DataFrame:
    pub = pd.read_csv(settings.harmonized_long)
    pub = pub[(pub["year"] == year) & (pub["segment_dimension"] == "overall")]
    if "is_imputed" in pub.columns:
        pub = pub[pub["is_imputed"] != True]
    return (
        pub[["canonical_var", "value", "n_base"]]
        .rename(columns={"value": "published_pct"})
        .reset_index(drop=True)
    )


def reproduce_gate(wave: int = 2024, tol_pp: float | None = None) -> pd.DataFrame:
    """Against published overall. Default tol 1.5 pp for 2020, 0.1 pp for 2022 and 2024."""
    if tol_pp is None:
        tol_pp = 1.5 if wave == 2020 else 0.1
    if wave in UNWEIGHTED_WAVES:
        est = weighted_attendance_overall(wave)
        est["published_pct"] = np.nan
        est["diff_pp"] = np.nan
        est["ok"] = True
        est["note"] = "unweighted_diagnostic_skip_pub"
        return est
    est = weighted_attendance_overall(wave)
    pub = load_published_overall(wave)
    out = pub.merge(est, on="canonical_var", how="inner")
    out["diff_pp"] = (out["estimate_pct"] - out["published_pct"]).round(4)
    out["ok"] = out["diff_pp"].abs() <= tol_pp
    out["tol_pp"] = tol_pp
    return out.sort_values("canonical_var").reset_index(drop=True)


def gate_all_eras() -> pd.DataFrame:
    rows = []
    for wave in ERA_WAVES:
        g = reproduce_gate(wave)
        g.insert(0, "wave", wave)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def depth_proxy_from_band(bands: pd.Series) -> pd.Series:
    """Attendance-count band → representative value. Apply the **same** map to
    synthetic and observed counts (see the schema comment).
    """
    return bands.map(lambda x: ATTEND_COUNT_BAND_MIDPOINTS.get(str(x), np.nan)).astype(float)
