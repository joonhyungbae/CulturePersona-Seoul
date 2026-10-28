"""Microdata → anchor frame.

This module is the only one that touches person-level rows. The return value stays
in memory and is not written to disk at person level. If preprocessing left a pickle
cache, read that; otherwise read the xlsx with openpyxl.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterable

import numpy as np
import pandas as pd

from sfac_persona.config import settings
from sfac_persona.schema import (
    ANCHORS,
    ATTEND_COUNT_EDGES,
    GENRES_2024,
    HOUSEHOLD_INCOME_EDGES_MANWON,
    LIKERT7_TO_3,
    SEOUL_YEARS_EDGES,
    SPEND_EDGES_MANWON,
    AnchorSpec,
)

logger = logging.getLogger(__name__)

PII_COLUMNS: frozenset[str] = frozenset({"NAME", "HP", "zipcode", "address", "SQ2_NAME", "SQ2_2"})
NA_LABEL = "해당 없음"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def _read_xlsx(path, sheet=0) -> pd.DataFrame:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    cols = [str(c) if c is not None else f"_c{i}" for i, c in enumerate(rows[0])]
    return pd.DataFrame(rows[1:], columns=cols)


def load_raw() -> pd.DataFrame:
    """2024 general-public information-disclosure edition. Direct-identifier columns
    are dropped as soon as they are read.
    """
    cache = settings.microdata_cache
    if cache.exists():
        with open(cache, "rb") as f:
            df = pickle.load(f)
        logger.info("microdata cache: %s (%d rows)", cache, len(df))
    else:
        files = sorted(settings.microdata_dir.glob("*일반시민_정보공개청구.xlsx"))
        if not files:
            raise FileNotFoundError(f"microdata not found under {settings.microdata_dir}")
        df = _read_xlsx(files[0])
        logger.info("microdata xlsx: %s (%d rows)", files[0].name, len(df))
    drop = [c for c in df.columns if c in PII_COLUMNS]
    if drop:
        logger.warning("dropping identifier-like columns: %s", drop)
        df = df.drop(columns=drop)
    return df


def load_value_labels() -> dict[str, dict[str, str]]:
    v = pd.read_csv(settings.value_labels_csv)
    v = v[(v.year == settings.survey_year) & (v.group == settings.microdata_group)]
    out: dict[str, dict[str, str]] = {}
    for r in v.itertuples():
        out.setdefault(str(r.var), {})[str(r.value)] = str(r.value_label)
    return out


# ----------------------------------------------------------------------------
# Derivation rules
# ----------------------------------------------------------------------------


def _band(x: float, edges: Iterable[tuple[int, int | None, str]]) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    for lo, hi, label in edges:
        if x >= lo and (hi is None or x <= hi):
            return label
    return None


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _likert3(s: pd.Series, labels: tuple[str, str, str] = ("낮음", "보통", "높음")) -> pd.Series:
    m = {k: v for k, v in LIKERT7_TO_3.items()}
    rename = {"낮음": labels[0], "보통": labels[1], "높음": labels[2]}
    return _num(s).map(lambda x: rename[m[int(x)]] if not np.isnan(x) and int(x) in m else None)


def _label_col(raw: pd.DataFrame, var: str, labels: dict[str, dict[str, str]]) -> pd.Series:
    lab = labels.get(var, {})
    return _num(raw[var]).map(lambda x: lab.get(str(int(x))) if not np.isnan(x) else None)


def _clean_likert_label(x: str | None) -> str | None:
    """Strip the circled number from a raw label such as '④ 보통'."""
    if not isinstance(x, str):
        return None
    return x.lstrip("①②③④⑤⑥⑦ ").strip()


def build_anchor_frame(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Frame of anchor values (strings) plus weight w. Person-level, but not written to disk."""
    raw = load_raw() if raw is None else raw
    labels = load_value_labels()
    out = pd.DataFrame(index=raw.index)
    out["w"] = _num(raw[settings.weight_var])

    for a in ANCHORS:
        out[a.name] = _derive(a, raw, labels)

    # Items blank for non-attenders are filled with '해당 없음' (a structural skip, not missingness)
    for col in ("viewing_companion", "viewing_satisfaction"):
        out.loc[out["attend_any"] == "아니오", col] = NA_LABEL

    missing = out.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        logger.info("anchor missing counts: %s", missing.to_dict())
    return out


def _derive(a: AnchorSpec, raw: pd.DataFrame, labels: dict[str, dict[str, str]]) -> pd.Series:
    if a.kind == "categorical":
        s = _label_col(raw, a.source[0], labels)
        if a.name == "living_area":
            # Survey label '동북2생활권 (성북구, …)' → schema LIVING_AREAS '동북2생활권'.
            # R0 and the generated rows use the short name, but the grounding tables used
            # the long name, so nodes parented on living area missed the key and backed
            # off every time (caught by the zero-cell check).
            return s.map(lambda x: x.split(" (")[0].strip() if isinstance(x, str) else None)
        return s.map(_clean_likert_label)
    if a.kind == "binary":
        return raw[a.source[0]].notna().map({True: "예", False: "아니오"})

    n = a.name
    if n == "seoul_years_band":
        return _num(raw["DQ1_1"]).map(lambda x: _band(x, SEOUL_YEARS_EDGES))
    if n == "child_cohabiting":
        none = raw["DQ7_6"].notna() | (_num(raw["DQ6"]) == 2)  # the survey skips the unmarried
        any_child = raw[["DQ7_1", "DQ7_2", "DQ7_3", "DQ7_4", "DQ7_5"]].notna().any(axis=1)
        return np.where(any_child, "있음", np.where(none, "없음", None))
    if n == "household_income_band":
        code = _num(raw["DQ8"])
        lower = (code - 1) * 50  # 1=under 500,000 won → 0, 2=50–100 → 50, … 19=900+ → 900
        return lower.map(lambda x: _band(x, HOUSEHOLD_INCOME_EDGES_MANWON))
    if n == "leisure_sufficiency":
        return _likert3(raw["A2"], ("부족", "보통", "충분"))
    if n == "attend_any":
        return raw[list(a.source)].notna().any(axis=1).map({True: "예", False: "아니오"})
    if n == "attend_count_band":
        total = raw[list(a.source)].apply(_num).fillna(0).sum(axis=1)
        return total.map(lambda x: _band(x, ATTEND_COUNT_EDGES))
    if n == "paid_spend_band":
        total = raw[list(a.source)].apply(_num).fillna(0).sum(axis=1)
        return total.map(lambda x: _band(x, SPEND_EDGES_MANWON))
    if n == "viewing_satisfaction":
        return _likert3(raw["B3"], ("불만족", "보통", "만족"))
    if n == "intent_max":
        mx = raw[list(a.source)].apply(_num).max(axis=1)
        return _likert3(mx)
    if n == "participation_any":
        none = raw["D1_1_10"].notna()
        any_p = raw[[f"D1_1_{i}" for i in range(1, 10)]].notna().any(axis=1)
        return np.where(none, "아니오", np.where(any_p, "예", None))
    if n == "activity_place":
        return _num(raw["E1"]).map(
            {
                1: "거주 생활권에서 주로 한다",
                2: "생활권 밖으로 찾아간다",
                3: "활동을 전혀 하지 않는다",
            }
        )
    if n == "seoul_culture_status":
        return _likert3(raw["H1"], ("비동의", "보통", "동의"))
    raise KeyError(f"no derive rule for anchor {n!r}")


def genre_columns() -> list[str]:
    return [f"attend_{v}" for v in GENRES_2024.values()]
