"""2022-wave anchor frame — same anchor names and values as 2024. For the error
study's replication check only.

The 2022 file mixes five groups in one xlsx via GRP. General public = GRP 1 (panel)
+ 3 (interview), weight wt3_b. The variable system differs from 2024 (B1_1xk is a
count, D1xk is 예/아니오, DQ9 is income, …). Only the 29 anchors that can be built
to the 2024 definition are built. Subjective class (no DQ), reading, activity place
(E1_2 options differ), and Seoul culture perception (H1 is a different item) are omitted.

Same rules as 2024: PII columns are dropped as soon as they are read, and person
rows are not written to disk.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sfac_persona.config import settings
from sfac_persona.grounding.microdata import (
    ATTEND_COUNT_EDGES,
    HOUSEHOLD_INCOME_EDGES_MANWON,
    NA_LABEL,
    PII_COLUMNS,
    SEOUL_YEARS_EDGES,
    SPEND_EDGES_MANWON,
    _band,
    _likert3,
    _num,
)
from sfac_persona.schema import GENRE_ANCHORS

logger = logging.getLogger(__name__)

WEIGHT = "wt3_b"
GENERAL_GROUPS = (1, 3)
AGE = {
    1: "10대",
    2: "20대",
    3: "30대",
    4: "40대",
    5: "50대",
    6: "60대",
    7: "70대 이상",
    8: "70대 이상",
}
AREA_PREFIX = {
    11: "도심생활권",
    21: "동북1생활권",
    22: "동북2생활권",
    31: "서북생활권",
    41: "서남1생활권",
    42: "서남2생활권",
    43: "서남3생활권",
    51: "동남1생활권",
    52: "동남2생활권",
}
EDU = {
    1: "고등학교 졸업(재학, 중퇴포함) 이하",
    2: "고등학교 졸업(재학, 중퇴포함) 이하",
    3: "고등학교 졸업(재학, 중퇴포함) 이하",
    4: "전문대 졸업(재학, 중퇴포함)",
    5: "대학 졸업(재학, 중퇴포함)",
    6: "대학원수료/졸업(재학, 중퇴포함)",
}
OCC = {
    1: "관리자(정부, 기업, 단체의 고위직 및 부서장 급 이상)",
    2: "전문가 및 관련 종사자(과학, 의료, 교육, 예체능 등 전문분야 종사자)",
    3: "사무종사자(일반회사 사무직, 기술직)",
    4: "서비스종사자(공공 사회서비스 및 개인 생활 서비스 등 제공)",
    5: "판매종사자(영업 및 통신, 상점을 통해 상품이나 서비스를 판매)",
    6: "기타",
    7: "기능원 및 관련 기능 종사자(손과 공구, 기계를 통해 제품을 가공)",
    8: "장치, 기계 조작 및 조립 종사자(기계 및 장비를 조작하는 직업)",
    9: "단순노무 종사자(단순하고 일상적인 육제적인 노동 투입)",
    10: "학생",
    11: "전업주부",
    12: "무직/은퇴",
    13: "기타",
}
MARITAL = {1: "기혼(동거 포함)", 2: "미혼", 3: "별거/이혼", 4: "사별"}
COMPANION = {
    1: "혼자서",
    2: "가족(친척 포함)과 함께",
    3: "친구(연인 포함)와 함께",
    4: "직장 동료",
    5: "동호회(종교단체 등 포함) 활동을 통해서",
}
GENRE_IDX = {g: i for i, g in enumerate(GENRE_ANCHORS, 1)}  # same order in 2022 (codebook, 09-07)


def load_raw_2022() -> pd.DataFrame:
    d = settings.preprocessing_root / "data" / "raw" / "microdata" / "2022"
    files = sorted(d.glob("데이터*.xlsx"))
    if not files:
        raise FileNotFoundError(f"2022 microdata not found under {d}")
    df = pd.read_excel(files[0])
    drop = [c for c in df.columns if c in PII_COLUMNS or c in ("SQ2",)]
    if drop:
        logger.warning("dropping identifier-like columns: %s", drop)
        df = df.drop(columns=drop)
    df = df[_num(df["GRP"]).isin(GENERAL_GROUPS)].reset_index(drop=True)
    return df


def build_anchor_frame_2022(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    raw = load_raw_2022() if raw is None else raw
    out = pd.DataFrame(index=raw.index)
    out["w"] = _num(raw[WEIGHT])
    out["sex"] = _num(raw["SQ3"]).map({1: "남성", 2: "여성"})
    out["age_band"] = _num(raw["SQ4_R"]).map(AGE)
    area_code = _num(raw["SQ1_R1"]).map(AREA_PREFIX)
    out["living_area"] = area_code  # aligned to the 2024 full label below
    out["seoul_years_band"] = _num(raw["DQ1x1_1"]).map(lambda x: _band(x, SEOUL_YEARS_EDGES))
    out["education"] = _num(raw["DQ2"]).map(EDU)
    out["occupation"] = _num(raw["DQ3"]).map(OCC)
    out["arts_worker"] = _num(raw["DQ4"]).map({1: "예", 2: "아니오"})
    out["marital"] = _num(raw["DQ6"]).map(MARITAL)
    any_child = raw[[f"DQ7x{i}" for i in range(1, 6)]].notna().any(axis=1)
    none = raw["DQ7x6"].notna() | (_num(raw["DQ6"]) == 2)
    out["child_cohabiting"] = np.where(any_child, "있음", np.where(none, "없음", None))
    out["household_income_band"] = ((_num(raw["DQ9"]) - 1) * 50).map(
        lambda x: _band(x, HOUSEHOLD_INCOME_EDGES_MANWON)
    )
    out["leisure_sufficiency"] = _likert3(raw["A2"], ("부족", "보통", "충분"))
    out["leisure_companion"] = _num(raw["A4"]).map(COMPANION)
    counts = pd.DataFrame({g: _num(raw[f"B1_1x{i}"]).fillna(0) for g, i in GENRE_IDX.items()})
    for g in GENRE_ANCHORS:
        out[g] = (counts[g] > 0).map({True: "예", False: "아니오"})
    out["attend_any"] = (counts.sum(axis=1) > 0).map({True: "예", False: "아니오"})
    out["attend_count_band"] = counts.sum(axis=1).map(lambda x: _band(x, ATTEND_COUNT_EDGES))
    spend = pd.DataFrame({g: _num(raw[f"B1_3x{i}"]).fillna(0) for g, i in GENRE_IDX.items()}).sum(
        axis=1
    )  # 10,000 won
    out["paid_spend_band"] = spend.map(lambda x: _band(x, SPEND_EDGES_MANWON))
    out["viewing_companion"] = _num(raw["B2"]).map({**COMPANION, 6: "기타", 7: "기타"})
    out["viewing_satisfaction"] = _likert3(raw["B3"], ("불만족", "보통", "만족"))
    out["intent_max"] = _likert3(
        pd.DataFrame({i: _num(raw[f"B1_5x{i}"]) for i in range(1, 11)}).max(axis=1)
    )
    part = pd.DataFrame({i: _num(raw[f"D1x{i}"]) for i in range(1, 9)})
    out["participation_any"] = np.where(
        (part == 1).any(axis=1), "예", np.where((part == 2).all(axis=1), "아니오", None)
    )
    for col in ("viewing_companion", "viewing_satisfaction"):
        out.loc[out["attend_any"] == "아니오", col] = NA_LABEL
    out = out[out["living_area"].notna()].reset_index(drop=True)  # drop '그 외' areas
    return out


def align_living_area(frame22: pd.DataFrame, frame24: pd.DataFrame) -> pd.DataFrame:
    """Align to 2024's full labels, of the form '도심생활권 (종로구, 중구, 용산구)'."""
    full = {lab.split(" ")[0]: lab for lab in frame24["living_area"].dropna().unique()}
    frame22 = frame22.copy()
    frame22["living_area"] = frame22["living_area"].map(full)
    return frame22


ANCHORS_2022 = [
    "sex",
    "age_band",
    "living_area",
    "seoul_years_band",
    "education",
    "occupation",
    "arts_worker",
    "marital",
    "child_cohabiting",
    "household_income_band",
    "leisure_sufficiency",
    "leisure_companion",
    *GENRE_ANCHORS,
    "attend_any",
    "attend_count_band",
    "paid_spend_band",
    "viewing_companion",
    "viewing_satisfaction",
    "intent_max",
    "participation_any",
]
