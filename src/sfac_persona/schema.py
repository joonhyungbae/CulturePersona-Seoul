"""Anchor codebook — rules that map 2024 survey variables onto persona quant anchors.

Principles
- Value labels are taken as-is from the codebook (VARIABLE/VALUE sheets). Do not invent names.
- Continuous values (counts, cost, years of residence) are folded into bands. Band edges live here.
- Multi-response items are expanded to 'any selected' binaries or per-genre binaries.
- Geography stops at the living area (9). Dong-level variables are not read at all.

AnchorSpec.kind
  categorical  value → label map
  binary       selected or not (one multi-response column)
  derived      computed from several columns (functions live in grounding.microdata)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

AnchorKind = Literal["categorical", "binary", "derived"]

# 2024 attendance-battery option order → canonical_var (same as the preprocessing item map)
GENRES_2024: dict[int, str] = {
    1: "exhibition",
    2: "museum",
    3: "western_classical",
    4: "popular_performance",
    5: "dance_performance",
    6: "traditional",
    7: "theater",
    8: "literature_event",
    9: "movie",
    10: "outdoor_event",
}
GENRE_KO: dict[str, str] = {
    "exhibition": "미술관·전시회",
    "museum": "박물관",
    "western_classical": "서양 고전 음악공연",
    "popular_performance": "대중공연",
    "dance_performance": "무용공연",
    "traditional": "전통예술공연",
    "theater": "연극(뮤지컬 포함)",
    "literature_event": "문학행사",
    "movie": "극장 영화",
    "outdoor_event": "야외문화행사",
}

# official item canonical order (independent of option order — ADR-007)
GENRE_LIST: tuple[str, ...] = tuple(GENRES_2024.values())

# Instrument era (questionnaire regime)
WAVE_REGIME: dict[int, str] = {
    2016: "R3",
    2018: "R3",
    2020: "R3",
    2022: "R4",
    2024: "R5",
}
WEIGHTED_WAVES: tuple[int, ...] = (2020, 2022, 2024)
UNWEIGHTED_WAVES: tuple[int, ...] = (2016, 2018)
ERA_WAVES: tuple[int, ...] = WEIGHTED_WAVES + UNWEIGHTED_WAVES

HOUSEHOLD_SIZE_EDGES: list[tuple[int, int | None, str]] = [
    (1, 1, "1명"),
    (2, 2, "2명"),
    (3, 3, "3명"),
    (4, None, "4명 이상"),
]

AGE_BANDS: list[str] = ["10대", "20대", "30대", "40대", "50대", "60대", "70대 이상"]
LIVING_AREAS: list[str] = [
    "도심생활권",
    "동북1생활권",
    "동북2생활권",
    "서북생활권",
    "서남1생활권",
    "서남2생활권",
    "서남3생활권",
    "동남1생활권",
    "동남2생활권",
]
SEXES: list[str] = ["남성", "여성"]

# Band edges (lower bound inclusive). Labels are built in grounding.microdata.
ATTEND_COUNT_EDGES: list[tuple[int, int | None, str]] = [
    (0, 0, "0회"),
    (1, 2, "1~2회"),
    (3, 5, "3~5회"),
    (6, 10, "6~10회"),
    (11, None, "11회 이상"),
]
# Band → representative value. Synthetic personas carry only the band, so depth
# (annual attendance count) is approximated by this value.
#
# The open top band's representative value is arbitrary (observed weighted means:
# 2020 21.1 · 2022 19.4 · 2024 19.2). So **when comparing to the observed data, fold
# the observed counts into the same bands and the same representatives**
# (`depth_proxy_from_band`). Subtracting raw counts measures discretization bias,
# not model error.
ATTEND_COUNT_BAND_MIDPOINTS: dict[str, float] = {
    "0회": 0.0,
    "1~2회": 1.5,
    "3~5회": 4.0,
    "6~10회": 8.0,
    "11회 이상": 15.0,
}

# B1_2_x_3 (annual amount paid by the respondent) is in 10,000-won units on the
# questionnaire. 2024 response median 11, max 550.
SPEND_EDGES_MANWON: list[tuple[int, int | None, str]] = [
    (0, 0, "0원"),
    (1, 4, "5만원 미만"),
    (5, 9, "5~10만원"),
    (10, 19, "10~20만원"),
    (20, 49, "20~50만원"),
    (50, None, "50만원 이상"),
]
# DQ8 has 19 bands (50 × 10,000 won) and the top is '900만원 이상', so a 1000 edge
# cannot be formed.
HOUSEHOLD_INCOME_EDGES_MANWON: list[tuple[int, int | None, str]] = [
    (0, 199, "200만원 미만"),
    (200, 399, "200~400만원"),
    (400, 599, "400~600만원"),
    (600, 799, "600~800만원"),
    (800, None, "800만원 이상"),
]
SEOUL_YEARS_EDGES: list[tuple[int, int | None, str]] = [
    (0, 4, "5년 미만"),
    (5, 9, "5~10년"),
    (10, 19, "10~20년"),
    (20, None, "20년 이상"),
]
# 7-point scale → 3 levels (1–3 / 4 / 5–7)
LIKERT7_TO_3: dict[int, str] = {
    1: "낮음",
    2: "낮음",
    3: "낮음",
    4: "보통",
    5: "높음",
    6: "높음",
    7: "높음",
}


@dataclass(frozen=True)
class AnchorSpec:
    name: str
    kind: AnchorKind
    source: tuple[str, ...]  # survey variables read
    label_ko: str
    root: bool = False  # variable whose root is the resident-registration population
    note: str = ""
    order: tuple[str, ...] = field(default_factory=tuple)  # value order (used in tables when set)


ANCHORS: list[AnchorSpec] = [
    AnchorSpec("sex", "categorical", ("SQ3",), "성별", root=True, order=tuple(SEXES)),
    AnchorSpec("age_band", "categorical", ("SQ4R",), "연령대", root=True, order=tuple(AGE_BANDS)),
    AnchorSpec(
        "living_area", "categorical", ("SQ1R",), "생활권", root=True, order=tuple(LIVING_AREAS)
    ),
    AnchorSpec("seoul_years_band", "derived", ("DQ1_1",), "서울 거주기간"),
    AnchorSpec("education", "categorical", ("DQ2",), "최종학력"),
    AnchorSpec("occupation", "categorical", ("DQ3",), "직업"),
    AnchorSpec("arts_worker", "categorical", ("DQ4",), "문화예술계 종사 여부"),
    AnchorSpec("marital", "categorical", ("DQ6",), "혼인상태"),
    AnchorSpec(
        "child_cohabiting",
        "derived",
        ("DQ6", "DQ7_1", "DQ7_2", "DQ7_3", "DQ7_4", "DQ7_5", "DQ7_6"),
        "동거 자녀",
        note="미혼(DQ6=2)에게는 묻지 않았다 → '없음' 으로 둔다",
    ),
    AnchorSpec("household_income_band", "derived", ("DQ8",), "월 가구소득"),
    # Derived anchor used only in the era core. Banded household_size from the
    # official canonical (2024 DQ5_1).
    AnchorSpec("household_size_band", "derived", ("DQ5_1",), "가구원 수 구간"),
    AnchorSpec("subjective_class", "categorical", ("DQ9",), "주관적 계층"),
    AnchorSpec("reading", "categorical", ("DQ10",), "지난 1년 독서"),
    AnchorSpec("leisure_sufficiency", "derived", ("A2",), "여가시간 충분도 (3단)"),
    AnchorSpec("leisure_companion", "categorical", ("A4",), "여가 동반자"),
    *[
        AnchorSpec(f"attend_{v}", "binary", (f"B1_1_{i}",), f"{GENRE_KO[v]} 관람")
        for i, v in GENRES_2024.items()
    ],
    AnchorSpec(
        "attend_any", "derived", tuple(f"B1_1_{i}" for i in GENRES_2024), "오프라인 관람 여부"
    ),
    AnchorSpec(
        "attend_count_band", "derived", tuple(f"B1_2_{i}_1" for i in GENRES_2024), "연간 관람 횟수"
    ),
    AnchorSpec(
        "paid_spend_band",
        "derived",
        tuple(f"B1_2_{i}_3" for i in GENRES_2024),
        "연간 본인 지불 관람비",
    ),
    AnchorSpec(
        "viewing_companion", "categorical", ("B2",), "관람 동반자", note="관람 경험자만 응답"
    ),
    AnchorSpec(
        "viewing_satisfaction", "derived", ("B3",), "관람 만족도 (3단)", note="관람 경험자만 응답"
    ),
    AnchorSpec(
        "intent_max",
        "derived",
        tuple(f"B1_6_{i}" for i in GENRES_2024),
        "향후 관람 의향 (최대, 3단)",
    ),
    AnchorSpec(
        "participation_any", "derived", tuple(f"D1_1_{i}" for i in range(1, 11)), "직접 참여 경험"
    ),
    AnchorSpec("activity_place", "derived", ("E1",), "주 활동 장소"),
    AnchorSpec("seoul_culture_status", "derived", ("H1",), "서울의 문화도시 위상 (3단)"),
]

ANCHOR_BY_NAME: dict[str, AnchorSpec] = {a.name: a for a in ANCHORS}
ROOT_ANCHORS: list[str] = [a.name for a in ANCHORS if a.root]
GENRE_ANCHORS: list[str] = [f"attend_{v}" for v in GENRES_2024.values()]

# ADR-007 Tier 0–2 — era-common core (official frame)
ERA_CORE_ANCHORS: list[str] = [
    *ROOT_ANCHORS,
    "education",
    "occupation",
    "household_income_band",
    "seoul_years_band",
    "household_size_band",
    "leisure_sufficiency",
    "leisure_companion",
    *GENRE_ANCHORS,
    "attend_any",
    "attend_count_band",
    "paid_spend_band",
    "viewing_companion",
]


class PersonaQuant(BaseModel):
    """Persona quantitative anchors. Every value is a Korean label string
    (binaries are '예'/'아니오').
    """

    persona_id: str
    survey_year: int = 2024
    values: dict[str, str] = Field(default_factory=dict)

    def get(self, name: str) -> str:
        return self.values[name]
