"""Unit tests for the anchor codebook and derived rules (runs without microdata)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sfac_persona.grounding.microdata import _band, _likert3
from sfac_persona.schema import (
    ANCHOR_BY_NAME,
    ANCHORS,
    ATTEND_COUNT_EDGES,
    GENRE_ANCHORS,
    GENRES_2024,
    HOUSEHOLD_INCOME_EDGES_MANWON,
    ROOT_ANCHORS,
)


def test_anchor_names_unique() -> None:
    names = [a.name for a in ANCHORS]
    assert len(names) == len(set(names))


def test_root_anchors() -> None:
    assert ROOT_ANCHORS == ["sex", "age_band", "living_area"]


def test_genre_anchor_count() -> None:
    assert len(GENRE_ANCHORS) == 10 == len(GENRES_2024)
    assert GENRES_2024[4] == "popular_performance"  # number 4 from 2022 on


def test_bands() -> None:
    assert _band(0, ATTEND_COUNT_EDGES) == "0회"
    assert _band(2, ATTEND_COUNT_EDGES) == "1~2회"
    assert _band(11, ATTEND_COUNT_EDGES) == "11회 이상"
    assert _band(0, HOUSEHOLD_INCOME_EDGES_MANWON) == "200만원 미만"
    assert _band(900, HOUSEHOLD_INCOME_EDGES_MANWON) == "800만원 이상"
    assert _band(float("nan"), ATTEND_COUNT_EDGES) is None


def test_likert3() -> None:
    s = pd.Series([1, 4, 7, np.nan])
    out = _likert3(s).tolist()
    assert out[:3] == ["낮음", "보통", "높음"] and pd.isna(out[3])


def test_every_anchor_has_source() -> None:
    for a in ANCHORS:
        assert a.source, a.name
        assert ANCHOR_BY_NAME[a.name] is a
