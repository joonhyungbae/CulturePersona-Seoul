"""Grounding-error ladder: routing rung (`cond_age_route`) and pair-set rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfac_persona.grounding.error_study import ladder_mask, pair_errors, routed_values


@pytest.fixture
def frame() -> pd.DataFrame:
    """Synthetic frame: two genres are conditionally independent of age band
    among attendees. Education is correlated with attendance.
    """
    rng = np.random.default_rng(0)
    n = 6000
    age = rng.choice(["20대", "40대", "60대"], n)
    p_att = pd.Series({"20대": 0.8, "40대": 0.7, "60대": 0.5})[age].to_numpy()
    attend = rng.random(n) < p_att
    edu = np.where(rng.random(n) < np.where(attend, 0.6, 0.3), "대졸", "고졸")
    g1 = np.where(attend & (rng.random(n) < 0.5), "예", "아니오")
    g2 = np.where(attend & (rng.random(n) < 0.3), "예", "아니오")
    count = np.where(attend, rng.choice(["1~2회", "3~5회"], n), "0회")
    return pd.DataFrame(
        {
            "w": 1.0,
            "sex": rng.choice(["남성", "여성"], n),
            "age_band": age,
            "attend_any": np.where(attend, "예", "아니오"),
            "attend_exhibition": g1,
            "attend_museum": g2,
            "attend_count_band": count,
            "education": edu,
        }
    )


ANCHORS = ["attend_any", "attend_exhibition", "attend_museum", "attend_count_band", "education"]


def test_routed_values_reads_the_questionnaire(frame: pd.DataFrame) -> None:
    r = routed_values(frame, ANCHORS)
    assert r == {
        "attend_any": "아니오",
        "attend_exhibition": "아니오",
        "attend_museum": "아니오",
        "attend_count_band": "0회",
    }


def test_route_equals_attend_inside_block_and_age_outside(frame: pd.DataFrame) -> None:
    p = pair_errors(frame, anchors=ANCHORS).set_index(["anchor_a", "anchor_b"])
    both = p.loc[("attend_exhibition", "attend_museum")]
    # Within-block pair: routing and attendance rungs share one assumption
    # (age-conditional independence among attendees) → the same error
    assert both.tvd_cond_age_route == pytest.approx(both.tvd_cond_age_attend, abs=1e-9)
    assert both.tvd_cond_age_route < both.tvd_cond_age
    # One-block pair: routing adds no attendance information, so it equals the
    # age rung, and the attendance rung is lower
    one = p.loc[("attend_exhibition", "education")]
    assert one.tvd_cond_age_route == pytest.approx(one.tvd_cond_age, abs=1e-9)
    assert one.tvd_cond_age_attend < one.tvd_cond_age_route


def test_ladder_mask_drops_structural_and_conditioning_pairs(frame: pd.DataFrame) -> None:
    p = pair_errors(frame, anchors=ANCHORS)
    assert "conditioning" in p.columns
    kept = p[ladder_mask(p)]
    assert not kept.structural.any()
    assert not (kept.anchor_a.eq("attend_any") | kept.anchor_b.eq("attend_any")).any()
    # Every rung is defined on ladder pairs
    assert kept.tvd_cond_age_attend.notna().all()
    assert kept.tvd_cond_age_route.notna().all()
