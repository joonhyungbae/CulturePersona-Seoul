"""official era frame · gate smoke."""

from __future__ import annotations

import pytest

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame, reproduce_gate
from sfac_persona.schema import ERA_CORE_ANCHORS, GENRE_ANCHORS

# Skip when source microdata (publisher approval required) is absent — the review copy has none.
needs_microdata = pytest.mark.skipif(
    not (settings.official_person_dir / "official_person_master.parquet").exists(),
    reason="licensed microdata not present",
)


@needs_microdata
def test_reproduce_gate_2024_pass():
    g = reproduce_gate(2024, tol_pp=0.1)
    assert g["ok"].all()


@needs_microdata
def test_anchor_frame_2024_core_columns():
    fr = build_official_anchor_frame(2024)
    assert len(fr) >= 5000
    for a in ERA_CORE_ANCHORS:
        assert a in fr.columns
    assert set(GENRE_ANCHORS).issubset(fr.columns)
    assert fr["living_area"].notna().all()
    assert fr["attend_any"].isin(["예", "아니오"]).all()
