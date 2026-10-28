"""When A equals B, out-of-sample ladder TVD equals in-sample (`pair_errors`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sfac_persona.grounding.error_study import RECON, ladder_mask, pair_errors

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "era_ladder_crossfit.py"
_spec = importlib.util.spec_from_file_location("era_ladder_crossfit", _PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["era_ladder_crossfit"] = _mod
_spec.loader.exec_module(_mod)

ANCHORS = ["attend_any", "attend_exhibition", "attend_museum", "attend_count_band", "education"]


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 4000
    age = rng.choice(["20대", "40대", "60대"], n)
    p_att = pd.Series({"20대": 0.8, "40대": 0.7, "60대": 0.5})[age].to_numpy()
    attend = rng.random(n) < p_att
    edu = np.where(rng.random(n) < np.where(attend, 0.6, 0.3), "대졸", "고졸")
    g1 = np.where(attend & (rng.random(n) < 0.5), "예", "아니오")
    g2 = np.where(attend & (rng.random(n) < 0.3), "예", "아니오")
    count = np.where(attend, rng.choice(["1~2회", "3~5회"], n), "0회")
    return pd.DataFrame(
        {
            "w": rng.uniform(0.5, 1.5, n),
            "sex": rng.choice(["남성", "여성"], n),
            "age_band": age,
            "attend_any": np.where(attend, "예", "아니오"),
            "attend_exhibition": g1,
            "attend_museum": g2,
            "attend_count_band": count,
            "education": edu,
        }
    )


def test_oos_equals_insample_when_halves_match() -> None:
    frame = _frame()
    scored = _mod.score_split(frame, frame.copy(), ANCHORS)
    ins = pair_errors(frame, anchors=ANCHORS)
    ins = ins[ladder_mask(ins)]
    assert scored.n_pairs == len(ins)
    assert float(ins["tvd_indep"].mean()) > 0.01
    for rung in RECON:
        mean = float(ins[f"tvd_{rung}"].mean())
        assert scored.tvd_b[rung] == pytest.approx(mean, abs=1e-8)
        assert scored.tvd_a[rung] == pytest.approx(mean, abs=1e-8)
    assert scored.floor == pytest.approx(0.0, abs=1e-12)
