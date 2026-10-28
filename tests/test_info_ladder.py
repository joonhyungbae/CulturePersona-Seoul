"""On a synthetic 2×2, KL(P || P(x)P(y)) = I(X;Y) and Pinsker holds."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "era_info_ladder.py"
_spec = importlib.util.spec_from_file_location("era_info_ladder", _PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_kl_equals_mi_and_pinsker_on_2x2() -> None:
    # P(X=Y=1)=0.4; off-diagonal cells are 0.1. Not independent.
    joint = np.array([[0.4, 0.1], [0.1, 0.4]], dtype=float)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    indep = px * py
    tvd, kl, n_zero = _mod.kl_tvd(joint, indep)
    mi = _mod.mutual_info(joint)
    assert n_zero == 0
    assert kl == pytest.approx(mi, abs=1e-12)
    bound = _mod.pinsker_bound(kl)
    assert tvd <= bound + 1e-12
    assert 0.0 < tvd < bound
