"""CAP: 1 if the key determines the target; the marginal base rate if the key is irrelevant."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "era_attribute_disclosure.py"
    spec = importlib.util.spec_from_file_location("era_attribute_disclosure", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


disclosure = _load()


def test_cap_is_one_when_determined_and_matches_baseline_when_random() -> None:
    keys = np.array(["a", "a", "b", "b"])
    target = np.array(["x", "x", "y", "y"])
    determined = disclosure.correct_attribution_probability(keys, target, keys, target)
    assert np.allclose(determined, 1.0)

    # Within each key the target share matches the margin (weight 3 on x, 1 on y).
    src_keys = np.array(["a", "a", "b", "b"])
    src_target = np.array(["x", "y", "x", "y"])
    src_w = np.array([3.0, 1.0, 3.0, 1.0])
    real_keys = np.array(["a", "b", "a"])
    real_target = np.array(["x", "y", "y"])
    cap = disclosure.correct_attribution_probability(
        real_keys, real_target, src_keys, src_target, source_weight=src_w
    )
    base = disclosure.marginal_attribution_probability(real_target, src_target, source_weight=src_w)
    assert np.allclose(cap, base)
    assert np.allclose(cap, [0.75, 0.25, 0.25])
