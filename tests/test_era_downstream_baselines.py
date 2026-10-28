"""Whether naive Bayes on the public tables recovers the public cells exactly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "era_downstream_baselines.py"
_spec = importlib.util.spec_from_file_location("era_downstream_baselines", _PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["era_downstream_baselines"] = _mod
_spec.loader.exec_module(_mod)


def _tables() -> SimpleNamespace:
    return SimpleNamespace(marg={}, joint={}, cond={})


def test_suppressed_binary_complement_is_one_minus() -> None:
    tables = _tables()
    tables.marg[("attend_any", "예")] = 0.6
    tables.marg[("attend_any", "아니오")] = 0.4
    tables.marg[("age_band", "10대")] = 0.05
    tables.cond[("attend_any", "예", "age_band", "10대")] = 0.876
    got = _mod.attendance_probability(tables, ["age_band"], ["10대"])
    assert got == pytest.approx(0.876)


def test_single_root_recovers_conditional() -> None:
    tables = _tables()
    tables.marg[("attend_any", "예")] = 0.6
    tables.marg[("attend_any", "아니오")] = 0.4
    tables.marg[("age_band", "20대")] = 0.3
    tables.cond[("attend_any", "예", "age_band", "20대")] = 0.8
    tables.cond[("attend_any", "아니오", "age_band", "20대")] = 0.2
    got = _mod.attendance_probability(tables, ["age_band"], ["20대"])
    assert got == pytest.approx(0.8)


def test_single_pair_joint_recovers_conditional() -> None:
    tables = _tables()
    tables.marg[("attend_any", "예")] = 0.5
    tables.marg[("attend_any", "아니오")] = 0.5
    tables.marg[("education", "대졸")] = 0.25
    tables.joint[("attend_any", "예", "education", "대졸")] = 0.20
    tables.joint[("attend_any", "아니오", "education", "대졸")] = 0.05
    got = _mod.attendance_probability(tables, ["education"], ["대졸"])
    assert got == pytest.approx(0.20 / 0.25)


def test_missing_joint_keeps_marginal() -> None:
    tables = _tables()
    tables.marg[("attend_any", "예")] = 0.7
    tables.marg[("attend_any", "아니오")] = 0.3
    tables.marg[("education", "대졸")] = 0.4
    got = _mod.attendance_probability(tables, ["education"], ["대졸"])
    assert got == pytest.approx(0.7)


def test_two_factors_multiply_odds() -> None:
    tables = _tables()
    tables.marg[("attend_any", "예")] = 0.5
    tables.marg[("attend_any", "아니오")] = 0.5
    tables.marg[("age_band", "20대")] = 0.25
    tables.cond[("attend_any", "예", "age_band", "20대")] = 0.8
    tables.cond[("attend_any", "아니오", "age_band", "20대")] = 0.2
    tables.marg[("education", "대졸")] = 0.2
    tables.joint[("attend_any", "예", "education", "대졸")] = 0.15
    tables.joint[("attend_any", "아니오", "education", "대졸")] = 0.05
    got = _mod.attendance_probability(tables, ["age_band", "education"], ["20대", "대졸"])
    assert got == pytest.approx(12 / 13)


def test_map_follows_pair_joint() -> None:
    tables = _tables()
    tables.marg[("education", "고졸")] = 0.4
    tables.marg[("education", "대졸")] = 0.6
    tables.marg[("occupation", "학생")] = 0.32
    tables.joint[("education", "고졸", "occupation", "학생")] = 0.30
    tables.joint[("education", "대졸", "occupation", "학생")] = 0.02
    frame = pd.DataFrame({"occupation": ["학생", "학생"]})
    pred = _mod.predict_labels(tables, "education", frame, ["occupation"])
    assert pred.tolist() == ["고졸", "고졸"]
    prior = _mod.predict_labels(tables, "education", frame, [])
    assert prior.tolist() == ["대졸", "대졸"]
