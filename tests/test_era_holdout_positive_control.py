"""Positive-control script: prompt, paired resample, bootstrap. Does not call an LLM."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "era_holdout_positive_control.py"
    spec = importlib.util.spec_from_file_location("era_holdout_positive_control", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


pc = _load()


def test_resolve_option_integer_code() -> None:
    refs = {"leisure_time_weekday": {"2": ["하루 여가시간은 대략 2시간이다.", "약 2시간"]}}
    key, text = pc.resolve_option("leisure_time_weekday", "2.0", refs)
    assert key == "2"
    assert "2시간" in text
    assert pc.resolve_option("leisure_time_weekday", None, refs) is None


def test_prompt_keeps_skeleton_and_fact() -> None:
    label = pc.FACT_LABELS["leisure_time_weekday"]
    text = "하루 여가시간은 대략 2시간이다."
    quant = pc.quant_with_facts(
        {"age_band": "30대", "attend_any": "예", "sex": "여성"},
        {"leisure_time_weekday": {"key": "2", "text": text}},
    )
    prompt = pc.prompt_for(quant)
    assert prompt.startswith("당신은 서울에 사는")
    assert "30대" in prompt
    assert text in prompt
    stripped = pc.strip_oracle_quant(quant)
    assert label not in stripped
    assert stripped["age_band"] == "30대"
    assert text not in pc.prompt_for(stripped)


def test_weave_is_idempotent() -> None:
    sentence = "평일 하루 여가시간: 하루 여가시간은 대략 2시간이다."
    once = pc.weave_facts({"summary": "한 줄.", "weekend_routine": "주말."}, sentence)
    twice = pc.weave_facts(once, sentence)
    assert once == twice
    assert pc.FACT_MARK in once["summary"]
    assert pc.FACT_MARK in once["weekend_routine"]
    assert once["summary"].count(pc.FACT_MARK) == 1


def test_resample_preserves_cell_size() -> None:
    ids = ["a", "b", "c", "d"]
    cell = {
        "a": ("30대", "예"),
        "b": ("30대", "예"),
        "c": ("40대", "아니오"),
        "d": ("40대", "아니오"),
    }
    twin = {("30대", "예"): ["t1", "t2"], ("40대", "아니오"): ["t3"]}
    shared, twins = pc.resample_paired(ids, cell, twin, np.random.default_rng(0))
    assert len(shared) == 4
    assert sum(1 for pid in shared if cell[pid] == ("30대", "예")) == 2
    assert len(twins) == 3
    assert set(shared) <= set(ids)


def _rec(pid: str, p_ga: float) -> dict:
    return {
        "persona_id": pid,
        "quant": {"age_band": "30대", "attend_any": "예", "sex": "여성"},
        "probs": {"leisure": {"가": p_ga, "나": 1.0 - p_ga}},
    }


def _arms(release_p: float, oracle_p: float, twin_p: float) -> dict[str, list[dict]]:
    return {
        "release": [_rec(f"p{i}", release_p) for i in range(8)],
        "oracle": [_rec(f"p{i}", oracle_p) for i in range(8)],
        "twin": [_rec(f"t{i}", twin_p) for i in range(8)],
    }


def _real() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age_band": ["30대"] * 10,
            "attend_any": ["예"] * 10,
            "leisure": ["가"] * 5 + ["나"] * 5,
            "w": [1.0] * 10,
        }
    )


def test_bootstrap_oracle_beats_point_mass() -> None:
    order = {"leisure": ["가", "나"]}
    arms = _arms(1.0, 0.5, 1.0)
    rows = pc.bootstrap_diffs(arms, _real(), ["leisure"], order, n_boot=80, seed=1)
    by = {r["contrast"]: r for r in rows}
    assert by["oracle-release"]["diff"] < -0.4
    assert by["oracle-release"]["ci_hi"] < 0
    assert by["release-twin"]["ci_lo"] <= 0 <= by["release-twin"]["ci_hi"]


def test_identical_arms_cover_zero() -> None:
    order = {"leisure": ["가", "나"]}
    arms = _arms(0.5, 0.5, 0.5)
    rows = pc.bootstrap_diffs(arms, _real(), ["leisure"], order, n_boot=40, seed=2)
    by = {r["contrast"]: r for r in rows}
    assert abs(by["oracle-release"]["diff"]) < 1e-9
    assert by["oracle-release"]["ci_lo"] <= 0 <= by["oracle-release"]["ci_hi"]


def test_point_stays_inside_when_cell_tvd_is_resample_biased() -> None:
    """When probabilities spread within a cell, resampled TVD is biased upward
    and the percentile interval misses the point.

    The basic interval reflects that percentile interval about the point, so it
    misses the point too. The bias-removed interval contains the point.
    """
    order = {"leisure": ["가", "나"]}
    probs = [0.05, 0.2, 0.35, 0.65, 0.8, 0.95]
    ages = ["20대", "30대", "40대", "50대"]
    release: list[dict] = []
    oracle: list[dict] = []
    twin: list[dict] = []
    k = 0
    for age in ages:
        for p in probs:
            release.append(_rec_cell(f"p{k}", 1.0, age))
            oracle.append(_rec_cell(f"p{k}", p, age))
            twin.append(_rec_cell(f"t{k}", 1.0, age))
            k += 1
    real_rows = []
    for age in ages:
        for lab in ("가", "나"):
            real_rows.extend(
                {"age_band": age, "attend_any": "예", "leisure": lab, "w": 1.0} for _ in range(6)
            )
    arms = {"release": release, "oracle": oracle, "twin": twin}
    rows = pc.bootstrap_diffs(arms, pd.DataFrame(real_rows), ["leisure"], order, n_boot=400, seed=1)
    for row in rows:
        assert row["ci_lo"] <= row["diff"] <= row["ci_hi"]
    ora = next(r for r in rows if r["contrast"] == "oracle-release")
    assert ora["boot_bias"] > 0.02
    q_lo = ora["ci_lo"] + ora["boot_bias"]
    q_hi = ora["ci_hi"] + ora["boot_bias"]
    assert not (q_lo <= ora["diff"] <= q_hi)
    basic_lo = 2 * ora["diff"] - q_hi
    basic_hi = 2 * ora["diff"] - q_lo
    assert not (basic_lo <= ora["diff"] <= basic_hi)


def _rec_cell(pid: str, p_ga: float, age: str) -> dict:
    rec = _rec(pid, p_ga)
    rec["quant"] = {"age_band": age, "attend_any": "예", "sex": "여성"}
    return rec
