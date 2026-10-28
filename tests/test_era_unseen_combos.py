"""Unobserved-combination rules. Does not run the 30,000-person sample."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


mod = _load("era_unseen_combos")
audit = _load("era_sampler_audit")

COLLEGE = "대학 졸업(재학, 중퇴포함)"
GRAD = "대학원수료/졸업(재학, 중퇴포함)"
JUNIOR = "전문대 졸업(재학, 중퇴포함)"
HIGH = "고등학교 졸업(재학, 중퇴포함) 이하"
MANAGER = "관리자(정부, 기업, 단체의 고위직 및 부서장 급 이상)"
CLERK = "사무종사자(일반회사 사무직, 기술직)"
SERVICE = "서비스종사자(공공 사회서비스 및 개인 생활 서비스 등 제공)"
UNEMPLOYED = "무직/은퇴"


def test_seed_and_axes_match_sampler_audit() -> None:
    assert mod.SEED == audit.SEED == 20260911
    assert mod.COVERAGE_KEYS == audit.COVERAGE_KEYS


def test_no_early_twenties_band_in_schema() -> None:
    from sfac_persona.schema import AGE_BANDS

    assert mod.early_twenties_labels(AGE_BANDS) == []
    assert "20대" in AGE_BANDS


def test_teen_college_and_graduate_are_rule_a() -> None:
    assert mod.flags_for("10대", GRAD, "학생") == {"a", "a_graduate"}
    assert mod.flags_for("10대", COLLEGE, "학생") == {"a", "a_college"}
    assert mod.flags_for("10대", JUNIOR, "학생") == set()
    assert mod.flags_for("10대", HIGH, "학생") == set()


def test_teen_jobs_other_than_student_or_unemployed_are_rule_b() -> None:
    assert mod.flags_for("10대", HIGH, MANAGER) == {"b", "b_white"}
    assert mod.flags_for("10대", HIGH, CLERK) == {"b", "b_white"}
    assert mod.flags_for("10대", HIGH, "전업주부") == {"b", "b_homemaker"}
    assert mod.flags_for("10대", HIGH, SERVICE) == {"b", "b_other_job"}
    assert mod.flags_for("10대", HIGH, "기타") == {"b", "b_other_job"}
    assert mod.flags_for("10대", HIGH, "학생") == set()
    assert mod.flags_for("10대", HIGH, UNEMPLOYED) == set()


def test_rule_c_does_not_flag_the_twenties() -> None:
    assert mod.flags_for("20대", GRAD, MANAGER) == set()
    assert mod.flags_for("20대", COLLEGE, CLERK) == set()


def test_elder_student_and_not_sixties() -> None:
    assert mod.flags_for("70대 이상", HIGH, "학생") == {"d"}
    assert mod.flags_for("60대", GRAD, "학생") == set()
    assert mod.flags_for("70대 이상", GRAD, UNEMPLOYED) == set()


def test_young_retiree_is_separate_from_bundled_unemployed() -> None:
    assert "e_young_retiree" in mod.flags_for("20대", HIGH, "은퇴")
    assert "e_young_retiree" not in mod.flags_for("10대", HIGH, UNEMPLOYED)
    teen_retire = mod.flags_for("10대", HIGH, "은퇴")
    assert "e_young_retiree" in teen_retire
    assert "b" in teen_retire


def test_college_and_white_collar_overlap() -> None:
    flags = mod.flags_for("10대", GRAD, CLERK)
    assert flags & mod.RULE_ANY == {"a", "b"}
    assert "a_graduate" in flags and "b_white" in flags


def test_level_used_prefers_the_longest_parent() -> None:
    node = {
        "name": "education",
        "levels": [
            {
                "parents": ["age_band", "sex"],
                "table": {"10대|여성": {"values": ["고등학교"], "weights": [1.0]}},
            },
            {
                "parents": ["age_band"],
                "table": {"10대": {"values": ["대학원"], "weights": [1.0]}},
            },
            {"parents": [], "table": {"": {"values": ["대학원"], "weights": [1.0]}}},
        ],
    }
    assert mod.level_used(node, {"age_band": "10대", "sex": "여성"}) == ["age_band", "sex"]
    assert mod.level_used(node, {"age_band": "10대", "sex": "남성"}) == ["age_band"]
    assert mod.parent_label(["age_band", "sex"], ["age_band"]) == "backoff:age_band|dropped:sex"
    assert mod.parent_label(["age_band", "sex"], ["age_band", "sex"]).startswith("full:")
