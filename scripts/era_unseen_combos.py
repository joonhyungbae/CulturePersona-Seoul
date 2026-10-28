"""Whether unobserved age×education×occupation×household-income combinations are plainly
unrealistic.

Resample with the same settings as the released sampler in `era_sampler_audit.py`.
Wave 2024, n=30,000, seed=20260911, `era_core_chain`, root_source=survey.
Coverage axes are those of that script's COVERAGE_KEYS that exist on the frame
(age band × education × occupation × household income. marital is absent from the era core, so it drops out).

Person rows stay in memory; only aggregates are written to disk.

Rules (applied only to unobserved personas. One person can hit several rules)
  (a) Teens (ages 15–19) whose education is college graduation or higher.
      The labels are '대학 졸업(재학, 중퇴포함)' and '대학원수료/졸업(재학, 중퇴포함)',
      so this splits into a_college and a_graduate. The college cell includes
      current enrollment, so ages 18–19 in college fall in. The graduate cell
      presupposes a completed bachelor's degree, so it does not fit ages 15–19.
  (b) Teens whose own occupation is neither student nor 무직/은퇴.
      b_white = manager, professional, or clerical; b_homemaker = homemaker;
      b_other_job = service, sales, self-employed, skilled, machine, elementary, or other.
      '무직/은퇴' is one cell, so retirement is not split off and the cell is allowed.
  (c) There is no band for the early twenties only (10대, 20대, …, 70대 이상). Not applied.
      Ages 20–29 commonly include graduate enrollment and managerial, professional, and
      clerical jobs, so the whole 20s band is not flagged.
  (d) Age 70+ and occupation student.
  (e) Teens or twenties, and the occupation label refers only to retirement (excluded if
      non-employment is in the same label).
      The 2024 label is the single cell '무직/은퇴', so this rule does not fire.
      The following were removed from violations after review. Counts are recorded only as a reference.
        Teens × junior college: age 19 can be enrolled, and the label includes enrollment and dropout.
        Sixties × student: graduate school and lifelong education are possible.
      Household income is a household quantity, so it is not treated as a contradiction with age.

Sensitivity strict is the union of a_graduate, b_white, b_homemaker, d, and e.
It drops (a)'s label that includes college enrollment, and the production, service, and
other jobs in (b) that are possible at age 19.
The violation rate the main text asks for is the union of (a)(b)(d)(e), not strict.

Run: uv run python scripts/era_unseen_combos.py
Outputs: outputs/reports/era_unseen_combos.csv · .md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sfac_persona.config import settings
from sfac_persona.grounding import chain as C
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import AGE_BANDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("era_unseen_combos")

# Must match era_sampler_audit.py for the 5.1% to reproduce.
SEED = 20260911
COVERAGE_KEYS = ["age_band", "education", "occupation", "household_income_band", "marital"]
TEEN = "10대"
ELDER = "70대 이상"
YOUNG_FOR_RETIREMENT = frozenset({TEEN, "20대"})
RULE_ANY = frozenset({"a", "b", "d", "e_young_retiree"})
RULE_STRICT = frozenset({"a_graduate", "b_white", "b_homemaker", "d", "e_young_retiree"})
BACKOFF_NODES = ("education", "occupation", "household_income_band")
EARLY_20_MARKERS = ("초반", "20-24", "20~24", "20–24")


def early_twenties_labels(ages: list[str] | set[str]) -> list[str]:
    """Age labels that refer only to ages 20–24. If none exist, (c) is not applied."""
    found = []
    for age in ages:
        if any(marker in age for marker in EARLY_20_MARKERS):
            found.append(age)
    return found


def education_class(education: str) -> str:
    if education.startswith("대학원"):
        return "graduate"
    if education.startswith("대학 졸업"):
        return "college"
    if education.startswith("전문대"):
        return "junior"
    return "other"


def occupation_class(occupation: str) -> str:
    if occupation in {"", "nan", "None", "NaN", "<NA>"}:
        return "missing"
    if occupation == "학생" or occupation.startswith("학생"):
        return "student"
    if occupation.startswith("무직"):
        return "nonemployed"
    if occupation.startswith(("관리자", "전문가", "사무")):
        return "white"
    if occupation.startswith("전업주부"):
        return "homemaker"
    if ("은퇴" in occupation or "퇴직" in occupation) and "무직" not in occupation:
        return "retired"
    return "other_job"


def flags_for(age: str, education: str, occupation: str) -> set[str]:
    """Violation tags for one person. (c) is not turned on here."""
    out: set[str] = set()
    edu = education_class(education)
    occ = occupation_class(occupation)
    if age == TEEN and edu == "graduate":
        out.update({"a", "a_graduate"})
    elif age == TEEN and edu == "college":
        out.update({"a", "a_college"})
    if age == TEEN and occ in {"white", "homemaker", "other_job", "retired"}:
        out.add("b")
        if occ == "white":
            out.add("b_white")
        elif occ == "homemaker":
            out.add("b_homemaker")
        else:
            out.add("b_other_job")
    if age == ELDER and occ == "student":
        out.add("d")
    if age in YOUNG_FOR_RETIREMENT and occ == "retired":
        out.add("e_young_retiree")
    return out


def level_used(node: dict, state: dict[str, str]) -> list[str]:
    """Like ChainSampler._draw_node, return the longest parent list that is in the table."""
    for level in node["levels"]:
        parents = list(level["parents"])
        key = C.SEP.join(state[p] for p in parents) if parents else ""
        if key in level["table"]:
            return parents
    raise RuntimeError(f"no table for {node.get('name', '?')}")


def parent_label(full: list[str], used: list[str]) -> str:
    if used == full:
        return "full:" + ("+".join(used) if used else "marginal")
    if not used:
        return "marginal"
    dropped = [p for p in full if p not in used]
    kept = "+".join(used)
    return f"backoff:{kept}|dropped:{'+'.join(dropped)}"


def _pair_set(frame: pd.DataFrame, left: str, right: str) -> set[tuple[str, str]]:
    return set(
        zip(frame[left].astype(str).tolist(), frame[right].astype(str).tolist(), strict=True)
    )


def _flag_frame(df: pd.DataFrame) -> pd.DataFrame:
    tags = [
        flags_for(age, edu, occ)
        for age, edu, occ in zip(
            df["age_band"].astype(str),
            df["education"].astype(str),
            df["occupation"].astype(str),
            strict=True,
        )
    ]
    columns = [
        "a",
        "a_college",
        "a_graduate",
        "b",
        "b_white",
        "b_homemaker",
        "b_other_job",
        "d",
        "e_young_retiree",
    ]
    out = pd.DataFrame({col: [col in tag for tag in tags] for col in columns})
    out["any"] = [bool(tag & RULE_ANY) for tag in tags]
    out["strict"] = [bool(tag & RULE_STRICT) for tag in tags]
    return out


def _add(
    rows: list[dict],
    *,
    section: str,
    key: str,
    n_personas: int,
    n_sample: int,
    n_unseen: int,
    note: str = "",
    n_combos: int | None = None,
    pct_of_rule: float | None = None,
    unseen_base: bool = True,
) -> None:
    rows.append(
        {
            "section": section,
            "key": key,
            "n_personas": int(n_personas),
            "n_combos": "" if n_combos is None else int(n_combos),
            "pct_sample": 100.0 * n_personas / n_sample,
            "pct_unseen": (100.0 * n_personas / n_unseen if unseen_base and n_unseen else np.nan),
            "pct_of_rule": "" if pct_of_rule is None else pct_of_rule,
            "note": note,
        }
    )


def _combo_count(combo: pd.Series, mask: pd.Series) -> int:
    return int(combo.loc[mask].nunique())


NOTES = {
    "any": "합집합 (a)∨(b)∨(d)∨(e). 논문이 묻는 위반 비율의 분자.",
    "strict": "민감도. a_graduate∨b_white∨b_homemaker∨(d)∨(e). 대학 재학 포함 라벨과 생산·서비스·기타 직업은 빼었다.",
    "a": "10대 × 학력 대학 졸업 이상 (재학·중퇴 포함 라벨).",
    "a_college": "10대 × '대학 졸업(재학, 중퇴포함)'. 만 18–19세 대학 재학이 이 칸에 섞인다.",
    "a_graduate": "10대 × '대학원수료/졸업(재학, 중퇴포함)'. 15–19세와 학제가 맞지 않는다.",
    "b": "10대 × 직업이 학생도 무직/은퇴도 아님.",
    "b_white": "10대 × 관리자·전문가·사무.",
    "b_homemaker": "10대 × 전업주부.",
    "b_other_job": "10대 × 서비스·판매·자영업·기능·장치·단순노무·기타. 19세 취업이 가능한 칸이 섞인다.",
    "d": "70대 이상 × 학생.",
    "e_young_retiree": "10대 또는 20대 × 은퇴·퇴직 단독 라벨. '무직/은퇴' 결합 라벨은 제외.",
    "c_skipped": "20대 초반 구간이 없어 (c)는 적용하지 않았다.",
}


def build_rows(
    frame: pd.DataFrame,
    syn: pd.DataFrame,
    spec: dict,
    *,
    n_sample: int,
) -> tuple[pd.DataFrame, dict]:
    keys = [k for k in COVERAGE_KEYS if k in syn.columns and k in frame.columns]
    real_combos = set(map(tuple, frame[keys].astype(str).to_numpy()))
    syn_tuples = list(map(tuple, syn[keys].astype(str).to_numpy()))
    syn_set = set(syn_tuples)
    new = syn_set - real_combos
    is_unseen = np.fromiter((c in new for c in syn_tuples), dtype=bool, count=len(syn_tuples))
    zero_share = float(is_unseen.mean()) if len(syn_tuples) else float("nan")
    n_unseen = int(is_unseen.sum())

    flags = _flag_frame(syn)
    flags.index = syn.index
    combo = pd.Series(["|".join(c) for c in syn_tuples], index=syn.index)
    unseen = pd.Series(is_unseen, index=syn.index)
    seen = ~unseen

    age_edu = _pair_set(frame, "age_band", "education")
    age_occ = _pair_set(frame, "age_band", "occupation")
    syn_age_edu = list(zip(syn["age_band"].astype(str), syn["education"].astype(str), strict=True))
    syn_age_occ = list(zip(syn["age_band"].astype(str), syn["occupation"].astype(str), strict=True))
    edu_missing = pd.Series([p not in age_edu for p in syn_age_edu], index=syn.index)
    occ_missing = pd.Series([p not in age_occ for p in syn_age_occ], index=syn.index)

    nodes = {n["name"]: n for n in spec["nodes"] if "levels" in n}
    full_parents = {name: list(nodes[name]["levels"][0]["parents"]) for name in BACKOFF_NODES}
    states = syn.to_dict("records")
    backoff = pd.DataFrame(index=syn.index)
    for name in BACKOFF_NODES:
        labels = [parent_label(full_parents[name], level_used(nodes[name], row)) for row in states]
        backoff[name] = labels

    rows: list[dict] = []
    _add(
        rows,
        section="summary",
        key="unseen",
        n_personas=n_unseen,
        n_sample=n_sample,
        n_unseen=n_unseen,
        n_combos=len(new),
        note=(
            f"응답자 조합 {len(real_combos)}개 대비 합성에만 있는 조합 {len(new)}개. "
            f"persona 비율 {100 * zero_share:.4f}% (원고 반올림 5.1%). "
            f"keys={keys}."
        ),
    )

    rule_masks = {
        "any": unseen & flags["any"],
        "strict": unseen & flags["strict"],
        "a": unseen & flags["a"],
        "a_college": unseen & flags["a_college"],
        "a_graduate": unseen & flags["a_graduate"],
        "b": unseen & flags["b"],
        "b_white": unseen & flags["b_white"],
        "b_homemaker": unseen & flags["b_homemaker"],
        "b_other_job": unseen & flags["b_other_job"],
        "d": unseen & flags["d"],
        "e_young_retiree": unseen & flags["e_young_retiree"],
    }
    for key, mask in rule_masks.items():
        _add(
            rows,
            section="rule",
            key=key,
            n_personas=int(mask.sum()),
            n_sample=n_sample,
            n_unseen=n_unseen,
            n_combos=_combo_count(combo, mask),
            note=NOTES[key],
        )

    overlaps = {
        "a_and_b": unseen & flags["a"] & flags["b"],
        "a_and_d": unseen & flags["a"] & flags["d"],
        "b_and_d": unseen & flags["b"] & flags["d"],
        "a_only": unseen & flags["a"] & ~flags["b"] & ~flags["d"] & ~flags["e_young_retiree"],
        "b_only": unseen & flags["b"] & ~flags["a"] & ~flags["d"] & ~flags["e_young_retiree"],
        "d_only": unseen & flags["d"] & ~flags["a"] & ~flags["b"] & ~flags["e_young_retiree"],
    }
    for key, mask in overlaps.items():
        _add(
            rows,
            section="overlap",
            key=key,
            n_personas=int(mask.sum()),
            n_sample=n_sample,
            n_unseen=n_unseen,
            n_combos=_combo_count(combo, mask),
            note="유형은 겹칠 수 있다. 합집합은 rule.any.",
        )

    ages = syn["age_band"].astype(str)
    edus = syn["education"].astype(str)
    occs = syn["occupation"].astype(str)
    not_flagged = {
        "teen_junior_college": unseen & ages.eq(TEEN) & edus.map(education_class).eq("junior"),
        "sixties_student": unseen & ages.eq("60대") & occs.map(occupation_class).eq("student"),
        "twenties_graduate": unseen & ages.eq("20대") & edus.map(education_class).eq("graduate"),
        "twenties_white": unseen & ages.eq("20대") & occs.map(occupation_class).eq("white"),
    }
    not_flagged_notes = {
        "teen_junior_college": "위반 아님. 10대×전문대(재학·중퇴 포함)는 만 19세 재학이 가능하다.",
        "sixties_student": "위반 아님. 60대 학생은 (d)에 넣지 않았다.",
        "twenties_graduate": "위반 아님. (c)를 적용하지 않아 20대×대학원은 세지 않는다.",
        "twenties_white": "위반 아님. 20대×관리자·전문가·사무도 세지 않는다.",
    }
    for key, mask in not_flagged.items():
        _add(
            rows,
            section="not_flagged",
            key=key,
            n_personas=int(mask.sum()),
            n_sample=n_sample,
            n_unseen=n_unseen,
            n_combos=_combo_count(combo, mask),
            note=not_flagged_notes[key],
        )

    margin_specs = [
        (
            "a",
            "age_x_education_absent",
            flags["a"],
            edu_missing,
            "연령×학력 쌍이 응답자 프레임에 없다.",
        ),
        (
            "a_graduate",
            "age_x_education_absent",
            flags["a_graduate"],
            edu_missing,
            "연령×대학원 쌍이 응답자 프레임에 없다.",
        ),
        (
            "a_college",
            "age_x_education_absent",
            flags["a_college"],
            edu_missing,
            "연령×대학 졸업(재학 포함) 쌍이 응답자 프레임에 없다.",
        ),
        (
            "b",
            "age_x_occupation_absent",
            flags["b"],
            occ_missing,
            "연령×직업 쌍이 응답자 프레임에 없다.",
        ),
        (
            "b_white",
            "age_x_occupation_absent",
            flags["b_white"],
            occ_missing,
            "연령×관리자·전문가·사무 쌍이 응답자 프레임에 없다.",
        ),
        (
            "d",
            "age_x_occupation_absent",
            flags["d"],
            occ_missing,
            "연령×학생 쌍이 응답자 프레임에 없다.",
        ),
    ]
    for rule, key, rule_flag, missing, note in margin_specs:
        mask = unseen & rule_flag & missing
        base = int((unseen & rule_flag).sum())
        _add(
            rows,
            section="margin",
            key=f"{rule}|{key}",
            n_personas=int(mask.sum()),
            n_sample=n_sample,
            n_unseen=n_unseen,
            n_combos=_combo_count(combo, mask),
            pct_of_rule=(100.0 * int(mask.sum()) / base) if base else np.nan,
            note=note + " 분모는 그 규칙의 미관측 페르소나.",
        )

    groups = {
        "violation": unseen & flags["any"],
        "unseen_ok": unseen & ~flags["any"],
    }
    for group, mask in groups.items():
        base_n = int(mask.sum())
        for name in BACKOFF_NODES:
            counts = backoff.loc[mask, name].value_counts()
            for label, n in counts.items():
                _add(
                    rows,
                    section="backoff",
                    key=f"{group}|{name}|{label}",
                    n_personas=int(n),
                    n_sample=n_sample,
                    n_unseen=n_unseen,
                    pct_of_rule=(100.0 * int(n) / base_n) if base_n else np.nan,
                    note=(
                        f"{group} 페르소나 {base_n}명 가운데 이 부모로 {name} 을 뽑은 비율. "
                        "백오프는 성별이 부모라 4원 조합만으로는 정해지지 않는다. "
                        "full 은 부모 칸 비가중 n≥cell_floor 인 조건부이고, 그 학력·직업 범주 "
                        "자체의 인원이 30명 이상이라는 뜻은 아니다."
                    ),
                )

    for rule, node in (
        ("a", "education"),
        ("a_graduate", "education"),
        ("a_college", "education"),
        ("b", "occupation"),
        ("b_white", "occupation"),
        ("d", "occupation"),
    ):
        mask = unseen & flags[rule]
        base_n = int(mask.sum())
        if base_n == 0:
            _add(
                rows,
                section="backoff_rule",
                key=f"{rule}|{node}|none",
                n_personas=0,
                n_sample=n_sample,
                n_unseen=n_unseen,
                pct_of_rule=np.nan,
                note="해당 위반 페르소나가 없다.",
            )
            continue
        counts = backoff.loc[mask, node].value_counts()
        for label, n in counts.items():
            _add(
                rows,
                section="backoff_rule",
                key=f"{rule}|{node}|{label}",
                n_personas=int(n),
                n_sample=n_sample,
                n_unseen=n_unseen,
                pct_of_rule=100.0 * int(n) / base_n,
                note=f"규칙 {rule} 의 미관측 페르소나가 {node} 를 뽑을 때 쓴 부모.",
            )

    in_support_n = int(seen.sum())
    for key in ("any", "a", "a_college", "a_graduate", "b", "b_white", "d"):
        mask = seen & flags[key]
        _add(
            rows,
            section="in_support",
            key=key,
            n_personas=int(mask.sum()),
            n_sample=n_sample,
            n_unseen=n_unseen,
            n_combos=_combo_count(combo, mask),
            pct_of_rule=(100.0 * int(mask.sum()) / in_support_n) if in_support_n else np.nan,
            unseen_base=False,
            note="응답자에 있는 4원 조합을 가진 페르소나. pct_of_rule 은 그 집단 대비. 위반 분자가 아니다.",
        )

    meta = {
        "zero_share": zero_share,
        "n_unseen": n_unseen,
        "n_new_combos": len(new),
        "n_real_combos": len(real_combos),
        "keys": keys,
        "n_any": int(rule_masks["any"].sum()),
        "n_strict": int(rule_masks["strict"].sum()),
        "early20": early_twenties_labels(sorted(set(ages) | set(AGE_BANDS))),
        "cell_floor": int(spec.get("cell_floor", settings.cell_floor)),
        "full_parents": {k: "+".join(v) if v else "marginal" for k, v in full_parents.items()},
    }
    return pd.DataFrame(rows), meta


def _fmt_pct(value: float) -> str:
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def write_markdown(
    path: Path, table: pd.DataFrame, meta: dict, *, wave: int, n_sample: int
) -> None:
    def take(section: str) -> pd.DataFrame:
        return table.loc[table["section"].eq(section)].copy()

    def line_for(section: str, key: str) -> pd.Series:
        hit = table.loc[table["section"].eq(section) & table["key"].eq(key)]
        if hit.empty:
            raise KeyError(f"{section}/{key}")
        return hit.iloc[0]

    any_row = line_for("rule", "any")
    strict_row = line_for("rule", "strict")
    zero = 100.0 * meta["zero_share"]
    lines = [
        "# 미관측 조합의 비현실 규칙 (2024)",
        "",
        (
            f"wave={wave} · n={n_sample:,} · seed={SEED} · "
            f"era_core_chain · root_source=survey · cell_floor={meta['cell_floor']}"
        ),
        "",
        (
            "대상은 2024년 서울 거주 온라인 패널 응답자 프레임에서 만든 샘플러다. "
            f"커버 축은 {' × '.join(meta['keys'])}."
        ),
        (
            f"미관측 페르소나 비율 {zero:.4f}% "
            f"(고유 조합 {meta['n_new_combos']}개 / 응답자 조합 {meta['n_real_combos']}개). "
            "원고 `samplingZeroShare` 는 이 비율을 소수 한 자리로 반올림한 5.1 이다."
        ),
        "",
        "## 규칙",
        "",
        "- (a) 10대 × 대학 졸업 이상. `a_college` 는 재학·중퇴가 라벨에 들어 있고, `a_graduate` 는 학사 이수가 전제다.",
        "- (b) 10대 × 학생·무직/은퇴가 아닌 직업. 관리자·전문가·사무, 전업주부, 그 밖을 나눈다.",
        "- (c) 적용하지 않음. 연령은 " + "·".join(AGE_BANDS) + " 이고 20대 초반 구간이 없다.",
        "- (d) 70대 이상 × 학생.",
        "- (e) 10대·20대 × 은퇴·퇴직 단독 라벨. `무직/은퇴` 결합 라벨은 무직으로 보아 제외한다.",
        "- 가구소득은 가구 단위라 규칙에 넣지 않았다. 10대 × 전문대, 60대 × 학생, 20대 × 대학원·사무직도 위반이 아니다.",
        "- `strict` 는 민감도다. 대학원·관리자/전문가/사무·전업주부·70대 학생·젊은 은퇴만 남긴다.",
        "",
        "## 미관측 페르소나 가운데 위반",
        "",
        (
            f"합집합 {int(any_row.n_personas):,}명, 미관측의 {any_row.pct_unseen:.2f}%, "
            f"전체의 {any_row.pct_sample:.2f}% "
            f"(고유 조합 {int(any_row.n_combos):,}개)."
        ),
        (
            f"민감도 strict {int(strict_row.n_personas):,}명, 미관측의 {strict_row.pct_unseen:.2f}%, "
            f"전체의 {strict_row.pct_sample:.2f}%."
        ),
        "",
        "| key | personas | combos | 전체 % | 미관측 % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in take("rule").iterrows():
        lines.append(
            f"| {r['key']} | {int(r.n_personas)} | {r.n_combos} | "
            f"{r.pct_sample:.2f} | {r.pct_unseen:.2f} |"
        )
    lines += [
        "",
        "겹침 (미관측 페르소나).",
        "",
        "| key | personas | 미관측 % |",
        "| --- | ---: | ---: |",
    ]
    for _, r in take("overlap").iterrows():
        lines.append(f"| {r['key']} | {int(r.n_personas)} | {r.pct_unseen:.2f} |")

    lines += [
        "",
        "## 위반 속성이 응답자 연령 주변에 있는가",
        "",
        "4원 조합이 없어도 연령×학력 또는 연령×직업 쌍은 응답자에 있을 수 있다. 아래는 그 쌍이 없는 경우다.",
        "",
        "| key | personas | 그 규칙 중 % |",
        "| --- | ---: | ---: |",
    ]
    for _, r in take("margin").iterrows():
        lines.append(f"| {r['key']} | {int(r.n_personas)} | {_fmt_pct(r.pct_of_rule)} |")

    lines += [
        "",
        "## 어느 부모로 뽑혔는가",
        "",
        (
            "교육의 완전 부모는 "
            f"`{meta['full_parents']['education']}`, 직업은 "
            f"`{meta['full_parents']['occupation']}`, 가구소득은 "
            f"`{meta['full_parents']['household_income_band']}` 이다. "
            "칸의 비가중 인원이 cell_floor 미만이면 뒤쪽 부모부터 뺀다."
        ),
        "수치는 페르소나 기준이다. 성별이 부모에 있어 같은 4원 조합도 백오프가 다를 수 있다.",
        "",
        "| group | node | parents | personas | 그룹 중 % |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for _, r in take("backoff").iterrows():
        group, node, parents = str(r["key"]).split("|", 2)
        lines.append(
            f"| {group} | {node} | {parents} | {int(r.n_personas)} | {_fmt_pct(r.pct_of_rule)} |"
        )
    lines += [
        "",
        "규칙별로, 그 규칙을 만든 노드만 본다. (a)는 학력, (b)와 (d)는 직업.",
        "",
        "| rule | node | parents | personas | 규칙 중 % |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for _, r in take("backoff_rule").iterrows():
        rule, node, parents = str(r["key"]).split("|", 2)
        lines.append(
            f"| {rule} | {node} | {parents} | {int(r.n_personas)} | {_fmt_pct(r.pct_of_rule)} |"
        )

    lines += [
        "",
        "## 위반으로 세지 않은 미관측",
        "",
        "| key | personas | 미관측 % |",
        "| --- | ---: | ---: |",
    ]
    for _, r in take("not_flagged").iterrows():
        lines.append(f"| {r['key']} | {int(r.n_personas)} | {r.pct_unseen:.2f} |")

    lines += [
        "",
        "## 응답자에 있는 조합",
        "",
        "같은 규칙이, 응답자에 이미 있는 4원 조합에서도 켜지는지 본다. 여기 인원은 위반 비율의 분자가 아니다.",
        "",
        "| key | personas | 전체 % |",
        "| --- | ---: | ---: |",
    ]
    for _, r in take("in_support").iterrows():
        lines.append(f"| {r['key']} | {int(r.n_personas)} | {r.pct_sample:.2f} |")

    lines += [
        "",
        "CSV: `era_unseen_combos.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2024)
    ap.add_argument("--sample", type=int, default=30000)
    args = ap.parse_args()

    for name in ("sfac_persona.grounding.chain", "sfac_persona.grounding.official"):
        logging.getLogger(name).setLevel(logging.WARNING)

    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    frame = C.add_derived_columns(build_official_anchor_frame(args.wave))
    spec = C.build_spec(frame, root_source="survey", nodes=C.era_core_chain())
    log.info("wave=%d frame n=%d sample=%d seed=%d", args.wave, len(frame), args.sample, SEED)
    syn = C.ChainSampler(spec).sample(args.sample, seed=SEED)
    table, meta = build_rows(frame, syn, spec, n_sample=args.sample)
    early = meta["early20"]
    if early:
        log.warning(
            "an early-twenties band exists, so the condition for skipping (c) is broken: %s", early
        )
    else:
        log.info("(c) skipped: no early-twenties band")

    csv_path = reports / "era_unseen_combos.csv"
    md_path = reports / "era_unseen_combos.md"
    table.to_csv(csv_path, index=False)
    write_markdown(md_path, table, meta, wave=args.wave, n_sample=args.sample)
    log.info(
        "unseen %.4f%% (%d) violation %d (%.2f%% of unseen, %.2f%% of sample) strict %d",
        100 * meta["zero_share"],
        meta["n_unseen"],
        meta["n_any"],
        100 * meta["n_any"] / meta["n_unseen"] if meta["n_unseen"] else float("nan"),
        100 * meta["n_any"] / args.sample,
        meta["n_strict"],
    )
    log.info("wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
