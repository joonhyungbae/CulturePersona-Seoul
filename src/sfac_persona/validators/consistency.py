"""Quantitative–qualitative consistency checker.

Checks
  attendance_claim   non-attender narrative claims attendance in the past year (error)
  genre_claim        a genre not attended is written as seen with
                     '지난 1년/올해/최근' (warning)
  age_number         age is written as a number, or another age band is named (error)
  minor_work         a teen narrative mentions workplace, salary, or company (warning)
  empty_or_short     a field is empty or too short (error)
  korean_ratio       a field has a low Hangul ratio (warning)
  han_characters     a field mixes in Han or Chinese characters (error)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from sfac_persona.schema import AGE_BANDS, GENRE_ANCHORS, GENRE_KO

Severity = Literal["error", "warning", "info"]


@dataclass
class Issue:
    check: str
    severity: Severity
    field: str
    detail: str


_ATTEND_VERBS = r"(관람했|관람을 했|보러 갔|보러 다녔|공연을 봤|전시를 봤|영화관에 갔|영화를 봤|다녀왔|찾아갔|관람하곤|관람한다|즐겨 본다|자주 본다)"
_RECENT = r"(지난 1년|지난 한 해|최근 1년|올해|요즘|최근에|이번 해)"
_GENRE_WORDS: dict[str, str] = {
    "exhibition": r"(미술관|전시회|전시)",
    "museum": r"(박물관)",
    "western_classical": r"(클래식|오페라|교향악|음악회)",
    "popular_performance": r"(콘서트|대중공연|가요 공연|페스티벌 무대)",
    "dance_performance": r"(무용|발레)",
    "traditional": r"(국악|전통예술|판소리|사물놀이)",
    "theater": r"(연극|뮤지컬)",
    "literature_event": r"(문학행사|낭독회|북토크|작가와의 만남)",
    "movie": r"(영화관|극장에서 영화|영화를 봤)",
    "outdoor_event": r"(야외 ?행사|축제|거리예술|공공예술)",
}
_AGE_NUM = re.compile(r"(?<![\d~])(\d{2})\s*(세|살)")
_WORK = re.compile(r"(직장|월급|회사 생활|출근|퇴근|연봉)")
_HANGUL = re.compile(r"[가-힣]")
_HAN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")  # Han/Chinese; 27B sometimes mixes Chinese words


def _age_band_of(n: int) -> str:
    if n < 20:
        return "10대"
    if n >= 70:
        return "70대 이상"
    return f"{n // 10 * 10}대"


def check_all(quant: dict[str, str], narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    text_all = " ".join(narratives.values())

    for k, v in narratives.items():
        if len(v.strip()) < 8:
            issues.append(Issue("empty_or_short", "error", k, f"len={len(v.strip())}"))
            continue
        if _HAN.search(v):
            issues.append(Issue("han_characters", "error", k, _HAN.search(v).group(0)))
        letters = re.sub(r"[\s\d\W]", "", v)
        if letters and len(_HANGUL.findall(v)) / max(len(letters), 1) < 0.6:
            issues.append(Issue("korean_ratio", "warning", k, "non-Korean text"))

    if quant.get("attend_any") == "아니오":
        for k in ("cultural_taste", "companions", "spending_attitude", "summary"):
            v = narratives.get(k, "")
            if re.search(_RECENT + r".{0,25}" + _ATTEND_VERBS, v) or re.search(
                _ATTEND_VERBS + r".{0,25}" + _RECENT, v
            ):
                issues.append(
                    Issue(
                        "attendance_claim", "error", k, "non-attender described recent attendance"
                    )
                )

    for g in GENRE_ANCHORS:
        if quant.get(g) == "예":
            continue
        pat = _GENRE_WORDS[g.replace("attend_", "")]
        v = narratives.get("cultural_taste", "")
        if re.search(_RECENT + r".{0,20}" + pat + r".{0,20}" + _ATTEND_VERBS, v):
            issues.append(
                Issue(
                    "genre_claim",
                    "warning",
                    "cultural_taste",
                    f"claims recent {GENRE_KO[g.replace('attend_', '')]}",
                )
            )

    band = quant.get("age_band", "")
    for m in _AGE_NUM.finditer(text_all):
        n = int(m.group(1))
        if 10 <= n <= 99 and _age_band_of(n) != band:
            issues.append(Issue("age_number", "error", "*", f"'{m.group(0)}' vs {band}"))
            break
    for other in AGE_BANDS:
        if other != band and re.search(rf"(본인은|그는|그녀는|이 인물은)\s*{other}", text_all):
            issues.append(Issue("age_number", "error", "*", f"mentions {other} vs {band}"))
            break

    if band == "10대" and _WORK.search(text_all):
        issues.append(Issue("minor_work", "warning", "*", "work vocabulary for a teenager"))

    return issues
