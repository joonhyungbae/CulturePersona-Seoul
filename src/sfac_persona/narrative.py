"""Narrative layer — field definitions, prompts, and response parsing.

Twelve fields for the cultural-enjoyment domain. One JSON
call per persona. Each field lists the anchors it must reflect (`anchors`), and
the prompt emphasizes only those anchors.
Contradiction branches:
  non_attender   attend_any=아니오 → do not narrate attendance in the past year;
                 narrate barriers and substitute activities
  minor          teens → do not narrate own job or household income as the subject
                 (student and family context)
  low_income_high_spend  household income under 4 million won × spend of 200,000 won
                 or more → narrate priorities and tradeoffs
  arts_worker    works in culture and the arts → include a producer's point of view
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from jinja2 import BaseLoader, Environment

from sfac_persona.schema import ANCHOR_BY_NAME, GENRE_ANCHORS, GENRE_KO


@dataclass(frozen=True)
class NarrativeField:
    key: str
    label_ko: str
    guide: str
    anchors: tuple[str, ...]
    max_sentences: int = 3


FIELDS: list[NarrativeField] = [
    NarrativeField(
        "summary",
        "한 줄 요약",
        "3인칭 한 문장. 나이대·지역·문화생활의 핵심 성격.",
        ("age_band", "living_area", "attend_any"),
        1,
    ),
    NarrativeField(
        "weekend_routine",
        "주말 루틴",
        "평소 주말을 어떻게 보내는지. 여가시간 충분도와 동반자를 반영.",
        ("leisure_sufficiency", "leisure_companion", "occupation"),
    ),
    NarrativeField(
        "cultural_taste",
        "관람 취향과 동기",
        "지난 1년 실제로 본 장르만 언급. 무관람자는 과거 경험이나 관심으로만 말한다.",
        ("attend_any", *GENRE_ANCHORS, "intent_max"),
    ),
    NarrativeField(
        "info_channel",
        "정보 채널",
        "문화 정보를 어디서 얻는지. 나이대에 맞는 매체.",
        ("age_band", "online_content_hint"),
    ),
    NarrativeField(
        "companions",
        "동반자",
        "문화활동을 누구와 하는지. 관람 동반자와 여가 동반자가 다르면 그 차이.",
        ("viewing_companion", "leisure_companion", "marital"),
    ),
    NarrativeField(
        "barriers",
        "장벽과 아쉬움",
        "가지 못하게 하는 것. 시간·비용·거리·정보·동반자 중 앵커에 맞는 것.",
        ("attend_any", "leisure_sufficiency", "household_income_band", "activity_place"),
    ),
    NarrativeField(
        "neighborhood_culture",
        "동네 문화공간",
        "거주 생활권 안에서 이용하는 공간과 생활권 밖으로 나가는지.",
        ("living_area", "activity_place", "seoul_years_band"),
    ),
    NarrativeField(
        "online_consumption",
        "온라인 문화소비",
        "OTT·유튜브·음원 등 온라인 소비 습관. 오프라인과의 관계.",
        ("age_band", "attend_any"),
    ),
    NarrativeField(
        "spending_attitude",
        "지불 태도",
        "1년 관람비 수준과 그에 대한 태도. 가구소득·주관적 계층과 모순 없이.",
        ("paid_spend_band", "household_income_band", "subjective_class", "attend_count_band"),
    ),
    NarrativeField(
        "family_context",
        "가족 맥락",
        "혼인·동거 자녀·가구가 문화생활에 미치는 영향.",
        ("marital", "child_cohabiting", "age_band"),
    ),
    NarrativeField(
        "participation_history",
        "직접 참여 이력",
        "악기·창작·동호회 등 직접 해 본 경험. 없으면 없다고.",
        ("participation_any", "arts_worker", "reading"),
    ),
    NarrativeField(
        "view_on_seoul",
        "서울 문화에 대한 견해",
        "서울이 문화도시라고 느끼는지와 그 이유. 만족도와 위상 앵커 반영.",
        ("seoul_culture_status", "viewing_satisfaction", "living_area"),
    ),
]
FIELD_KEYS: list[str] = [f.key for f in FIELDS]

# Table-grounded field (ADR-033): expands the anchor `wish_theme` sampled from the
# free-response theme table T7. Present only in the 2024 wave (item H3), and kept
# apart from the 12 audited fields (FIELDS) — the 12-field generate/check path does
# not know this field.
WISH_FIELD = NarrativeField(
    "wish_for_seoul",
    "서울이 문화도시가 되기 위해 바라는 점",
    "이 사람이 서울에 바라는 문화 정책·환경을 본인 상황과 목소리로. 주제 앵커를 반드시 반영.",
    ("wish_theme", "living_area", "attend_any"),
)
TEXT_GROUNDED_FIELDS: list[NarrativeField] = [WISH_FIELD]

# Anchor lines inserted into the prompt (survey labels as-is, genres grouped)
_LOW_INCOME = {"200만원 미만", "200~400만원"}
_HIGH_SPEND = {"20~50만원", "50만원 이상"}


def branches(q: dict[str, str]) -> list[str]:
    b = []
    if q.get("attend_any") == "아니오":
        b.append("non_attender")
    if q.get("age_band") == "10대":
        b.append("minor")
    if q.get("household_income_band") in _LOW_INCOME and q.get("paid_spend_band") in _HIGH_SPEND:
        b.append("low_income_high_spend")
    if q.get("arts_worker") == "예":
        b.append("arts_worker")
    return b


def anchor_block(q: dict[str, str]) -> str:
    lines = []
    for name, val in q.items():
        if name in GENRE_ANCHORS or name in ("n_genres_band",):
            continue
        spec = ANCHOR_BY_NAME.get(name)
        label = spec.label_ko if spec else name
        lines.append(f"- {label}: {val}")
    attended = [GENRE_KO[g.replace("attend_", "")] for g in GENRE_ANCHORS if q.get(g) == "예"]
    not_attended = [GENRE_KO[g.replace("attend_", "")] for g in GENRE_ANCHORS if q.get(g) != "예"]
    lines.append(f"- 지난 1년 오프라인 관람 장르: {', '.join(attended) if attended else '없음'}")
    lines.append(
        f"- 지난 1년 관람하지 않은 장르: {', '.join(not_attended) if not_attended else '없음'}"
    )
    return "\n".join(lines)


_TEMPLATE = Environment(loader=BaseLoader(), autoescape=False).from_string(
    """당신은 서울에 사는 가상 인물의 문화생활을 서술하는 작가입니다. 아래 [정량 정보]는 설문 응답에서 온 사실이며, 서술은 이 사실과 한 글자도 모순되어서는 안 됩니다. 정보에 없는 것은 지어내되 정보와 충돌하지 않게 하십시오. 실제 인물·상호·주소를 쓰지 마십시오.

[규칙]
- 3인칭, 자연스러운 한국어 문어체. 각 필드는 최대 {{ max_sent }}문장, 한 줄 요약은 한 문장. 한자·중국어·영어 단어를 섞지 마십시오.
- 나이대·지역·관람 장르·가구소득·혼인 등 정량 정보를 바꾸거나 넘겨짚지 마십시오. 나이는 숫자로 쓰지 말고 [정량 정보]의 연령대로만 언급하십시오.
- 상투어 금지: '문화생활을 즐기는', '바쁜 일상 속에서', '소소한 행복', '힐링', '여유를 만끽'.
- 무관람자(지난 1년 오프라인 관람 장르: 없음)이면 지난 1년의 관람 경험을 서술하지 말고, 관람 취향은 과거 경험이나 관심·의향으로만 쓰며, 장벽 필드에 왜 가지 않았는지를 구체적으로 쓰십시오.
- 10대이면 직업·가구소득은 본인의 것이 아니라 가족의 상황이며, 학생의 관점에서 쓰고 직장·월급·회사 이야기를 하지 마십시오.
- 가구소득에 비해 관람비 지출이 크면(소득 400만원 미만, 지출 20만원 이상) 무리하지 않는 것처럼 꾸미지 말고 무엇을 우선하고 무엇을 포기하는지 쓰십시오.
- 문화예술계 종사자이면 관람자이면서 만드는 사람의 관점이 들어가야 합니다.

[출력]
아래 키를 모두 가진 JSON 객체 하나만 출력하십시오. 설명·머리말 없이 JSON 만.
{
{% for f in fields %}  "{{ f.key }}": "<{{ f.label_ko }} — {{ f.guide }}>"{{ "," if not loop.last }}
{% endfor %}}

[정량 정보]
{{ anchors }}
{% if br %}
[이 인물에 해당하는 규칙] {{ br_ko }}
{% endif %}"""
)
# Put the fixed part (role, rules, output schema) first and the per-persona anchors
# last → the KV prefix cache is reused.
# All branch rules are written in the fixed part; only the ones that apply are
# restated in one line at the end.
_BRANCH_KO = {
    "non_attender": "무관람자 규칙",
    "minor": "10대 규칙",
    "low_income_high_spend": "저소득·고지출 규칙",
    "arts_worker": "문화예술계 종사자 규칙",
}


def render_prompt(q: dict[str, str]) -> str:
    br = branches(q)
    return _TEMPLATE.render(
        anchors=anchor_block(q),
        br=br,
        br_ko=", ".join(_BRANCH_KO[b] for b in br),
        fields=FIELDS,
        max_sent=2,
    )


_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def parse_narrative(text: str) -> dict[str, str]:
    """Extract JSON only. If a thinking trace (<think>) is mixed in, take from the
    first { to the last }.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _FENCE.search(text)
    raw = m.group(1) if m else text[text.find("{") : text.rfind("}") + 1]
    obj: dict[str, Any] = json.loads(raw)
    out = {k: str(obj.get(k, "")).strip() for k in FIELD_KEYS}
    missing = [k for k, v in out.items() if not v]
    if missing:
        raise ValueError(f"missing narrative fields: {missing}")
    return out
