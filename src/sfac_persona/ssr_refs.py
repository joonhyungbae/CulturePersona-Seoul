"""SSR (semantic similarity rating) reference sentences.

Literature: arXiv 2510.08338, 2607.28550. Instruct models cannot draw a
sample from a distribution (2607.25292), so instead of showing the options and
asking for one, **hide the options, elicit a sentence, then** measure embedding
cosine against each option's reference sentence to place a distribution on the scale.

This module has no dependencies (imported by path from the embedding-only venv).

  QUESTION  question used to ask the item as a first-person free response. Options are not shown.
  SCALE     per-option reference sentences for Likert and ordinal items (written by hand —
            the meaning of the scale lives here)
  TEMPLATE  categorical and binary items: the option wording is the content word, so a template is used
  OVERRIDE  only options where the template sounds wrong ('기타', '하지 않는다') are written apart

One option may have **several** reference sentences (embeddings are averaged). Two reasons:
  1. If the frame sentence is long and identical across options, that shared part dominates
     the embedding and the options do not separate. So the framed sentence and **the option
     wording itself** are kept together.
  2. For binaries ('예'/'아니오'), two sentences that differ by only a negation
     barely separate in embedding space (cosine 0.87). Add one more sentence whose
     vocabulary actually differs.
"""

from __future__ import annotations

QUESTION: dict[str, str] = {
    "A3_1": "주말이나 휴일에는 주로 무엇을 하며 시간을 보내십니까?",
    "A5_1": "여건이 된다면 앞으로 가장 하고 싶은 여가활동은 무엇입니까?",
    **{
        f"C1_1_{i}": f"지난 1년 동안 온라인으로 {lab}을(를) 이용한 적이 있습니까?"
        for i, lab in enumerate(
            [
                "영화·드라마·애니메이션",
                "공연이나 전시 같은 문화예술 행사 영상",
                "음원",
                "웹툰·웹소설",
                "게임",
                "전자책이나 오디오북",
            ],
            1,
        )
    },
    "E2_3": "서울시에서 할 수 있는 문화예술 활동에 전반적으로 얼마나 만족하십니까?",
    "E3_1": "사시는 동네 주변에서 자주 이용하는 문화시설은 어디입니까?",
    "F2": "지난 1년 동안의 본인 문화생활에 스스로 얼마나 만족하십니까?",
    "F3": "문화예술은 당신의 삶에서 얼마나 중요합니까?",
    "F4": "요즘 얼마나 행복하다고 느끼십니까?",
    "F5": "문화예술이 당신이 사는 지역사회를 더 행복하게 만든다고 보십니까?",
    "F6_1": "요즘 외롭다고 느끼십니까?",
    "F6_2": "요즘 사람들 사이에서 소외되어 있다고 느끼십니까?",
    "F7": "한 달에 한 번 이상 만나거나 일주일에 한 번 이상 연락하는 가까운 사람이 몇 명쯤 됩니까?",
    "F9": "사람들과 관계를 맺거나 외로움을 덜기 위해 문화예술 활동에 참여할 생각이 있습니까?",
    "F11": "생성형 AI 가 만든 작품의 전시나 공연에 가 볼 생각이 있습니까?",
    "F12": "생성형 AI 가 만든 작품을 보는 데 돈을 낼 생각이 있습니까?",
    **{
        f"G1_{i}": f"{cond} 문화예술 활동을 지금보다 더 하시겠습니까?"
        for i, cond in enumerate(
            [
                "함께 참여할 가족이나 친구가 있다면",
                "본인 관심에 맞는 활동이 있다면",
                "동네에 문화시설이 더 많다면",
                "비용이 지금보다 낮다면",
                "시간 여유가 있다면",
            ],
            1,
        )
    },
    "H2": "서울이 문화예술 도시가 되려면 무엇이 가장 필요하다고 보십니까?",
}

_L = ("낮음", "보통", "높음")
_O4 = ("전혀 그렇지 않다", "대체로 그렇지 않다", "대체로 그렇다", "매우 그렇다")


def _triple(lo, mid, hi) -> dict:
    return dict(zip(_L, (lo, mid, hi)))


SCALE: dict[str, dict[str, str]] = {
    "E2_3": _triple(
        "서울에서 할 수 있는 문화예술 활동에 나는 전반적으로 불만이다. 기대에 못 미친다.",
        "서울의 문화예술 활동은 그럭저럭 괜찮다. 아주 만족스럽지도 크게 불만스럽지도 않다.",
        "서울에서 누릴 수 있는 문화예술 활동에 나는 전반적으로 아주 만족한다.",
    ),
    "F2": _triple(
        "지난 1년 동안 내 문화생활은 아쉬웠다. 스스로 만족스럽지 않다.",
        "지난 1년 동안의 내 문화생활은 보통이었다. 특별히 만족스럽지도 아쉽지도 않다.",
        "지난 1년 동안 나는 문화생활을 충분히 누렸고 스스로 아주 만족스럽다.",
    ),
    "F3": _triple(
        "문화예술은 내 삶에서 그다지 중요하지 않다. 없어도 사는 데 지장이 없다.",
        "문화예술은 내 삶에서 어느 정도 의미가 있는 정도다. 아주 중요하다고까지는 못 하겠다.",
        "문화예술은 내 삶에서 매우 중요하다. 내 삶을 이루는 큰 부분이다.",
    ),
    "F4": _triple(
        "요즘 나는 행복하지 않다. 삶이 힘들고 지쳐 있다.",
        "요즘 내 삶은 그저 그렇다. 특별히 행복하지도 불행하지도 않다.",
        "요즘 나는 아주 행복하다. 지금의 삶에 만족한다.",
    ),
    "F5": _triple(
        "문화예술이 우리 지역사회를 더 행복하게 만든다고 보기는 어렵다.",
        "문화예술이 지역사회 행복에 어느 정도는 도움이 될 것이다.",
        "문화예술은 우리 지역사회를 훨씬 더 행복하게 만든다고 생각한다.",
    ),
    "F9": _triple(
        "사람을 사귀거나 외로움을 달래려고 문화예술 활동에 참여할 생각은 없다.",
        "관계를 넓히려고 문화예술 활동에 참여하는 것도 상황이 되면 해볼 수 있다.",
        "사람들과 어울리고 외로움을 덜기 위해 문화예술 활동에 꼭 참여해 보고 싶다.",
    ),
    "F11": _triple(
        "생성형 AI 가 만든 작품의 전시나 공연에는 가고 싶지 않다.",
        "AI 가 만든 작품의 전시나 공연도 기회가 되면 한번쯤 가 볼 수 있다.",
        "생성형 AI 가 만든 작품의 전시나 공연에 꼭 가 보고 싶다. 관심이 많다.",
    ),
    "F12": _triple(
        "AI 가 만든 작품을 보려고 돈을 낼 생각은 없다.",
        "AI 가 만든 작품이라도 값이 적당하면 돈을 낼 수도 있다.",
        "AI 가 만든 작품을 보기 위해서라면 기꺼이 돈을 낼 것이다.",
    ),
    "G1_1": _triple(
        "함께 갈 가족이나 친구가 생겨도 내 문화예술 활동이 늘지는 않을 것이다.",
        [
            "같이 갈 사람이 있다면 문화예술 활동을 조금은 더 하게 될 것 같다.",
            "같이 갈 사람이 있으면 조금은 더 할 것 같지만, 크게 달라지지는 않을 것이다.",
        ],
        [
            "함께 갈 가족이나 친구만 있다면 문화예술 활동을 훨씬 더 많이 할 것이다.",
            "함께 갈 사람만 있으면 훨씬 더 많이 할 것이다. 지금 못 하는 이유가 바로 그것이다.",
        ],
    ),
    "G1_2": _triple(
        "내 관심에 맞는 활동이 생겨도 참여가 더 늘 것 같지는 않다.",
        [
            "관심에 맞는 활동이 있으면 조금 더 참여하게 될 것 같다.",
            "관심에 맞는 활동이 있으면 조금은 더 하겠지만, 크게 달라지지는 않을 것이다.",
        ],
        [
            "내 관심에 맞는 활동만 있다면 문화예술 활동을 훨씬 더 많이 할 것이다.",
            "관심에 맞는 활동만 있으면 훨씬 더 많이 할 것이다. 지금 못 하는 이유가 바로 그것이다.",
        ],
    ),
    "G1_3": _triple(
        "동네에 문화시설이 더 생겨도 내 참여가 늘지는 않을 것이다.",
        [
            "가까운 곳에 시설이 더 있으면 조금 더 다니게 될 것 같다.",
            "가까운 곳에 시설이 더 있으면 조금은 더 가겠지만, 크게 달라지지는 않을 것이다.",
        ],
        [
            "동네에 문화시설이 더 많아지면 훨씬 더 자주 이용할 것이다.",
            "동네에 시설만 더 있으면 훨씬 더 자주 갈 것이다. 지금 못 가는 이유가 바로 그것이다.",
        ],
    ),
    "G1_4": _triple(
        "값이 싸진다고 해서 내 문화예술 활동이 늘지는 않을 것이다.",
        [
            "비용이 좀 낮아지면 조금 더 자주 가게 될 것 같다.",
            "비용이 좀 낮아지면 조금은 더 가겠지만, 크게 달라지지는 않을 것이다.",
        ],
        [
            "비용만 낮아진다면 문화예술 활동을 훨씬 더 많이 할 것이다.",
            "값만 싸지면 훨씬 더 많이 할 것이다. 지금 못 하는 이유가 바로 돈이다.",
        ],
    ),
    "G1_5": _triple(
        "시간이 더 생겨도 문화예술 활동을 더 하지는 않을 것 같다.",
        [
            "시간 여유가 생기면 조금 더 문화예술 활동을 하게 될 것 같다.",
            "시간이 나면 조금은 더 하겠지만, 크게 달라지지는 않을 것이다.",
        ],
        [
            "시간 여유만 있다면 문화예술 활동을 훨씬 더 많이 할 것이다.",
            "시간만 나면 훨씬 더 많이 할 것이다. 지금 못 하는 이유가 순전히 시간이다.",
        ],
    ),
    "F6_1": dict(
        zip(
            _O4,
            (
                "나는 외롭다고 느끼는 일이 전혀 없다.",
                "나는 외롭다고 느끼는 편은 아니다. 가끔 그럴 때가 있어도 드물다.",
                "나는 외롭다고 느낄 때가 자주 있다.",
                "나는 늘 외롭다. 외로움이 아주 크다.",
            ),
        )
    ),
    "F6_2": dict(
        zip(
            _O4,
            (
                "나는 사람들에게서 소외되어 있다고 느끼는 일이 전혀 없다.",
                "나는 소외감을 느끼는 편은 아니다.",
                "나는 사람들 사이에서 소외되어 있다고 느낄 때가 자주 있다.",
                "나는 늘 사람들에게서 밀려나 있다고 느낀다. 소외감이 아주 크다.",
            ),
        )
    ),
    # Embeddings separate quantities poorly (sentences that differ only by a number
    # have cosine 0.92). Describe **circumstances** instead of numbers so they separate.
    "F7": {
        "0명": [
            "아무도 없다. 만나는 사람도 연락하는 사람도 없다.",
            "완전히 혼자다. 가깝다고 할 사람이 한 명도 없다.",
        ],
        "1~2명": [
            "딱 한 명, 많아야 두 명이다. 그 밖에는 없다.",
            "한두 명뿐이다. 손에 꼽을 만큼 적다.",
        ],
        "3~6명": [
            "서너 명은 된다. 대여섯 명쯤 꾸준히 본다.",
            "몇 명 정도다. 세 명에서 여섯 명 사이다.",
        ],
        "6명 이상": [
            "일고여덟 명이 넘는다. 여러 무리와 두루 어울린다.",
            "아주 많다. 열 명 가까이 되고 늘 북적인다.",
        ],
    },
}

# Items whose option wording is the content word. {opt} is filled with the option.
# Keep the frame short — a long stretch that is identical across options dominates
# the embedding and the options do not separate.
TEMPLATE: dict[str, str] = {
    "A3_1": "주말에는 주로 {opt}.",
    "A5_1": "가장 하고 싶은 것은 {opt}.",
    "E3_1": "동네에서 자주 가는 곳은 {opt}.",
    "H2": "가장 필요한 것은 {opt}.",
}

OVERRIDE: dict[tuple[str, str], list[str]] = {
    ("A3_1", "기타"): ["주말에는 앞에 없는 다른 일을 한다.", "그 밖의 활동, 기타."],
    ("A5_1", "기타"): ["가장 하고 싶은 것은 앞에 없는 다른 것이다.", "그 밖의 활동, 기타."],
    ("H2", "기타"): ["앞에 없는 다른 것이 가장 필요하다.", "그 밖의 것, 기타."],
    ("E3_1", "기타"): ["앞에 없는 다른 시설에 간다.", "그 밖의 시설, 기타."],
    ("E3_1", "거주지 주변에서 문화예술 활동을 하지 않는다"): [
        "동네 주변에서는 문화예술 활동을 하지 않는다.",
        "자주 가는 문화시설이 없다. 그런 곳에 가지 않는다.",
    ],
}


def _binary_label(ko: str) -> str:
    return ko.split(": ", 1)[1] if ": " in ko else ko


def references(var: str, kind: str, ko: str, options: list[str]) -> dict[str, list[str]]:
    """Per-option reference sentences for one item. Raises if any option has none."""
    if var in SCALE:
        refs = {o: SCALE[var][o] for o in options if o in SCALE[var]}
    elif kind == "binary_multi":
        lab = _binary_label(ko)
        refs = {
            "예": [
                f"지난 1년 동안 온라인으로 {lab}을(를) 이용한 적이 있다.",
                f"{lab}은(는) 자주 본다. 즐겨 이용하는 편이다.",
            ],
            "아니오": [
                f"지난 1년 동안 온라인으로 {lab}을(를) 이용한 적이 없다.",
                f"{lab}은(는) 거의 보지 않는다. 관심이 없다.",
            ],
        }
    elif var in TEMPLATE:
        # Frame sentence plus the option wording itself. The latter preserves the
        # difference between options.
        refs = {
            o: (OVERRIDE[(var, o)] if (var, o) in OVERRIDE else [TEMPLATE[var].format(opt=o), o])
            for o in options
        }
    else:
        raise KeyError(f"{var}: no reference sentences (kind={kind})")
    missing = [o for o in options if o not in refs]
    if missing:
        raise KeyError(f"{var}: options have no reference sentences — {missing}")
    return {o: ([refs[o]] if isinstance(refs[o], str) else list(refs[o])) for o in options}
