"""Anchors rendered as fixed Korean sentence templates: the template arm of the legibility test.

The release prose beats the same personas' anchor lists for other readers. That gain could come from any
sentence-form rendering of the anchors, or from what the generator wrote into the prose. This module renders
the anchors with one fixed sentence per anchor (no adjective, no inference, no model call) in the two slots
the reader sees (summary, weekend routine), so that release minus template isolates what the generated
prose adds beyond sentence form.

Design fixed before running (2026-10-10):
  template    one sentence per anchor, "<label>은/는 <value>이다.", demographic and household anchors in the
              summary slot, leisure and attendance anchors with the genre lists in the weekend-routine slot
  arms        template beside release and synthanchors, same personas (2024 evaluation bank), twins, items,
              reference statements and frozen tau
  readers     EXAONE and Gemma under SSR (primary: release minus template, cell-stratified paired bootstrap,
              2,000 draws, per reader); the generator under option log-probabilities (secondary)
Every result is reported, whatever its sign.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfac_persona.narrative import GENRE_ANCHORS, GENRE_KO
from sfac_persona.schema import ANCHOR_BY_NAME

SUMMARY_ANCHORS = (
    "sex",
    "age_band",
    "living_area",
    "education",
    "occupation",
    "household_income_band",
    "household_size_band",
    "seoul_years_band",
)
ROUTINE_ANCHORS = (
    "leisure_sufficiency",
    "leisure_companion",
    "attend_any",
    "attend_count_band",
    "paid_spend_band",
    "viewing_companion",
)


def _topic(word: str) -> str:
    """Topic particle by the final syllable's batchim: 은 after a consonant, 는 after a vowel."""
    if not word:
        return "는"
    ch = word.rstrip(")）]")[-1:] or word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        return "은" if (code - 0xAC00) % 28 else "는"
    return "는"


def _sentence(name: str, value: str) -> str:
    label = ANCHOR_BY_NAME[name].label_ko
    return f"{label}{_topic(label)} {value}이다."


def template_fields(quant: dict[str, str]) -> dict[str, str]:
    """The two fields the reader sees, filled with fixed sentences only."""
    summary = " ".join(_sentence(a, quant[a]) for a in SUMMARY_ANCHORS if quant.get(a))
    routine = [_sentence(a, quant[a]) for a in ROUTINE_ANCHORS if quant.get(a)]
    attended = [GENRE_KO[g.replace("attend_", "")] for g in GENRE_ANCHORS if quant.get(g) == "예"]
    not_attended = [
        GENRE_KO[g.replace("attend_", "")] for g in GENRE_ANCHORS if quant.get(g) != "예"
    ]
    routine.append(
        f"지난 1년 오프라인 관람 장르는 {', '.join(attended) if attended else '없음'}이다."
    )
    routine.append(
        f"지난 1년 관람하지 않은 장르는 {', '.join(not_attended) if not_attended else '없음'}이다."
    )
    return {"summary": summary, "weekend_routine": " ".join(routine)}


if __name__ == "__main__":
    import json

    sys.path.insert(0, str(ROOT / "scripts"))
    import era_holdout_run as H

    rec = H.load_eval_bank(2024)[0]
    print(json.dumps(template_fields(rec["quant"]), ensure_ascii=False, indent=1))
