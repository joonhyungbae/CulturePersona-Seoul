"""Human-audit traps v2 — one field of the persona's own narrative, off on one axis only
(ADR-023 supplement, before first scoring).

v1 traps pasted B's whole narrative onto A's anchors, so about 7 of 14 anchor cells mismatched —
too easy for S1 and S2 to be informative (agent review, 2026-09-23). v2:

  1. Build A′ by changing **one axis** of clean persona A's anchors (values that belong to that
     axis are made consistent inside A′).
  2. Write A′'s narrative with the **same prompt, model, and temperature** as production
     (render_prompt, qwen3.8:27b-mtp, 0.6).
  3. Splice **only the one narrative field** where that axis shows, taking it from A′, into A's
     original narrative. The other 11 fields stay A's.
  → Style comes from the same generator, and the anchor mismatch is one field and one axis.

Difficulty tiers (16 each):
  T1 explicit   age band (summary) · attendance (cultural taste) · one extra genre
                (cultural taste)
  T2 degree     spend (spending attitude) · viewing companion (companions) · leisure time
                (weekend routine)
  T3 inference  household size (family context) 6 · occupation (weekend routine) 10 — company and
                commute on a not-working/retired or homemaker anchor, retirement on an employed or
                self-employed anchor. The changed value's wording must not appear in that field
                (the contradiction is visible only by reading and inferring).
                (Years in Seoul and life-stage age were dropped — the production model almost
                never surfaces them in the field, so a change yields traps that are invisible
                in principle.
                Trial generation: 0 of 4 revealed per axis.)
Manipulation check, 6: v1 style (wholesale replacement) — not tested; only whether raters applied
the rubric.

Whether the generated field actually reveals the axis is filtered by per-axis rules (`evident`
below); if it does not, rewrite. Raters and the LLM rater (Gemma) are not used to select traps —
if selection resembles the evaluation target, sensitivity is inflated.

Output: outputs/anno/traps_v2.jsonl (gitignore — same class as the answer key)
Run: uv run python scripts/anno_make_traps.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfac_persona.config import settings
from sfac_persona.llm_client import get_client
from sfac_persona.narrative import parse_narrative, render_prompt
from sfac_persona.schema import GENRE_ANCHORS
from sfac_persona.validators.consistency import _GENRE_WORDS, check_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_traps")

SEED = 20260924
WAVES = (2020, 2022, 2024)
OUT = settings.outputs_dir / "anno" / "traps_v2.jsonl"
TEMPERATURE = 0.6  # matches production generation (generate.GenerateConfig)
MAX_TOKENS = 2048
TRIES_PER_RECIPIENT = 3

AGE = ["10대", "20대", "30대", "40대", "50대", "60대", "70대 이상"]
SPEND = ["0원", "5만원 미만", "5~10만원", "10~20만원", "20~50만원", "50만원 이상"]
COMPANIONS = ["혼자서", "가족(친척 포함)과 함께", "친구(연인 포함)와 함께"]
OFFICE = "사무종사자(일반회사 사무직, 기술직)"

# (tier, axis, field to splice, count)
PLAN = [
    ("T1", "age", "summary", 6),
    ("T1", "attend", "cultural_taste", 5),
    ("T1", "genre", "cultural_taste", 5),
    ("T2", "spend", "spending_attitude", 6),
    ("T2", "companion", "companions", 5),
    ("T2", "leisure", "weekend_routine", 5),
    ("T3", "household", "family_context", 6),
    ("T3", "occupation", "weekend_routine", 10),
]
N_GROSS = 6  # wholesale-replacement manipulation checks are built v1-style by anno_build_sample.py
OUT_PRACTICE = settings.outputs_dir / "anno" / "traps_v2_practice.jsonl"
PLAN_PRACTICE = [
    ("T1", "age", "summary", 1),
    ("T2", "spend", "spending_attitude", 1),
    ("T3", "household", "family_context", 1),
]


def load_bank(wave: int) -> list[dict]:
    p = settings.data_root / "synthetic" / f"era_{wave}_narrative" / "personas.jsonl"
    rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    for r in rows:
        r["_wave"] = wave
    return rows


def is_clean(r: dict) -> bool:
    return bool(r.get("ok")) and not r.get("issues")


# --------------------------------------------------------------------------- A′ by axis


def alter(q: dict, axis: str, rng: np.random.Generator) -> dict | None:
    """A′ with one axis of q changed. None when the conditions do not hold."""
    a = dict(q)
    if axis == "age":
        if q["age_band"] == "10대" or q["occupation"] == "학생":
            return None
        i = AGE.index(q["age_band"])
        opts = [
            j for j in (i - 2, i + 2) if 1 <= j < len(AGE)
        ]  # do not send to the teens band (the branching rules change)
        if not opts:
            return None
        a["age_band"] = AGE[int(rng.choice(opts))]
    elif axis == "attend":
        if q["attend_any"] == "예":
            a.update(
                attend_any="아니오",
                n_genres_band="0",
                attend_count_band="0회",
                paid_spend_band="0원",
            )
            a["viewing_companion"] = "해당 없음"
            for g in GENRE_ANCHORS:
                a[g] = "아니오"
        else:
            g = str(rng.choice(["attend_movie", "attend_exhibition", "attend_popular_performance"]))
            a.update(
                attend_any="예",
                n_genres_band="1",
                attend_count_band="1~2회",
                paid_spend_band="5만원 미만",
            )
            a[g] = "예"
            lc = q["leisure_companion"]
            a["viewing_companion"] = lc if lc in COMPANIONS else "친구(연인 포함)와 함께"
    elif axis == "genre":
        if q["attend_any"] != "예":
            return None
        unseen = [g for g in GENRE_ANCHORS if q.get(g) != "예" and g != "attend_movie"]
        if not unseen:
            return None
        g = str(rng.choice(unseen))
        a[g] = "예"
        a["_added_genre"] = g
    elif axis == "spend":
        if q["attend_any"] != "예" or q["paid_spend_band"] not in SPEND:
            return None
        i = SPEND.index(q["paid_spend_band"])
        opts = [
            j for j in range(1, len(SPEND)) if abs(j - i) >= 2
        ]  # attendees are not sent to zero spend
        if not opts:
            return None
        a["paid_spend_band"] = SPEND[int(rng.choice(opts))]
    elif axis == "companion":
        # The companion field also mentions the leisure companion — if that matches the old or new
        # viewing companion, the field alone cannot separate them.
        if (
            q["viewing_companion"] not in COMPANIONS
            or q["leisure_companion"] == q["viewing_companion"]
        ):
            return None
        opts = [c for c in COMPANIONS if c not in (q["viewing_companion"], q["leisure_companion"])]
        if not opts:
            return None
        a["viewing_companion"] = str(rng.choice(opts))
    elif axis == "leisure":
        if q["leisure_sufficiency"] == "보통":
            return None
        a["leisure_sufficiency"] = "부족" if q["leisure_sufficiency"] == "충분" else "충분"
    elif axis == "household":
        hs = q["household_size_band"]
        if hs == "1명":
            a["household_size_band"] = "4명 이상"
        elif hs == "4명 이상":
            a["household_size_band"] = "1명"
        else:
            return None
        if q["age_band"] == "10대" and a["household_size_band"] == "1명":
            return None
    elif axis == "occupation":
        # Only directions that cannot both be true: a person who does not work ↔ a person who goes
        # to an office. Weekend housework is not used, because office workers do it too (the first
        # generation produced non-contradictory fields in the homemaker and student directions —
        # a human read them and filtered them out).
        if q["age_band"] in ("10대", "70대 이상"):
            return None
        occ = q["occupation"]
        if occ in ("무직/은퇴", "전업주부"):
            a["occupation"] = OFFICE
        elif occ in (OFFICE, "자영업 및 프리랜서") and q["age_band"] in ("40대", "50대", "60대"):
            a["occupation"] = "무직/은퇴"
        else:
            return None
    elif axis == "seoul_years":
        sy = q["seoul_years_band"]
        if sy == "20년 이상" and q["age_band"] not in ("10대",):
            a["seoul_years_band"] = "5년 미만"
        elif sy == "5년 미만":
            a["seoul_years_band"] = "20년 이상"
        else:
            return None
    return a


# --------------------------------------------------------------------------- revealed?


def _has(p: str, t: str) -> bool:
    return re.search(p, t) is not None


def evident(axis: str, q: dict, a: dict, text: str) -> bool:
    """Does the changed axis actually show in the new field (and, for T3, without copying the
    changed value's wording)?
    """
    if axis == "age":
        return (
            a["age_band"].replace(" 이상", "") in text
            and q["age_band"].replace(" 이상", "") not in text
        )
    if axis == "attend":
        if a["attend_any"] == "아니오":
            return _has(
                r"(관람하지 않|보지 않았|찾지 않았|발길이 끊|가지 않았|관람 경험이 없)", text
            )
        g = next(k for k in GENRE_ANCHORS if a[k] == "예")
        return _has(_GENRE_WORDS[g.replace("attend_", "")], text)
    if axis == "genre":
        g = a["_added_genre"].replace("attend_", "")
        return _has(_GENRE_WORDS[g], text)
    if axis == "spend":
        v = a["paid_spend_band"]
        nums = re.findall(r"\d+", v)
        own = re.findall(r"\d+", q["paid_spend_band"])
        hit = any(f"{n}만" in text for n in nums)
        clash = any(f"{n}만" in text for n in own if n not in nums)
        return hit and not clash
    if axis == "companion":
        pat = {
            "혼자서": r"(혼자|홀로)",
            "가족(친척 포함)과 함께": r"(가족|배우자|아내|남편|자녀|아이|부모)",
            "친구(연인 포함)와 함께": r"(친구|연인|애인)",
        }
        return _has(pat[a["viewing_companion"]], text) and not _has(
            pat[q["viewing_companion"]], text
        )
    if axis == "leisure":
        if a["leisure_sufficiency"] == "부족":
            return _has(r"(부족|빠듯|쪼개|틈이 없|시간이 없|쉴 틈)", text)
        return _has(r"(충분|넉넉|여유롭|여유가 있|여유 있)", text)
    if axis == "household":
        if "1명" in text or "4명" in text or "가구원" in text:
            return False
        if a["household_size_band"] == "1명":
            return _has(r"(혼자 살|홀로 살|혼자 지내|1인 가구|독립해 살|자취)", text)
        return (
            len(re.findall(r"(배우자|아내|남편|자녀|아이들|아들|딸|부모님|어머니|아버지)", text))
            >= 2
        )
    if axis == "occupation":
        if a["occupation"] in text:
            return False
        if a["occupation"] == OFFICE:
            return _has(r"(회사|직장|출근|퇴근|사무실|근무)", text)
        return _has(r"(은퇴|퇴직|일을 그만|현역에서 물러|일을 하지 않)", text)
    if axis == "age_life":
        if "70대" in text or "노년" in text:
            return False
        return _has(r"(손주|손자|손녀|은퇴|출가한|자녀들이 모두 독립|장성한 자녀)", text)
    if axis == "seoul_years":
        if a["seoul_years_band"] in text:
            return False
        if a["seoul_years_band"] == "5년 미만":
            return _has(
                r"(이사 온 지|이주한 지|옮겨 온 지|얼마 되지 않|아직 낯|익숙하지 않|새로 정착)",
                text,
            )
        return _has(r"(오래|수십 년|토박이|평생|오랜 세월|수년째|오랫동안)", text)
    raise ValueError(axis)


# --------------------------------------------------------------------------- generate


def generate(client, model: str, a: dict) -> dict | None:
    q = {k: v for k, v in a.items() if not k.startswith("_")}
    res = client.chat(
        model=model,
        system=None,
        user=render_prompt(q),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    try:
        narr = parse_narrative(res.text)
    except Exception:  # noqa: BLE001
        return None
    if any(i.severity == "error" for i in check_all(q, narr)):
        return None  # use only a narrative that passes production checks for A′ (same conditions)
    return narr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show one A′ per axis and stop")
    ap.add_argument(
        "--practice",
        action="store_true",
        help="3 practice traps (T1, T2, T3) — kept disjoint from the main traps",
    )
    args = ap.parse_args()
    out = OUT_PRACTICE if args.practice else OUT
    plan = PLAN_PRACTICE if args.practice else PLAN

    key = pd.read_csv(settings.outputs_dir / "anno" / "answer_key.csv", dtype=str)
    exclude = set(key.quant_id.dropna()) | set(
        key.donor_quant_id.dropna()
    )  # the whole v1 sample (clean and flagged stay as they are)
    if args.practice and OUT.exists():
        exclude |= {
            json.loads(line)["quant_id"] for line in OUT.open(encoding="utf-8") if line.strip()
        }
    rng = np.random.default_rng(SEED + (1 if args.practice else 0))
    banks = {
        w: [r for r in load_bank(w) if is_clean(r) and r["quant_id"] not in exclude] for w in WAVES
    }
    orders = {w: [banks[w][i] for i in rng.permutation(len(banks[w]))] for w in WAVES}
    used: set[str] = set()

    done: list[dict] = []
    if out.exists() and not args.dry_run:
        done = [json.loads(line) for line in out.open(encoding="utf-8") if line.strip()]
        used |= {d["quant_id"] for d in done}
        log.info("resuming: %d already present", len(done))

    model = settings.llm_default_model
    client = None if args.dry_run else get_client()
    slot = 0
    for tier, axis, field, n in plan:
        have = sum(1 for d in done if d["axis"] == axis)
        for k in range(have, n):
            w = WAVES[(slot + k) % len(WAVES)]
            made = False
            for r in orders[w]:
                if r["quant_id"] in used:
                    continue
                a = alter(r["quant"], axis, rng)
                if a is None:
                    continue
                if args.dry_run:
                    diff = {
                        x: (r["quant"].get(x), a.get(x)) for x in a if r["quant"].get(x) != a.get(x)
                    }
                    log.info("%s %s %s: %s", tier, axis, field, diff)
                    made = True
                    break
                used.add(r["quant_id"])
                for t in range(TRIES_PER_RECIPIENT):
                    narr = generate(client, model, a)
                    if narr is None or not evident(axis, r["quant"], a, narr[field]):
                        continue
                    spliced = dict(r["narrative"])
                    spliced[field] = narr[field]
                    rec = {
                        "wave": w,
                        "quant_id": r["quant_id"],
                        "tier": tier,
                        "axis": axis,
                        "field": field,
                        "from": {
                            x: r["quant"].get(x)
                            for x in a
                            if not x.startswith("_") and r["quant"].get(x) != a.get(x)
                        },
                        "to": {
                            x: a.get(x)
                            for x in a
                            if not x.startswith("_") and r["quant"].get(x) != a.get(x)
                        },
                        "original_field": r["narrative"][field],
                        "new_field": narr[field],
                        "quant": r["quant"],
                        "narrative": spliced,
                        "validator_checks": sorted(
                            {i.check for i in check_all(r["quant"], spliced)}
                        ),
                        "model": model,
                        "temperature": TEMPERATURE,
                        "tries": t + 1,
                    }
                    with out.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.append(rec)
                    log.info(
                        "%s %s #%d (%s, try %d): %s → %s",
                        tier,
                        axis,
                        k + 1,
                        w,
                        t + 1,
                        rec["from"],
                        rec["to"],
                    )
                    made = True
                    break
                if made:
                    break
            if not made:
                raise SystemExit(f"{axis}: could not build one")
        slot += n
    if args.dry_run:
        return 0
    log.info(
        "traps v2: %d · caught by the validator %d",
        len(done),
        sum(1 for d in done if d["validator_checks"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
