"""Build the human-audit sample and trap items (ADR-023, the registered analysis plan).

The sample has four strata. Sizes were set by `scripts/human_audit_sample_size.py`.

  clean     Personas the validator did not flag. 81 per wave, allocated across 8 cells
            (4 age groups × attendance) in proportion to the bank. The **primary estimation
            stratum**, standing for 99.6% of the bank.
  flagged   46 personas the validator marked warning or error (proportional to wave). They are
            0.4% of the bank but oversampled on purpose — that many are needed to measure whether
            validator warnings match human judgement (precision).
  trap      48 personas with a planted contradiction — **v2 (2026-09-23, before first scoring)**.
            Rewrite the narrative of A′, a clean persona A with one anchor axis changed, using the
            same prompt and model as production, then splice **only the one narrative field** where
            that axis shows into A's original narrative (`anno_make_traps.py`). Difficulty tiers
            T1 explicit, T2 degree, T3 inference, 16 each. v1 (A's anchors + another person B's
            narrative wholesale) mismatched about 7 of 14 anchor cells on average and was too easy.
  trap_gross  6 wholesale replacements in the v1 style (2 per wave). **Manipulation check** —
            only whether raters applied the rubric.
  practice  12 practice items. Excluded from analysis. 9 boundary cases (hard cases that are not
            contradictions) + 3 local-edit traps (T1–T3). On save, a written key and explanation
            are shown (data_private/practice_explain.json; no oral briefing).

**Stratum is not shown to raters.** Full items are written to
`annotation/data_private/items_full.json`, and `npm run build:data` in annotation/ builds the
public file `public/data/items.json` (masked narrative, anchors and source sealed). Neither file
contains stratum, trap status, or validator results. The answer key lives only in
`outputs/anno/answer_key.csv` and stays gitignored until adjudication ends. Traps also go through
stage 1 (guessing from the masked narrative) — skipping it would reveal that they are traps.

Stage-1 masked narratives use the **same rule** as `era_probes.py` (strip anchor-value strings
that appear in the bank, and genre names). Humans and the classifier then see the same text, so
the two results can be placed side by side.

Run: uv run python scripts/anno_build_sample.py
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era_probes import genre_terms

from sfac_persona.config import settings
from sfac_persona.narrative import FIELDS
from sfac_persona.schema import ANCHOR_BY_NAME, ERA_CORE_ANCHORS, GENRE_ANCHORS
from sfac_persona.validators.consistency import check_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_sample")

SEED = 20260923
WAVES = (2020, 2022, 2024)
N_CLEAN_PER_WAVE = 81
N_FLAGGED = 46
N_TRAP_PER_TYPE = 4  # v1 (unused — kept for the record)
N_GROSS_PER_WAVE = 2  # manipulation-check wholesale replacement: attend 1 + age 1
N_PRACTICE_BOUNDARY = 9
TRAPS_V2 = settings.outputs_dir / "anno" / "traps_v2.jsonl"
TRAPS_V2_PRACTICE = settings.outputs_dir / "anno" / "traps_v2_practice.jsonl"
ID_SEED = (
    SEED + 555
)  # item codes are seeded too (reproducibility). v1 codes are unused — plaintext remains
# in old commits.

# Practice boundary cases: (name, condition) — one pass over hard cases that are **not**
# contradictions.
BOUNDARY = [
    (
        "online_nonattender",
        lambda q, n: (
            q["attend_any"] == "아니오"
            and re.search(r"(유튜브|OTT|스트리밍|온라인)", n["online_consumption"])
        ),
    ),
    (
        "past_experience",
        lambda q, n: (
            q["attend_any"] == "아니오"
            and re.search(r"(예전|과거|젊었을|학창|어릴 적|어렸을)", n["cultural_taste"])
        ),
    ),
    (
        "interest_unattended",
        lambda q, n: (
            q["attend_any"] == "예"
            and re.search(r"(관심|가 보고 싶|언젠가|기회가 되면)", n["cultural_taste"])
        ),
    ),
    (
        "companion_frequency",
        lambda q, n: (
            q["viewing_companion"] in ("가족(친척 포함)과 함께", "친구(연인 포함)와 함께")
            and re.search(r"(가끔|때때로|종종|이따금)", n["companions"])
        ),
    ),
    (
        "household_apart",
        lambda q, n: (
            q["household_size_band"] == "1명"
            and re.search(r"(따로|독립|떨어져|멀리 사는)", n["family_context"])
        ),
    ),
    (
        "leisure_vs_viewing",
        lambda q, n: (
            q["attend_any"] == "예"
            and q["leisure_companion"] == "혼자서"
            and q["viewing_companion"] in ("가족(친척 포함)과 함께", "친구(연인 포함)와 함께")
        ),
    ),
    (
        "movie_only",
        lambda q, n: (
            q["attend_any"] == "예"
            and q.get("attend_movie") == "예"
            and str(q.get("n_genres_band")) == "1"
        ),
    ),
    (
        "meta_utterance",
        lambda q, n: any(re.search(r"(정량 정보|명시되지|설문에서)", v) for v in n.values()),
    ),
    ("plain", lambda q, n: True),
]
N_PRACTICE = 10
RATERS = ("R1", "R2")
AGE_ORDER = ["10대", "20대", "30대", "40대", "50대", "60대", "70대 이상"]
AGE_GROUP = {"10대": 0, "20대": 0, "30대": 1, "40대": 1, "50대": 2, "60대": 2, "70대 이상": 3}
SEX_SYNONYMS = ["남자", "여자", "남성", "여성"]

WEB_DATA = (
    ROOT / "annotation" / "public" / "data"
)  # audit-tool repo (GitHub Pages) — order files only
WEB_PRIVATE = (
    ROOT / "annotation" / "data_private"
)  # full source — build_public.mjs seals it before publication
KEY_DIR = settings.outputs_dir / "anno"


# --------------------------------------------------------------------------- load


def load_bank(wave: int) -> list[dict]:
    p = settings.data_root / "synthetic" / f"era_{wave}_narrative" / "personas.jsonl"
    rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    for r in rows:
        r["_wave"] = wave
    return rows


def is_clean(r: dict) -> bool:
    return bool(r.get("ok")) and not r.get("issues")


def genres_of(q: dict) -> set[str]:
    return {g for g in GENRE_ANCHORS if str(q.get(g)) == "예"}


# --------------------------------------------------------------------------- masking


def mask_pattern(banks: list[list[dict]]) -> re.Pattern[str]:
    vals: set[str] = set()
    for rows in banks:
        for r in rows:
            for v in r["quant"].values():
                if isinstance(v, str) and len(v) >= 2 and v not in ("예", "아니오"):
                    vals.add(v)
    terms = sorted(vals | set(genre_terms()) | set(SEX_SYNONYMS), key=len, reverse=True)
    return re.compile("|".join(re.escape(t) for t in terms))


MASK_HUMAN = "○○"


def masked(narr: dict, pat: re.Pattern[str]) -> dict:
    """Strip the same vocabulary as era_probes. The classifier blanks the span (so the mask count
    is not a feature), but a human cannot read a sentence left as blanks only. Humans therefore
    see "○○". What was removed (which neighbourhood, which age band) is not revealed. The number
    of masked slots can cue the attendance guess, so that guess is descriptive only in the
    preregistration.
    """
    return {
        k: re.sub(r"(○○\s*)+", MASK_HUMAN + " ", pat.sub(MASK_HUMAN, v)).strip()
        for k, v in narr.items()
    }


# --------------------------------------------------------------------------- stratum draws


def stratified_clean(pool: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """8 cells of 4 age groups × attendance, proportional to the bank (largest-remainder)."""
    cells: dict[tuple[int, str], list[dict]] = {}
    for r in pool:
        key = (AGE_GROUP[r["quant"]["age_band"]], str(r["quant"]["attend_any"]))
        cells.setdefault(key, []).append(r)
    keys = sorted(cells)
    sizes = np.array([len(cells[k]) for k in keys], dtype=float)
    raw = sizes / sizes.sum() * n
    alloc = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - alloc))[: n - alloc.sum()]:
        alloc[i] += 1
    out: list[dict] = []
    for k, a in zip(keys, alloc, strict=True):
        idx = rng.choice(len(cells[k]), size=int(a), replace=False)
        out += [cells[k][i] for i in idx]
    return out


def find_donor(recipient: dict, pool: list[dict], kind: str, used: set[str]) -> dict | None:
    """A donor that differs on only the axis being swapped. The rest of the demographics match."""
    q = recipient["quant"]
    best, best_score = None, -1
    for d in pool:
        if d["quant_id"] in used or d["quant_id"] == recipient["quant_id"]:
            continue
        p = d["quant"]
        if kind == "attend":
            same = all(p[a] == q[a] for a in ("sex", "age_band", "living_area"))
            if same and p["attend_any"] != q["attend_any"]:
                return d
        elif kind == "genre":
            if str(p["attend_any"]) != "예" or str(q["attend_any"]) != "예":
                continue
            if not all(p[a] == q[a] for a in ("sex", "age_band", "living_area")):
                continue
            diff = len(genres_of(p) ^ genres_of(q))
            if diff >= 3 and diff > best_score:
                best, best_score = d, diff
        elif kind == "age":
            same = all(p[a] == q[a] for a in ("sex", "living_area", "attend_any"))
            gap = abs(AGE_ORDER.index(p["age_band"]) - AGE_ORDER.index(q["age_band"]))
            if same and gap >= 2 and gap > best_score:
                best, best_score = d, gap
    return best


def make_traps(
    pool: list[dict],
    wave: int,
    rng: np.random.Generator,
    used: set[str],
    plan: list[tuple[str, str | None]] | None = None,
) -> list[dict]:
    traps: list[dict] = []
    order = [pool[i] for i in rng.permutation(len(pool))]
    plan = plan or (
        [("attend", "아니오")] * 2
        + [("attend", "예")] * 2
        + [("genre", "예")] * 4
        + [("age", None)] * 4
    )
    for kind, recipient_attend in plan:
        for r in order:
            if r["quant_id"] in used:
                continue
            if recipient_attend and str(r["quant"]["attend_any"]) != recipient_attend:
                continue
            donor = find_donor(r, pool, kind, used)
            if donor is None:
                continue
            used.update({r["quant_id"], donor["quant_id"]})
            issues = check_all(r["quant"], donor["narrative"])
            traps.append(
                {
                    "recipient": r,
                    "donor": donor,
                    "trap_type": kind,
                    "wave": wave,
                    "validator_flags": [i.check for i in issues],
                }
            )
            break
        else:
            raise SystemExit(f"wave {wave}: no donor found for a {kind} trap")
    return traps


# --------------------------------------------------------------------------- display


def anchor_card(q: dict) -> list[dict]:
    card = []
    for a in ERA_CORE_ANCHORS:
        if a in GENRE_ANCHORS or a not in q:
            continue
        spec = ANCHOR_BY_NAME.get(a)
        card.append({"label": spec.label_ko if spec else a, "value": str(q[a])})
    return card


def genre_lists(q: dict) -> tuple[list[str], list[str]]:
    def name(g: str) -> str:
        return ANCHOR_BY_NAME[g].label_ko.replace(" 관람", "")

    yes = [name(g) for g in GENRE_ANCHORS if str(q.get(g)) == "예"]
    no = [name(g) for g in GENRE_ANCHORS if str(q.get(g)) == "아니오"]
    return yes, no


def item_payload(
    item_id: str, quant: dict, narr: dict, pat: re.Pattern[str], practice: bool
) -> dict:
    yes, no = genre_lists(quant)
    fields = [{"key": f.key, "label": f.label_ko} for f in FIELDS]
    return {
        "item_id": item_id,
        "practice": practice,
        "anchors": anchor_card(quant),
        "genres_attended": yes,
        "genres_not": no,
        "fields": fields,
        "narrative": {f.key: narr.get(f.key, "") for f in FIELDS},
        "masked": masked({f.key: narr.get(f.key, "") for f in FIELDS}, pat),
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force", action="store_true", help="overwrite the preregistered sample (do not)"
    )
    args = ap.parse_args()
    if (KEY_DIR / "answer_key.csv").exists() and not args.force:
        raise SystemExit(
            "sample already exists (preregistration e8ad295). Item codes are random, so redrawing "
            "would desync the answer key and the scores. Use --force only to draw anew."
        )
    rng = np.random.default_rng(SEED)
    banks = {w: load_bank(w) for w in WAVES}
    pat = mask_pattern(list(banks.values()))

    items, key_rows = [], []
    used: set[str] = set()

    id_rng = np.random.default_rng(ID_SEED)

    def new_id() -> str:
        while True:
            c = "".join(id_rng.choice(list("0123456789ABCDEF"), 6))
            if c not in {i["item_id"] for i in items}:
                return c

    def add(stratum: str, quant: dict, narr: dict, meta: dict, practice: bool = False) -> None:
        iid = new_id()
        items.append(item_payload(iid, quant, narr, pat, practice))
        key_rows.append({"item_id": iid, "stratum": stratum, **meta})

    # clean
    for w in WAVES:
        pool = [r for r in banks[w] if is_clean(r)]
        for r in stratified_clean(pool, N_CLEAN_PER_WAVE, rng):
            used.add(r["quant_id"])
            q = r["quant"]
            add(
                "clean",
                q,
                r["narrative"],
                {
                    "wave": w,
                    "quant_id": r["quant_id"],
                    "donor_quant_id": "",
                    "trap_type": "",
                    "validator_checks": "",
                    "true_age_band": q["age_band"],
                    "true_living_area": q["living_area"],
                    "true_attend": q["attend_any"],
                },
            )

    # flagged — proportional to wave
    flagged = {w: [r for r in banks[w] if r.get("issues")] for w in WAVES}
    tot = sum(len(v) for v in flagged.values())
    raw = np.array([len(flagged[w]) / tot * N_FLAGGED for w in WAVES])
    alloc = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - alloc))[: N_FLAGGED - alloc.sum()]:
        alloc[i] += 1
    for w, a in zip(WAVES, alloc, strict=True):
        pick = rng.choice(len(flagged[w]), size=int(a), replace=False)
        for i in pick:
            r = flagged[w][i]
            used.add(r["quant_id"])
            q = r["quant"]
            checks = ";".join(
                sorted({str(x.get("check")) for x in r["issues"] if isinstance(x, dict)})
            )
            add(
                "flagged",
                q,
                r["narrative"],
                {
                    "wave": w,
                    "quant_id": r["quant_id"],
                    "donor_quant_id": "",
                    "trap_type": "",
                    "validator_checks": checks,
                    "true_age_band": q["age_band"],
                    "true_living_area": q["living_area"],
                    "true_attend": q["attend_any"],
                },
            )

    # traps v2 — local edits (produced by anno_make_traps.py)
    t2 = [json.loads(line) for line in TRAPS_V2.open(encoding="utf-8") if line.strip()]
    assert len(t2) == 48, (
        f"trap v2 has {len(t2)} items (must be 48) — run anno_make_traps.py to completion"
    )
    for t in t2:
        used.add(t["quant_id"])
        q = t["quant"]
        add(
            "trap",
            q,
            t["narrative"],
            {
                "wave": t["wave"],
                "quant_id": t["quant_id"],
                "donor_quant_id": "",
                "trap_type": t["axis"],
                "tier": t["tier"],
                "trap_field": t["field"],
                "trap_from": json.dumps(t["from"], ensure_ascii=False),
                "trap_to": json.dumps(t["to"], ensure_ascii=False),
                "validator_checks": ";".join(t["validator_checks"]),
                "true_age_band": q["age_band"],
                "true_living_area": q["living_area"],
                "true_attend": q["attend_any"],
            },
        )

    # trap_gross — v1-style wholesale replacement, manipulation check (attend 1 + age 1 per wave)
    for w in WAVES:
        pool = [r for r in banks[w] if is_clean(r) and r["quant_id"] not in used]
        for t in make_traps(pool, w, rng, used, plan=[("attend", None), ("age", None)]):
            rq, dq = t["recipient"]["quant"], t["donor"]["quant"]
            add(
                "trap_gross",
                rq,
                t["donor"]["narrative"],
                {
                    "wave": w,
                    "quant_id": t["recipient"]["quant_id"],
                    "donor_quant_id": t["donor"]["quant_id"],
                    "trap_type": t["trap_type"],
                    "tier": "gross",
                    "validator_checks": ";".join(sorted(set(t["validator_flags"]))),
                    "true_age_band": dq["age_band"],
                    "true_living_area": dq["living_area"],
                    "true_attend": dq["attend_any"],
                },
            )

    # practice — excluded from analysis. 9 boundary cases + 3 local-edit traps (T1, T2, T3),
    # rotating across waves.
    pin_path = settings.outputs_dir / "anno" / "practice_pin.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8")) if pin_path.exists() else {}
    for k, (name, cond) in enumerate(BOUNDARY):
        # Once an explanation is written, pin that persona (practice_pin.json — items a human
        # read and wrote a written explanation for).
        if name in pin:
            r = next(x for w_ in WAVES for x in banks[w_] if x["quant_id"] == pin[name])
            w = r["_wave"]
            used.add(r["quant_id"])
            q = r["quant"]
            add(
                "practice",
                q,
                r["narrative"],
                {
                    "wave": w,
                    "quant_id": r["quant_id"],
                    "donor_quant_id": "",
                    "trap_type": "",
                    "tier": "",
                    "practice_case": name,
                    "validator_checks": "",
                    "true_age_band": q["age_band"],
                    "true_living_area": q["living_area"],
                    "true_attend": q["attend_any"],
                },
                practice=True,
            )
            continue
        # If the assigned wave has no candidate, try the next (some boundary cases are rare
        # in a wave)
        for dw in range(len(WAVES)):
            w = WAVES[(k + dw) % len(WAVES)]
            pool = [r for r in banks[w] if is_clean(r) and r["quant_id"] not in used]
            cands = [r for r in pool if cond(r["quant"], r["narrative"])]
            if cands:
                break
        if not cands:
            raise SystemExit(f"practice boundary case {name}: no candidates")
        log.info("practice %s: %d candidates (%s)", name, len(cands), w)
        r = cands[int(rng.integers(len(cands)))]
        used.add(r["quant_id"])
        q = r["quant"]
        add(
            "practice",
            q,
            r["narrative"],
            {
                "wave": w,
                "quant_id": r["quant_id"],
                "donor_quant_id": "",
                "trap_type": "",
                "tier": "",
                "practice_case": name,
                "validator_checks": "",
                "true_age_band": q["age_band"],
                "true_living_area": q["living_area"],
                "true_attend": q["attend_any"],
            },
            practice=True,
        )
    tp = [json.loads(line) for line in TRAPS_V2_PRACTICE.open(encoding="utf-8") if line.strip()]
    assert len(tp) == 3, f"practice traps: {len(tp)} (want 3) — anno_make_traps.py --practice"
    for t in tp:
        q = t["quant"]
        add(
            "practice",
            q,
            t["narrative"],
            {
                "wave": t["wave"],
                "quant_id": t["quant_id"],
                "donor_quant_id": "",
                "trap_type": t["axis"],
                "tier": t["tier"],
                "practice_case": f"trap_{t['tier']}",
                "trap_field": t["field"],
                "trap_from": json.dumps(t["from"], ensure_ascii=False),
                "trap_to": json.dumps(t["to"], ensure_ascii=False),
                "validator_checks": ";".join(t["validator_checks"]),
                "true_age_band": q["age_band"],
                "true_living_area": q["living_area"],
                "true_attend": q["attend_any"],
            },
            practice=True,
        )

    key = pd.DataFrame(key_rows)
    counts = key.stratum.value_counts().to_dict()
    log.info("strata: %s", counts)
    assert counts == {"clean": 243, "flagged": 46, "trap": 48, "trap_gross": 6, "practice": 12}, (
        counts
    )

    # Per-rater order: practice first (same order), then main items shuffled with a
    # rater-specific seed.
    practice_ids = key[key.stratum == "practice"].item_id.tolist()
    practice_ids = [
        practice_ids[i] for i in np.random.default_rng(SEED + 11).permutation(len(practice_ids))
    ]
    main_ids = key[key.stratum != "practice"].item_id.tolist()
    orders = {}
    for j, rater in enumerate(RATERS):
        r_rng = np.random.default_rng(SEED + 100 + j)
        orders[rater] = practice_ids + [main_ids[i] for i in r_rng.permutation(len(main_ids))]

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    WEB_PRIVATE.mkdir(parents=True, exist_ok=True)
    # Array order must not reveal stratum (clean→flagged→trap would show trap positions from the
    # public file alone).
    shuffled = [items[i] for i in np.random.default_rng(SEED + 7).permutation(len(items))]
    (WEB_PRIVATE / "items_full.json").write_text(
        json.dumps(shuffled, ensure_ascii=False), encoding="utf-8"
    )
    for rater, order in orders.items():
        (WEB_DATA / f"order_{rater}.json").write_text(json.dumps(order), encoding="utf-8")

    KEY_DIR.mkdir(parents=True, exist_ok=True)
    key.to_csv(KEY_DIR / "answer_key.csv", index=False)
    raters_file = KEY_DIR / "raters.json"
    if not raters_file.exists():
        # Codes for the local test server. Deployed codes are created separately by
        # spreadsheet setup().
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        codes = {
            r: "".join(secrets.choice(alphabet) for _ in range(12))
            for r in (*RATERS, "ADJ", "OWNER")
        }
        raters_file.write_text(json.dumps(codes, indent=2), encoding="utf-8")
        log.info("created new local access codes: %s", raters_file)

    trap = key[key.stratum == "trap"]
    log.info(
        "traps v2: %s · traps caught by the validator %d/%d",
        trap.tier.value_counts().to_dict(),
        int((trap.validator_checks != "").sum()),
        len(trap),
    )
    log.info(
        "wrote %s (%d items) — next: npm run build:data in annotation/",
        WEB_PRIVATE / "items_full.json",
        len(items),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
