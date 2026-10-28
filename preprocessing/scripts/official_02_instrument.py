"""Fingerprint the attendance battery wave by wave, and derive the regimes from the
data rather than declaring them.

The series split this repository used before (2014-2016 against 2018-2024) was settled
by reading the questionnaire text in the report PDFs word by word. The arrival of the
official codebooks makes it possible to redo that judgement from the data itself. Here
the regimes are **computed, not declared**:

  1. Find the battery's variables in each wave (total count, self-paid count, annual
     spending, attendance).
  2. Map each choice's codebook label to a canonical item, keeping the label verbatim
     as evidence.
  3. Take (item set, item order, sub-measures, response format) as the fingerprint.
  4. Group waves that share a fingerprint into a regime.

Outputs:
  official_item_battery.csv    wave x choice position x canonical item x variable x
                               codebook label
  official_instrument.csv      wave x fingerprint components x regime_id
  official_regime_summary.csv  regime x waves x item set x why it differs
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

# The canonical item vocabulary, with names shared across waves. The
# values are the instrument's own Korean wording, glossed in English alongside.
CANON = {
    "exhibition": "미술관·전시회 관람",            # art museum / exhibition
    "museum": "박물관 관람",                      # museum
    "western_classical": "음악공연 관람 (클래식·오페라 등)",  # classical concert, opera
    "dance_performance": "무용공연 관람",          # dance performance
    "traditional": "전통예술공연 관람",            # traditional performing arts
    "theater": "연극공연 관람 (뮤지컬 포함)",      # play, musicals included
    "literature_event": "문학행사 관람",           # literary event
    "movie": "극장에서 영화 관람",                 # film, in a cinema
    "popular_performance": "대중공연 관람",        # popular music concert
    "outdoor_event": "축제·야외문화행사 관람",     # festival / outdoor cultural event
    "music_dance_combined": "음악·무용 발표회 관람 (합산 보기)",  # music and dance, one combined choice
    "sports_event": "스포츠 경기 관람",            # sporting event
}

# The battery as declared per wave: (choice position, canonical) and the variable
# pattern for each sub-measure. The position is the order of the choice on the
# questionnaire -- a central axis of the regime judgement, so it is stated explicitly.
BATTERY = {
    2014: {
        "group": "mixed",
        "items": [
            (1, "exhibition"), (2, "museum"), (3, "music_dance_combined"),
            (4, "traditional"), (5, "theater"), (6, "movie"),
            (7, "popular_performance"), (8, "outdoor_event"), (9, "sports_event"),
        ],
        "measures": {"total_count": "QA5A{i}", "self_paid_count": "QA5B{i}", "annual_cost": "QA5C{i}"},
        "response_format": "count_grid",
    },
    2015: {
        "group": "mixed",
        "items": [
            (1, "exhibition"), (2, "museum"), (3, "music_dance_combined"),
            (4, "traditional"), (5, "theater"), (6, "movie"),
            (7, "popular_performance"), (8, "outdoor_event"),
        ],
        "measures": {"total_count": "B2_A{i}", "self_paid_count": "B2_B{i}", "annual_cost": "B2_C{i}"},
        "response_format": "count_grid",
    },
    2016: {
        "group": "mixed",
        "items": [
            (1, "exhibition"), (2, "museum"), (3, "western_classical"), (4, "dance_performance"),
            (5, "traditional"), (6, "theater"), (7, "literature_event"), (8, "movie"),
            (9, "popular_performance"), (10, "outdoor_event"),
        ],
        "measures": {"total_count": "B1_1_{i}", "self_paid_count": "B1_2_{i}", "annual_cost": "B1_3_{i}"},
        "response_format": "count_grid",
    },
    2018: {
        "group": "general",
        "items": [
            (1, "exhibition"), (2, "museum"), (3, "western_classical"), (4, "dance_performance"),
            (5, "traditional"), (6, "theater"), (7, "literature_event"), (8, "movie"),
            (9, "popular_performance"), (10, "outdoor_event"),
        ],
        "measures": {
            "total_count": "B1_1_{i}#1", "self_paid_count": "B1_1_{i}#2",
            "annual_cost": "B1_1_{i}#3m", "attended": "B1a_m{i:02d}a",
        },
        "response_format": "count_grid",
    },
    2020: {
        "group": "general",
        "items": [
            (1, "exhibition"), (2, "museum"), (3, "western_classical"), (4, "dance_performance"),
            (5, "traditional"), (6, "theater"), (7, "literature_event"), (8, "movie"),
            (9, "popular_performance"), (10, "outdoor_event"),
        ],
        "measures": {"total_count": "B1A_{i}", "self_paid_count": "B1B_{i}", "annual_cost": "B1C_{i}"},
        "response_format": "count_grid",
    },
    2022: {
        "group": "mixed",
        "items": [
            (1, "exhibition"), (2, "museum"), (3, "western_classical"), (4, "popular_performance"),
            (5, "dance_performance"), (6, "traditional"), (7, "theater"), (8, "literature_event"),
            (9, "movie"), (10, "outdoor_event"),
        ],
        "measures": {"total_count": "B1_1x{i}", "self_paid_count": "B1_2x{i}", "annual_cost": "B1_3x{i}"},
        "response_format": "count_grid",
    },
    # 2024 keeps the 2022 choice order (checked against the B1_1_* value labels:
    # position 4 is the popular-music concert). Attendance is not a per-item variable
    # here but a choice code sitting in the multiple-response slots B1_1_1..B1_1_11.
    2024: {
        "group": "general",
        "items": [
            (1, "exhibition"), (2, "museum"), (3, "western_classical"), (4, "popular_performance"),
            (5, "dance_performance"), (6, "traditional"), (7, "theater"), (8, "literature_event"),
            (9, "movie"), (10, "outdoor_event"),
        ],
        "measures": {
            "total_count": "B1_2_{i}_1", "self_paid_count": "B1_2_{i}_2", "annual_cost": "B1_2_{i}_3",
        },
        "multi_response_slots": [f"B1_1_{k}" for k in range(1, 12)],
        "response_format": "multi_select_then_count",
    },
}


# Within one wave, a separate file per group carries different variable names.
# The 2024 disabled file has one level less of measure prefix (B1_2_i / B1_3_i /
# B1_4_i). Its choice order matches the 2024 general file (checked against the SPSS
# value labels).
GROUP_VARIANTS = {
    (2024, "disabled"): {
        "total_count": "B1_2_{i}", "self_paid_count": "B1_3_{i}", "annual_cost": "B1_4_{i}",
    },
}

# Every (wave, group) that carries the attendance battery. Samples outside the
# general public are folded in too.
ALL_GROUPS = [
    (2014, "mixed"), (2015, "mixed"), (2016, "mixed"),
    (2018, "general"), (2018, "interest"),
    (2020, "general"), (2020, "interest"),
    (2022, "mixed"),
    (2024, "general"), (2024, "interest"), (2024, "disabled"),
]


def battery_for(wave: int, group: str) -> dict:
    """The battery spec for a (wave, group). A group variant swaps in its measures."""
    spec = dict(BATTERY[wave])
    var = GROUP_VARIANTS.get((wave, group))
    if var:
        spec["measures"] = var
        spec = {k: v for k, v in spec.items() if k != "multi_response_slots"} | (
            {"multi_response_slots": BATTERY[wave]["multi_response_slots"]}
            if "multi_response_slots" in BATTERY[wave] else {})
    spec["group"] = group
    return spec


def label_lookup() -> dict:
    v = pd.read_csv(C.OUT / "official_variable_dictionary.csv")
    return {
        (int(r.wave), str(r.group), str(r.variable)): ("" if pd.isna(r.label) else str(r.label))
        for r in v.itertuples()
    }


def main():
    labels = label_lookup()
    rows = []
    for wave, grp in ALL_GROUPS:
        spec = battery_for(wave, grp)
        df, _m = C.load(wave, grp)
        cols = set(df.columns)
        for pos, canon in spec["items"]:
            for measure, pat in spec["measures"].items():
                var = pat.format(i=pos)
                rows.append(
                    dict(
                        wave=wave, group=grp, position=pos, canonical_item=canon,
                        measure=measure, variable=var, in_data=var in cols,
                        codebook_label=labels.get((wave, grp, var), ""),
                        response_format=spec["response_format"],
                    )
                )
    bat = pd.DataFrame(rows)

    missing = bat[~bat.in_data]
    if len(missing):
        print("!! battery variables absent from the data:")
        for r in missing.itertuples():
            print(f"   {r.wave} {r.measure:16s} {r.variable}")

    # ---- compute the fingerprints --------------------------------------------
    fp = []
    for wave, spec in BATTERY.items():
        items = spec["items"]
        item_set = tuple(sorted(c for _, c in items))
        item_order = tuple(c for _, c in items)
        measures = tuple(sorted(spec["measures"]))
        present = bat[(bat.wave == wave) & bat.in_data]
        fp.append(
            dict(
                wave=wave,
                n_items=len(items),
                item_set="|".join(item_set),
                item_order="|".join(item_order),
                measures="|".join(measures),
                response_format=spec["response_format"],
                battery_vars_present=int(len(present)),
                set_hash=hashlib.sha1("|".join(item_set).encode()).hexdigest()[:8],
                order_hash=hashlib.sha1("|".join(item_order).encode()).hexdigest()[:8],
            )
        )
    ins = pd.DataFrame(fp).sort_values("wave").reset_index(drop=True)

    # A regime is the set of waves agreeing on item set, item order and response format
    key = ins[["item_order", "response_format"]].agg("~".join, axis=1)
    codes = {k: i for i, k in enumerate(dict.fromkeys(key))}
    ins["regime_id"] = [f"R{codes[k] + 1}" for k in key]

    # regime summary, plus why each adjacent pair of regimes differs
    summ = []
    for rid, g in ins.groupby("regime_id", sort=True):
        summ.append(
            dict(
                regime_id=rid,
                waves="·".join(str(w) for w in sorted(g.wave)),
                n_items=int(g.n_items.iloc[0]),
                item_order=g.item_order.iloc[0],
                response_format=g.response_format.iloc[0],
            )
        )
    summ = pd.DataFrame(summ)

    diffs = []
    ordered = ins.sort_values("wave")
    prev = None
    for r in ordered.itertuples():
        if prev is not None and prev.regime_id != r.regime_id:
            a, b = set(prev.item_set.split("|")), set(r.item_set.split("|"))
            reasons = []
            if a != b:
                if b - a:
                    reasons.append("items added: " + ", ".join(sorted(b - a)))
                if a - b:
                    reasons.append("items removed: " + ", ".join(sorted(a - b)))
            if a == b and prev.item_order != r.item_order:
                pa, pb = prev.item_order.split("|"), r.item_order.split("|")
                moved = [f"{c} {pa.index(c) + 1}->{pb.index(c) + 1}"
                         for c in pa if pa.index(c) != pb.index(c)]
                reasons.append("choice order moved: " + ", ".join(moved))
            if prev.response_format != r.response_format:
                reasons.append(
                    f"response format: {prev.response_format} -> {r.response_format}")
            diffs.append(
                dict(from_wave=prev.wave, to_wave=r.wave, from_regime=prev.regime_id,
                     to_regime=r.regime_id, reason=" · ".join(reasons))
            )
        prev = r
    diffs = pd.DataFrame(diffs)

    C.OUT.mkdir(parents=True, exist_ok=True)
    bat.to_csv(C.OUT / "official_item_battery.csv", index=False)
    ins.to_csv(C.OUT / "official_instrument.csv", index=False)
    summ.to_csv(C.OUT / "official_regime_summary.csv", index=False)
    diffs.to_csv(C.OUT / "official_regime_breaks.csv", index=False)

    print(f"\n{len(bat)} battery variable rows ({int(bat.in_data.sum())} found in the data)")
    print("\n== fingerprint by wave ==")
    print(ins[["wave", "n_items", "set_hash", "order_hash", "response_format", "regime_id"]].to_string(index=False))
    print("\n== regime ==")
    for r in summ.itertuples():
        print(f"  {r.regime_id}  {r.waves:22s} {r.n_items:2d} items  {r.response_format}")
    print("\n== regime boundaries and why ==")
    for r in diffs.itertuples():
        print(f"  {r.from_wave} -> {r.to_wave}  [{r.from_regime}->{r.to_regime}]  {r.reason}")


if __name__ == "__main__":
    main()
