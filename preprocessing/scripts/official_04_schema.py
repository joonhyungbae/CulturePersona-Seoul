"""One person-level schema -- seven waves folded into a single person x item table.

The aggregate track reads published tables; this track reads one respondent at a time.
Variable names, choice positions and units of measurement differ by wave, so everything
is mapped to the canonical vocabulary before being combined.

Units:
  Annual spending is in KRW for 2014, 2015, 2016 and 2020, and in units of ten
  thousand KRW for 2018, 2022 and 2024. The codebooks record no unit, so this was
  settled from the magnitude of the values and then confirmed against published
  figures: total annual spending per person in 2014 and 2015 matches the published
  value to the won (559,632 and 477,358, official_early_published_check.csv).
  Everything is converted to KRW, and cost_unit_original keeps the source unit as the
  instrument wrote it.

The instrument decides what counts as attendance:
  count_grid               total annual uses > 0
  multi_select_then_count  the choice code appears in a multiple-response slot

Outputs (person level -> data/interim/official/, kept out of the repository):
  official_person_item.parquet    person x item x (attended, total count, self-paid
                                  count, annual spending)
  official_person_master.parquet  person x demographics, frame, weight, summaries
Outputs (aggregate -> data/processed/official/):
  official_schema_coverage.csv  wave x canonical item x measure x n x missing rate
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C
import official_02_instrument as I

# The unit of annual spending, settled from the magnitude of the values (see the
# module docstring). Confirmed by the author on 2026-09-08: that judgement stands.
COST_UNIT = {2014: "원", 2015: "원", 2016: "원", 2018: "만원",
             2020: "원", 2022: "만원", 2024: "만원"}
COST_MULT = {"원": 1, "만원": 10_000}

# Upper bounds on counts and spending. The raw data carries free-entry outliers, so
# anything above the bound is set missing.
MAX_COUNT, MAX_COST = 365, 50_000_000


def num(s):
    return pd.to_numeric(s, errors="coerce")


def build_wave(wave: int, group: str, keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = I.battery_for(wave, group)
    df, _m = C.load(wave, group)
    k = keys[(keys.wave == wave) & (keys.source_group == group)].reset_index(drop=True)
    df = df[df[list(df.columns)[0]].notna()].reset_index(drop=True) if wave == 2016 else df.reset_index(drop=True)
    assert len(df) == len(k), f"{wave} {group}: row counts differ, {len(df)} vs {len(k)}"

    multi = spec["response_format"] == "multi_select_then_count"
    slots = df[spec["multi_response_slots"]].apply(num) if multi else None

    # Separate respondents who were never asked the battery from those who did not
    # attend. In 2014, 795 respondents living outside Seoul were screened out; the 795
    # non-Seoul cases and the 795 with no battery at all are exactly the same people.
    if multi:
        asked = slots.notna().any(axis=1)
    else:
        cnt_cols = [spec["measures"]["total_count"].format(i=pos) for pos, _ in spec["items"]]
        asked = df[cnt_cols].apply(num).notna().any(axis=1)

    recs = []
    for pos, canon in spec["items"]:
        m = spec["measures"]
        tot = num(df[m["total_count"].format(i=pos)]) if "total_count" in m else pd.Series(np.nan, index=df.index)
        paid = num(df[m["self_paid_count"].format(i=pos)]) if "self_paid_count" in m else pd.Series(np.nan, index=df.index)
        cost = num(df[m["annual_cost"].format(i=pos)]) if "annual_cost" in m else pd.Series(np.nan, index=df.index)

        tot = tot.where(tot.between(0, MAX_COUNT))
        paid = paid.where(paid.between(0, MAX_COUNT))
        cost = cost.where(cost.between(0, MAX_COST / COST_MULT[COST_UNIT[wave]]))

        if multi:
            attended = (slots == pos).any(axis=1).astype(float)
            # multiple-response waves do not ask counts of non-attenders -> fill with 0
            tot = tot.where(attended == 1, 0.0)
            paid = paid.where(attended == 1, 0.0)
            cost = cost.where(attended == 1, 0.0)
        else:
            attended = (tot.fillna(0) > 0).astype(float)
        attended = attended.where(asked)          # never asked stays missing
        tot = tot.where(asked)
        paid = paid.where(asked)
        cost = cost.where(asked)

        recs.append(pd.DataFrame({
            "wave": wave, "source_group": group, "row_uid": k.row_uid.values,
            "person_id": k.person_id.values, "position": pos,
            "canonical_item": canon, "attended": attended.values,
            "total_count": tot.values, "self_paid_count": paid.values,
            "annual_cost_krw": (cost * COST_MULT[COST_UNIT[wave]]).values,
            "cost_unit_original": COST_UNIT[wave],
        }))

    item = pd.concat(recs, ignore_index=True)

    master = k.copy()
    master["regime_id"] = REGIME[wave]
    master["n_items_asked"] = len(spec["items"])
    agg = item.groupby("row_uid").agg(
        items_attended=("attended", "sum"),
        items_asked=("attended", "count"),
        total_visits=("total_count", "sum"),
        self_paid_visits=("self_paid_count", "sum"),
        total_cost_krw=("annual_cost_krw", "sum"),
    )
    master = master.merge(agg, left_on="row_uid", right_index=True, how="left")
    master["battery_asked"] = master.items_asked > 0
    master["attended_any"] = np.where(master.battery_asked, (master.items_attended > 0).astype(float), np.nan)
    return item, master


REGIME: dict[int, str] = {}


def main():
    ins = pd.read_csv(C.OUT / "official_instrument.csv")
    REGIME.update(dict(zip(ins.wave, ins.regime_id)))

    keys = C.read_person("official_person_keys")
    items, masters = [], []
    for wave, group in I.ALL_GROUPS:
        it, ms = build_wave(wave, group, keys)
        items.append(it)
        masters.append(ms)

    ITEM = pd.concat(items, ignore_index=True)
    MASTER = pd.concat(masters, ignore_index=True)

    C.PERSON.mkdir(parents=True, exist_ok=True)
    C.write_person(ITEM, "official_person_item")
    C.write_person(MASTER, "official_person_master")

    cov = (ITEM.groupby(["wave", "canonical_item"])
           .agg(n=("attended", "size"),
                attended_obs=("attended", "count"),
                count_obs=("total_count", "count"),
                paid_obs=("self_paid_count", "count"),
                cost_obs=("annual_cost_krw", "count"))
           .reset_index())
    for c in ["attended", "count", "paid", "cost"]:
        cov[f"{c}_missing_pct"] = (100 * (1 - cov[f"{c}_obs"] / cov.n)).round(2)
    cov.to_csv(C.OUT / "official_schema_coverage.csv", index=False)

    print(f"{len(ITEM):,} person-item rows · {len(MASTER):,} people")
    print("\n== people, items and regime by wave ==")
    print(MASTER.groupby(["wave", "source_group", "regime_id"]).agg(
        persons=("person_id", "size"), asked=("battery_asked", "sum"),
        items=("n_items_asked", "first"),
        attended_any=("attended_any", "mean")).round(3).to_string())
    print("\n== missing rate by measure (%) ==")
    p = cov.pivot_table(index="wave", values=["count_missing_pct", "paid_missing_pct", "cost_missing_pct"],
                        aggfunc="mean").round(2)
    print(p.to_string())


if __name__ == "__main__":
    main()
