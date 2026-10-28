"""Harmonise the domains outside the battery: check the declarations against the data
and build the long person x item table.

The attendance battery (official_02) is regular enough to fingerprint automatically.
Outside it the question structure changes wholesale from wave to wave and automatic
matching does not work. So the declarations live in a data file
(`config/domain_declarations.csv`) and **are checked by machine here.**

Why the declaration needs each of these columns -- every one of them has silently
corrupted something in practice:

  universe   who was asked. In 2020 `D1A_*` went not to all 5,095 respondents but only
             to the 812 who said their media-based cultural activity had risen since
             the pandemic. Read as a population share that puts exhibitions at 53.8 per
             cent, while the question asked of everyone (`D4A`) gives 12.8.
  measure    separates `experienced` (a genuine binary) from `threshold_recode` (a
             derived cut of a scale). The 2018 `ZD1_*a` variables cut a four-point
             frequency scale at "occasionally or more", so (ZD1==1) and (D1>=3) agree
             exactly. Filed under the same measure as the other waves' binary
             experience items, 2018 alone would be systematically off.
  polarity   a 1 = yes / 2 = no coding means the larger value is "no" -- the opposite
             direction from a scale item.
  unit       online spending is in KRW in 2020 and in units of ten thousand KRW in
             2022. Combined without conversion they are out by a factor of 10,000.

primary_family stops double counting when one variable belongs to two conceptual
families (2020 `D4B_*` is both online and satisfaction, for instance). The rest are
recorded in `also_in`.

Outputs:
  official_domain_declarations.csv     declarations that pass, with the check columns
  official_domain_family_coverage.csv  wave x family x measure coverage
  official_person_domain.parquet       person x item (person level -> interim, kept
                                       out of the repository)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C
from official_03_samples import KEYS, GROUP_KEYS

# A wave does not have just one raw file -- 2018, 2020 and 2024 split by group.
# Sweeping only one file leaves 10,011 people with no variables outside the battery,
# which is how things stood before 2026-09-09. The list is never written by hand; the
# file register (C.FILES) is the only source.
GROUPS: dict[int, list[str]] = {}
for _w, _g in C.groups():
    GROUPS.setdefault(_w, []).append(_g)

# The representative file for each wave; the declarations were written against it. For
# the other group files it applies **only to variables of the same name** -- the
# culture-interested files for 2018, 2020 and 2024 share nearly all their columns with
# the representative file (25, 16 and 1 unique columns), and only the 2024 disabled
# file uses a substantially different instrument (582 unique columns).
SRC = {w: C.default_group(w) for w in GROUPS}

DECL = C.ROOT / "config" / "domain_declarations.csv"

# Measures whose 1 = yes / 2 = no coding is corrected to 1/0
BINARY_MEASURES = {"experienced", "threshold_recode"}


def slugify(lab: str, k: float) -> str:
    """Turn a choice label into a canonical suffix. The Korean is left as it is; only
    whitespace and punctuation are cleaned up."""
    t = re.sub(r"[\s/·,()\[\]]+", "_", str(lab).strip())
    t = re.sub(r"_+", "_", t).strip("_")
    return t[:40] if t else f"opt{int(k)}"


def load_wave(wave: int, group: str | None = None) -> pd.DataFrame:
    """Filter rows exactly as official_03 does -- any divergence shifts row_uid."""
    group = group or SRC[wave]
    df, _m = C.load(wave, group)
    k = dict(KEYS[wave])
    k.update(GROUP_KEYS.get((wave, group), {}))
    pid = k["pid"]
    return (df[df[pid].notna()] if pid else df).reset_index(drop=True)


def main():
    decl = pd.read_csv(DECL)
    keys = C.read_person("official_person_keys")
    _vl = pd.read_csv(C.OUT / "official_value_labels.csv")
    vlab = {}
    for (w, v), gg in _vl.groupby(["wave", "variable"]):
        vlab[(int(w), v)] = dict(zip(gg.value.astype(str), gg.label.astype(str)))
    _bat = pd.read_csv(C.OUT / "official_item_battery.csv")
    off = set(zip(_bat.wave, _bat.variable))

    checked, records = [], []
    for wave, g in decl.groupby("wave"):
        wave = int(wave)
        # A wave has more than one file. Outside the representative file, only
        # variables of the same name apply.
        for group in GROUPS[wave]:
            primary = group == SRC[wave]
            df = load_wave(wave, group)
            k = keys[(keys.wave == wave) & (keys.source_group == group)].reset_index(drop=True)
            if len(df) != len(k):
                raise AssertionError(
                    f"{wave} {group}: row counts differ, {len(df)} vs {len(k)}")
            cols = set(df.columns)

            # A declaration expanded into several variables whose values are choice
            # codes is a multiple-response slot set. Mistaking slots for items binds
            # one canonical to several variables and breaks the key.
            # (2014 QA7_1 to QA7_5 is not a grid but a slot set -- values 1 to 15 are
            # the choice codes for routes into arts education.)
            slot_groups = {}
            for key, gg in g.groupby(["primary_family", "canonical_item", "measure"]):
                vs = [x for x in gg.variable if x in cols]
                if len(vs) < 2:
                    continue
                # Several variables under one canonical item means a slot set, not a
                # grid: a grid would have had a different canonical declared for each
                # variable. Judging by the values alone would miss cases such as the
                # 2018 C1_2_3#1/#2 slots, whose values are only 1 and 2.
                slot_groups[key] = vs

            for r in g.itertuples():
                exists = r.variable in cols
                overlaps = (int(wave), r.variable) in off
                n_obs = int(df[r.variable].notna().sum()) if exists else 0
                # Every declaration must exist in the representative file (a
                # contract). Absence from another group file is normal and not
                # recorded.
                if not primary and not exists:
                    continue
                row = dict(wave=int(wave), source_group=group, is_primary=primary,
                           variable=r.variable, primary_family=r.primary_family,
                           also_in=r.also_in if isinstance(r.also_in, str) else "",
                           canonical_item=r.canonical_item, measure=r.measure,
                           universe=r.universe, polarity=r.polarity, unit=r.unit,
                           declared_n=r.n_valid, observed_n=n_obs,
                           in_data=exists, overlaps_offline=overlaps,
                           # declared_n was written against the representative file,
                           # so a different count in another group file is normal and
                           # is not compared.
                           n_matches_declared=(abs(n_obs - r.n_valid) <= 1
                                               if primary and exists and pd.notna(r.n_valid)
                                               else None),
                           is_routed=(n_obs < len(df) * 0.95) if exists else None)
                checked.append(row)
                if not exists or overlaps:
                    continue

                gkey = (r.primary_family, r.canonical_item, r.measure)
                if gkey in slot_groups:
                    if r.variable != slot_groups[gkey][0]:
                        continue  # handle each group once
                    slots = pd.concat([pd.to_numeric(df[x], errors="coerce") for x in slot_groups[gkey]], axis=1)
                    labs = vlab.get((int(wave), r.variable), {})
                    codes = sorted({v for v in pd.unique(slots.values.ravel()) if pd.notna(v)})
                    for code in codes:
                        lab = labs.get(str(code)) or labs.get(str(int(code))) or ""
                        records.append(pd.DataFrame({
                            "wave": int(wave), "source_group": group,
                            "row_uid": k.row_uid.values,
                            "family": r.primary_family,
                            "canonical_item": f"{r.canonical_item}__{slugify(lab, code)}",
                            "measure": "experienced", "variable": "|".join(slot_groups[gkey]),
                            "value": (slots == code).any(axis=1).astype(float).values,
                            "unit": "na", "polarity": "na",
                        }))
                    continue

                s = pd.to_numeric(df[r.variable], errors="coerce")
                val = s
                uniq = set(s.dropna().unique())
                # Fixed-position multiple response: chosen gives value k, not chosen
                # gives missing. The evidence is that the non-missing values are a
                # single constant (in 2024, D1_1_4 holds the value 4 only for people
                # who chose crafts). Here missing means **not chosen**, not
                # non-response, so it must be filled with 0. Otherwise every remaining
                # cell for someone who picked two or three choices is misclassified as
                # item non-response.
                if len(uniq) == 1 and 0.0 not in uniq and s.notna().sum() < len(s) * 0.95:
                    val = s.notna().astype(float)
                elif r.measure in BINARY_MEASURES and uniq <= {1.0, 2.0}:
                    # 1 = yes / 2 = no -> 1/0
                    val = s.map({1.0: 1.0, 2.0: 0.0})
                records.append(pd.DataFrame({
                    "wave": int(wave), "source_group": group, "row_uid": k.row_uid.values,
                    "family": r.primary_family, "canonical_item": r.canonical_item,
                    "measure": r.measure, "variable": r.variable,
                    "value": val.values, "unit": r.unit, "polarity": r.polarity,
                }))

    CH = pd.DataFrame(checked)
    bad = CH[~CH.in_data]
    if len(bad):
        print("!! declared but absent from the data:")
        print(bad[["wave", "source_group", "variable", "primary_family"]].to_string(index=False))
    dupo = CH[CH.overlaps_offline]
    if len(dupo):
        print(f"!! overlapping the offline battery: {len(dupo)}")

    LONG = pd.concat(records, ignore_index=True)
    LONG = LONG[LONG.value.notna()].reset_index(drop=True)

    # Record a summary so that uniqueness on the output side can be left to a contract
    kk = LONG.groupby(["wave", "source_group", "family", "canonical_item",
                       "measure"]).variable.nunique()
    LONGKEY = kk.reset_index().rename(columns={"variable": "n_variables"})
    LONGKEY.to_csv(C.OUT / "official_domain_item_keys.csv", index=False)

    cov = (CH[CH.in_data].groupby(["wave", "source_group", "primary_family", "measure"],
                                  as_index=False)
           .agg(variables=("variable", "nunique"), items=("canonical_item", "nunique"),
                routed=("is_routed", "sum")))

    C.OUT.mkdir(parents=True, exist_ok=True)
    C.PERSON.mkdir(parents=True, exist_ok=True)
    CH.to_csv(C.OUT / "official_domain_declarations.csv", index=False)
    cov.to_csv(C.OUT / "official_domain_family_coverage.csv", index=False)
    C.write_person(LONG, "official_person_domain")

    print(f"{len(CH)} declaration rows (from {len(decl)} declarations across "
          f"{sum(len(v) for v in GROUPS.values())} raw files) · "
          f"{int(CH.in_data.sum())} found in the data · "
          f"{int(CH.overlaps_offline.sum())} overlapping the offline battery")
    mism = CH[(CH.n_matches_declared == False)]
    print(f"declared n disagreeing with observed n: {len(mism)}")
    print("routed questions (answered by under 95 per cent): "
          f"{int(CH.is_routed.fillna(False).sum())}")
    print(f"\nlong person x item table: {len(LONG):,} rows · "
          f"{LONG.row_uid.nunique():,} people")
    print("\n== variables per wave and family ==")
    print(CH[CH.in_data].pivot_table(index="primary_family", columns="wave",
                                     values="variable", aggfunc="nunique").fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
