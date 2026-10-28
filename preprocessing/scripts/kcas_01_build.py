"""KCAS person table, variable map and coverage.

The hand-written declarations in config/kcas_declarations.csv are checked by machine,
then this writes
  data/interim/kcas/kcas_person.parquet
  data/processed/kcas/kcas_variable_map.csv
  data/processed/kcas/kcas_coverage.csv

The rules that put edu_ord, inc_ord and age on one scale across waves are fixed below.
Museums and outdoor events are not part of the attendance battery and are not made
attendance canonicals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kcas_00_common as C

# Age band 1-7 -> band midpoint in years. Every wave is put on this scale.
AGE_MID = {1: 17.0, 2: 25.0, 3: 35.0, 4: 45.0, 5: 55.0, 6: 65.0, 7: 75.0}


def age_to_mid(series: pd.Series, wave: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if wave == 2016:
        # 2016 records age in years; cut it into the same seven bands
        bins = [0, 20, 30, 40, 50, 60, 70, 200]
        labels = [1, 2, 3, 4, 5, 6, 7]
        band = pd.cut(s, bins=bins, labels=labels, right=False)
        return band.astype(float).map(AGE_MID)
    return s.map(AGE_MID)


def edu_to_ord(series: pd.Series, wave: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if wave == 2016:
        # 8 categories -> 4: none/primary -> 1, middle -> 2, high -> 3, tertiary -> 4
        return s.map({1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4})
    # Other waves already carry the analytic 4 categories
    return s.where(s.isin([1, 2, 3, 4]))


def income_to_ord(series: pd.Series, wave: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if wave == 2014:
        # Keep 1-7; 8 is "no answer" -> NA
        return s.where(s.isin([1, 2, 3, 4, 5, 6, 7]))
    if wave == 2016:
        # 12 bands -> 7: 1-2 -> 1 (under 1m KRW), 3 -> 2, ..., 8-12 -> 7 (6m KRW and up)
        m = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
             8: 7, 9: 7, 10: 7, 11: 7, 12: 7}
        return s.map(m)
    # 2018 onwards carries the analytic 7 categories; 9 is "no answer" -> NA
    return s.where(s.isin([1, 2, 3, 4, 5, 6, 7]))


def build_wave(wave: int, decl: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    d = decl[decl.wave == wave]
    needed = d.source_column.tolist()
    raw = C.load_wave(wave)
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{wave}: declared columns absent from the CSV -- {missing[:3]}")

    n = len(raw)
    out = pd.DataFrame({"wave": wave, "unit_id": [f"{wave}:{i}" for i in range(n)]})
    map_rows: list[dict] = []

    # ---- attendance counts -> binary
    att = d[d.family == "attendance"]
    for _, r in att.iterrows():
        src = r.source_column
        can = r.canonical
        counts = pd.to_numeric(raw[src], errors="coerce")
        null_rate = float(counts.isna().mean())
        if wave in C.NULL_NONATTEND:
            counts = counts.fillna(0)
        # guard against negative counts
        if (counts.dropna() < 0).any():
            raise ValueError(f"{wave} {can}: negative count")
        binary = (counts > 0).astype(float)
        # only the items that go into Spec.items reach the person table
        if r.in_spec_items == "yes":
            out[can] = binary
            out[f"{can}_count"] = counts
        map_rows.append({
            "wave": wave,
            "source_column": src,
            "canonical": can,
            "family": "attendance",
            "measure": "count",
            "position": r.position,
            "in_spec_items": r.in_spec_items,
            "zero_coding": "null_nonattend" if wave in C.NULL_NONATTEND else "explicit_zero",
            "n_raw": n,
            "null_rate_before_fill": round(null_rate, 4),
            "attend_rate": round(float(binary.mean()), 4),
            "note": r.note,
        })

    # ---- demographics
    demo = {r.canonical: r.source_column for _, r in d[d.family == "demographics"].iterrows()}
    notes = {r.canonical: r.note for _, r in d[d.family == "demographics"].iterrows()}

    out["sido"] = pd.to_numeric(raw[demo["sido"]], errors="coerce")
    sex = pd.to_numeric(raw[demo["sex"]], errors="coerce")
    out["female"] = (sex == 2).astype(float)
    out.loc[~sex.isin([1, 2]), "female"] = np.nan
    out["age"] = age_to_mid(raw[demo["age"]], wave)
    out["edu_ord"] = edu_to_ord(raw[demo["edu"]], wave)
    out["inc_ord"] = income_to_ord(raw[demo["income"]], wave)
    out["weight"] = pd.to_numeric(raw[demo["weight"]], errors="coerce")

    for can, src in demo.items():
        map_rows.append({
            "wave": wave,
            "source_column": src,
            "canonical": can,
            "family": "demographics",
            "measure": "code",
            "position": "",
            "in_spec_items": "no",
            "zero_coding": "",
            "n_raw": n,
            "null_rate_before_fill": round(float(pd.to_numeric(raw[src], errors="coerce").isna().mean()), 4),
            "attend_rate": "",
            "note": notes.get(can, ""),
        })

    return out, map_rows


def coverage_from_person(person: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave, g in person.groupby("wave"):
        n = len(g)
        for item in C.ITEMS:
            s = g[item]
            rows.append({
                "wave": int(wave),
                "item": item,
                "n": n,
                "n_attend": int(s.sum()),
                "attend_rate": round(float(s.mean()), 4),
                "n_demo_complete": int(g[["age", "female", "edu_ord", "inc_ord"]].notna().all(axis=1).sum()),
            })
    return pd.DataFrame(rows)


def main():
    decl = C.declarations()
    assert set(decl.wave) == set(C.WAVES), "a wave is missing from the declarations"

    parts = []
    all_maps = []
    for wave in C.WAVES:
        print(f"  {wave} …", flush=True)
        part, maps = build_wave(wave, decl)
        parts.append(part)
        all_maps.extend(maps)

    person = pd.concat(parts, ignore_index=True)
    # all nine common items must be present
    for item in C.ITEMS:
        assert item in person.columns, f"{item} absent from the person table"
    assert "museum" not in person.columns
    assert "outdoor_event" not in person.columns

    C.write_person(person)
    vmap = pd.DataFrame(all_maps)
    C.write_out(vmap, "kcas_variable_map")
    cov = coverage_from_person(person)
    C.write_out(cov, "kcas_coverage")

    print(f"{len(person):,} people · {person.wave.nunique()} waves · "
          f"map {len(vmap)} rows · coverage {len(cov)} rows")
    print(cov.groupby("wave").attend_rate.mean().round(3).to_string())


if __name__ == "__main__":
    main()
