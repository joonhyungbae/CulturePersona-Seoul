"""Enforce the KCAS track's contracts. A violation exits non-zero.

Kept separate from the official validator so that the 212 official contracts are never
disturbed.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kcas_00_common as C

FAILS: list[str] = []
CHECKS = 0


def check(cond: bool, msg: str):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)


def git_ignored(path: Path) -> bool:
    r = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=C.ROOT,
    )
    return r.returncode == 0


def main():
    O, P = C.OUT, C.PERSON

    # ---- raw inputs ---------------------------------------------------------
    zips = C.zip_paths()
    check(len(zips) == 2, f"expected 2 ZIPs, found {len(zips)}")
    members = []
    for zp in zips:
        import zipfile
        with zipfile.ZipFile(zp) as zf:
            members.extend([n for n in zf.namelist() if n.endswith(".csv")])
    check(len(members) == 9, f"expected 9 CSV members, found {len(members)}")
    years = sorted(int(m[:4]) for m in members)
    check(years == C.WAVES, f"the wave list does not match: {years}")

    # ---- outputs exist ------------------------------------------------------
    for f in ["kcas_variable_map.csv", "kcas_coverage.csv"]:
        check((O / f).exists(), f"output missing: {f}")
    check(C.person_path().exists(), "kcas_person.parquet is missing")
    check(git_ignored(C.person_path()),
          "kcas_person.parquet is not gitignored -- person-level data could leak")

    decl = C.declarations()
    vmap = pd.read_csv(O / "kcas_variable_map.csv")
    cov = pd.read_csv(O / "kcas_coverage.csv")
    person = C.read_person()

    # ---- declarations against the CSV columns --------------------------------
    for wave, g in decl.groupby("wave"):
        cols = set(C.wave_columns(int(wave)))
        for src in g.source_column:
            check(src in cols, f"{wave}: declared column absent -- {src[:50]}")

    # ---- map coverage -------------------------------------------------------
    check(set(vmap.wave) == set(C.WAVES), "variable_map is missing a wave")
    for wave in C.WAVES:
        g = vmap[(vmap.wave == wave) & (vmap.family == "attendance")
                 & (vmap.in_spec_items == "yes")]
        check(set(g.canonical) == set(C.ITEMS),
              f"{wave}: the nine Spec items are not all in the map ({sorted(g.canonical)})")
    # 2016 history_philosophy is wave-local
    h = vmap[(vmap.wave == 2016) & (vmap.canonical == "history_philosophy")]
    check(len(h) == 1 and h.iloc[0].in_spec_items == "no",
          "2016 history_philosophy is not marked wave-local")

    # museums and outdoor events must not be attendance canonicals
    bad = set(vmap.canonical) & {"museum", "outdoor_event"}
    check(not bad, f"museum/outdoor is an attendance canonical: {bad}")
    check("museum" not in person.columns and "outdoor_event" not in person.columns,
          "the person table carries a museum/outdoor_event column")

    # ---- person level -------------------------------------------------------
    check(set(person.wave) == set(C.WAVES), "the person table is missing a wave")
    for wave in C.WAVES:
        n_map = int(vmap.loc[vmap.wave == wave, "n_raw"].iloc[0])
        n_p = int((person.wave == wave).sum())
        check(n_p == n_map, f"{wave}: person {n_p} != map n_raw {n_map}")

    for item in C.ITEMS:
        check(item in person.columns, f"{item} absent from the person table")
        vals = set(person[item].dropna().unique())
        check(vals <= {0.0, 1.0, 0, 1}, f"{item} is not 0/1: {vals}")

    # a null_nonattend wave must have had nulls before the fill (recorded in the map)
    for wave in C.NULL_NONATTEND:
        g = vmap[(vmap.wave == wave) & (vmap.family == "attendance")
                 & (vmap.in_spec_items == "yes")]
        check((g.null_rate_before_fill > 0).all(),
              f"{wave}: marked null_nonattend yet the pre-fill null rate is 0")

    # ---- coverage -----------------------------------------------------------
    check(len(cov) == len(C.WAVES) * len(C.ITEMS),
          "coverage rows do not equal waves x items")
    for _, r in cov.iterrows():
        check(0 < r.attend_rate < 1, f"{r.wave} {r.item}: attendance rate {r.attend_rate}")

    # ---- theater and musical stay separate -----------------------------------
    check("theater" in C.ITEMS and "musical" in C.ITEMS,
          "theater/musical absent from ITEMS")

    # ---- contract snapshot --------------------------------------------------
    snapshot = {
        "n_checks": CHECKS,
        "n_fails": len(FAILS),
        "n_person": int(len(person)),
        "waves": C.WAVES,
        "items": C.ITEMS,
        "null_nonattend": sorted(C.NULL_NONATTEND),
    }
    (C.CONFIG / "kcas_contracts.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"{CHECKS} contracts · {len(FAILS)} failures")
    if FAILS:
        for f in FAILS:
            print(f"  !! {f}")
        sys.exit(1)
    print("contracts pass · config/kcas_contracts.json refreshed")


if __name__ == "__main__":
    main()
