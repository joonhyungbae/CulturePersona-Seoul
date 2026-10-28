"""A registry of every variable -- each one gets a status, and none may be left out.

Counting harmonisation as "what carries a canonical name" stalls coverage at 63 per
cent. The remaining 37 per cent is not useless but **of a different kind**: slot
siblings (only the first slot carries a label), open-ended "other" responses, derived
recodes, identifiers and weights, and questions that mean something only within their
own wave.

So there are three tiers, and **every variable gets a status.**

  A  canonical   harmonised under one name across waves (the attendance battery, the
                 single canonicals, the battery families)
  B  wave_local  meaning is settled within this wave and it can be used numerically;
                 there is no cross-wave counterpart
  C  catalogued  registered only -- open text, identifiers, variables with no values

A slot sibling (`QA7_6` is the sixth cell of `QA7_1`) inherits its label and status
from the first slot. A derived recode is linked back to its source variable.

Outputs:
  official_variable_registry.csv    wave x variable x tier x status x basis, for every
                                    variable
  official_registry_summary.csv     wave x tier x status summary
  official_person_wavelocal.parquet person-level values of the tier-B numeric variables
                                    (person level -> interim)
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
from official_15_domain_items import load_wave, SRC

TEXT_PAT = re.compile(r"_OP$|ETC|기타내용|^T[A-Z]|_NAME$|NOANSWER")
ID_PAT = re.compile(r"^(userid|ID|IDX|id|코딩ID|DEPART|GRP|TYPE\d?|구분|WT|WTB|wt\w*|Raw_)")
RECODE_PAT = re.compile(r"(R|RR)$|^Z[A-Z]|^m[A-Z]")


def slot_base(v: str) -> str | None:
    """The first-slot name of a slot sibling. QA7_6 -> QA7_1, D3#4R -> D3#1R."""
    m = re.match(r"^(.*?[_#])(\d+)([A-Za-z]*)$", str(v))
    if not m:
        return None
    return f"{m.group(1)}1{m.group(3)}"


def recode_source(v: str, cols: set) -> str | None:
    """Find the source variable a derived recode came from."""
    for pat, rep in ((r"RR$", ""), (r"R$", ""), (r"^Z", ""), (r"^m", "")):
        cand = re.sub(pat, rep, str(v))
        if cand != v and cand in cols:
            return cand
    return None


def main():
    v = pd.read_csv(C.OUT / "official_variable_dictionary.csv")
    bat = pd.read_csv(C.OUT / "official_item_battery.csv")
    cm = pd.read_csv(C.OUT / "official_canonical_map.csv")
    bd = pd.read_csv(C.OUT / "official_domain_declarations.csv")
    keys = C.read_person("official_person_keys")

    # ---- variables a rule can promote to tier A --------------------------------
    # These would otherwise sit at tier B, but each is really part of a structure that
    # is already harmonised, or a second or third choice, a total, or a derived measure
    # of an existing canonical. The reason is recorded in basis.
    import official_02_instrument as INS

    RULES = []
    # The 2024 battery collects answers through multiple-response slots. official_04
    # already uses them, but the slot variables themselves are not in
    # official_item_battery, so they were counted as unharmonised.
    for slot in INS.BATTERY[2024].get("multi_response_slots", []):
        RULES.append((2024, slot, "battery_offline", "offline_attendance_slot",
                      "2024 battery multiple-response slot (official_04 uses it to "
                      "derive attendance)"))
    # battery totals
    for w, var, canon in [(2018, "B1a_SUM", "offline_total_count"),
                          (2018, "B1b_SUM", "offline_total_self_paid_count"),
                          (2018, "B1cR_SUM", "offline_total_cost"),
                          (2022, "B1_1x99", "offline_total_count"),
                          (2022, "B1_2x99", "offline_total_self_paid_count"),
                          (2022, "B1_3x99", "offline_total_cost")]:
        RULES.append((w, var, "battery_total", canon, "sum over the battery items"))
    # attended-nothing flag
    for w, var in [(2018, "B1a_m99a"), (2018, "B1b_m99a")]:
        RULES.append((w, var, "battery_total", "offline_none", "attended nothing flag"))
    # 2018 per-item self-paid attendance dummies
    for i, canon in enumerate([c for _, c in INS.BATTERY[2018]["items"]], start=1):
        RULES.append((2018, f"B1b_m{i:02d}a", "battery_offline", canon,
                      "per-item self-paid attendance, derived from B1_1_i#2 > 0"))
    # second and third choices of an existing canonical
    for w, var, canon in [(2014, "QA1A2", "leisure_activity_2nd"), (2014, "QA1A3", "leisure_activity_3rd"),
                          (2014, "QA3A2", "leisure_activity_wanted_2nd"),
                          (2015, "A1_2", "leisure_activity_2nd"), (2015, "A1_3", "leisure_activity_3rd"),
                          (2015, "A3_2", "leisure_activity_wanted_2nd"),
                          (2016, "A3#2", "leisure_activity_2nd"), (2016, "A5#2", "leisure_activity_wanted_2nd"),
                          (2018, "A3#2RR", "leisure_activity_2nd"), (2018, "A5#2RR", "leisure_activity_wanted_2nd"),
                          (2020, "A3_2", "leisure_activity_2nd"), (2020, "A5_2", "leisure_activity_wanted_2nd"),
                          (2022, "A3x2", "leisure_activity_2nd"), (2022, "A5x2", "leisure_activity_wanted_2nd"),
                          (2024, "A3_2", "leisure_activity_2nd"), (2024, "A5_2", "leisure_activity_wanted_2nd")]:
        RULES.append((w, var, "canonical_single", canon,
                      "second or third choice of an existing canonical"))
    # questions repeated across waves, confirmed by comparing labels
    for w, var, canon in [(2020, "DQ3A", "arts_sector_field"), (2022, "DQ4_1", "arts_sector_field"),
                          (2024, "DQ4_1", "arts_sector_field"),
                          (2020, "DQ5", "marital_status"), (2022, "DQ6", "marital_status"),
                          (2024, "DQ6", "marital_status"),
                          (2016, "DQ3_2", "arts_sector_years"), (2022, "DQ4_2_0", "arts_sector_years"),
                          (2024, "DQ4_2", "arts_sector_years")]:
        RULES.append((w, var, "canonical_single", canon,
                      "repeated across waves, confirmed by comparing labels"))

    A = {}
    for r in bat.itertuples():
        A[(int(r.wave), r.variable)] = ("battery_offline", r.canonical_item)
    for r in cm[cm.in_data].itertuples():
        A.setdefault((int(r.wave), r.source_var), ("canonical_single", r.canonical_var))
    for r in bd[bd.in_data].itertuples():
        A.setdefault((int(r.wave), r.variable), ("battery_family", r.canonical_item))
    RULE_BASIS = {}
    for w, var, status, canon, why in RULES:
        A.setdefault((w, var), (status, canon))
        RULE_BASIS[(w, var)] = why

    # Do not mask the variable dictionary twice per (wave, group) -- read it once and
    # split it up.
    LBL = {kk: dict(zip(gg.variable, gg.label.fillna("")))
           for kk, gg in v.groupby(["wave", "group"])}

    rows, wavelocal, textvals = [], [], []
    for wave, group in C.groups():
        df = load_wave(wave, group)
        k = keys[(keys.wave == wave) & (keys.source_group == group)].reset_index(drop=True)
        cols = set(df.columns)
        labels = LBL.get((wave, group), {})
        # A group-specific file has different key variables (the 2024 disabled file
        # uses pid=ID, for instance). Failing to mirror official_03's overrides here
        # loses the identifiers.
        keyvars = {x for x in {**KEYS[wave], **GROUP_KEYS.get((wave, group), {})}.values() if x}

        for c in df.columns:
            lab = str(labels.get(c, "") or "")
            s = df[c]
            n_obs = int(s.notna().sum())
            numeric = pd.to_numeric(s, errors="coerce")
            n_num = int(numeric.notna().sum())
            is_num = n_obs > 0 and n_num >= n_obs * 0.9

            hit = A.get((wave, c))
            if hit:
                tier, status, canon = "A", hit[0], hit[1]
                basis = RULE_BASIS.get((wave, c), "canonical mapping")
            elif n_obs == 0:
                # A variable with no values at all. Most are empty slots nobody filled
                # that far down (2014 QA7_6 and QA7_7, for instance). They must be
                # caught before anything is classified as open text.
                tier, status, canon, basis = ("C", "empty", "",
                                              "no values at all (an unused slot, say)")
            elif c in keyvars or ID_PAT.match(c):
                tier, status, canon, basis = ("C", "identifier_weight_frame", "",
                                              "identifier, weight or frame (held in "
                                              "person_keys)")
            elif TEXT_PAT.search(c) or not is_num:
                tier, status, canon, basis = ("C", "open_text", "",
                                              "open-ended or non-numeric")
            else:
                base = slot_base(c)
                src = recode_source(c, cols)
                if base and base != c and A.get((wave, base)):
                    bt = A[(wave, base)]
                    tier, status, canon = "A", "slot_sibling", bt[1]
                    basis = f"slot sibling -- inherits the declaration of {base}"
                    if not lab:
                        lab = str(labels.get(base, "") or "") + " [slot]"
                elif src and A.get((wave, src)):
                    tier, status, canon = "A", "derived_recode", A[(wave, src)][1]
                    basis = f"derived recode -- source variable {src}"
                elif src:
                    tier, status, canon, basis = ("B", "derived_recode", "",
                                                  f"derived recode -- source variable {src}")
                elif n_obs == 0:
                    tier, status, canon, basis = "C", "empty", "", "no values"
                else:
                    tier, status, canon, basis = ("B", "wave_local", "",
                                                  "numeric question, wave-local")

            rows.append(dict(wave=wave, source_group=group, variable=c, label=lab[:200],
                             tier=tier, status=status,
                             canonical_item=canon, n_obs=n_obs, is_numeric=is_num, basis=basis))

            if status == "open_text" and n_obs > 0:
                # Open text goes into the person-level table too. It is data, merely
                # not numeric -- to code it or analyse it, one first has to know where
                # it is.
                t = s.astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
                m = t.notna()
                if m.any():
                    textvals.append(pd.DataFrame({
                        "wave": wave, "source_group": group,
                        "row_uid": k.row_uid.values[m.values],
                        "variable": c, "text": t.values[m.values], "label": lab[:120]}))

            if tier == "B" and is_num:
                wavelocal.append(pd.DataFrame({
                    "wave": wave, "source_group": group,
                    "row_uid": k.row_uid.values, "variable": c,
                    "value": numeric.values, "label": lab[:120]}))

    R = pd.DataFrame(rows)
    W = pd.concat(wavelocal, ignore_index=True) if wavelocal else pd.DataFrame()
    W = W[W.value.notna()].reset_index(drop=True) if len(W) else W

    S = (R.groupby(["wave", "source_group", "tier", "status"], as_index=False)
         .agg(variables=("variable", "nunique")))

    # Count only what actually has values in a person-level table. Counting
    # status != empty reported that the canonical_single variables (education, income,
    # occupation) had a home when they were nowhere at all (corrected 2026-09-09 --
    # in that state every stratification analysis was blocked).
    # Write this run's person-level outputs first: the check below reads them back.
    if len(W):
        C.write_person(W, "official_person_wavelocal")
    T = pd.concat(textvals, ignore_index=True) if textvals else pd.DataFrame()
    if len(T):
        C.write_person(T, "official_person_text")
    _homes = set()
    for _n, _c in [("official_person_canonical", "source_var"),
                   ("official_person_wavelocal", "variable"),
                   ("official_person_text", "variable"),
                   ("official_person_domain", "variable")]:
        _t = C.read_person(_n, ["wave", "source_group", _c])
        _homes |= set(zip(_t.wave, _t.source_group, _t[_c]))
    _bat = set(zip(*[R[R.status.str.startswith("battery")][c]
                     for c in ("wave", "source_group", "variable")]))
    _homes |= _bat        # the battery has its home in person_item
    R["homed"] = [t in _homes for t in zip(R.wave, R.source_group, R.variable)]

    R.to_csv(C.OUT / "official_variable_registry.csv", index=False)
    S.to_csv(C.OUT / "official_registry_summary.csv", index=False)

    tot = len(R)
    print(f"{tot} variables registered (every variable in every wave)")
    print("\n== tier by wave ==")
    print(R.pivot_table(index="tier", columns="wave", values="variable", aggfunc="count").fillna(0).astype(int).to_string())
    print("\n== by status ==")
    print(R.status.value_counts().to_string())
    a = int((R.tier == "A").sum()); b = int((R.tier == "B").sum())
    print(f"\nA canonical {a} ({a/tot*100:.0f}%) · B wave_local {b} ({b/tot*100:.0f}%) "
          f"· C catalogued {tot-a-b} ({(tot-a-b)/tot*100:.0f}%)")
    print(f"registry coverage {tot}/{tot} = 100% -- no variable left out")
    if len(W):
        print(f"\ntier-B person-level values: {len(W):,} rows · "
              f"{W.variable.nunique()} variables")
    if len(T):
        print(f"open text: {len(T):,} rows · {T.variable.nunique()} variables")
    homed = int(R.homed.sum())
    print(f"variables with a home at person level: {homed}/{tot} = "
          f"{homed/tot*100:.0f}%  (the rest have no values at all)")


if __name__ == "__main__":
    main()
