"""A cross-wave concordance for the domains outside the attendance battery -- the body
of the question-change table.

The battery (official_02) is regular enough to fingerprint automatically. Outside it,
the question structure changes wholesale from wave to wave and automatic matching does
not work. So the canonical variables here are **declared by hand and checked by
machine**:

  · confirm each declared variable really exists in that wave's data (fail if not);
  · carry the codebook label through verbatim, so the basis for the mapping is visible;
  · where the number of choices differs across waves (a scale change), catch it from
    the value labels and reflect it in the grade.

**Sharing a canonical name does not make the values comparable.** This table shows only
what exists in which wave and what its original wording was. Comparability is what the
comparability column speaks to.

Outputs:
  official_canonical_map.csv     canonical x wave x source variable x codebook wording
                                 x number of choices
  official_domain_coverage.csv   domain x wave x present
  official_person_canonical      person x canonical variable (person level -> interim)

A concordance that carries no values stays a document and nothing more. That is exactly
what happened: education, income and occupation were in no person-level table at all,
which blocked every stratification analysis (found 2026-09-09). The values now ship
under their canonical names, so a consumer calls it `education` however the source
variable happens to be named in each wave.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

# The representative file for each wave. The source variables in the concordance were
# read off these files.
SRC = {w: C.default_group(w) for w, _ in C.groups()}

# canonical_var -> {domain, kind, description, source variable per wave}
# A wave with no source variable is simply not listed -- that wave did not ask it.
MAP: dict[str, dict] = {
    # ------------------------------------------------------------------ leisure
    "leisure_time_weekday": dict(domain="leisure", kind="numeric",
        desc="average daily leisure hours, weekday",
        vars={2016: "A1#1", 2018: "mA1_1", 2020: "A1_1", 2022: "A1x1_1", 2024: "A1_1_1_1"}),
    "leisure_time_weekend": dict(domain="leisure", kind="numeric",
        desc="average daily leisure hours, weekend or holiday",
        vars={2016: "A1#2", 2018: "mA1_2", 2020: "A1_3", 2022: "A1x2_1", 2024: "A1_2_1_1"}),
    "leisure_time_sufficiency": dict(domain="leisure", kind="ordinal",
        desc="whether weekend or holiday leisure time feels sufficient",
        vars={2016: "A2", 2018: "A2", 2020: "A2", 2022: "A2", 2024: "A2"}),
    "leisure_activity_1st": dict(domain="leisure", kind="categorical",
        desc="main leisure activity, first choice",
        vars={2014: "QA1A1", 2015: "A1_1", 2016: "A3#1", 2018: "A3#1RR",
              2020: "A3_1", 2022: "A3x1", 2024: "A3_1"}),
    "leisure_activity_wanted_1st": dict(domain="leisure", kind="categorical",
        desc="most desired leisure activity, first choice",
        vars={2014: "QA3A1", 2015: "A3_1", 2016: "A5#1", 2018: "A5#1RR",
              2020: "A5_1", 2022: "A5x1", 2024: "A5_1"}),
    "leisure_companion": dict(domain="leisure", kind="categorical",
        desc="who they spend leisure time with",
        vars={2014: "QA2", 2015: "A2", 2016: "A4", 2018: "A4RR",
              2020: "A4", 2022: "A4", 2024: "A4"}),
    # ----------------------------------------------------------------- attitude
    "arts_interest": dict(domain="attitude", kind="ordinal",
        desc="general level of interest in the arts",
        vars={2014: "QA4", 2015: "A4", 2016: "A6", 2018: "A6"}),
    "attendance_companion": dict(domain="attitude", kind="categorical",
        desc="who they attend the arts with",
        vars={2015: "B4", 2016: "B2", 2018: "B3RR", 2020: "B3", 2022: "B2", 2024: "B2"}),
    # ------------------------------------------------------------- demographics
    "education": dict(domain="demographics", kind="ordinal", desc="highest education",
        vars={2014: "DQ4", 2016: "DQ2", 2018: "DQ2", 2020: "DQ2", 2022: "DQ2", 2024: "DQ2"}),
    "occupation": dict(domain="demographics", kind="categorical", desc="occupation",
        vars={2014: "DQ5", 2016: "DQ3", 2018: "DQ3", 2020: "DQ3", 2022: "DQ3", 2024: "DQ3"}),
    "household_income": dict(domain="demographics", kind="ordinal",
        desc="average monthly household income",
        vars={2014: "DQ8", 2016: "DQ7", 2018: "DQ7", 2020: "DQ7", 2022: "DQ9", 2024: "DQ8"}),
    "household_size": dict(domain="demographics", kind="numeric",
        desc="household size, respondent included",
        vars={2016: "DQ4", 2018: "DQ4", 2020: "DQ4", 2022: "DQ5X1", 2024: "DQ5_1"}),
    "marital_status": dict(domain="demographics", kind="categorical",
        desc="marital status",
        vars={2014: "DQ2", 2016: "DQ5", 2018: "DQ5", 2020: "BA05"}),
    "years_in_seoul": dict(domain="demographics", kind="numeric",
        desc="total years lived in Seoul",
        vars={2014: "DQ1_1", 2016: "DQ1#1", 2018: "DQ1#1", 2020: "DQ1_1", 2022: "DQ1x1_1", 2024: "DQ1_1"}),
}

# Records only whether a domain exists in a wave, including blocks that cannot be
# folded into a canonical. Existence is judged by variable prefix.
DOMAIN_PROBE = {
    "online_attendance": {2018: ["D1_"], 2020: ["D3_", "D4_"], 2022: ["C2", "C3"], 2024: ["C2", "C3"]},
    "participation": {2014: ["QA8"], 2015: ["C1_"], 2016: ["C1_"], 2018: ["C1_"],
                      2020: ["C1_"], 2022: ["D1", "D2"], 2024: ["D1", "D2"]},
    "arts_education": {2014: ["QA7"], 2015: ["C1_"], 2016: ["C2"], 2018: ["C2_"],
                       2020: ["C2_", "C3"], 2022: ["G1", "G2"], 2024: ["G1"]},
    # 2024 has no barriers-to-attendance battery of the 2018/2020 kind. It carries E8
    # (suggestions for improvement) instead, a different construct, so that is not
    # folded into the same domain.
    "barriers": {2014: ["QA6"], 2015: ["B5_"], 2016: ["B4#", "B4_"], 2018: ["E2_"],
                 2020: ["E2"], 2022: ["C4"]},
    "satisfaction": {2014: ["QA10", "QA12", "QA13"], 2015: ["D1", "D2_"], 2016: ["D1"],
                     2018: ["D4", "E3"], 2020: ["E1"], 2022: ["E4", "F1"], 2024: ["E4", "F1"]},
    "facility_use": {2018: ["D2_"], 2020: ["F1_"], 2022: ["E2"], 2024: ["E2"]},
}


def main():
    rows = []
    for canon, spec in MAP.items():
        for wave, var in spec["vars"].items():
            df, _m = C.load(wave, SRC[wave])
            present = var in df.columns
            vd = pd.read_csv(C.OUT / "official_variable_dictionary.csv")
            lab = vd[(vd.wave == wave) & (vd.group == SRC[wave]) & (vd.variable == var)]
            label = "" if not len(lab) else str(lab.label.iloc[0])
            vl = pd.read_csv(C.OUT / "official_value_labels.csv")
            n_opt = len(vl[(vl.wave == wave) & (vl.group == SRC[wave]) & (vl.variable == var)])
            rows.append(dict(canonical_var=canon, domain=spec["domain"], kind=spec["kind"],
                             description=spec["desc"], wave=wave, source_var=var,
                             in_data=present, codebook_label=label.replace("\n", " ")[:120],
                             n_options=n_opt))
    M = pd.DataFrame(rows)

    missing = M[~M.in_data]
    if len(missing):
        print("!! declared but absent from the data:")
        print(missing[["canonical_var", "wave", "source_var"]].to_string(index=False))

    # A differing number of choices across waves means the scale changed
    comp = []
    for canon, g in M[M.in_data].groupby("canonical_var"):
        waves = sorted(g.wave)
        opts = {int(r.wave): int(r.n_options) for r in g.itertuples() if r.n_options}
        stable = len(set(opts.values())) <= 1 if opts else None
        comp.append(dict(canonical_var=canon, domain=g.domain.iloc[0], n_waves=len(waves),
                         waves="·".join(str(w) for w in waves),
                         option_counts=str(opts),
                         scale_stable=stable))
    K = pd.DataFrame(comp)
    M = M.merge(K[["canonical_var", "scale_stable"]], on="canonical_var", how="left")

    # domain coverage
    cov = []
    for dom, probe in DOMAIN_PROBE.items():
        for wave in SRC:
            pre = probe.get(wave)
            if pre is None:
                cov.append(dict(domain=dom, wave=wave, present=False, matched_vars=0,
                                note="not in this wave"))
                continue
            df, _m = C.load(wave, SRC[wave])
            n = sum(1 for c in df.columns if any(str(c).startswith(p) for p in pre))
            cov.append(dict(domain=dom, wave=wave, present=n > 0, matched_vars=n,
                            note="·".join(pre)))
    for canon, spec in MAP.items():
        for wave in SRC:
            cov.append(dict(domain=spec["domain"], wave=wave,
                            present=wave in spec["vars"], matched_vars=int(wave in spec["vars"]),
                            note=canon))
    V = pd.DataFrame(cov)
    DC = (V.groupby(["domain", "wave"], as_index=False)
            .agg(present=("present", "any"), vars=("matched_vars", "sum")))

    # ---- person-level values --------------------------------------------------
    # Source variable names differ by wave. Shipping the values under their canonical
    # names lets a consumer call it `education` throughout. A group file carrying the
    # same variable is included too -- sweeping only the representative file would drop
    # the culture-interested and disabled samples.
    from official_15_domain_items import load_wave
    keys = C.read_person("official_person_keys", ["wave", "source_group", "row_uid"])
    prows = []
    for wave, group in C.groups():
        df = load_wave(wave, group)
        k = keys[(keys.wave == wave) & (keys.source_group == group)].reset_index(drop=True)
        if len(df) != len(k):
            raise AssertionError(f"{wave} {group}: row counts differ, {len(df)} vs {len(k)}")
        for r in M[M.in_data & (M.wave == wave)].itertuples():
            if r.source_var not in df.columns:
                continue
            v = pd.to_numeric(df[r.source_var], errors="coerce")
            keep = v.notna().values
            if not keep.any():
                continue
            prows.append(pd.DataFrame({
                "wave": wave, "source_group": group, "row_uid": k.row_uid.values[keep],
                "domain": r.domain, "canonical_var": r.canonical_var,
                "source_var": r.source_var, "value": v.values[keep]}))
    PC = pd.concat(prows, ignore_index=True)
    C.write_person(PC, "official_person_canonical")

    M.to_csv(C.OUT / "official_canonical_map.csv", index=False)
    DC.to_csv(C.OUT / "official_domain_coverage.csv", index=False)

    print(f"{M.canonical_var.nunique()} canonical variables · {len(M)} mapping rows "
          f"({int(M.in_data.sum())} found in the data)")
    print(f"person x canonical: {len(PC):,} rows · {PC.row_uid.nunique():,} people · "
          f"{PC.groupby(['wave','source_group']).ngroups} (wave, group) pairs")
    print("\n== canonical by wave ==")
    print(M[M.in_data].pivot_table(index=["domain", "canonical_var"], columns="wave",
                                   values="source_var", aggfunc="first").fillna("").to_string())
    print("\n== variables whose scale moves (the number of choices changes) ==")
    for r in K[K.scale_stable == False].itertuples():
        print(f"  {r.canonical_var:<28} {r.option_counts}")
    print("\n== domain coverage (number of variables held) ==")
    print(DC.pivot_table(index="domain", columns="wave", values="vars").fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
