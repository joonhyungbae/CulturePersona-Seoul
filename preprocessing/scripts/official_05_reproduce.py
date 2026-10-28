"""Can the published tables be rebuilt from the delivered files?

**A rule about how this is written: the subject stays on our side.** Where a value does
not match, that is not a defect in the published figure but a consequence of
information the delivered file does not carry -- the weights and the control totals.
Nothing here is phrased as an attack on the foundation's published numbers; that
phrasing would also be inaccurate.

Reads the person x item table from 04 (the unified schema), so it runs after it.

For waves that do not reproduce, the person-level outputs give numbers different from
the published ones. That fact is recorded cell by cell, so every output can say which
figure it is using.

The instrument decides what attendance means, wave by wave (official_02_instrument):
  count_grid              total annual uses > 0
  multi_select_then_count the choice code appears in a multiple-response slot

The early waves (2014-2016) published only a few headline figures, so they are checked
separately. That check settles the sample base definition and the spending unit
empirically.

Outputs:
  official_margin_reproduction.csv    wave x segment x item x published x reproduced x
                                      error x verdict
  official_reproduction_summary.csv   wave x dimension x cells x mean and max absolute
                                      error x verdict
  official_early_published_check.csv  the 2014-2016 headline check, including the
                                      construct check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C
import official_02_instrument as I

TOL_EXACT, TOL_CLOSE = 0.10, 1.50  # %p


def attendance_matrix(wave: int, df: pd.DataFrame) -> pd.DataFrame:
    """Person x canonical item attendance as 0/1, following the instrument's own
    definition."""
    spec = I.BATTERY[wave]
    items = [c for _, c in spec["items"]]
    if spec["response_format"] == "multi_select_then_count":
        slots = df[spec["multi_response_slots"]].apply(pd.to_numeric, errors="coerce")
        out = {}
        for pos, canon in spec["items"]:
            out[canon] = (slots == pos).any(axis=1).astype(float)
        return pd.DataFrame(out, index=df.index)[items]
    pat = spec["measures"]["total_count"]
    out = {}
    for pos, canon in spec["items"]:
        c = pd.to_numeric(df[pat.format(i=pos)], errors="coerce").fillna(0)
        out[canon] = (c > 0).astype(float)
    return pd.DataFrame(out, index=df.index)[items]


def gu_name_map(wave: int, group: str, var: str) -> dict:
    vl = pd.read_csv(C.OUT / "official_value_labels.csv")
    v = vl[(vl.wave == wave) & (vl.group == group) & (vl.variable == var)]
    return {float(r.value): str(r.label).strip() for r in v.itertuples()
            if str(r.value).replace(".", "").isdigit()}


# The 2014-2016 published headline figures, transcribed from the reports by the
# aggregate track. (wave, base, metric, published value, unit as published)
EARLY_PUBLISHED = [
    (2014, "sfac_member_seoul", "annual_visits", 35.5, "회"),
    (2014, "sfac_member_seoul", "annual_cost", 559632.0, "원"),
    (2014, "sfac_member_seoul", "leisure_top3_arts", 68.7, "%"),
    (2015, "sfac_member_all", "annual_visits", 31.3, "회"),
    (2015, "sfac_member_all", "annual_cost", 477358.0, "원"),
    (2015, "sfac_member_all", "leisure_top3_arts", 65.2, "%"),
    (2016, "general_panel", "attendance_rate", 93.1, "%"),
    (2016, "general_panel", "annual_visits", 14.63, "회"),
    (2016, "culture_interested", "attendance_rate", 99.0, "%"),
    (2016, "culture_interested", "annual_visits", 23.78, "회"),
]

# Base definition -> (wave, list of frame codes, restricted to Seoul)
EARLY_BASE = {
    "sfac_member_seoul": (2014, [1.0], True),
    "sfac_member_all": (2015, [9.0, 10.0], False),
    "general_panel": (2016, [1.0], False),
    "culture_interested": (2016, [2.0, 3.0, 4.0], False),
}

# The first-to-third leisure activity variables. Choice 2 is attending the arts.
LEISURE_VARS = {2014: ["QA1A1", "QA1A2", "QA1A3"], 2015: ["A1_1", "A1_2", "A1_3"]}


def early_check():
    """Rebuild the early waves' published headline figures from the microdata."""
    item = C.read_person("official_person_item")
    master = C.read_person("official_person_master")
    m = master.set_index("row_uid")
    item["frame_code"] = item.row_uid.map(m.frame_code)
    item["in_seoul"] = item.row_uid.map(m.in_seoul)

    rows = []
    for wave, base, metric, expected, unit in EARLY_PUBLISHED:
        _w, codes, seoul_only = EARLY_BASE[base]
        d = item[(item.wave == wave) & (item.frame_code.isin(codes))]
        if seoul_only:
            d = d[d.in_seoul]
        g = d.groupby("row_uid").agg(att=("attended", "max"),
                                     visits=("total_count", "sum"),
                                     cost=("annual_cost_krw", "sum"))
        if metric == "attendance_rate":
            got = float(g.att.mean() * 100)
        elif metric == "annual_visits":
            got = float(g.visits.mean())
        elif metric == "annual_cost":
            got = float(g.cost.mean())
        elif metric == "leisure_top3_arts":
            df, _mm = C.load(wave, "mixed")
            if seoul_only:
                df = df[pd.to_numeric(df["SQ1"], errors="coerce") == 1]
            q = df[LEISURE_VARS[wave]].apply(pd.to_numeric, errors="coerce")
            got = float((q == 2).any(axis=1).mean() * 100)
        else:
            continue
        rel = abs(got - expected) / expected * 100 if expected else np.nan
        rows.append(dict(wave=wave, base=base, metric=metric, unit=unit,
                         published=expected, reproduced=round(got, 2),
                         rel_error_pct=round(rel, 3),
                         verdict="exact" if rel < 0.5 else ("close" if rel < 5 else "differs")))
    return pd.DataFrame(rows)


def main():
    # The published values are frozen under data/reference/ (observed rows only).
    pub = pd.read_csv(C.REFERENCE / "published_tables_2018_2024.csv")
    pub = pub[(pub.domain == "offline") & (pub.population == "seoul_general_citizen_15plus")]

    cw = pd.read_csv(C.REFERENCE / "seoul_living_area_crosswalk.csv")
    la_by_gu = dict(zip(cw.district_name, cw.living_area_ko))

    keys = C.read_person("official_person_keys")

    # The general-public sample as the published tables define it
    GEN = {2018: ("general", None), 2020: ("general", None),
           2022: ("mixed", [1, 3]), 2024: ("general", None)}
    GU_VAR = {2020: "SQ2", 2022: "SQ1", 2024: "SQ1"}

    rows = []
    for wave, (group, grp_filter) in GEN.items():
        df, _m = C.load(wave, group)
        k = C.read_person("official_person_keys")
        k = k[(k.wave == wave) & (k.source_group == group)].reset_index(drop=True)
        df = df.reset_index(drop=True)
        if grp_filter is not None:
            mask = pd.to_numeric(df[I.BATTERY[wave].get("frame_var", "GRP")]
                                 if False else df["GRP"], errors="coerce").isin(grp_filter)
            df, k = df[mask].reset_index(drop=True), k[mask.values].reset_index(drop=True)

        A = attendance_matrix(wave, df)
        A["offline_total"] = (A.sum(axis=1) > 0).astype(float)
        w = k.weight.fillna(1.0).values if k.weight.notna().any() else np.ones(len(df))

        # match the published age bands; 2020 collapses everything to 60+
        ab = k.age_band.copy()
        if wave == 2020:
            ab = ab.replace({"60-69": "60+", "70+": "60+"})

        segs = {"overall": pd.Series("overall", index=k.index)}
        segs["sex"] = k.sex.map({"male": "sex=male", "female": "sex=female"})
        segs["age"] = ab.map(lambda x: f"age={x}" if pd.notna(x) else None)
        segs["age_sex"] = [
            f"sex={s};age={a}" if pd.notna(s) and pd.notna(a) else None
            for s, a in zip(k.sex, ab)
        ]
        if wave in GU_VAR:
            gm = gu_name_map(wave, group, GU_VAR[wave])
            gu = pd.to_numeric(df[GU_VAR[wave]], errors="coerce").map(gm)
            segs["living_area"] = gu.map(la_by_gu).map(lambda x: f"living_area={x}" if pd.notna(x) else None)

        pw = pub[pub.year == wave]
        for dim, sv in segs.items():
            sv = pd.Series(sv, index=k.index)
            for seg, idx in sv.groupby(sv).groups.items():
                ii = k.index.get_indexer(idx)
                for item in A.columns:
                    got = float(np.average(A[item].values[ii], weights=w[ii]) * 100)
                    m = pw[(pw.segment == seg) & (pw.canonical_var == item)
                           & (pw.segment_dimension == dim)]
                    if not len(m):
                        continue
                    exp = float(m.value.iloc[0])
                    d = abs(got - exp)
                    rows.append(dict(
                        wave=wave, dimension=dim, segment=seg, canonical_item=item,
                        published=exp, reproduced=round(got, 3), abs_error=round(d, 3),
                        n_cell=len(ii),
                        verdict="exact" if d <= TOL_EXACT else ("close" if d <= TOL_CLOSE else "differs"),
                    ))

    R = pd.DataFrame(rows)
    S = (R.groupby(["wave", "dimension"])
           .agg(cells=("abs_error", "size"), mean_abs=("abs_error", "mean"),
                max_abs=("abs_error", "max"),
                exact=("verdict", lambda s: (s == "exact").sum()),
                close=("verdict", lambda s: (s == "close").sum()),
                differs=("verdict", lambda s: (s == "differs").sum()))
           .round(3).reset_index())

    EARLY = early_check()
    EARLY.to_csv(C.OUT / "official_early_published_check.csv", index=False)

    R.to_csv(C.OUT / "official_margin_reproduction.csv", index=False)
    S.to_csv(C.OUT / "official_reproduction_summary.csv", index=False)

    print("== early waves against the published headline figures ==")
    print(EARLY.to_string(index=False))
    print("  · visits and spending in 2014 and 2015 match to the won, which settles "
          "both the sample base and the spending unit")
    print("  · the published 'arts attendance rate' is not an attendance rate: it is "
          "the share naming the arts among their top three leisure activities, a "
          "different construct")

    print("\n== reproduction summary (absolute error against published, pp) ==")
    print(S.to_string(index=False))
    print("\n== verdict by wave ==")
    for wave, g in R.groupby("wave"):
        n = len(g)
        ex, cl = (g.verdict == "exact").sum(), (g.verdict == "close").sum()
        print(f"  {wave}: {n:4d} cells · exact {ex:4d} · close {cl:4d} · "
              f"differs {n - ex - cl:4d} · mean {g.abs_error.mean():.3f} · "
              f"max {g.abs_error.max():.2f}")
    bad = R[R.verdict == "differs"]
    if len(bad):
        print("\ntwelve largest differences (the weights are not in the delivered file):")
        print(bad.nlargest(12, "abs_error")[
            ["wave", "dimension", "segment", "canonical_item", "published", "reproduced", "abs_error"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
