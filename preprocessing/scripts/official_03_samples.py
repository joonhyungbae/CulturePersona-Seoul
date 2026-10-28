"""Sample definitions, the frame register, the weight register, and person-level keys.

What counts as a "sample of Seoul residents" differs by wave. Three waves (2015, 2016
and 2022) mix several frames in one file, 2014 includes respondents living outside
Seoul, and in 2020 the file's row count disagrees with the sample size in the report.
Those definitions are settled here once, with the evidence recorded.

If the weights are raked on an attendance item itself, that item's published figure is
closer to a control total than to an estimate. To make that visible, the effect of the
weights is measured item by item -- unweighted against weighted, and the ratio of mean
weights between attenders and non-attenders.

Outputs:
  official_sample_frames.csv    wave x frame x n x group x mode x in Seoul
  official_weight_inventory.csv wave x weight variable x label x sum x whether used
  official_weight_effect.csv    wave x item x unweighted and weighted rate x weight
                                ratio
  official_person_keys.parquet  person-level keys (sex, birth year, age, district,
                                frame, weight)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

# The key variables in each wave. None means the wave does not carry it.
KEYS = {
    2014: dict(pid="userid", sex="SQ3", birth=None, age_band="SQ4", gu="SQ2",
               dong=None, seoul="SQ1", frame=None, weight=None),
    2015: dict(pid="ID", sex="SEX", birth="YEAR", age_band=None, gu="SQ1_1",
               dong="DONG", seoul="SQ1", frame="DEPART", weight=None),
    2016: dict(pid="코딩ID", sex="SQ2", birth="SQ3", age_band=None, gu="SQ1_1",
               dong="SQ1_2", seoul=None, frame="TYPE", weight=None),
    2018: dict(pid="ID", sex="SQ2", birth="SQ3", age_band=None, gu="SQ1_GU",
               dong=None, seoul="SQ1", frame=None, weight=None),
    2020: dict(pid="IDX", sex="SQ3", birth="SQ4", age_band=None, gu="SQ2",
               dong=None, seoul="SQ1", frame="TYPE", weight="WT"),
    2022: dict(pid="IDX", sex="SQ3", birth="SQ4", age_band=None, gu="SQ1",
               dong=None, seoul=None, frame="GRP", weight="wt3_b"),
    2024: dict(pid="id", sex="SQ3", birth="SQ4_1", age_band=None, gu="SQ1",
               dong="SQ2_2", seoul=None, frame=None, weight="WT1_2"),
}

# Which (year, group) files to read. Samples outside the general public -- the
# institutional members, the culture-interested group, disabled respondents -- are
# folded in too: they do not project to a population, but they were measured with the
# same instrument and serve as the comparison group for studying frame effects.
SOURCE = [(2014, "mixed"), (2015, "mixed"), (2016, "mixed"),
          (2018, "general"), (2018, "interest"),
          (2020, "general"), (2020, "interest"),
          (2022, "mixed"),
          (2024, "general"), (2024, "interest"), (2024, "disabled")]

# Which frames are general-public frames, in the sense that they can project to a
# population. The evidence is each report's description of the sample design
# (official_09_wave_metadata.FRAME_DESIGN carries the source line numbers). The
# microdata alone cannot tell you this.
#   2014 a census of foundation members, emailed to 72,348 people -> not general public
#   2015 foundation members plus the existing cultural policy panel -> not general
#   2016 only TYPE=1, the general panel of n=2,000, is general public
GENERAL_POPULATION = {
    (2014, 1.0): False,
    (2018, 2.0): False, (2020, 2.0): False,
    (2024, 2.0): False, (2024, 3.0): False,
    (2015, 9.0): False, (2015, 10.0): False,
    (2016, 1.0): True, (2016, 2.0): False, (2016, 3.0): False, (2016, 4.0): False,
    (2018, 1.0): True,
    (2020, 1.0): True, (2020, 4.0): True,
    (2022, 1.0): True, (2022, 2.0): False, (2022, 3.0): True,
    (2022, 4.0): False, (2022, 5.0): False,
    (2024, 1.0): True,
}

# The single-frame label for waves with no frame variable. The label values stay in
# the instrument's own wording: they are join keys for later steps.
# 2014 is a census of the foundation's website members and people who had taken part
# in its programmes -- not a general-public sample, so it must not be called general.
SINGLE_FRAME = {
    (2014, "mixed"): ("sfac_member", "서울문화재단 회원 전수조사"),
    (2018, "general"): ("general", "서울시민 온라인 패널"),
    (2018, "interest"): ("interest", "기관회원"),
    (2020, "interest"): ("interest", "문화관심집단"),
    (2024, "general"): ("general", "서울시민 표본"),
    (2024, "interest"): ("interest", "문화관심집단"),
    (2024, "disabled"): ("disabled", "장애인"),
}

# In 2020, TYPE is the mode (1 = panel, 4 = offline), and both are general public.
FRAME_LABEL = {
    2015: C.FRAME_MAPS[2015], 2016: C.FRAME_MAPS[2016], 2022: C.FRAME_MAPS[2022],
    2020: {1: ("general", "패널"), 2: ("general", "배너"), 3: ("general", "-"),
           4: ("general", "오프라인")},
}

AGE_BANDS = [(15, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 200)]

# Where one wave has several files, each group gets its own number so the frame_code
# values do not collide.
SINGLE_FRAME_CODE = {(2018, "general"): 1, (2018, "interest"): 2,
                     (2020, "interest"): 2,
                     (2024, "general"): 1, (2024, "interest"): 2, (2024, "disabled"): 3,
                     (2014, "mixed"): 1}

# A separate file per group may carry different key variables. Only the differences
# are overridden.
GROUP_KEYS = {
    (2024, "disabled"): {"pid": "ID", "gu": "SQ1", "dong": "SQ2_2"},
    (2018, "interest"): {"pid": "ID"},
    (2020, "interest"): {"pid": "IDX"},
    (2024, "interest"): {"pid": "id"},
}


def band(age):
    if pd.isna(age):
        return None
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    return "under15" if age < 15 else None


def main():
    frames, weights, persons = [], [], []

    for wave, group in SOURCE:
        df, meta = C.load(wave, group)
        k = dict(KEYS[wave])
        if (wave, group) in GROUP_KEYS:
            k.update(GROUP_KEYS[(wave, group)])
        df = df[df[k["pid"]].notna()].copy() if k["pid"] else df.copy()  # 2016 trailing blanks

        # ---- frame -------------------------------------------------------
        # Splitting on frame_var applies only where one file mixes several frames
        # (the 2015, 2016 and 2022 mixed files; in 2020 general, TYPE is the mode). A
        # file dedicated to one group is itself the frame, so SINGLE_FRAME applies.
        use_frame_var = bool(k["frame"]) and (wave, group) not in SINGLE_FRAME
        if use_frame_var:
            fv = pd.to_numeric(df[k["frame"]], errors="coerce")
            fmap = FRAME_LABEL.get(wave, {})
            frame_group = fv.map(lambda x: fmap.get(x, ("unknown", "unknown"))[0])
            frame_label = fv.map(lambda x: fmap.get(x, ("unknown", str(x)))[1])
        else:
            fg_, fl_ = SINGLE_FRAME[(wave, group)]
            fv = pd.Series(float(SINGLE_FRAME_CODE.get((wave, group), 1)), index=df.index)
            frame_group = pd.Series(fg_, index=df.index)
            frame_label = pd.Series(fl_, index=df.index)

        # ---- living in Seoul ---------------------------------------------
        if k["seoul"]:
            in_seoul = pd.to_numeric(df[k["seoul"]], errors="coerce") == 1
        else:
            in_seoul = pd.Series(True, index=df.index)

        # ---- demographics ------------------------------------------------
        sex = pd.to_numeric(df[k["sex"]], errors="coerce").map({1: "male", 2: "female"})
        if k["birth"]:
            birth = pd.to_numeric(df[k["birth"]], errors="coerce")
            birth = birth.where((birth >= 1900) & (birth <= wave))
            age = wave - birth
            ab = age.map(band)
        else:
            birth = pd.Series(np.nan, index=df.index)
            age = pd.Series(np.nan, index=df.index)
            # 2014 carries bands only. Codebook: 1 under 20, 2 twenties, 3 thirties,
            # 4 forties, 5 fifties, 6 sixty and over
            m = {1: "under20", 2: "20-29", 3: "30-39", 4: "40-49", 5: "50-59", 6: "60+"}
            ab = pd.to_numeric(df[k["age_band"]], errors="coerce").map(m)

        # A weight may or may not exist for a given group; the 2020 culture-interested
        # file and the 2024 culture-interested and disabled files carry none.
        wcol = k["weight"] if (k["weight"] and k["weight"] in df.columns) else None
        w = pd.to_numeric(df[wcol], errors="coerce") if wcol else pd.Series(np.nan, index=df.index)

        persons.append(pd.DataFrame({
            "wave": wave, "source_group": group,
            # person_id is not unique: in 2020 the panel and offline ID spaces overlap
            # in IDX, giving 890 duplicates. And since a wave can have several group
            # files, the group has to enter the key too -- otherwise 2018 general and
            # interest would share row_uid values.
            "row_uid": [f"{wave}:{group}:{i}" for i in range(len(df))],
            "person_id": df[k["pid"]].astype(str) if k["pid"] else np.arange(len(df)).astype(str),
            "frame_code": fv.values, "frame_group": frame_group.values, "frame_label": frame_label.values,
            "in_seoul": in_seoul.values, "sex": sex.values, "birth_year": birth.values,
            "age": age.values, "age_band": ab.values,
            "gu_code": pd.to_numeric(df[k["gu"]], errors="coerce").values if k["gu"] else np.nan,
            # Depending on the wave this is a code or a neighbourhood name, and 2016
            # mixes both within one wave (19 numeric, 7,687 by name). Do not try to
            # read it as a number; keep it as a string.
            "dong_code": (df[k["dong"]].astype("string") if k["dong"]
                          else pd.Series(pd.NA, index=df.index, dtype="string")),
            "weight_var": wcol or "", "weight": w.values,
        }))

        # ---- frame register ------------------------------------------------
        tmp = pd.DataFrame({"frame_code": fv, "frame_group": frame_group,
                            "frame_label": frame_label, "in_seoul": in_seoul})
        for (fc, fg, fl), g in tmp.groupby(["frame_code", "frame_group", "frame_label"], dropna=False):
            frames.append(dict(wave=wave, frame_code=fc, frame_group=fg, frame_label=fl,
                               n=len(g), n_seoul=int(g.in_seoul.sum()),
                               general_population=GENERAL_POPULATION.get((wave, fc)),
                               weight_var=wcol or ""))

        # ---- weight register -----------------------------------------------
        cand = [c for c in df.columns if c.upper().startswith("WT") or c.lower().startswith("wt")]
        labels = (meta.column_names_to_labels if meta else {})
        vd = pd.read_csv(C.OUT / "official_variable_dictionary.csv")
        vd = vd[(vd.wave == wave) & (vd.group == group)]
        lab_map = dict(zip(vd.variable, vd.label.fillna("")))
        for c in cand:
            s = pd.to_numeric(df[c], errors="coerce")
            weights.append(dict(
                wave=wave, variable=c,
                label=labels.get(c) or lab_map.get(c, ""),
                n=int(s.notna().sum()), sum=round(float(s.sum()), 2),
                mean=round(float(s.mean()), 4), min=round(float(s.min()), 4),
                max=round(float(s.max()), 4),
                is_published_weight=(c == k["weight"]),
            ))
        if not cand:
            weights.append(dict(wave=wave, variable="", label="no weight variable", n=0,
                                sum=np.nan, mean=np.nan, min=np.nan, max=np.nan,
                                is_published_weight=False))

    # ---- diagnose the effect of the weights ----------------------------------
    import official_02_instrument as I

    eff = []
    GEN = {2020: ("general", None), 2022: ("mixed", [1, 3]), 2024: ("general", None)}
    for wave, (grp, gf) in GEN.items():
        spec = I.BATTERY[wave]
        df, _m = C.load(wave, grp)
        if gf is not None:
            df = df[pd.to_numeric(df["GRP"], errors="coerce").isin(gf)]
        wv = pd.to_numeric(df[KEYS[wave]["weight"]], errors="coerce")
        for pos, canon in spec["items"]:
            if spec["response_format"] == "multi_select_then_count":
                slots = df[spec["multi_response_slots"]].apply(pd.to_numeric, errors="coerce")
                y = (slots == pos).any(axis=1)
            else:
                y = pd.to_numeric(df[spec["measures"]["total_count"].format(i=pos)],
                                  errors="coerce").fillna(0) > 0
            uw = float(y.mean() * 100)
            ww = float(np.average(y.astype(float), weights=wv) * 100)
            w_att, w_non = float(wv[y].mean()), float(wv[~y].mean())
            eff.append(dict(
                wave=wave, canonical_item=canon, weight_var=KEYS[wave]["weight"],
                unweighted=round(uw, 2), weighted=round(ww, 2), shift_pp=round(ww - uw, 2),
                mean_wt_attender=round(w_att, 3), mean_wt_non_attender=round(w_non, 3),
                wt_ratio=round(w_att / w_non, 3) if w_non else np.nan,
            ))
    EFF = pd.DataFrame(eff)

    P = pd.concat(persons, ignore_index=True)
    F = pd.DataFrame(frames)
    W = pd.DataFrame(weights)

    C.OUT.mkdir(parents=True, exist_ok=True)
    C.PERSON.mkdir(parents=True, exist_ok=True)
    assert P.row_uid.is_unique, (
        f"row_uid is not unique -- {int(P.row_uid.duplicated().sum())} colliding rows. "
        "A broken join key invalidates every downstream output")
    C.write_person(P, "official_person_keys")
    F.to_csv(C.OUT / "official_sample_frames.csv", index=False)
    W.to_csv(C.OUT / "official_weight_inventory.csv", index=False)
    EFF.to_csv(C.OUT / "official_weight_effect.csv", index=False)

    print("== frame register ==")
    print(F.to_string(index=False))
    assert F.general_population.notna().all(), \
        "some frame has no general_population setting"
    print("\n== weight register ==")
    print(W.to_string(index=False))
    print("\n== effect of the weights (unweighted -> weighted, and the ratio of mean "
          "weights between attenders and non-attenders) ==")
    print(EFF.sort_values("wt_ratio")[
        ["wave", "canonical_item", "unweighted", "weighted", "shift_pp", "wt_ratio"]
    ].head(8).to_string(index=False))

    print("\n== person-level keys ==")
    g = P.groupby("wave").agg(n=("person_id", "size"), seoul=("in_seoul", "sum"),
                              birth=("birth_year", "count"), sex=("sex", "count"),
                              weighted=("weight", "count"))
    print(g.to_string())
    print("\nwaves with no birth year:", sorted(P[P.birth_year.isna()].wave.unique()))
    dup = P.groupby("wave").person_id.apply(lambda s: int(s.duplicated().sum()))
    print("duplicate person_id counts:", {int(k): v for k, v in dup.items() if v})


if __name__ == "__main__":
    main()
