"""Birth cohort coverage -- which waves hold which generations.

Comparing generations, Gen Z above all, starts with checking whether the cohort is
observed in every wave at all. The survey covers ages 15 and over, so young cohorts
appear only in recent waves, and even there not across their whole range.

**2014 carries no birth year**, only six age bands, so it enters cohort analysis by
band alone.

Outputs:
  official_birth_cohort_counts.csv  wave x frame x birth year x n x weighted n
  official_generation_coverage.csv  wave x generation definition x observed cohort
                                    range x coverage
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

# There is no single accepted definition of a generation. Both are carried, so that
# whichever is used has to be stated.
GENERATIONS = {
    "gen_z_1997_2012": (1997, 2012),
    "gen_z_1995_2009": (1995, 2009),
    "millennial_1981_1996": (1981, 1996),
    "gen_x_1965_1980": (1965, 1980),
    "boomer_1955_1964": (1955, 1964),
}


def main():
    keys = C.read_person("official_person_keys")
    fr = pd.read_csv(C.OUT / "official_sample_frames.csv")
    gen_codes = {(int(r.wave), float(r.frame_code)) for r in fr.itertuples() if r.general_population}

    keys["is_general_frame"] = [
        (int(w), float(f)) in gen_codes for w, f in zip(keys.wave, keys.frame_code)
    ]

    cc = (keys.dropna(subset=["birth_year"])
          .groupby(["wave", "frame_label", "is_general_frame", "birth_year"], as_index=False)
          .agg(n=("row_uid", "size"), n_weighted=("weight", "sum")))
    cc["age_at_survey"] = cc.wave - cc.birth_year
    cc.to_csv(C.OUT / "official_birth_cohort_counts.csv", index=False)

    rows = []
    for wave, g in keys.groupby("wave"):
        gen_only = g[g.is_general_frame]
        base = gen_only if len(gen_only) else g
        by = base.birth_year.dropna()
        for name, (lo, hi) in GENERATIONS.items():
            if not len(by):
                rows.append(dict(wave=wave, generation=name, gen_from=lo, gen_to=hi,
                                 observed_min=np.nan, observed_max=np.nan, n=0,
                                 cohorts_expected=hi - lo + 1, cohorts_observed=0,
                                 coverage=0.0,
                                 note="no birth year variable (age bands only)"))
                continue
            sel = by[(by >= lo) & (by <= hi)]
            obs = sorted(sel.unique())
            rows.append(dict(
                wave=wave, generation=name, gen_from=lo, gen_to=hi,
                observed_min=(min(obs) if obs else np.nan),
                observed_max=(max(obs) if obs else np.nan),
                n=int(len(sel)), cohorts_expected=hi - lo + 1,
                cohorts_observed=len(obs),
                coverage=round(len(obs) / (hi - lo + 1), 3),
                note=("general-public frame" if len(gen_only)
                      else "no general-public frame -- all frames")))
    G = pd.DataFrame(rows)
    G.to_csv(C.OUT / "official_generation_coverage.csv", index=False)

    print("== observed range of birth years (general-public frame) ==")
    for wave, g in keys.groupby("wave"):
        gen_only = g[g.is_general_frame]
        base = gen_only if len(gen_only) else g
        by = base.birth_year.dropna()
        tag = "" if len(gen_only) else "  (no general-public frame)"
        if not len(by):
            print(f"  {wave}: no birth year -- age bands only{tag}")
        else:
            print(f"  {wave}: born {int(by.min())}-{int(by.max())}  n={len(by):,}{tag}")

    print("\n== generation coverage (cohorts observed / cohorts in the definition) ==")
    p = G.pivot_table(index="generation", columns="wave", values="coverage")
    print(p.round(2).to_string())

    z = G[G.generation == "gen_z_1997_2012"]
    full = z[z.coverage >= 0.999]
    print(f"\nwaves covering all of Gen Z (1997-2012): "
          f"{sorted(full.wave.tolist()) if len(full) else 'none'}")
    obs = z[z.n > 0]
    if len(obs):
        lo, hi = int(obs.observed_min.max()), int(obs.observed_max.min())
        print(f"cohorts of Gen Z observed in every wave: born {lo}-{hi}")
    gp = z[z.wave.isin([2016, 2018, 2020, 2022, 2024]) & (z.n > 0)]
    if len(gp):
        lo, hi = int(gp.observed_min.max()), int(gp.observed_max.min())
        print(f"cohorts common to the general-public frames (2016-2024): born "
              f"{lo}-{hi} -- a generational comparison reaching outside that range "
              f"compares different people in each wave")


if __name__ == "__main__":
    main()
