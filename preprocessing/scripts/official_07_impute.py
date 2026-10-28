"""Classify person-level missingness and impute -- but only what may be imputed.

The aggregate track needed a great deal of imputation to fill the blanks in the
published segment tables. In the microdata those blanks do not exist: any segment can
be computed from the respondents directly. So only three kinds of missingness remain
here, and only one of them may be imputed.

  not_asked_in_wave  the wave did not ask that item. Imputing would invent a trend
                     point for a wave that has none, so it is forbidden -- the same
                     principle as the aggregate track's "never impute at wave level".
  skip_logic         the questionnaire's flow did not ask it, and the value follows
                     logically. In 2024 annual spending is asked only of those with at
                     least one self-paid visit, so a self-paid count of 0 means
                     spending of 0 (the cross-tabulation is perfectly diagonal, with
                     no exceptions).
  item_nonresponse   it was asked and not answered. Only this is imputed.

In practice item_nonresponse is 126 respondents in the 2016 spending and self-paid
blocks, across every item.

The method is a hot deck: donors are drawn from respondents with a similar total
annual use count, within the same wave, frame and item. M = 5 draws are kept, giving a
mean and a spread. A hold-out check measures interval coverage -- measured, not
claimed at its nominal value.

Outputs:
  official_attendance_missingness.csv        wave x item x measure x mechanism x cells
  official_attendance_impute_validation.csv  the hold-out check
  official_person_item_imputed.parquet       (person level -> interim) the imputed
                                             values plus is_imputed
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C
import official_02_instrument as I

RNG = np.random.default_rng(20260908)
M = 5  # number of imputation draws


def classify(item: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Attach a mechanism to every missing cell."""
    asked_waves = {c: {w for w, s in I.BATTERY.items() if c in [x for _, x in s["items"]]}
                   for c in item.canonical_item.unique()}
    m = master.set_index("row_uid")
    rows = []
    for (wave, canon), g in item.groupby(["wave", "canonical_item"]):
        battery_asked = m.loc[g.row_uid, "battery_asked"].values
        for meas in ["attended", "total_count", "self_paid_count", "annual_cost_krw"]:
            na = g[meas].isna().values
            if not na.any():
                continue
            # respondents who were never asked the battery at all
            n_screen = int((na & ~battery_asked).sum())
            rest = na & battery_asked
            n_skip = 0
            if meas == "annual_cost_krw" and rest.any():
                paid = g.self_paid_count.values
                n_skip = int((rest & (paid == 0)).sum())
                rest = rest & ~(paid == 0)
            if n_screen:
                rows.append(dict(wave=wave, canonical_item=canon, measure=meas,
                                 mechanism="not_asked_screened_out", n_cells=n_screen))
            if n_skip:
                rows.append(dict(wave=wave, canonical_item=canon, measure=meas,
                                 mechanism="skip_logic", n_cells=n_skip))
            if rest.sum():
                rows.append(dict(wave=wave, canonical_item=canon, measure=meas,
                                 mechanism="item_nonresponse", n_cells=int(rest.sum())))
    # Items a wave never asked. The cell count is that wave's respondent count, since
    # it applies to every one of them.
    n_by_wave = master.groupby("wave").size().to_dict()
    for canon, ws in asked_waves.items():
        for w in set(I.BATTERY) - ws:
            rows.append(dict(wave=w, canonical_item=canon, measure="attended",
                             mechanism="not_asked_in_wave", n_cells=int(n_by_wave[w])))
    return pd.DataFrame(rows)


def hotdeck(target: pd.Series, donor_key: pd.Series, need: np.ndarray, m: int):
    """Draw from observed cases with a nearby donor_key. Returns an (M, n_need) array."""
    obs = ~target.isna().values & ~need
    if obs.sum() == 0:
        return None
    dk_obs, y_obs = donor_key.values[obs], target.values[obs]
    out = np.empty((m, int(need.sum())))
    dk_need = donor_key.values[need]
    for j, k in enumerate(dk_need):
        d = np.abs(dk_obs - k)
        pool = np.where(d == d.min())[0]
        if len(pool) < 5:  # thin donor pool: widen to the five nearest
            pool = np.argsort(d)[:5]
        out[:, j] = RNG.choice(y_obs[pool], size=m, replace=True)
    return out


def impute_wave(g: pd.DataFrame, meas: str, m: int):
    need = g[meas].isna().values & g.total_count.notna().values
    if not need.any():
        return None, need
    draws = hotdeck(g[meas], g.total_count.fillna(0), need, m)
    return draws, need


def main():
    item = C.read_person("official_person_item")
    master = C.read_person("official_person_master")

    tax = classify(item, master)
    tax.to_csv(C.OUT / "official_attendance_missingness.csv", index=False)

    print("== cells by missingness mechanism ==")
    print(tax.groupby("mechanism").n_cells.agg(["count", "sum"]).to_string())
    print("\n== where item_nonresponse occurs ==")
    inr = tax[tax.mechanism == "item_nonresponse"]
    print(inr.groupby(["wave", "measure"]).n_cells.sum().to_string() if len(inr) else "  none")

    # ---- skip_logic: fill the values that follow logically --------------------
    item["is_imputed"] = False
    item["impute_method"] = ""
    skip = item.annual_cost_krw.isna() & (item.self_paid_count == 0)
    item.loc[skip, "annual_cost_krw"] = 0.0
    item.loc[skip, "impute_method"] = "skip_logic_zero"
    print(f"\nskip_logic filled: {int(skip.sum()):,} cells (self-paid 0 -> spending 0)")

    # ---- impute item_nonresponse ----------------------------------------------
    mfr = master.set_index("row_uid").frame_code
    item["frame_code"] = item.row_uid.map(mfr)
    filled = 0
    spread = []
    for (wave, canon, frame), g in item.groupby(["wave", "canonical_item", "frame_code"]):
        for meas in ["self_paid_count", "annual_cost_krw"]:
            draws, need = impute_wave(g, meas, M)
            if draws is None:
                continue
            idx = g.index[need]
            item.loc[idx, meas] = draws.mean(axis=0)
            item.loc[idx, "is_imputed"] = True
            item.loc[idx, "impute_method"] = f"hotdeck_m{M}_within_frame"
            filled += len(idx)
            sd = draws.std(axis=0, ddof=1)
            spread.append(pd.DataFrame(dict(wave=wave, canonical_item=canon, measure=meas,
                                            row_uid=g.row_uid.values[need],
                                            imputed=draws.mean(axis=0), sd=sd)))
    print(f"item_nonresponse imputed: {filled:,} cells")

    # ---- hold-out check: hide observed cells and measure interval coverage -----
    val = []
    for (wave, canon), g in item.groupby(["wave", "canonical_item"]):
        for meas in ["self_paid_count", "annual_cost_krw"]:
            obs = g[g[meas].notna() & ~g.is_imputed]
            if len(obs) < 200:
                continue
            hold = obs.sample(min(150, len(obs) // 5), random_state=7)
            gg = g.copy()
            mask = gg.index.isin(hold.index)
            truth = gg.loc[mask, meas].values
            gg.loc[mask, meas] = np.nan
            draws = hotdeck(gg[meas], gg.total_count.fillna(0), mask, M)
            if draws is None:
                continue
            lo, hi = draws.min(axis=0), draws.max(axis=0)
            val.append(dict(wave=wave, canonical_item=canon, measure=meas, n_held=len(truth),
                            coverage=float(((truth >= lo) & (truth <= hi)).mean()),
                            mae=float(np.abs(draws.mean(axis=0) - truth).mean())))
    V = pd.DataFrame(val)
    V.to_csv(C.OUT / "official_attendance_impute_validation.csv", index=False)
    C.write_person(item, "official_person_item_imputed")

    if len(V):
        print("\n== hold-out check (coverage of the M=5 donor range) ==")
        print(V.groupby("measure").agg(cells=("n_held", "sum"),
                                       coverage=("coverage", "mean"),
                                       mae=("mae", "mean")).round(3).to_string())
    print("\nleft unimputed by rule (not_asked_in_wave):")
    na = tax[tax.mechanism == "not_asked_in_wave"]
    for canon, g in na.groupby("canonical_item"):
        print(f"  {canon:<22} {sorted(g.wave)}")


if __name__ == "__main__":
    main()
