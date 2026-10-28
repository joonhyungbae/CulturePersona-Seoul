"""Attendance series built directly from the microdata, with comparability grades.

Rather than transcribing the published tables, the rates are recomputed from the
respondents. That is the only way to guarantee the definition is the same in every
wave, and it allows segments beyond the ones that were published.

A caution: this series guarantees the same question, not the same population. Frame and
mode differ by wave (2014 and 2015 are convenience-sample panels, 2016 and 2018 online
panels, 2020 panel plus offline, 2022 panel plus face-to-face, 2024 rake-weighted). The
regime speaks only to the instrument being identical. Always read frame_note alongside.

2016 and 2022 each measured **two populations with one instrument in the same wave**.
That is the only place a frame effect can be measured with no instrument difference in
the way, and having it twice, six years apart, shows whether the effect is stable. (It
replaces the old aggregate track's survey_early_bridge.)

**The comparison must be unweighted.** 2016 carries no weight and 2022 carries `wt3_b`,
which is entangled with the outcome (movies and theater). Comparing a weighted 2022
with an unweighted 2016 measures the difference in weighting design, not the frame
effect: done that way the between-wave correlation falls to 0.55, while putting both on
an unweighted footing gives 0.89 (checked 2026-09-08). The weighted version is exported
alongside, but the verdict rests on the unweighted one.

Outputs:
  official_series_estimates.csv     wave x population x segment x item x estimate x
                                    standard error
  official_series_comparability.csv item x adjacent wave pair x grade x reasons
  official_frame_effect.csv         the frame effect measured within one wave on one
                                    instrument (2016 and 2022)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C
import official_02_instrument as I

# The population branches are built per wave. Which frame counts as general public is
# decided by general_population in official_sample_frames.csv, which rests on the
# reports' sample design.
#
# Important -- **2014 and 2015 have no general-public frame at all.** Both are
# foundation members and the cultural policy panel. So the general-public series runs
# over five waves, starting in 2016.
POP_NOTE = {
    2014: "census of foundation members · Seoul residents only (795 non-Seoul cases "
          "screened out) · unweighted",
    2015: "foundation members plus the existing cultural policy panel · unweighted",
    2016: "general panel n=2,000 · online panel · unweighted",
    2018: "online panel n=2,545 · unweighted. The cultural-activity adjustment factor "
          "used in the published figures is not in the delivered file",
    2020: "online panel plus offline · weighted by WT",
    2022: "online panel plus face-to-face · weighted by wt3_b, raked on movies and "
          "theater",
    2024: "rake-weighted, WT1_2",
}


def wprop(y: np.ndarray, w: np.ndarray):
    """A weighted proportion, its linearised standard error, and Kish's effective n."""
    m = ~np.isnan(y)
    y, w = y[m], w[m]
    if len(y) == 0 or w.sum() == 0:
        return np.nan, np.nan, 0, 0.0
    sw = w.sum()
    p = float((w * y).sum() / sw)
    var = float((w ** 2 * (y - p) ** 2).sum() / sw ** 2)
    n_eff = float(sw ** 2 / (w ** 2).sum())
    return p, np.sqrt(var), int(len(y)), n_eff


def main():
    ins = pd.read_csv(C.OUT / "official_instrument.csv")
    regime = dict(zip(ins.wave, ins.regime_id))
    order = {r.wave: r.item_order.split("|") for r in ins.itertuples()}
    fmt = dict(zip(ins.wave, ins.response_format))

    item = C.read_person("official_person_item")
    master = C.read_person("official_person_master")

    # The music_dance_combined bridge: from 2016 it is built as western_classical OR
    # dance_performance. That carries a format effect.
    bridge = []
    for wave in [2016, 2018, 2020, 2022, 2024]:
        sub = item[(item.wave == wave) & item.canonical_item.isin(["western_classical", "dance_performance"])]
        g = sub.groupby("row_uid").agg(attended=("attended", "max"),
                                         total_count=("total_count", "sum"),
                                         self_paid_count=("self_paid_count", "sum"),
                                         annual_cost_krw=("annual_cost_krw", "sum")).reset_index()
        g["wave"] = wave
        g["canonical_item"] = "music_dance_combined"
        g["position"] = np.nan
        g["derived"] = True
        bridge.append(g)
    item["derived"] = False
    item = pd.concat([item] + bridge, ignore_index=True)

    md = master.set_index("row_uid")
    assert md.index.is_unique, "row_uid is not unique"
    n_before = len(item)
    item = item.join(md[["frame_code", "frame_group", "sex", "age_band", "weight", "in_seoul"]],
                     on="row_uid")
    assert len(item) == n_before, "the join added rows"

    fr = pd.read_csv(C.OUT / "official_sample_frames.csv")
    pops = {}
    for wave, g in fr.groupby("wave"):
        gen = sorted(g[g.general_population].frame_code)
        if gen:
            pops[(wave, "general_population")] = gen
        non = sorted(g[~g.general_population].frame_code)
        if non:
            pops[(wave, "engaged_frame")] = non

    rows = []
    for (wave, pop), codes in sorted(pops.items()):
        note = POP_NOTE[wave]
        d = item[(item.wave == wave) & (item.frame_code.isin(codes))].copy()
        if wave == 2014:
            d = d[d.in_seoul]
        if not len(d):
            continue
        weighted = d.weight.notna().any()
        d["w"] = d.weight.fillna(1.0) if weighted else 1.0

        segs = {"overall": pd.Series("overall", index=d.index),
                "sex": d.sex.map(lambda x: f"sex={x}" if pd.notna(x) else None),
                "age": d.age_band.map(lambda x: f"age={x}" if pd.notna(x) else None)}
        for dim, sv in segs.items():
            d["_seg"] = sv
            for (it, seg), g in d.dropna(subset=["_seg"]).groupby(["canonical_item", "_seg"]):
                y = g.attended.values.astype(float)
                p, se, n, ne = wprop(y, g.w.values.astype(float))
                # some waves weight on the outcome, so the unweighted figure always
                # ships alongside
                pu, seu, _n, _ne = wprop(y, np.ones(len(y)))
                if np.isnan(p):
                    continue
                rows.append(dict(
                    wave=wave, population=pop, regime_id=regime[wave], frame_note=note,
                    dimension=dim,
                    segment=seg, canonical_item=it, derived=bool(g.derived.iloc[0]),
                    n=n, n_eff=round(ne, 1), weighted=weighted,
                    estimate=round(p * 100, 3), se=round(se * 100, 3),
                    estimate_unweighted=round(pu * 100, 3), se_unweighted=round(seu * 100, 3),
                    weight_shift_pp=round((p - pu) * 100, 3),
                    ci_low=round(max(0, p - 1.96 * se) * 100, 3),
                    ci_high=round(min(1, p + 1.96 * se) * 100, 3),
                    position=(order[wave].index(it) + 1) if it in order[wave] else None,
                ))
    E = pd.DataFrame(rows)

    # ---- comparability across adjacent wave pairs ------------------------
    waves = sorted(I.BATTERY)
    comp = []
    for it in E.canonical_item.unique():
        for a, b in zip(waves, waves[1:]):
            ea = E[(E.wave == a) & (E.canonical_item == it) & (E.dimension == "overall")].head(1)
            eb = E[(E.wave == b) & (E.canonical_item == it) & (E.dimension == "overall")].head(1)
            if not len(ea) or not len(eb):
                comp.append(dict(canonical_item=it, from_wave=a, to_wave=b,
                                 grade="not_asked",
                                 reasons="the item is absent from one of the waves"))
                continue
            # The grade separates "did this item change" from "did the battery around
            # it change".
            #   comparable        neither the item nor the battery moved
            #   context_changed   the item held, but another choice in the same battery
            #                     changed
            #   design_perturbed  this item's choice position or response format changed
            #   not_comparable    the item is differently constituted (a combined choice
            #                     against a derived union)
            reasons, sev = [], 0
            if bool(ea.derived.iloc[0]) != bool(eb.derived.iloc[0]):
                reasons.append("one side is derived (western_classical OR dance) -- "
                               "a different form from the combined choice")
                sev = max(sev, 3)
            pa, pb = ea.position.iloc[0], eb.position.iloc[0]
            if pd.notna(pa) and pd.notna(pb) and pa != pb:
                reasons.append(f"choice position {int(pa)}->{int(pb)}")
                sev = max(sev, 2)
            if fmt[a] != fmt[b]:
                reasons.append(f"response format {fmt[a]}->{fmt[b]}")
                sev = max(sev, 2)
            set_a, set_b = set(order[a]), set(order[b])
            if set_a != set_b:
                d = sorted((set_a - set_b) | (set_b - set_a))
                reasons.append("battery composition changed (this item did not): "
                               + ", ".join(d))
                sev = max(sev, 1)
            elif order[a] != order[b]:
                reasons.append("another choice in the battery moved position (this "
                               "item did not)")
                sev = max(sev, 1)
            grade = ["comparable", "context_changed", "design_perturbed", "not_comparable"][sev]
            comp.append(dict(canonical_item=it, from_wave=a, to_wave=b, grade=grade,
                             reasons=" · ".join(reasons) or "instrument identical"))
    K = pd.DataFrame(comp)

    # The derived item (music_dance_combined) inherits the grade of its constituents.
    SEV = {"comparable": 0, "context_changed": 1, "design_perturbed": 2,
           "not_comparable": 3, "not_asked": 9}
    for i, r in K[K.canonical_item == "music_dance_combined"].iterrows():
        parts = K[(K.canonical_item.isin(["western_classical", "dance_performance"]))
                  & (K.from_wave == r.from_wave) & (K.to_wave == r.to_wave)]
        parts = parts[parts.grade != "not_asked"]
        if not len(parts) or SEV[r.grade] == 9:
            continue
        worst = max(parts.grade, key=lambda g: SEV[g])
        if SEV[worst] > SEV[r.grade]:
            K.at[i, "grade"] = worst
            K.at[i, "reasons"] = (r.reasons + " · inherited from the constituents: "
                                  + "; ".join(sorted(set(parts.reasons))))

    # ---- frame effect: one wave, one instrument -------------------------------
    def _rate(sub, wcol_used):
        return sub.set_index("canonical_item")

    fe = []
    ov = E[E.dimension == "overall"]
    for wave in sorted(set(ov.wave)):
        g = ov[ov.wave == wave]
        a = g[g.population == "general_population"].set_index("canonical_item")
        b = g[g.population == "engaged_frame"].set_index("canonical_item")
        for it in sorted(set(a.index) & set(b.index)):
            pa, pb = a.loc[it, "estimate"] / 100, b.loc[it, "estimate"] / 100
            ua, ub = a.loc[it, "estimate_unweighted"] / 100, b.loc[it, "estimate_unweighted"] / 100
            rec = dict(wave=wave, canonical_item=it,
                       general=round(pa * 100, 2), engaged=round(pb * 100, 2),
                       general_unweighted=round(ua * 100, 2), engaged_unweighted=round(ub * 100, 2))
            if 0 < pa < 1 and 0 < pb < 1:
                rec["log_odds_ratio_weighted"] = round(
                    np.log(pb / (1 - pb)) - np.log(pa / (1 - pa)), 3)
            if 0 < ua < 1 and 0 < ub < 1:
                rec["log_odds_ratio"] = round(
                    np.log(ub / (1 - ub)) - np.log(ua / (1 - ua)), 3)
            rec["diff_pp"] = round((ub - ua) * 100, 2)
            fe.append(rec)
    FE = pd.DataFrame(fe).dropna(subset=["log_odds_ratio"])
    FE.to_csv(C.OUT / "official_frame_effect.csv", index=False)
    if len(FE):
        print("\n== frame effect (one wave, one instrument, unweighted; engaged "
              "against general) ==")
        pv = FE.pivot_table(index="canonical_item", columns="wave", values="log_odds_ratio")
        print(pv.round(2).to_string())
        for w, g in FE.groupby("wave"):
            print(f"  {w}: median log-odds difference {g.log_odds_ratio.median():.2f} "
                  f"· range {g.log_odds_ratio.min():.2f} to "
                  f"{g.log_odds_ratio.max():.2f}")
        both = FE.pivot_table(index="canonical_item", columns="wave", values="log_odds_ratio").dropna()
        if both.shape[1] == 2:
            c = both.iloc[:, 0].corr(both.iloc[:, 1])
            cw_ = both.iloc[:, 0].corr(both.iloc[:, 1])
            bw = FE.pivot_table(index="canonical_item", columns="wave",
                                values="log_odds_ratio_weighted").dropna()
            cwe = bw.iloc[:, 0].corr(bw.iloc[:, 1]) if bw.shape[1] == 2 else float("nan")
            print(f"  correlation of the per-item frame effect between 2016 and 2022 "
                  f"-- unweighted {cw_:.3f} / weighted {cwe:.3f}")
            print("  (the weighted version measures the difference in weighting "
                  "design, not the frame effect, because 2022's wt3_b is entangled "
                  "with the outcome. The verdict rests on the unweighted one.)")

    E.to_csv(C.OUT / "official_series_estimates.csv", index=False)
    K.to_csv(C.OUT / "official_series_comparability.csv", index=False)

    for pop, lab in [("general_population", "general-public frames"),
                     ("engaged_frame", "culture-engaged frames")]:
        sub = E[(E.dimension == "overall") & (E.population == pop)]
        if not len(sub):
            continue
        print(f"\n== attendance rate, {lab} (computed from the microdata, %) ==")
        print(sub.pivot_table(index="canonical_item", columns="wave",
                              values="estimate").round(1).to_string())
    print("\n== grade summary across adjacent waves ==")
    print(K.pivot_table(index="canonical_item", columns=["from_wave"], values="grade",
                        aggfunc="first").to_string())
    print(f"\n{len(E):,} estimate rows · {len(K)} comparability rows")


if __name__ == "__main__":
    main()
