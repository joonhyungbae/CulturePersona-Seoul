"""Enforce the official track's contracts. A violation exits non-zero.

A dataset with no contract violates this repository's practice. Adding a new output
means updating the invariants here alongside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C
import official_02_instrument as I

FAILS: list[str] = []
CHECKS = 0


def check(cond: bool, msg: str):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)


def main():
    O, P = C.OUT, C.PERSON

    # ---- outputs exist --------------------------------------------------
    required = [
        "official_variable_dictionary.csv", "official_value_labels.csv",
        "official_item_battery.csv", "official_instrument.csv",
        "official_regime_summary.csv", "official_regime_breaks.csv",
        "official_sample_frames.csv", "official_weight_inventory.csv",
        "official_margin_reproduction.csv", "official_reproduction_summary.csv",
        "official_schema_coverage.csv", "official_series_estimates.csv",
        "official_series_comparability.csv", "official_attendance_missingness.csv",
        "official_attendance_impute_validation.csv", "official_weight_effect.csv",
        "official_wave_metadata.csv", "official_frame_design.csv",
        "official_frame_effect.csv", "official_early_published_check.csv",
        "official_canonical_map.csv", "official_domain_coverage.csv",
        "official_living_area_estimates.csv", "official_living_area_tests.csv",
        "official_birth_cohort_counts.csv", "official_generation_coverage.csv",
        "official_weight_provenance.csv", "official_weight_item_residuals.csv",
        "official_domain_declarations.csv", "official_domain_family_coverage.csv",
        "official_domain_item_keys.csv", "official_domain_missingness.csv",
        "official_domain_impute_validation.csv",
        "official_variable_registry.csv", "official_registry_summary.csv",
        "official_file_manifest.csv", "official_column_profile.csv",
        "official_join_map.csv", "official_person_coverage.csv",
    ]
    for f in required:
        check((O / f).exists(), f"output missing: {f}")

    bat = pd.read_csv(O / "official_item_battery.csv")
    ins = pd.read_csv(O / "official_instrument.csv")
    est = pd.read_csv(O / "official_series_estimates.csv")
    rep = pd.read_csv(O / "official_margin_reproduction.csv")
    frames = pd.read_csv(O / "official_sample_frames.csv")
    tax = pd.read_csv(O / "official_attendance_missingness.csv")
    comp = pd.read_csv(O / "official_series_comparability.csv")

    # ---- the battery ------------------------------------------------------
    check(bat.in_data.all(), "a battery variable is absent from the data")
    check(set(bat.wave) == set(I.BATTERY), "the battery is missing a wave")
    for wave, g in bat.groupby("wave"):
        pos = sorted(g.position.unique())
        check(pos == list(range(1, len(pos) + 1)),
              f"{wave}: choice positions are not a run of 1..n")

    # ---- fingerprints and regimes -----------------------------------------
    check(ins.wave.is_unique, "a wave appears twice in the fingerprints")
    # One regime means one item order and one response format -- that is its definition
    for rid, g in ins.groupby("regime_id"):
        check(g.item_order.nunique() == 1, f"{rid}: one regime, differing item orders")
        check(g.response_format.nunique() == 1,
              f"{rid}: one regime, differing response formats")
    # And conversely, one fingerprint must mean one regime
    check(ins.groupby(["item_order", "response_format"]).regime_id.nunique().eq(1).all(),
          "one fingerprint has been split across regimes")
    # A fact the data states; if it flips, the report text is void
    r3 = set(ins[ins.regime_id == ins.set_index("wave").regime_id[2016]].wave)
    check({2016, 2018, 2020} <= r3, "2016, 2018 and 2020 are no longer one regime")

    # ---- reproduction -----------------------------------------------------
    for wave, tol in [(2022, 0.10), (2024, 0.10)]:
        g = rep[rep.wave == wave]
        check(len(g) > 0, f"no reproduction result for {wave}")
        check(g.abs_error.max() <= tol,
              f"{wave} reproduction error exceeds {tol} pp "
              f"(max {g.abs_error.max():.3f})")
    check((rep[rep.wave == 2018].verdict == "differs").all(),
          "2018 now appears to rebuild from the delivered file -- this contradicts "
          "the report")

    # ---- samples ----------------------------------------------------------
    check(int(frames[frames.wave == 2014].n_seoul.sum()) == 2110,
          "the 2014 Seoul sample is not 2,110")
    # Every group must be harmonised in, including the institutional members, the
    # culture-interested group and the disabled sample
    _keys = C.read_person("official_person_keys")
    check(_keys.row_uid.is_unique, "row_uid is not unique -- the join key is broken")
    check(len(_keys) == 48433,
          f"the harmonised sample is not 48,433 people ({len(_keys)})")
    check(set(zip(_keys.wave, _keys.source_group)) == set(I.ALL_GROUPS),
          "the harmonised (wave, group) set differs from ALL_GROUPS")
    check(int(frames[(frames.wave == 2022) & (frames.frame_code.isin([1.0, 3.0]))].n.sum()) == 5011,
          "the 2022 general-public sample is not 5,011")

    # ---- estimates --------------------------------------------------------
    check(est.estimate.between(0, 100).all(), "an estimate falls outside 0 to 100")
    check((est.se >= 0).all(), "a standard error is negative")
    check((est.ci_low <= est.estimate + 1e-9).all() and (est.estimate <= est.ci_high + 1e-9).all(),
          "an estimate falls outside its confidence interval")
    check((est.n_eff <= est.n + 1e-6).all(),
          "the effective sample size exceeds the actual one")
    # A derived item must always be marked as one
    md = est[est.canonical_item == "music_dance_combined"]
    check(md[md.wave.isin([2016, 2018, 2020, 2022, 2024])].derived.all(),
          "music_dance_combined from 2016 onwards is not marked derived")
    check(not md[md.wave.isin([2014, 2015])].derived.any(),
          "music_dance_combined in 2014 and 2015 is wrongly marked derived")

    # ---- comparability ----------------------------------------------------
    ok = {"comparable", "context_changed", "design_perturbed", "not_comparable", "not_asked"}
    check(set(comp.grade) <= ok, f"grade vocabulary violated: {set(comp.grade) - ok}")
    for a, b in [(2016, 2018), (2018, 2020)]:
        g = comp[(comp.from_wave == a) & (comp.to_wave == b) & (comp.grade != "not_asked")]
        g = g[g.canonical_item != "music_dance_combined"]
        check((g.grade == "comparable").all(),
              f"{a} to {b} is not comparable throughout -- this contradicts the "
              "regime decision")

    # ---- frames and populations ---------------------------------------------
    fd = pd.read_csv(O / "official_frame_design.csv")
    check(fd.general_population.notna().all(),
          "some frame has no general_population setting")
    check(fd.sampling.notna().all(),
          "some frame has no sample design description")
    # That 2014 and 2015 have no general-public frame underpins everything this track
    # says
    for wave in (2014, 2015):
        check(not fd[(fd.wave == wave) & fd.general_population].shape[0],
              f"{wave} now has a general-public frame -- this contradicts the report")
    check(fd[(fd.wave == 2016) & fd.general_population].n.sum() == 2000,
          "the 2016 general-public frame is not 2,000 people")
    check(set(est.population) <= {"general_population", "engaged_frame"},
          "population vocabulary violated")
    gp = sorted(set(est[est.population == "general_population"].wave))
    check(gp == [2016, 2018, 2020, 2022, 2024],
          f"the general-public series is not the five waves 2016 to 2024: {gp}")

    # ---- wave metadata --------------------------------------------------------
    wm = pd.read_csv(O / "official_wave_metadata.csv")
    check(sorted(wm.wave) == sorted(I.BATTERY), "wave metadata is missing a wave")
    r2020 = wm[wm.wave == 2020].iloc[0]
    check(str(r2020.fielding_start).startswith("2021"),
          "the 2020 survey is not recorded as fielded in 2021")
    check(set(wm.recall_basis) <= {"stated", "inferred"},
          "recall_basis vocabulary violated")

    # ---- the early waves against the published figures ------------------------
    ep = pd.read_csv(O / "official_early_published_check.csv")
    for wave in (2014, 2015):
        g = ep[(ep.wave == wave) & ep.metric.isin(["annual_visits", "annual_cost"])]
        check(len(g) == 2 and (g.verdict == "exact").all(),
              f"{wave} visits and spending no longer reproduce exactly -- the evidence "
              "for the sample base and the spending unit collapses")
    lt = ep[ep.metric == "leisure_top3_arts"]
    check(len(lt) == 2 and (lt.verdict == "exact").all(),
          "the finding that the published 'participation rate' is the top-three "
          "leisure share no longer reproduces")

    # ---- frame effect ---------------------------------------------------------
    fe = pd.read_csv(O / "official_frame_effect.csv")
    # Harmonising every group raised the number of waves where a frame effect can be
    # measured from two to five.
    check(set(fe.wave) >= {2016, 2018, 2020, 2022, 2024},
          f"the frame effect does not cover five waves: {sorted(set(fe.wave))}")
    # The engaged frame must sit above the general public. Only popular performance in
    # 2020 goes the other way, at -0.013, which is effectively zero.
    neg = fe[fe.log_odds_ratio < -0.05]
    check(neg.empty,
          "an item puts the engaged frame clearly below the general public: "
          f"{neg[['wave','canonical_item']].to_dict('records')}")
    # Corrected 2026-09-08: the earlier invariant was "correlation < 0.8", but that
    # correlation compared an unweighted 2016 with a weighted 2022 (0.37, 95% CI
    # [-0.12, 0.88] -- indistinguishable from the threshold). Put on the same
    # unweighted footing it is 0.87. Two things are now enforced:
    #   (a) the between-wave correlation, measured unweighted, must be high -- the
    #       effect transports
    #   (b) it must still scatter across items -- there is no single constant
    #       coefficient
    import itertools as _it
    both = fe.pivot_table(index="canonical_item", columns="wave", values="log_odds_ratio").dropna()
    _rs = [both[a].corr(both[b]) for a, b in _it.combinations(sorted(both.columns), 2)]
    check(len(_rs) >= 10 and min(_rs) > 0.6,
          "the lowest wave-pair frame-effect correlation is at or below 0.6 "
          f"(min {min(_rs):.2f}) -- this contradicts the report")
    check(sum(_rs) / len(_rs) > 0.75,
          f"the mean wave-pair correlation is at or below 0.75 "
          f"({sum(_rs)/len(_rs):.2f})")
    for w in sorted(set(fe.wave)):
        g = fe[fe.wave == w].log_odds_ratio
        check(g.max() - g.min() > 0.5,
              f"{w}: the frame effect no longer scatters across items -- this "
              "contradicts the claim that no single coefficient exists")
    check("log_odds_ratio_weighted" in fe.columns,
          "the weighted frame effect is missing -- the weighting sensitivity cannot "
          "be shown")

    # ---- effect of the weights --------------------------------------------
    eff = pd.read_csv(O / "official_weight_effect.csv")
    th = eff[(eff.wave == 2022) & (eff.canonical_item == "theater")]
    check(len(th) == 1 and float(th.wt_ratio.iloc[0]) < 0.35,
          "the 2022 theater weight ratio is no longer under 0.35 -- this contradicts "
          "the report")
    check(eff.wt_ratio.min() == th.wt_ratio.iloc[0],
          "2022 theater is no longer the smallest weight ratio -- this contradicts "
          "the report")

    # ---- weight provenance ----------------------------------------------------
    wp = pd.read_csv(O / "official_weight_provenance.csv")
    adj = wp[wp.adjustment == "demographic+edu+income"]
    for wave, wcol in [(2020, "WT"), (2022, "wt3_b"), (2024, "WT1_2")]:
        r = adj[(adj.wave == wave) & (adj.weight_var == wcol)]
        check(len(r) == 1 and r.verdict.iloc[0] == "outcome_entangled",
              f"{wave} {wcol} no longer reads as entangled with the outcome -- this "
              "contradicts the report")
    # The 2020 WTB is the internal control. If this breaks, doubt the diagnostic itself.
    wtb = adj[(adj.wave == 2020) & (adj.weight_var == "WTB")]
    check(len(wtb) == 1 and wtb.verdict.iloc[0] == "demographic_only",
          "2020 WTB is no longer demographics-only -- the diagnostic has lost its "
          "internal control")

    # ---- living areas: unweighted sensitivity ---------------------------------
    lau = pd.read_csv(O / "official_living_area_estimates.csv")
    check("unweighted" in lau.columns,
          "the living-area estimates carry no unweighted version")
    lt2 = pd.read_csv(O / "official_living_area_tests.csv")
    bu = lt2[lt2.test == "rank_null_bootstrap_unweighted"]
    check(len(bu) == 1 and float(bu.p_value.iloc[0]) < 0.05,
          "the break in rank order disappears when unweighted -- possible weight "
          "contamination; revisit the text")

    # ---- out-of-battery domain harmonisation ----------------------------------
    bd = pd.read_csv(O / "official_domain_declarations.csv")
    check(bd.in_data.all(),
          f"{int((~bd.in_data).sum())} declarations absent from the data")
    check(not bd.overlaps_offline.any(),
          f"{int(bd.overlaps_offline.sum())} declarations overlap the offline battery")
    dup = bd.groupby(["wave", "source_group", "variable"]).size()
    check(int((dup > 1).sum()) == 0,
          "a (wave, group, variable) is declared twice")
    # Every declaration must exist in the representative file; absence from another
    # group file is normal.
    check(bd[bd.is_primary].in_data.all(),
          f"{int((~bd[bd.is_primary].in_data).sum())} declarations absent from the "
          "representative file")
    # This pins "every declaration appears exactly once in the representative file",
    # not today's declaration count.
    _n_decl = len(pd.read_csv(C.ROOT / "config" / "domain_declarations.csv"))
    check(int(bd.is_primary.sum()) == _n_decl,
          f"representative-file declarations ({int(bd.is_primary.sum())}) differ from "
          f"the config declarations ({_n_decl})")
    # Uniqueness is judged on the **output**, not the declarations. One row per
    # variable is normal in a declaration, and a multiple-response slot set binds
    # several variables into one item before expanding into one item per choice.
    lk = pd.read_csv(O / "official_domain_item_keys.csv")
    check(int((lk.n_variables > 1).sum()) == 0,
          "the (wave, family, item, measure) key is not unique in the person x item "
          f"table: {int((lk.n_variables > 1).sum())} cases")
    check(len(lk) > 400, f"too few battery item keys ({len(lk)})")
    # Universe, polarity and unit were sources of silent corruption. None may be blank.
    for col in ("universe", "polarity", "unit"):
        check(bd[col].notna().all() and (bd[col].astype(str).str.strip() != "").all(),
              f"a declaration has an empty {col}")
    # Routed questions really are numerous; if that stops being true, recording the
    # universe has lost its point
    check(bd.is_routed.fillna(False).sum() > 100,
          "fewer than 100 routed questions -- the universe decision may have broken")
    check(set(bd.measure) <= {
        "experienced", "threshold_recode", "frequency", "count", "cost", "satisfaction",
        "satisfaction_frequency", "satisfaction_content", "intention", "intention_recommend",
        "burden", "duration", "channel", "route", "rank", "other"},
        f"measure vocabulary violated: {sorted(set(bd.measure))}")

    # ---- out-of-battery missingness and imputation ----------------------------
    bm = pd.read_csv(O / "official_domain_missingness.csv")
    _gt = pd.read_csv(O / "official_domain_gates.csv")
    tot = int(bm.cells_total.sum()); inr = int(bm.cells_item_nonresponse.sum())
    check(tot > 5_000_000, f"too few cells outside the battery ({tot})")
    # Most missingness is questionnaire flow, not non-response. A large rise in this
    # share means the classification has broken.
    check(inr / tot < 0.10,
          f"item non-response exceeds 10 per cent ({inr/tot*100:.1f}%) -- the routing, "
          "slot or conditional classification may have broken")
    check(bm.slot_cascade.notna().all(), "the slot cascade decision is empty")
    check(int(bm.n_gated_items.sum()) > 50,
          "fewer than 50 conditionally routed items -- the gate decision may have "
          "broken")
    # The gate count in the missingness table and the row count in the audit table
    # count the same thing. A mismatch means one of them silently went stale.
    check(int(bm.n_gated_items.sum()) == len(_gt),
          f"the missingness table's gate count ({int(bm.n_gated_items.sum())}) differs "
          f"from the audit table's row count ({len(_gt)})")
    bv = pd.read_csv(O / "official_domain_impute_validation.csv")
    check(len(bv) > 500 and bv.coverage.mean() > 0.80,
          f"hold-out coverage for the out-of-battery imputation is low "
          f"({bv.coverage.mean():.3f})")
    # Satisfaction is asked per genre, only of people who attended that genre -- the
    # basis of section 5-6-1 of the report. That observation and attendance agree
    # perfectly in both directions is what supports calling it a closed gate rather
    # than non-response. This used to re-read 566 MB and recompute; official_16 already
    # measures it under the same rule and writes it to the audit table, so that is what
    # is pinned -- no wave is listed by hand, so new waves are picked up.
    _mask_gates = _gt[_gt.gate != "exclusive_branch"]
    check((_mask_gates.agreement_reverse == 1.0).all(),
          "reverse agreement for the mask gates is not 1.0 "
          f"(min {_mask_gates.agreement_reverse.min()}) -- these may be non-response "
          "rather than a gate")
    _sat = _gt[(_gt.gate == "attendance_battery") & (_gt.family == "satisfaction")]
    check(len(_sat) >= 8 and _sat.wave.nunique() >= 3
          and (_sat.agreement == 1.0).all() and (_sat.agreement_reverse == 1.0).all(),
          "satisfaction observed and that genre attended no longer agree at 1.0 in "
          f"both directions ({len(_sat)} cases · waves {sorted(_sat.wave.unique())}) "
          "-- this contradicts section 5-6-1 of the report")

    # The missingness section below reuses inr with a different meaning. Hold on to the
    # values the contract file needs here.
    _dom_cells, _dom_inr = tot, inr
    _sat_inr = int(bm[bm.family == "satisfaction"].cells_item_nonresponse.sum())
    # With the item-level gates applied, satisfaction's share must be small. A rise
    # means the gate decision has broken (before 2026-09-09 it was 25 per cent).
    check(_sat_inr / _dom_inr < 0.05,
          "satisfaction's share of out-of-battery item non-response has risen again "
          f"({_sat_inr / _dom_inr * 100:.0f}%) -- the item-level gates have broken")
    check(_dom_inr / _dom_cells < 0.05,
          "out-of-battery item non-response exceeds 5 per cent "
          f"({_dom_inr / _dom_cells * 100:.2f}%)")

    # ---- item-level gates -------------------------------------------------------
    # The result of finding, from the data, the cases where the gate lives outside this
    # table. The evidence for each decision is kept so it can be audited.
    GT = _gt
    check(len(GT) > 50, f"too few gate decisions ({len(GT)})")
    check(set(GT.gate) == {"sibling_experienced", "attendance_battery", "exclusive_branch"},
          f"gate vocabulary violated: {sorted(set(GT.gate))}")
    check((GT.agreement >= 0.95).all(),
          f"a gate decision falls below 0.95 agreement (min {GT.agreement.min():.3f})")
    # All three kinds must survive. A count threshold per kind would need retuning
    # every time a wave is added, and the contract would lose its meaning that way.
    for kind in ("sibling_experienced", "attendance_battery", "exclusive_branch"):
        check(int((GT.gate == kind).sum()) > 0, f"no {kind} gate at all")
    # That satisfaction opens off the attendance battery is the basis of section 5-6-1,
    # and the agreement was 1.0.
    _sg = GT[(GT.gate == "attendance_battery") & (GT.family == "satisfaction")]
    check(len(_sg) > 0 and (_sg.agreement == 1.0).all(),
          f"satisfaction no longer opens off the attendance battery ({len(_sg)} cases, "
          f"lowest agreement {_sg.agreement.min() if len(_sg) else float('nan')})")

    # ---- does the person-level table agree with the raw data? -------------------
    # official_04's attended_any once disagreed with a direct computation from the raw
    # data (0.789 against 0.688 in 2020). A shifted sort or key is silently wrong, so
    # it is checked against the raw data every time.
    _ms = C.read_person("official_person_master")
    for _w, _g in [(2016, "mixed"), (2018, "general"), (2020, "general"),
                   (2022, "mixed"), (2024, "general")]:
        _spec = I.battery_for(_w, _g)
        _df, _ = C.load(_w, _g)
        if _w == 2016:
            _df = _df[_df[list(_df.columns)[0]].notna()]
        if _spec["response_format"] == "multi_select_then_count":
            _slots = _df[_spec["multi_response_slots"]].apply(pd.to_numeric, errors="coerce")
            _att = pd.concat([(_slots == p).any(axis=1) for p, _ in _spec["items"]], axis=1).any(axis=1)
        else:
            _pat = _spec["measures"]["total_count"]
            _cnt = pd.concat([pd.to_numeric(_df[_pat.format(i=p)], errors="coerce").fillna(0)
                              for p, _ in _spec["items"]], axis=1)
            _att = _cnt.sum(axis=1) > 0
        _raw = float(_att.mean())
        _sub = _ms[(_ms.wave == _w) & (_ms.source_group == _g)]
        _got = float(_sub.attended_any.mean())
        check(abs(_raw - _got) < 0.002,
              f"{_w} {_g}: attended_any disagrees with the raw data -- table "
              f"{_got:.4f} against raw {_raw:.4f}")

    # ---- the variable registry --------------------------------------------------
    # This track's aim is that no variable is left out, and the registry enforces it.
    reg = pd.read_csv(O / "official_variable_registry.csv")
    from official_15_domain_items import load_wave, SRC as _SRC, GROUPS as _GRP
    for w, gs in _GRP.items():
        for g in gs:
            cols = set(load_wave(w, g).columns)
            got = set(reg[(reg.wave == w) & (reg.source_group == g)].variable)
            missing = cols - got
            check(not missing,
                  f"{w} {g}: {len(missing)} variables missing from the registry -- "
                  f"{sorted(missing)[:5]}")
            check(not (got - cols),
                  f"{w} {g}: the registry holds a variable that does not exist")
    # This checks that the registry really swept the whole file register, not how big
    # that register is.
    check(set(zip(reg.wave, reg.source_group)) == set(C.groups()),
          "the files the registry swept differ from the register: "
          f"{sorted(set(C.groups()) - set(zip(reg.wave, reg.source_group)))}")
    check(set(reg.tier) <= {"A", "B", "C"},
          f"tier vocabulary violated: {sorted(set(reg.tier))}")
    check(reg.basis.notna().all() and (reg.basis.astype(str).str.strip() != "").all(),
          "a registry row has an empty basis")
    # On the representative files the share must hold as before. Overall it falls once
    # the 2024 disabled file enters (582 unique columns, a different instrument), and
    # that is expected.
    _prim = reg[reg.source_group == reg.wave.map(_SRC)]
    check((_prim.tier == "A").mean() > 0.60,
          "the canonical share on the representative files is under 60 per cent "
          f"({(_prim.tier == 'A').mean()*100:.0f}%)")
    a_share = (reg.tier == "A").mean()
    check(a_share > 0.55,
          f"the overall canonical share is under 55 per cent ({a_share*100:.0f}%)")
    # A variable with no values yet mapped as canonical or wave_local signals a
    # mapping error. (This check caught the merge bug that left all 20 columns of the
    # 2018 DQ block empty.) An empty column in a declared battery is legitimate,
    # though: either nobody chose that slot, or that group has no attenders of that
    # genre (no one in the 2024 disabled sample attended a dance performance).
    ghost = reg[(reg.n_obs == 0) & reg.tier.isin(["A", "B"])
                & ~reg.status.isin(["battery_family", "battery_offline",
                                    "battery_total", "slot_sibling"])]
    check(ghost.empty,
          f"{len(ghost)} variables are mapped yet hold no values -- suspect a merge or "
          f"mapping error: {ghost[['wave','variable']].head(5).to_dict('records')}")

    # ---- manifest and data dictionary -------------------------------------------
    # Stops outputs appearing or disappearing silently. The manifest does for files
    # what the registry does for variables.
    MF = pd.read_csv(O / "official_file_manifest.csv")
    check(bool(MF.declared.all()),
          f"an output is not declared: {MF[~MF.declared].file.tolist()[:5]} -- "
          "update config/output_manifest.csv")
    check(bool(MF.exists.all()),
          f"a declared output does not exist: {MF[~MF.exists].file.tolist()[:5]}")
    # Premise 1, enforced by machine: no person-level file may enter the repository.
    # Unlike a filename-pattern check this uses git check-ignore, so it still catches
    # the file if it is renamed.
    leak = MF[MF.sensitivity.isin(["person_level", "free_text"]) & MF.in_repo.fillna(False)]
    check(leak.empty,
          f"a person-level file is inside the repository: {leak.file.tolist()} -- "
          "check .gitignore")
    # These files are handed out, so every column needs a description.
    # Addressed by (layer, file): `2018_general.pkl` exists both as a raw cache and as
    # an export.
    CPF = pd.read_csv(O / "official_column_profile.csv")
    _pf = MF[MF.sensitivity.isin(["person_level", "free_text"])
             & (MF.layer == "interim") & MF.file.str.endswith(".csv")]
    person_csv = set(zip(_pf.layer, _pf.file))
    _is_person = pd.MultiIndex.from_frame(CPF[["layer", "file"]]).isin(person_csv)
    nodesc = CPF[_is_person & CPF.description.isna()]
    check(nodesc.empty,
          f"{len(nodesc)} columns in person-level files have no description: "
          f"{nodesc[['file','column']].head(5).to_dict('records')} -- "
          "config/person_columns.csv")
    # The join spine. A match rate below 1 means a key has shifted.
    JM = pd.read_csv(O / "official_join_map.csv")
    spine = JM[JM.to_file.str.startswith("official_person_keys")]
    check(len(spine) >= 7 and (spine.match_rate == 1.0).all(),
          "the row_uid join match rate is not 1: "
          f"{spine[spine.match_rate < 1][['from_file','match_rate']].to_dict('records')}")
    # The out-of-battery harmonisation now sweeps all 11 raw files. A gap fails here.
    PCV = pd.read_csv(O / "official_person_coverage.csv")
    gap = PCV[PCV.file.str.startswith("official_person_domain.") & ~PCV.covered]
    check(gap.empty,
          f"a gap has opened in the out-of-battery harmonisation: "
          f"{sorted(zip(gap.wave, gap.source_group))}")
    check((C.ROOT / "reports" / "official_data_dictionary.md").exists(),
          "reports/official_data_dictionary.md is missing")

    # ---- missingness ------------------------------------------------------
    check("not_asked_in_wave" in set(tax.mechanism),
          "the not_asked_in_wave classification is missing")
    inr = tax[tax.mechanism == "item_nonresponse"]
    check(set(inr.wave) <= {2016},
          f"item_nonresponse has appeared outside 2016: {sorted(set(inr.wave))}")

    # ---- the out-of-battery concordance ---------------------------------------
    cm = pd.read_csv(O / "official_canonical_map.csv")
    check(cm.in_data.all(),
          "the canonical concordance holds a variable absent from the data")
    check(cm.groupby(["canonical_var", "wave"]).size().max() == 1,
          "a canonical appears twice in one wave")
    # Variables whose scale moves must always be flagged
    check(cm.scale_stable.notna().any(), "scale_stable was not computed")

    # ---- living areas --------------------------------------------------------
    lt = pd.read_csv(O / "official_living_area_tests.csv")
    la = pd.read_csv(O / "official_living_area_estimates.csv")
    check(set(la.wave) == {2022, 2024} and la.living_area.nunique() == 9,
          "the living-area estimates are not 2022 and 2024 across nine areas")
    boot = lt[lt.test == "rank_null_bootstrap_std"]
    check(len(boot) == 1 and float(boot.p_value.iloc[0]) < 0.05,
          "the break in rank order now reads as explained by sampling error -- this "
          "contradicts the report")
    het = lt[lt.test.str.startswith("heterogeneity_Q_")]
    check(len(het) == 4 and (het.p_value.astype(float) < 0.05).all(),
          "heterogeneity across living areas is no longer significant -- this "
          "contradicts the report")

    # ---- cohorts ---------------------------------------------------------------
    gc = pd.read_csv(O / "official_generation_coverage.csv")
    z = gc[gc.generation == "gen_z_1997_2012"]
    check(len(z) == len(I.BATTERY), "generation coverage is missing a wave")
    check((z.coverage < 0.999).all(),
          "a wave now covers all of Gen Z -- this contradicts the report")
    check(float(gc[(gc.wave == 2014)].coverage.max()) == 0.0,
          "2014 now has a birth year -- it should carry age bands only")

    # ---- scope: the official track alone ------------------------------------------
    # Fails if scripts from an earlier, retired track reappear.
    stale = [p.name for pat in ("microdata_*.py", "harmonize_*.py")
             for p in (C.ROOT / "scripts").glob(pat)]
    check(not stale, f"a retired track's scripts are back: {stale}")
    check(not (C.ROOT / "data" / "processed" / "harmonized").exists(),
          "data/processed/harmonized should not exist -- the published values come from the "
          "frozen copy in data/reference")
    check(not (C.ROOT / "ref").exists(),
          "ref/ should not exist under preprocessing/")
    for f in ("published_tables_2018_2024.csv", "seoul_living_area_crosswalk.csv"):
        check((C.REFERENCE / f).exists(), f"reference material missing: data/reference/{f}")
    pub = pd.read_csv(C.REFERENCE / "published_tables_2018_2024.csv")
    check(not pub.is_imputed.astype(bool).any(),
          "the frozen published copy contains imputed values")
    check(sorted(pub.year.unique()) == [2018, 2020, 2022, 2024],
          "the frozen published copy does not cover 2018 to 2024")

    # ---- exports read by the persona pipeline (src/sfac_persona) ----------------
    legacy = C.ROOT / "data" / "processed" / "microdata" / "microdata_value_labels.csv"
    check(legacy.exists(),
          "microdata_value_labels.csv, which the persona pipeline reads, is missing")
    if legacy.exists():
        lv = pd.read_csv(legacy)
        check(list(lv.columns) == ["year", "group", "var", "value", "value_label"],
              f"the columns differ from the old contract: {list(lv.columns)}")
        # The consumer looks up str(int(x)). With '1.0' everything becomes None, with
        # no crash.
        check(lv.value.astype(str).str.fullmatch(r"-?\d+").all(),
              "a value is not an integer string -- the persona pipeline's lookup breaks "
              "silently")
        check({2018, 2020, 2022, 2024} <= set(lv.year), "the export is missing a wave")
    for y in (2018, 2020, 2022, 2024):
        pk = C.ROOT / "data" / "interim" / "microdata" / f"{y}_general.pkl"
        check(pk.exists(), f"{pk.name}, which the persona pipeline reads, is missing")

    # ---- does the report's table agree with the outputs? ------------------------
    # 2026-09-08: the headline table in section 4 was collapsing population and giving
    # the mean of the general-public and engaged frames (theater 2022 read 27.2 instead
    # of 8.7). None of the 103 contracts then compared the rendered report against the
    # CSVs. This one does.
    rp = C.ROOT / "reports" / "official_overview.md"
    check(rp.exists(), "reports/official_overview.md is missing")
    if rp.exists():
        text = rp.read_text(encoding="utf-8")
        gen_ov = est[(est.dimension == "overall") & (est.population == "general_population")]
        spot = [("theater", 2022), ("exhibition", 2016), ("movie", 2016), ("museum", 2024)]
        for item, wave in spot:
            row = gen_ov[(gen_ov.canonical_item == item) & (gen_ov.wave == wave)]
            if not len(row):
                continue
            val = f"{row.estimate.iloc[0]:.1f}"
            line = next((l for l in text.splitlines()
                         if l.startswith(f"| {item} |")), None)
            check(line is not None and f"| {val} " in line + " ",
                  f"the report table gives {item} {wave} as something other than "
                  f"{val} -- it disagrees with the outputs")

    # ---- has a person-level file leaked into the repository? -------------------
    leaked = list((O).glob("*person_item*")) + list((O).glob("*person_master*")) \
        + list((O).glob("*person_keys*"))
    check(not leaked,
          f"a person-level file sits in a public path: {[p.name for p in leaked]}")

    # ---- the contract file ------------------------------------------------
    contracts = {
        "generated_by": "scripts/validate_official.py",
        "waves": sorted(int(w) for w in ins.wave),
        "regimes": {r: sorted(int(w) for w in g.wave) for r, g in ins.groupby("regime_id")},
        "battery_vars_confirmed": int(bat.in_data.sum()),
        "domain_declarations": {"n": int(len(bd)), "routed": int(bd.is_routed.sum())},
        "domain_item_nonresponse": {
            "cells": _dom_cells, "item_nonresponse": _dom_inr,
            "pct": round(_dom_inr / _dom_cells * 100, 2),
            "satisfaction_gate": _sat_inr,
            "pct_excl_satisfaction_gate": round(
                (_dom_inr - _sat_inr) / _dom_cells * 100, 2),
        },
        "reproduction": {
            str(int(w)): dict(cells=int(len(g)), max_abs_error=float(g.abs_error.max()),
                              exact=int((g.verdict == "exact").sum()))
            for w, g in rep.groupby("wave")
        },
        "general_frame_n": {str(int(r.wave)): int(r.n)
                            for r in frames[frames.frame_group == "general"].itertuples()},
        "general_population_waves": gp,
        "domain_coverage_gap": [
            {"wave": int(r.wave), "source_group": r.source_group, "n": int(r.n_frame)}
            for r in gap.itertuples(index=False)
        ],
        "domain_item_gates": {k: int(v) for k, v in GT.gate.value_counts().items()},
        "registry": {"rows": int(len(reg)), "files": sum(len(v) for v in _GRP.values()),
                     "tier_A_share_primary": round(float((_prim.tier == "A").mean()), 4)},
        "living_area_rank_test": {
            r.test: dict(statistic=float(r.statistic), p_value=float(r.p_value))
            for r in pd.read_csv(O / "official_living_area_tests.csv")
            .query("test.str.startswith('rank_null_bootstrap') or "
                   "test.str.startswith('baseline_vs_change')").itertuples()
        },
        "invariants": [
            "the break in rank order across the nine living areas is not explained by "
            "sampling error (bootstrap p<0.05)",
            "2014 and 2015 have no general-public frame (foundation members and the "
            "cultural policy panel)",
            "the general-public series is the five waves 2016 to 2024",
            "the published arts participation rate for 2014 and 2015 is not an "
            "attendance rate but the share naming the arts among their top three "
            "leisure activities",
            "2016, 2018 and 2020 are one regime (same item set, order and response "
            "format)",
            "the published 2022 and 2024 figures reproduce from the microdata to "
            "within 0.10 pp",
            "2018 does not rebuild: the cultural-activity adjustment factor used in "
            "the published figures is not in the delivered file",
            "the analysis base for 2014 is the 2,110 Seoul residents",
            "not_asked_in_wave is never imputed",
            "nothing is phrased as an attack on the foundation's published figures -- "
            "the subject stays on the limits of the data we hold",
            "person-level files never sit in data/processed/official/",
            "every output is declared in config/output_manifest.csv",
            "person-level files are outside the repository by git check-ignore",
            "every person-level table joins to the spine on row_uid alone, at 100 "
            "per cent",
            "the out-of-battery harmonisation and the registry sweep all 11 raw files "
            "(2026-09-09)",
            "item-level gates are decided from the data, and applied only at 0.95 "
            "agreement or above",
        ],
    }
    (C.ROOT / "config" / "official_contracts.json").write_text(
        json.dumps(contracts, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{CHECKS} contracts · {len(FAILS)} failures")
    for f in FAILS:
        print(f"  !! {f}")
    if FAILS:
        sys.exit(1)
    print("contracts pass · config/official_contracts.json refreshed")


if __name__ == "__main__":
    main()
