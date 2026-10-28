"""Generate the official track's report. Do not edit the output by hand -- this script
overwrites it."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

REPORTS = C.REPORTS
md_table = C.md_table


def main():
    O = C.OUT
    ins = pd.read_csv(O / "official_instrument.csv")
    summ = pd.read_csv(O / "official_regime_summary.csv")
    breaks = pd.read_csv(O / "official_regime_breaks.csv")
    rep = pd.read_csv(O / "official_reproduction_summary.csv")
    frames = pd.read_csv(O / "official_sample_frames.csv")
    wts = pd.read_csv(O / "official_weight_inventory.csv")
    eff = pd.read_csv(O / "official_weight_effect.csv")
    est = pd.read_csv(O / "official_series_estimates.csv")
    comp = pd.read_csv(O / "official_series_comparability.csv")
    tax = pd.read_csv(O / "official_attendance_missingness.csv")
    cov = pd.read_csv(O / "official_schema_coverage.csv")
    wmeta = pd.read_csv(O / "official_wave_metadata.csv")
    fdesign = pd.read_csv(O / "official_frame_design.csv")
    fe = pd.read_csv(O / "official_frame_effect.csv")
    early = pd.read_csv(O / "official_early_published_check.csv")
    cmap = pd.read_csv(O / "official_canonical_map.csv")
    dcov = pd.read_csv(O / "official_domain_coverage.csv")
    la = pd.read_csv(O / "official_living_area_estimates.csv")
    lt = pd.read_csv(O / "official_living_area_tests.csv")
    gcov = pd.read_csv(O / "official_generation_coverage.csv")
    wprov = pd.read_csv(O / "official_weight_provenance.csv")
    # The declarations have two grains: the declarations themselves, and the rows they
    # produce once applied to the files. Settle it once here -- when the report says
    # "N declarations" it must mean the row count of the hand-maintained config.
    _bdecl_all = pd.read_csv(O / "official_domain_declarations.csv")
    bdecl = _bdecl_all[_bdecl_all.is_primary]
    bmiss = pd.read_csv(O / "official_domain_missingness.csv")
    bimp = pd.read_csv(O / "official_domain_impute_validation.csv")
    regsum = pd.read_csv(O / "official_registry_summary.csv")
    gates = pd.read_csv(O / "official_domain_gates.csv")

    # The population must not be collapsed. 2016 and 2022 hold both a general-public
    # and a culture-engaged frame, so pivoting without population gives the arithmetic
    # mean of two populations (found 2026-09-08).
    _ov = est[est.dimension == "overall"]
    o = (_ov[_ov.population == "general_population"]
         .pivot_table(index="canonical_item", columns="wave", values="estimate")
         .round(1).reset_index())
    o_eng = (_ov[_ov.population == "engaged_frame"]
             .pivot_table(index="canonical_item", columns="wave", values="estimate")
             .round(1).reset_index())
    comp = comp.assign(pair=comp.from_wave.astype(str) + "->" + comp.to_wave.astype(str))
    grades = comp.pivot_table(index="canonical_item", columns="pair",
                              values="grade", aggfunc="first").reset_index()

    _mr = pd.read_csv(O / "official_margin_reproduction.csv")
    n_exact = int((_mr.verdict == "exact").sum())
    n_cells = len(_mr)

    tax_md = md_table(
        tax.groupby("mechanism", as_index=False)
           .agg(**{"item-wave combinations": ("n_cells", "size"),
                   "cells": ("n_cells", "sum")})
           .assign(**{"cells": lambda d: d["cells"].astype(int)}))

    gcov_md = md_table(
        gcov.pivot_table(index="generation", columns="wave", values="coverage")
        .round(2).reset_index())

    cmap_md = md_table(
        cmap[cmap.in_data].pivot_table(index=["domain", "canonical_var"], columns="wave",
                                       values="source_var", aggfunc="first")
        .fillna("").reset_index())
    _sc = cmap[cmap.scale_stable == False][["canonical_var", "wave", "n_options"]]
    scale_md = md_table(
        _sc.pivot_table(index="canonical_var", columns="wave", values="n_options")
        .fillna(0).astype(int).replace(0, "").reset_index())
    dcov_md = md_table(
        dcov.pivot_table(index="domain", columns="wave", values="vars")
        .fillna(0).astype(int).reset_index())
    la_md = md_table(
        la.pivot_table(index="living_area", columns="wave", values=["crude", "std"])
        .round(1).reset_index())
    lt_md = md_table(lt[lt.test.notna() & lt.living_area.isna()][
        ["test", "statistic", "p_value", "n", "note"]].head(10))

    fe_md = md_table(
        fe.pivot_table(index="canonical_item", columns="wave",
                       values="log_odds_ratio").round(2).reset_index())
    early_md = md_table(early, ["wave", "base", "metric", "unit", "published",
                                "reproduced", "rel_error_pct", "verdict"])

    wprov_md = md_table(
        wprov[wprov.adjustment == "demographic+edu+income"][
            ["wave", "weight_var", "r2_log_weight", "residual_corr_any_attend",
             "top_item", "top_item_corr", "verdict"]])

    # Out-of-battery harmonisation (15) -- variables per family and wave.
    # The same variable appears in several group files, so count distinct variables
    # rather than summing.
    bcov_md = md_table(
        _bdecl_all[_bdecl_all.in_data].pivot_table(index="primary_family", columns="wave",
                                         values="variable", aggfunc="nunique")
        .fillna(0).astype(int).reset_index().rename(columns={"primary_family": "family"}))
    n_decl = len(bdecl)
    n_routed = int(bdecl.is_routed.sum())
    routed_pct = n_routed / n_decl * 100
    meas_md = md_table(
        bdecl.measure.value_counts().rename_axis("measure").reset_index(name="declarations")
        .head(8))

    # Out-of-battery missingness (16), by wave
    _bm = (bmiss.groupby("wave", as_index=False)
           .agg(**{"cells": ("cells_total", "sum"),
                   "item non-response": ("cells_item_nonresponse", "sum"),
                   "routed out (person-blocks)": ("routed_out_persons", "sum"),
                   "gated items": ("n_gated_items", "sum")}))
    _bm["item non-response %"] = (_bm["item non-response"] / _bm["cells"] * 100).round(2)
    bmiss_md = md_table(_bm.assign(**{c: _bm[c].map("{:,.0f}".format)
                                      for c in ("cells", "item non-response",
                                                "routed out (person-blocks)")}))
    cells_tot = int(bmiss.cells_total.sum())
    cells_inr = int(bmiss.cells_item_nonresponse.sum())

    # Item-level gates (16) -- where the gate turned out to be
    GATE_LABEL = {"sibling_experienced": "an experience item in the same family",
                  "attendance_battery": "the attendance battery (another table)",
                  "exclusive_branch": "an exclusive branch (an earlier question split it)"}
    _g = gates.gate.value_counts().rename_axis("gate").reset_index(name="items")
    _g["what opens it"] = _g.gate.map(GATE_LABEL)
    _g["lowest agreement"] = _g.gate.map(gates.groupby("gate").agreement.min()).map("{:.3f}".format)
    gate_md = md_table(_g, ["what opens it", "items", "lowest agreement"])
    n_gates = len(gates)
    sat_inr = int(bmiss[bmiss.family == "satisfaction"].cells_item_nonresponse.sum())

    # The variable registry (17)
    reg_md = md_table(
        regsum.pivot_table(index="wave", columns="tier", values="variables",
                           aggfunc="sum").fillna(0).astype(int).reset_index())
    tier_n = regsum.groupby("tier").variables.sum()
    _reg = pd.read_csv(O / "official_variable_registry.csv")
    prim_a = (_reg[_reg.source_group == _reg.wave.map(C.default_group)].tier == "A").mean() * 100
    n_vars = int(tier_n.sum())
    status_md = md_table(
        regsum.groupby(["tier", "status"], as_index=False).variables.sum()
        .rename(columns={"tier": "tier", "status": "status", "variables": "variables"}))
    files_md = md_table(
        regsum.groupby(["wave", "source_group"], as_index=False).variables.sum()
        .rename(columns={"wave": "wave", "source_group": "group",
                         "variables": "variables"}))
    n_files = len(C.groups())
    n_decl_rows = len(_bdecl_all)

    weight_effect_md = md_table(
        eff.sort_values("wt_ratio").head(10),
        ["wave", "canonical_item", "unweighted", "weighted", "shift_pp",
         "mean_wt_attender", "mean_wt_non_attender", "wt_ratio"])

    doc = f"""# Official raw data track -- person-level data for seven waves, 2014 to 2024

Generated automatically · {date.today().isoformat()} · `scripts/official_*.py`

> **This document is generated. Do not edit it by hand.**
> Person-level data never enters a published output -- the repository is an external
> boundary. Every figure here is an aggregate.

## 0. What changed

The foundation's official delivery **contains person-level data for 2014, 2015 and
2016** as well as 2018 to 2024.

This track recomputes from the respondents. The values printed in the reports are
frozen in `data/reference/published_tables_2018_2024.csv` and used as the reference.

## 1. Instrument regimes -- computed, not declared

Waves are grouped by a fingerprint of the attendance battery: its item set, item order
and response format.

{md_table(summ, ["regime_id", "waves", "n_items", "response_format"])}

Boundaries and why:

{md_table(breaks, ["from_wave", "to_wave", "from_regime", "to_regime", "reason"])}

**2016's fingerprint is identical to 2018's and 2020's.** The basis of the old series
split -- that the early waves asked about "times you went somewhere, free admission
included" while 2018 onwards asked about "attendance, excluding viewing on TV or a
computer" -- is contradicted by the 2016 codebook, whose items already read
`극장에서 영화관람(TV, 컴퓨터를 이용한 영화시청은 제외)` and
`연극공연 관람 (뮤지컬 포함)`. The real break is **2015 to 2016**.

The item set has not changed since 2016. What changed after that is the choice order
(2022) and the response format (2024).

## 2. Rebuilding the published tables -- which waves come back from the delivered files

> **A rule about how this is written.** A value that does not match reflects
> information the delivered file does not carry -- the weights and the control totals
> -- not a defect in the published figure. This section records what can be rebuilt
> from the files we hold. It passes no judgement on the validity of the foundation's
> published numbers.

{md_table(rep, ["wave", "dimension", "cells", "mean_abs", "max_abs", "exact", "close", "differs"])}

- **2022 and 2024 rebuild completely** from the delivered files: every cell of sex, age,
  age by sex and the nine living areas, with a maximum error of 0.05 pp. The weights
  came with the file.
- **2020 comes close** (mean 0.56 pp). The report's sample of 5,000 cannot be recovered
  from the 5,095 rows in the delivered file, and that is what remains.
- **2018 does not rebuild from the delivered file alone** (mean 23.96 pp). The reason is
  documented: section VII of the 2018 report states that the final weight is the design
  weight times a cultural-activity adjustment factor, and that the factor was built
  against the cultural-activity distribution in the Seoul Survey. **Neither that
  reference distribution nor the weight is in the delivered file.** This is a limit of
  the data we hold, not a problem with the published figures. The practical conclusion
  is single: for 2018, **state in every output whether the microdata value or the
  published value is being used**, and never mix the two in one series.

## 3. Samples and frames

Three waves mix several frames in one file (2015, 2016 and 2022).
**Which frame is the general public cannot be told from the microdata** -- the evidence
is each report's description of the sample design.

{md_table(fdesign, ["wave", "frame_label", "n", "n_seoul", "general_population", "sampling", "mode"])}

**2014 and 2015 have no general-public frame.** 2014 is a census of foundation members;
2015 is foundation members plus the existing cultural policy panel, and the codebook
reads `9 '재단회원' 10 '기존패널'`. So **the general-public series runs over five waves,
from 2016.**

### 3-0. Fielding period and recall window

{md_table(wmeta, ["wave", "fielding_start", "fielding_end", "recall_start", "recall_end", "recall_basis"])}

**The '2020' survey was fielded in January and February 2021, and its recall window is
the whole of 2020** -- the pandemic year. 2015 is not a calendar year but 2014-06 to
2015-05. The recall windows for 2014, 2016 and 2018 are not stated in the reports and
are inferred backwards from fieldwork (`recall_basis=inferred`).

Weights:

{md_table(wts, ["wave", "variable", "label", "n", "sum", "is_published_weight"])}

- Four waves, 2014 through 2018, carry no weight variable.
- 2020 carries two. `WTB` matches the report's sample size of 5,000, but it is `WT`
  that reproduces the published figures (mean error 0.65 against 3.79 pp).

### 3-1. The weights are raked on the attendance items themselves -- in all three waves

**This is not a hidden practice. The foundation states it in the reports.** The 2018
report gives the final weight as the design weight times a cultural-activity adjustment
factor, and says raking was applied on the **four most representative activities:
film, exhibitions, museums and concerts** -- the aim being to correct the online
survey's cultural-activity participation rates against the face-to-face Seoul Survey.
The 2022 report likewise says it uses a design weight and a cultural-activity weight.

So the diagnostic below is **not a discovery but two things**: (a) recovering from the
data which genres the published practice actually raked on and how hard, and (b)
measuring how much that moves the published numbers. It regresses `log(weight)` on sex,
age and district (plus education and income) and asks whether the **residual**
correlates with attendance.

{wprov_md}

The diagnostic agrees with the documents: exhibitions are among the targets 2018 names
and the 2020 `WT` puts exhibitions at -0.92; the 2022 variable is named for film and
theater and the diagnostic puts theater at -0.87. **The raking targets can be recovered
from the data alone**, which means the method also works on surveys that do not
disclose them.

But **the target genres change from wave to wave** -- film, exhibitions, museums and
concerts by the 2018 documentation, film and theater by the 2022 variable name. The
diagnostic moves with them, from exhibitions in 2020 to theater in 2022 and 2024.
**A genre-level trend therefore crosses a moving adjustment boundary.** That is the
substantive problem.

**2020 is an internal control.** One file carries two weights, and `WTB` shows a
residual correlation of only -0.06, which looks like a design weight on demographics
alone, with a higher R-squared as well (0.363 against 0.303). `WT` looks like that
multiplied by the cultural-activity adjustment factor -- matching the documented "final
weight = design weight x cultural-activity adjustment factor". (A conjecture; the
foundation would have to confirm it.)

So **the attendance rate for each wave's raked genres is not an independent estimate
but a figure reflecting an external control total.** The adjustment itself may be a
legitimate procedure -- it pulls an online panel's over-reporting of cultural activity
back to a face-to-face benchmark. The problems are that **the result has never been
published** (theater in 2022: 27.1 per cent unweighted, 8.7 weighted) and that **the
target genres differ by wave**, so genre-level trends cannot be read on one scale.
That is why every attendance output in this track carries `estimate_unweighted` and
`weight_shift_pp` next to `estimate`.

A caution: correlation is not proof. Settling it requires the **control totals** from
the foundation (limitation 7 in section 6).

### 3-2. The effect of the weights, item by item

The SPSS label on the 2022 variable `wt3_b` names film and theater as raking margins.
Whether they really are control variables was checked against the data. The further the
ratio of mean weights between attenders and non-attenders is from 1, the more the
weight is entangled with that item.

{weight_effect_md}

`theater` in 2022 has a ratio of **0.256**, the most extreme value across every wave
and every item: a mean weight of 0.321 for theatergoers against 1.253 for everyone else.
The effect appears **separately within both frames** -- the panel (GRP1: 32.8 to 10.3
per cent) and the face-to-face sample (GRP3: 7.2 to 1.9) -- so it is an adjustment on
the item itself, not an allocation across strata.

**The published 2022 figures for film and theater are therefore closer to control
totals than to estimates.** Both are among the five items the old aggregate track left
marked "usable for trends". The KOPIS comparison flagged the 2020-to-2022 fall in
`theater` as `contradicted`, and that too may be explained by the weighting design
rather than by any change in behaviour. Say so whenever these are read as a trend.

The 2020 `WT` points the same way, more weakly (`exhibition` ratio 0.453).

- 2014 includes 795 respondents living outside Seoul, and they were screened out of the
  attendance battery (795 non-Seoul cases and 795 with no battery at all -- exactly the
  same people). **The analysis base for 2014 is 2,110.**

## 4. Attendance rates, computed from the microdata

**General-public frames** (five waves, 2016 to 2024). Waves with a weight are weighted;
waves without are not. 2014 and 2015 have no general-public frame at all and do not
appear in this table.

{md_table(o)}

**Culture-engaged frames** (foundation and institutional members, and the like). Do not
read these in the same column as the table above.

{md_table(o_eng)}

**The frame differs by wave. Do not read this table as one trend line.** 2014 and 2015
are convenience-sample panels (2014 is 70 per cent women), 2016 and 2018 online panels,
2020 panel plus offline, 2022 panel plus face-to-face, and 2024 is rake-weighted.
A regime speaks **only to the instrument being identical**.

### 4-1. Frame effect -- one wave, one instrument

2016 and 2022 each surveyed a general-public frame and a culture-engaged frame **with
the same instrument in the same wave**. That is the only place a frame effect can be
measured with no instrument difference in the way, and it happens twice, six years
apart.

{fe_md}

**It is measured unweighted.** 2016 has no weight and 2022's `wt3_b` is entangled with
the outcome, so comparing a weighted 2022 with an unweighted 2016 would measure the
difference in weighting design rather than the frame effect.

The result splits in two.

- **Across items it scatters.** 2016 runs 0.25 to 1.49, 2022 0.63 to 1.40. **There is
  no single frame coefficient** that can be applied to every item.
- **Across waves it is stable.** The medians sit between 0.64 and 0.89, and **all ten
  wave pairs correlate between +0.69 and +0.91 (mean +0.82)**. A different coefficient
  per item, but one that holds across eight years. This is not a two-wave coincidence.

Computed on the weighted version the correlation falls to 0.370. The earlier draft's
claim that the effect is "unstable across waves too" **mistook a difference in
weighting design for a frame effect** (corrected 2026-09-08). Item-level frame
adjustment is not ruled out at the outset; it is **something to test**.

The call for a 2026 split ballot stands unchanged, however: a frame effect and a
**question effect** are different things, and this data still does not identify the
latter.

Comparability grades between adjacent waves:

{md_table(grades)}

The grade vocabulary, which keeps "this item changed" separate from "the battery around
it changed":

| grade | meaning |
| --- | --- |
| `comparable` | neither the item nor the battery moved |
| `context_changed` | the item held, but another choice in the same battery changed |
| `design_perturbed` | this item's choice position or response format changed |
| `not_comparable` | the item is differently constituted |
| `not_asked` | the item is absent from one of the waves |

**2016 to 2018 and 2018 to 2020 are `comparable` on all ten items.**

## 5. Person-level missingness -- only what may be imputed was imputed

{tax_md}

| mechanism | treatment |
| --- | --- |
| `not_asked_in_wave` | **Never imputed.** It would invent a trend point for a wave that has none |
| `not_asked_screened_out` | left missing (the 795 non-Seoul cases in 2014) |
| `skip_logic` | filled where the value follows. Spending in 2024 is asked only of those with at least one self-paid visit, so a self-paid count of 0 means spending of 0 (the cross-tabulation is perfectly diagonal, no exceptions) |
| `item_nonresponse` | imputed. It amounts to 126 respondents in the 2016 spending and self-paid blocks |

**The microdata barely needs imputation.** The old aggregate track needed a great deal
of it because the published segment tables had blanks. Computing from the respondents
directly, those blanks do not exist.

The hold-out check (coverage of the M=5 donor range) is in
`official_attendance_impute_validation.csv`.

## 5-1. The early waves against the published headline figures

{early_md}

- **Annual visits and spending for 2014 and 2015 match to the won.** That settles the
  sample base definitions empirically (2014 = the 2,110 Seoul residents, 2015 = 1,498
  across both frames) and **the spending unit** as KRW.
- **The published "arts attendance rate" (68.7 per cent in 2014, 65.2 in 2015) is not an
  attendance rate.** It is the share naming attending the arts among their top three
  leisure activities (2014 `QA1`, 2015 `A1`). The actual attendance rate from the same
  battery is 99.9 and 99.2 per cent. Putting the two side by side is a category error.
  The twelve legacy rows the old aggregate track carried were retired for this reason.

## 5-2. Outside the attendance battery -- the question-change concordance

The battery is regular enough to fingerprint automatically, but outside it the question
structure changes wholesale from wave to wave and automatic matching does not work. The
canonical variables are declared and then checked by machine: each declared variable
must really exist in that wave, and the codebook's own wording is carried through
verbatim.

{cmap_md}

**Sharing a name does not make the values comparable.** For most of these variables the
number of choices changes between waves -- a scale change:

{scale_md}

Domain coverage (number of variables held; 0 means absent from that wave):

{dcov_md}

2024 has no barriers-to-attendance battery of the 2018/2020 kind. The 2015 file has no
demographics block at all, only frame, sex and birth year. The 2018 demographics are in
the disclosure `.xlsx` rather than the source-of-record `.xls` and were restored by
joining on ID (all 2,545 rows matched).

## 5-3. Rank order across the nine living areas

The 2022-to-2024 rank correlation of living-area attendance rates is 0.067. With the
microdata, cell-level standard errors can be computed directly to check whether
sampling error explains it.

{la_md}

{lt_md}

Three verdicts:

1. **The break in rank order is not explained by sampling error.** Under the null that
   the true values are unchanged, the rank correlation has a median of 0.68 (5th
   percentile 0.32). The observed -0.05 lies outside the lower tail of that
   distribution (p=0.002). The ranking really did change.
2. **It is not composition.** Direct standardisation on sex and age does not restore
   the correlation; it pushes it further down, from 0.067 to -0.05.
3. **The differences between living areas are themselves real.** The heterogeneity Q is
   significant in both waves (p<0.0001).

The shape of the change is **convergence**. The correlation between the 2022 level and
the 2022-to-2024 change is -0.79 (p=0.011): the areas that were low in 2022 rose most.
Five of the nine rose significantly after Bonferroni (the city centre, Southeast 1,
Southwest 2, Southwest 3 and Northwest), and all five had been in the lower half in
2022.

**So reading the 11.6 pp gap across living areas in 2022 as a structural regional
disparity cannot be sustained.** The 2022 recall window (2021-11 to 2022-10) is still
inside the pandemic's reach and the 2024 one (2023-10 to 2024-09) is not. The observed
convergence reads more naturally as differing speeds of recovery by area. Claiming a
structural gap would need the 2026 wave.

## 5-4. Generational cohort coverage

Comparing generations starts with checking whether the cohort is observed in every
wave. The survey covers ages 15 and over, so young cohorts appear only in recent waves.

{gcov_md}

- **No wave covers all of Gen Z (1997-2012).** Even 2024 reaches only 0.81, missing
  those born 2010 to 2012.
- **The cohorts common to the general-public waves (2016-2024) are those born 1997 to
  2001.** Reaching outside that range compares different people in each wave.
- **2014 carries no birth year.** With only six age bands it enters cohort analysis by
  band alone.

## 5-5. Outside the battery -- {n_decl} declarations checked against the data

The concordance in section 5-2 covers only the single canonical variables that run
across waves. Beyond those the question structure changes wholesale and automatic
matching does not work. So the declarations live in a data file
(`config/domain_declarations.csv`, {n_decl} rows) and the script checks them by machine:
that each declared variable really exists in that wave, that it does not overlap the
offline battery, and that no `(wave, group, variable)` is declared twice. The
declarations are written against each wave's representative file; applied across the
{n_files} raw files they become {n_decl_rows:,} rows.

Variables per family and wave:

{bcov_md}

The measure vocabulary (top 8):

{meas_md}

**{routed_pct:.0f} per cent of the declarations ({n_routed}/{n_decl}) are routed
questions.** That is why `universe` is in the declaration schema at all: used without
knowing the base, the figure is not a population share but a share of those who passed
the routing.

- **2020 `D1A_*` was not asked of everyone.** It went only to the 812 who said their
  media-based cultural activity had risen since the pandemic. The question asked of
  everyone is `D4A_*`, and exhibitions come out at 53.8 per cent against 12.8.
- **2018 `ZD1_*a` is not an experience item.** It is a four-point frequency scale cut at
  "occasionally or more", so `(ZD1==1)` and `(D1>=3)` agree exactly. Filed under the
  same measure as the other waves' binary experience items, 2018 alone would be
  systematically off -- hence the separate `threshold_recode`.
- **Polarity and units were silent contaminants too.** In a `1 = yes / 2 = no` coding
  the larger value is "no", and online spending is in KRW in 2020 but in units of ten
  thousand KRW in 2022. Combined without conversion they are out by a factor of 10,000.

## 5-6. Missingness outside the battery -- first find where the gate is

{bmiss_md}

Of {cells_tot:,} cells in total, item non-response is {cells_inr:,}, or
**{cells_inr / cells_tot * 100:.2f} per cent**. The rest is structural missingness from
routing, unselected slots and conditional gates, and is not a candidate for imputation.

- **A blank in a fixed-position multiple response is not non-response but non-selection.**
  Slots such as `D1_1_4` (value 4 for those who chose crafts, missing otherwise) are
  filled with 0. Counted as non-response they inflate the missingness rate wholesale.
- Imputation applies to item non-response only. The hold-out check (coverage of the M=5
  donor range) covers {len(bimp):,} items with a mean of {bimp.coverage.mean():.3f} --
  a measured value, not a claim at nominal.

### 5-6-1. When the gate is in another table

Knowing who received a question only at family level **mistakes an item-level gate for
non-response.** Satisfaction is asked per genre, only of people who attended that genre,
and the attendance that opens the gate is in the battery, not this table. Searching
within the family alone never finds it.

Three kinds of gate are found **from the data**. The free-text `universe` is not
trusted; a gate applies only where agreement exceeds 0.95. The evidence for each item
is recorded in `official_domain_gates.csv`.

{gate_md}

- **The attendance battery** -- satisfaction, frequency and spending open only for
  people who attended that genre. Where item names overlap the battery, observation and
  attendance agree at **1.0, with no exception in either direction** (2014, 2015, 2016).
  No gate is applied to the experience items themselves: those are what open gates.
- **Exclusive branches** -- an earlier question sends respondents down mutually exclusive
  paths (2022 `E1` splits inside and outside the living area; 2022 and 2024 `G` split
  the youth-policy and ageing modules). **Non-response does not create exclusivity**, so
  a branch is inferred only where a pair of items has observed sets that do not overlap
  at all. An item asked of everyone overlaps with all of them and is never caught.
- These judgements agree with the hand-written `universe` text independently: every item
  judged to be a branch is declared as routed or as not asked of everyone.

**Before the gates went in, item non-response stood at 6.26 per cent.** A quarter of it
sat in satisfaction alone, and was not non-response at all but a closed gate. It now
stands at **{cells_inr / cells_tot * 100:.2f} per cent**, of which satisfaction is
{sat_inr / cells_inr * 100:.1f} per cent.

## 5-7. A registry of every variable -- so that nothing can be left out

Counting canonical variables alone stalls coverage at 63 per cent. What remains is not
useless but of a different kind. So **every variable in the raw data gets a tier.**

{reg_md}

{status_md}

**{n_vars:,} in total** -- A canonical {int(tier_n['A']):,}
({tier_n['A'] / n_vars * 100:.0f} per cent) · B wave_local {int(tier_n['B']):,}
({tier_n['B'] / n_vars * 100:.0f} per cent) · C catalogued {int(tier_n['C']):,}
({tier_n['C'] / n_vars * 100:.0f} per cent). Tier B is not an omission but questions
that are genuinely specific to their wave, and tier C is identifiers, weights and free
text -- not analysis variables. Every variable that has values has a home in a
person-level table.

**The registry and the out-of-battery harmonisation sweep all {n_files} raw files.** A
wave does not have just one: 2018, 2020 and 2024 split by group.

{files_md}

The declarations were written against each wave's representative file, and apply to the
other group files **only for variables of the same name**. The culture-interested files
share nearly all their columns with the representative file (25, 16 and 1 unique
columns), but **the 2024 disabled file has 842 columns and a substantially different
instrument** (the general file has 361). Its 582 unique columns are in the registry at
tiers B and C with no canonical counterpart yet -- which is why the overall tier-A share
of {tier_n['A'] / n_vars * 100:.0f} per cent is below the {prim_a:.0f} per cent for the
representative files.

Per-file coverage is in `official_person_coverage.csv` and the
[data dictionary](official_data_dictionary.md).

## 6. Known limitations

1. **2018's published figures cannot be rebuilt from the delivered file.** The
   cultural-activity adjustment factor the report describes (benchmarked to the Seoul
   Survey) and its reference distribution are not in the file. The microdata and
   published values differ by a factor of three to six, so **every output must state
   which of the two it uses**, and the two are never mixed in one series. This is a
   limit of data access, not a judgement on the published figures.
2. **The 2020 report's sample of 5,000 cannot be recovered.** The file has 5,095 rows,
   and the 890 duplicate `IDX` values are the panel (TYPE=1) and offline (TYPE=4) ID
   spaces overlapping. What the 95 excluded rows were has not been established.
3. **The 2022 weights are raked on film and theater.** See section 3.
4. **2014 has no birth year.** With only six age bands it enters cohort analysis by band
   alone.
5. **The 2015 file has no district-contact frames.** The codebook has eight district
   sheets, but the data holds only foundation members (1,041) and the general panel
   (457).
6. ~~The spending unit is not in the codebooks.~~ **Resolved.** It was settled from the
   magnitude of the values and then confirmed against the published figures: total
   annual spending per person in 2014 and 2015 matches to the won (section 5-1).
7. **The weight diagnostic is a correlation, not a proof.** Raking on an unobserved
   demographic axis could also leave a residual correlated with attendance. That it
   survives controlling for education and income strengthens the evidence, but settling
   it requires the raking control totals per wave from the foundation.
8. **Frame effect and question effect are still not separated.** The microdata controls
   design effects only in part and does not identify question effects. The call for a
   2026 split ballot stands.
9. **The item-level gates are inferred from the data, not declared.** The threshold is
   0.95 agreement, so items near it could be judged either way. The evidence is kept in
   `official_domain_gates.csv`; check it before citing (section 5-6-1).
10. **The 582 unique columns in the 2024 disabled file have no canonical counterpart.**
   They are all in the registry, at tiers B and C. The instrument differs enough from
   the general-public one that mapping them across waves would take hand work at the
   level of the question wording (section 5-7).

## 7. Outputs

| file | contents |
| --- | --- |
| `official_variable_dictionary.csv` | wave x variable x question wording x check against the data |
| `official_value_labels.csv` | value labels |
| `official_item_battery.csv` | wave x choice position x canonical item x measure x variable |
| `official_instrument.csv` | the instrument fingerprint and regime per wave |
| `official_regime_summary.csv` · `official_regime_breaks.csv` | regimes and why they break |
| `official_sample_frames.csv` · `official_frame_design.csv` | frame register and sample design |
| `official_wave_metadata.csv` | fielding period, recall window, source |
| `official_weight_inventory.csv` | weight register |
| `official_frame_effect.csv` | the frame effect measured within one wave on one instrument |
| `official_early_published_check.csv` | 2014-2016 against the published headline figures |
| `official_canonical_map.csv` · `official_domain_coverage.csv` | the question-change concordance outside the battery |
| `official_living_area_estimates.csv` · `official_living_area_tests.csv` | tests across the nine living areas |
| `official_birth_cohort_counts.csv` · `official_generation_coverage.csv` | birth cohorts and generational coverage |
| `official_weight_effect.csv` | the effect of the weights per item, and the attender/non-attender weight ratio |
| `official_weight_provenance.csv` · `official_weight_item_residuals.csv` | weight provenance -- demographics or the outcome |
| `official_margin_reproduction.csv` · `official_reproduction_summary.csv` | the published-table reproduction check |
| `official_schema_coverage.csv` | observed and missing in the unified schema |
| `official_series_estimates.csv` · `official_series_comparability.csv` | attendance rates and grades |
| `official_attendance_missingness.csv` · `official_attendance_impute_validation.csv` | battery missingness mechanisms and the hold-out check |
| `official_domain_declarations.csv` · `official_domain_family_coverage.csv` | the {n_decl} out-of-battery declarations ({n_decl_rows:,} applied rows) and family coverage |
| `official_domain_item_keys.csv` | uniqueness check on the person x item key |
| `official_domain_missingness.csv` · `official_domain_impute_validation.csv` | out-of-battery missingness mechanisms and the hold-out check |
| `official_domain_gates.csv` | item-level gate decisions and their evidence ({n_gates}) |
| `official_variable_registry.csv` · `official_registry_summary.csv` | the variable registry and its summary |
| `official_file_manifest.csv` · `official_column_profile.csv` | the output manifest and the column-level dictionary |
| `official_join_map.csv` · `official_person_coverage.csv` | join checks and the sample coverage of each person-level file |

**What is in which file and how they join is in the
[data dictionary](official_data_dictionary.md).**
The person-level files (`official_person_*.csv`) live in `data/interim/official/` and
are kept out of the repository.
"""
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "official_overview.md").write_text(doc, encoding="utf-8")
    print(f"reports/official_overview.md written ({len(doc):,} characters)")
    print(f"{n_exact} of {n_cells} reproduction cells exact")


if __name__ == "__main__":
    main()
