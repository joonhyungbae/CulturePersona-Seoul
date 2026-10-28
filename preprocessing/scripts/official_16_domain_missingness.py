"""Classify and impute missingness outside the attendance battery -- here too, only
what may be imputed is imputed.

What the battery showed holds more strongly outside it. Of 907 declarations, **62 per
cent (559) are routed**. Most missingness is therefore not a respondent declining to
answer but **the questionnaire not asking**.

Three mechanisms, decided **from the data** rather than from the free-text universe
field:

  not_asked_in_wave   that (family, item, measure) does not exist in the wave at all.
                      Imputing would invent a value for a wave that has none:
                      forbidden.
  not_asked_gated     the question was asked, but **this item's** gate never opened.
                      Satisfaction, frequency and spending are asked only of people who
                      attended that genre, and the attendance that opens the gate lives
                      in the battery rather than this table, so looking only within the
                      family never finds it. Not imputed.
  not_asked_routed    the question exists in the wave but this respondent never
                      received it. Answering nothing anywhere in the same battery block
                      places the respondent outside it. The question has a defined
                      universe, so it must not be rescaled to the full sample. Not
                      imputed.
  item_nonresponse    inside the block -- sibling items were answered -- yet this item
                      alone is empty. Only this is imputed.

Imputation is a hot deck within one wave, frame and item, and a hold-out check measures
coverage. Nothing is claimed at its nominal value.

Outputs:
  official_domain_missingness.csv        wave x family x measure x mechanism x cells
  official_domain_gates.csv              what opens which item's gate, with the evidence
  official_domain_impute_validation.csv  the hold-out check
  official_person_domain_imputed.parquet (person level -> interim)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

# Measures that open gates rather than sit behind one. No attendance gate applies.
OPENER_MEASURES = {"experienced", "threshold_recode"}

SEED = 20260909


def rng_for(key) -> np.random.Generator:
    """Seed each block from its own key. Drawing from one shared RNG in sequence makes
    **the order the blocks are visited change the results** -- moving the group key
    from a string to several columns did in fact change every imputed value. Seeding
    from the key reproduces regardless of order or parallelism. Python's `hash` cannot
    be used: it is salted per process."""
    h = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return np.random.default_rng(SEED ^ int.from_bytes(h, "big"))
M = 5
MIN_DONORS = 20


def main():
    L = C.read_person("official_person_domain")
    keys = C.read_person("official_person_keys")
    frame = dict(zip(keys.row_uid, keys.frame_code))

    # Precompute the positive set of each gate-opening (experienced) item, for deciding
    # conditional routing.
    n_by_group = keys.groupby(["wave", "source_group"]).size()
    gate_pos = {}
    for (w, fam), gg in L[L.measure == "experienced"].groupby(["wave", "family"]):
        d = {}
        for it, hh in gg.groupby("canonical_item"):
            d[it] = hh.set_index("row_uid").value.fillna(0).astype(bool)
        gate_pos[(int(w), fam)] = d

    # Sometimes the gate lives in **another table**. Satisfaction, frequency and
    # spending are asked only of people who attended that genre, and that attendance is
    # in the battery (official_person_item), not here. Looking only at the family's own
    # experienced items never finds the gate, and all of that missingness is booked as
    # item non-response -- a quarter of all non-response outside the battery was this
    # (2026-09-09). These enter only as candidates; they apply only where the data
    # clears the 0.95 agreement threshold below.
    _it = C.read_person("official_person_item", ["wave", "row_uid", "canonical_item", "attended"])
    att_gate = {}
    for (w, _item), gg in _it.groupby(["wave", "canonical_item"]):
        att_gate[(int(w), _item)] = gg.set_index("row_uid").attended.fillna(0).astype(bool)

    # A battery block must be keyed on the **original question**. (wave, family,
    # measure) is too coarse: when one family and measure holds several questions with
    # different routing, a respondent who got A but not B is misclassified as item
    # non-response on B. Blocks are formed on the stem of the source variable
    # (C3A_1_1 and C3A_2_1 -> C3A, D4B_3 -> D4B). Expanded slot sets carry their
    # variable names joined by '|'.
    import re

    def qstem(v: str) -> str:
        base = str(v).split("|")[0]
        return re.sub(r"[_#x]?\d+[A-Za-z]*$", "", re.sub(r"[_#x]?\d+$", "", base)) or base

    # There are only 724 distinct variables across 5.7 million rows. Running the regex
    # per row does 11.5 million matches for what needs 1,448 -- apply it to the unique
    # values and spread the result with map.
    L["qstem"] = L.variable.map({v: qstem(v) for v in L.variable.unique()})
    wide = {}
    rows = []
    gate_rows = []
    imputed_rows = []

    # Routing can differ between group files: leave (wave, group) out of the block key
    # and a question one group never received is mixed in as another group's
    # non-response.
    # The key is not joined into a string and split apart again; groupby hands it back
    # as a tuple. sort=True stays. Imputation and the hold-out check share an RNG, so
    # **the order the blocks are visited changes the results.** The older string-key
    # groupby was sorted too, and matching that order is what keeps this refactor from
    # changing the outputs.
    GKEY = ["wave", "source_group", "family", "measure", "qstem"]
    for gkey, g in L.groupby(GKEY, sort=True):
        wave, sgrp, fam, meas, stem = gkey
        wave = int(wave)
        items = sorted(g.canonical_item.unique())
        piv = g.pivot_table(index="row_uid", columns="canonical_item", values="value", aggfunc="first")

        # Detect a multiple-response slot cascade. Choosing three of ten leaves the
        # remaining cells empty, which is structure, not non-response. In a slot set,
        # "if a cell is empty, every later cell is empty" holds. Where that monotonicity
        # is strong (>= 0.95) the block is treated as a cascade and its missingness is
        # excluded from imputation.
        order = list(piv.columns)
        cascade = False
        if len(order) >= 3:
            miss = piv[order].isna().values
            ok = 0
            tot = 0
            for j in range(len(order) - 1):
                prev, nxt = miss[:, j], miss[:, j + 1]
                tot += int(prev.sum())
                ok += int((prev & nxt).sum())
            cascade = tot > 0 and (ok / tot) >= 0.95

        in_universe = piv.notna().any(axis=1)
        n_univ = int(in_universe.sum())
        # every respondent in that wave
        n_wave = int(n_by_group[(wave, sgrp)])

        n_routed = n_wave - n_univ
        # Conditional routing per item. Satisfaction, frequency and spending are asked
        # only of people who experienced that item. A block-level universe cannot catch
        # it: someone who attended two of ten is inside the block, yet the other eight
        # satisfaction cells are legitimately empty. Whether a sibling experience item
        # opens the gate is decided from the data.
        gated: dict[str, str] = {}
        gates = gate_pos.get((wave, fam), {})
        for it in items:
            obs = piv[it].notna()
            if not obs.any():
                continue
            pos, src = gates.get(it), "sibling_experienced"
            if pos is None and meas not in OPENER_MEASURES:
                # experience items open gates rather than sit behind one, so no
                # attendance gate applies
                pos, src = att_gate.get((wave, it)), "attendance_battery"
            if pos is None:
                continue
            _pos = pos.reindex(piv.index).fillna(False)
            inter, n_obs_i, n_pos = int((obs & _pos).sum()), int(obs.sum()), int(_pos.sum())
            if inter / max(n_obs_i, 1) >= 0.95:
                gated[it] = src
                gate_rows.append(dict(wave=wave, source_group=sgrp, family=fam, measure=meas,
                                      question_stem=stem, canonical_item=it, gate=src,
                                      counterpart="", n_observed=n_obs_i,
                                      n_in_universe=n_univ,
                                      # forward: did people who answered this item
                                      # pass the gate?
                                      agreement=round(inter / max(n_obs_i, 1), 4),
                                      # reverse: did people who passed the gate answer
                                      # this item? Both at 1.0 means no exception in
                                      # either direction.
                                      n_gate_positive=n_pos,
                                      agreement_reverse=round(inter / max(n_pos, 1), 4)))

        # Branching questions, where an earlier question sends respondents down
        # mutually exclusive paths (in 2022, E1 routes to E2x1 and E2x2 for outside the
        # living area and E2x3 for inside; in 2024, E1 routes to E2_1 and E2_2). If the
        # variable that opens that gate is not in this table, one branch's structural
        # blank is booked entirely as non-response.
        # **Non-response does not create exclusivity** -- a pair of items whose observed
        # sets do not overlap at all is taken as a branch. An item asked of everyone
        # (2022 E2x4) overlaps with all of them and is never caught.
        obs_sets = {it: piv[it].notna().values for it in items}
        floor = max(30, 0.05 * n_univ)
        for it in items:
            a = obs_sets[it]
            na = int(a.sum())
            if it in gated or na < floor:
                continue
            for jt in items:
                if jt == it:
                    continue
                b = obs_sets[jt]
                nb = int(b.sum())
                if nb < floor:
                    continue
                if int((a & b).sum()) <= 0.02 * min(na, nb):
                    gated[it] = "exclusive_branch"
                    gate_rows.append(dict(wave=wave, source_group=sgrp, family=fam,
                                          measure=meas,
                                          question_stem=stem, canonical_item=it,
                                          gate="exclusive_branch", counterpart=jt,
                                          n_observed=na, n_in_universe=n_univ,
                                          agreement=round(1 - int((a & b).sum()) / min(na, nb), 4),
                                          n_gate_positive=nb, agreement_reverse=None))
                    break

        cells_inr = 0
        if not cascade:
            for it in items:
                if it in gated:
                    continue
                col = piv[it]
                cells_inr += int((col.isna() & in_universe).sum())
        rows.append(dict(wave=wave, source_group=sgrp, family=fam, measure=meas,
                         question_stem=stem, n_items=len(items),
                         n_wave=n_wave, n_in_universe=n_univ,
                         routed_out_persons=n_routed,
                         cells_item_nonresponse=cells_inr,
                         slot_cascade=bool(cascade), n_gated_items=len(gated),
                         cells_total=n_wave * len(items)))

        # ---- impute: within the block only, a hot deck per item -----------------
        if cascade or cells_inr == 0:
            wide[gkey] = piv
            continue
        rng = rng_for(gkey)
        fr = piv.index.to_series().map(frame)
        for it in items:
            if it in gated:
                continue
            col = piv[it]
            need = col.isna() & in_universe
            if not need.any():
                continue
            for f in fr.unique():
                sel = need & (fr == f)
                if not sel.any():
                    continue
                donors = col[(fr == f) & col.notna()].values
                if len(donors) < MIN_DONORS:
                    donors = col[col.notna()].values
                if len(donors) < MIN_DONORS:
                    continue
                draws = rng.choice(donors, size=(M, int(sel.sum())), replace=True)
                piv.loc[sel, it] = draws.mean(axis=0)
                imputed_rows.append(pd.DataFrame({
                    "wave": wave, "source_group": sgrp, "family": fam, "measure": meas,
                    "canonical_item": it,
                    "row_uid": piv.index[sel.values], "sd": draws.std(axis=0, ddof=1),
                }))
        wide[gkey] = piv

    MISS = pd.DataFrame(rows)
    GATES = pd.DataFrame(gate_rows)

    # ---- hold-out check -------------------------------------------------------
    val = []
    for gkey, piv in wide.items():
        wave, sgrp, fam, meas = gkey[:4]
        for it in piv.columns:
            obs = piv[it].dropna()
            if len(obs) < 300:
                continue
            vrng = rng_for((gkey, it))
            hold = obs.sample(min(150, len(obs) // 5), random_state=11)
            donors = obs.drop(hold.index).values
            if len(donors) < MIN_DONORS:
                continue
            draws = vrng.choice(donors, size=(M, len(hold)), replace=True)
            lo, hi = draws.min(axis=0), draws.max(axis=0)
            truth = hold.values
            val.append(dict(wave=int(wave), source_group=sgrp, family=fam, measure=meas,
                            canonical_item=it,
                            n_held=len(truth),
                            coverage=float(((truth >= lo) & (truth <= hi)).mean()),
                            mae=float(np.abs(draws.mean(axis=0) - truth).mean())))
    V = pd.DataFrame(val)

    _K = ["wave", "source_group", "family", "measure", "canonical_item", "row_uid"]
    OUT = pd.concat([p.stack().rename("value").reset_index().assign(
        wave=int(k[0]), source_group=k[1], family=k[2], measure=k[3])
        for k, p in wide.items()], ignore_index=True)
    IMP = (pd.concat(imputed_rows, ignore_index=True) if imputed_rows
           else pd.DataFrame(columns=_K + ["sd"]))
    OUT = OUT.merge(IMP.assign(is_imputed=True)[_K + ["is_imputed"]], on=_K, how="left")
    OUT["is_imputed"] = OUT.is_imputed.fillna(False)

    MISS.to_csv(C.OUT / "official_domain_missingness.csv", index=False)
    GATES.to_csv(C.OUT / "official_domain_gates.csv", index=False)
    V.to_csv(C.OUT / "official_domain_impute_validation.csv", index=False)
    C.write_person(OUT, "official_person_domain_imputed")

    tot_cells = int(MISS.cells_total.sum())
    inr = int(MISS.cells_item_nonresponse.sum())
    print(f"{len(MISS)} battery blocks · {tot_cells:,} cells in total")
    print(f"  routed out of the block: {int(MISS.routed_out_persons.sum()):,} "
          "person-blocks -- never imputed")
    print(f"  item non-response: {inr:,} cells = {inr/tot_cells*100:.2f}% -- the only "
          "thing imputed")
    print(f"  item-level gates: {len(GATES)} · " +
          " · ".join(f"{k} {v}" for k, v in GATES.gate.value_counts().items()))
    print(f"  actually imputed: {len(IMP):,} cells")
    if len(V):
        print(f"\nhold-out check: {len(V)} items · mean coverage {V.coverage.mean():.3f}")
    print(f"\nimputed person x item table: {len(OUT):,} rows")


if __name__ == "__main__":
    main()
