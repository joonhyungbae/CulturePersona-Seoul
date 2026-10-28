"""What were the weights built from -- demographics, or the outcome itself?

Every downstream conclusion in this track (the attendance series, the frame effects,
the living-area gaps) stands on the weights. And the SPSS label on the 2022 variable
`wt3_b` names movies and theater as raking margins. If that name is accurate, the
attendance rates for those two genres in that wave are not estimates at all but
**raking control totals**.

For waves whose weights carry no such name, three diagnostics are used.

  1. Regress log(weight) on sex, age band and district. Pure demographic raking gives
     a high R-squared.
  2. Ask whether the **residual** of that regression correlates with attendance. If
     the part of the weight demographics cannot explain tracks the outcome, the weight
     is entangled with the outcome.
  3. Measure the residual correlation item by item, to name **which genre** was a
     control variable.

A caution: this is correlation, not proof. Raking on an unobserved demographic axis
that itself correlates with attendance -- education or income, say -- would also leave
a residual correlated with attendance. So a second version that puts education and
income in as auxiliary variables is reported alongside. What survives that is stronger
evidence of raking on the outcome. Settling it requires the control totals from the
foundation.

Outputs:
  official_weight_provenance.csv       wave x weight x R-squared x residual
                                       correlation x the genre named
  official_weight_item_residuals.csv   wave x item x residual correlation, in both
                                       versions
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C
import official_05_reproduce as R
from official_03_samples import KEYS

# (wave, group, weight variable, general-public GRP filter, education/income variables)
WEIGHTED = [
    (2020, "general", "WT", None, {"education": "DQ2", "income": "DQ7"}),
    (2020, "general", "WTB", None, {"education": "DQ2", "income": "DQ7"}),
    (2022, "mixed", "wt3_b", [1, 3], {"education": "DQ2", "income": "DQ9"}),
    (2024, "general", "WT1_2", None, {"education": "DQ2", "income": "DQ8"}),
]


def _design_matrix(k: pd.DataFrame, df: pd.DataFrame, extra: dict | None) -> pd.DataFrame:
    cols = {"sex": k.sex.astype(str),
            "age": k.age_band.astype(str),
            "gu": k.gu_code.astype("Int64").astype(str)}
    if extra:
        for name, var in extra.items():
            if var in df.columns:
                cols[name] = pd.to_numeric(df[var], errors="coerce").astype("Int64").astype(str)
    X = pd.get_dummies(pd.DataFrame(cols), drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    return X


def _fit(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1.0 - res.var() / y.var() if y.var() > 0 else np.nan
    return res, float(r2)


def main():
    prov, item_rows = [], []
    for wave, grp, wcol, gf, extra in WEIGHTED:
        df, _m = C.load(wave, grp)
        df = df[df[KEYS[wave]["pid"]].notna()].reset_index(drop=True)
        k = C.read_person("official_person_keys")
        k = k[(k.wave == wave) & (k.source_group == grp)].reset_index(drop=True)
        if gf is not None:
            mask = pd.to_numeric(df["GRP"], errors="coerce").isin(gf)
            df, k = df[mask].reset_index(drop=True), k[mask.values].reset_index(drop=True)

        w = pd.to_numeric(df[wcol], errors="coerce")
        A = R.attendance_matrix(wave, df)
        any_att = (A.sum(axis=1) > 0).astype(float).values

        for label, ex in (("demographic", None), ("demographic+edu+income", extra)):
            X = _design_matrix(k, df, ex)
            lw = np.log(w.values)
            ok = np.isfinite(lw) & X.notna().all(axis=1).values
            res, r2 = _fit(X.values[ok], lw[ok])
            r_any, p_any = stats.pointbiserialr(any_att[ok], res)

            best, best_r = None, 0.0
            for it in A.columns:
                rr, pp = stats.pointbiserialr(A[it].values[ok], res)
                item_rows.append(dict(wave=wave, weight_var=wcol, adjustment=label,
                                      canonical_item=it, residual_corr=round(float(rr), 3),
                                      p_value=float(pp)))
                if abs(rr) > abs(best_r):
                    best, best_r = it, rr

            prov.append(dict(
                wave=wave, weight_var=wcol, adjustment=label, n=int(ok.sum()),
                r2_log_weight=round(r2, 3),
                residual_corr_any_attend=round(float(r_any), 3),
                p_any=float(p_any),
                top_item=best, top_item_corr=round(float(best_r), 3),
                verdict=("outcome_entangled" if abs(best_r) >= 0.30 else "demographic_only"),
            ))

    P = pd.DataFrame(prov)
    I = pd.DataFrame(item_rows)
    P.to_csv(C.OUT / "official_weight_provenance.csv", index=False)
    I.to_csv(C.OUT / "official_weight_item_residuals.csv", index=False)

    print("== weight provenance diagnostics ==")
    print(P[["wave", "weight_var", "adjustment", "r2_log_weight",
             "residual_corr_any_attend", "top_item", "top_item_corr", "verdict"]]
          .to_string(index=False))
    print("\n== item residual correlations surviving education and income (|r| >= 0.3) ==")
    adj = I[(I.adjustment == "demographic+edu+income") & (I.residual_corr.abs() >= 0.30)]
    print(adj.sort_values("residual_corr")[
        ["wave", "weight_var", "canonical_item", "residual_corr"]].to_string(index=False))
    print("\nCaution: correlation is not proof. Settling this requires the control "
          "totals from the foundation.")


if __name__ == "__main__":
    main()
