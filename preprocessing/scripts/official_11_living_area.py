"""Is the 2022-to-2024 failure of rank order across the nine living areas explained by
sampling error?

The rank correlation of living-area attendance rates between the two waves is 0.067.
Three checks are done here:

  (a) a heterogeneity test that accounts for the sampling error in each cell
  (b) a multiple-testing correction
  (c) control for population composition -- composition or a genuine area effect

With the microdata in hand the cell-level variance can be computed directly, which the
published tables alone did not allow.

Part (c) is done by direct standardisation. Reweighting each wave's living-area rates
to a common standard population (the 2024 sex x age distribution) removes the share of
the rank change that composition explains.

Outputs:
  official_living_area_estimates.csv  wave x living area x crude and standardised rate
                                      x standard error
  official_living_area_tests.csv      rank correlation, heterogeneity tests, and
                                      between-wave difference tests with corrections
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

WAVES = {2022: ("mixed", [1.0, 3.0], "SQ1"), 2024: ("general", [1.0], "SQ1")}


def wmean_var(y, w):
    """A weighted proportion and its linearised variance."""
    sw = w.sum()
    if sw <= 0 or len(y) == 0:
        return np.nan, np.nan
    p = float((w * y).sum() / sw)
    v = float((w ** 2 * (y - p) ** 2).sum() / sw ** 2)
    return p, v


def main():
    cw = pd.read_csv(C.REFERENCE / "seoul_living_area_crosswalk.csv")
    la_by_gu = dict(zip(cw.district_name, cw.living_area_ko))
    keys = C.read_person("official_person_keys")

    frames = {}
    for wave, (grp, codes, guvar) in WAVES.items():
        df, _m = C.load(wave, grp)
        k = keys[(keys.wave == wave) & (keys.source_group == grp)].reset_index(drop=True)
        df = df.reset_index(drop=True)
        if grp == "mixed":
            mask = pd.to_numeric(df["GRP"], errors="coerce").isin(codes)
            df, k = df[mask].reset_index(drop=True), k[mask.values].reset_index(drop=True)
        A = R.attendance_matrix(wave, df)
        y = (A.sum(axis=1) > 0).astype(float).values          # offline_total
        gm = R.gu_name_map(wave, grp, guvar)
        la = pd.to_numeric(df[guvar], errors="coerce").map(gm).map(la_by_gu)
        frames[wave] = pd.DataFrame({
            "living_area": la.values, "y": y,
            "w": k.weight.fillna(1.0).values,
            "sex": k.sex.values, "age_band": k.age_band.values,
        }).dropna(subset=["living_area"])

    # ---- standard population: the 2024 sex x age distribution ------------------
    std = frames[2024].groupby(["sex", "age_band"]).w.sum()
    std = std / std.sum()

    rows = []
    for wave, d in frames.items():
        for la, g in d.groupby("living_area"):
            p, v = wmean_var(g.y.values, g.w.values)
            # direct standardisation: remix the cell rates by the standard weights
            cell = g.groupby(["sex", "age_band"]).apply(
                lambda x: pd.Series(wmean_var(x.y.values, x.w.values), index=["p", "v"]),
                include_groups=False)
            common = cell.index.intersection(std.index)
            if len(common):
                wt = std.loc[common] / std.loc[common].sum()
                ps = float((cell.loc[common, "p"] * wt).sum())
                vs = float(((wt ** 2) * cell.loc[common, "v"]).sum())
            else:
                ps, vs = np.nan, np.nan
            pu, vu = wmean_var(g.y.values, np.ones(len(g)))
            rows.append(dict(wave=wave, living_area=la, n=len(g),
                             crude=round(p * 100, 2), crude_se=round(np.sqrt(v) * 100, 3),
                             std=round(ps * 100, 2), std_se=round(np.sqrt(vs) * 100, 3),
                             unweighted=round(pu * 100, 2),
                             unweighted_se=round(np.sqrt(vu) * 100, 3)))
    E = pd.DataFrame(rows)

    tests = []
    piv = E.pivot(index="living_area", columns="wave", values="crude")
    pivs = E.pivot(index="living_area", columns="wave", values="std")
    pivu = E.pivot(index="living_area", columns="wave", values="unweighted")
    # The 2022 wt3_b weight is entangled with the outcome (official_14), so an
    # unweighted version is always reported alongside.
    for lab, P in [("crude", piv), ("age_sex_standardised", pivs), ("unweighted", pivu)]:
        rho, pv = stats.spearmanr(P[2022], P[2024])
        tests.append(dict(test=f"spearman_rank_{lab}", statistic=round(float(rho), 4),
                          p_value=round(float(pv), 4), df="", n=len(P),
                          note="how far the living-area rank order survives between "
                               "the two waves"))

    # Heterogeneity across living areas within each wave, using the weighted variance
    for wave in WAVES:
        g = E[E.wave == wave]
        for lab, col, se in [("crude", "crude", "crude_se"), ("std", "std", "std_se")]:
            m = g[col].values / 100
            s = g[se].values / 100
            wgt = 1 / np.maximum(s ** 2, 1e-12)
            mu = float((wgt * m).sum() / wgt.sum())
            Q = float((wgt * (m - mu) ** 2).sum())
            dfree = len(m) - 1
            tests.append(dict(test=f"heterogeneity_Q_{wave}_{lab}", statistic=round(Q, 2),
                              p_value=round(float(stats.chi2.sf(Q, dfree)), 6), df=dfree,
                              n=len(m),
                              note="are the differences across living areas explained "
                                   "by sampling error? (H0: all equal)"))

    # Significance of each living area's 2022-to-2024 change, with Bonferroni and BH
    diffs = []
    for la in piv.index:
        a = E[(E.wave == 2022) & (E.living_area == la)].iloc[0]
        b = E[(E.wave == 2024) & (E.living_area == la)].iloc[0]
        for lab, c, se in [("crude", "crude", "crude_se"), ("std", "std", "std_se"),
                           ("unweighted", "unweighted", "unweighted_se")]:
            d = b[c] - a[c]
            sed = np.sqrt(a[se] ** 2 + b[se] ** 2)
            z = d / sed if sed > 0 else np.nan
            diffs.append(dict(living_area=la, basis=lab, diff_pp=round(float(d), 2),
                              se=round(float(sed), 3), z=round(float(z), 3),
                              p_raw=float(2 * stats.norm.sf(abs(z)))))
    D = pd.DataFrame(diffs)
    for lab, g in D.groupby("basis"):
        m = len(g)
        D.loc[g.index, "p_bonferroni"] = np.minimum(g.p_raw * m, 1.0)
        order = g.p_raw.rank(method="first")
        D.loc[g.index, "p_bh"] = np.minimum(g.p_raw * m / order, 1.0)
    D["sig_bonferroni"] = D.p_bonferroni < 0.05
    D = D.round({"p_raw": 5, "p_bonferroni": 5, "p_bh": 5})

    # ---- could sampling error alone break the rank order? ---------------------
    # H0: each living area's true rate is the same in both waves (taken as the mean of
    # the two observed values). Drawing twice with the observed standard errors builds
    # a distribution of rank correlations. If the observed rho sits comfortably inside
    # it, the claim that the ranking "flipped" is explained by sampling error.
    rng = np.random.default_rng(20260908)
    for lab, col, se in [("crude", "crude", "crude_se"), ("std", "std", "std_se"),
                         ("unweighted", "unweighted", "unweighted_se")]:
        a = E[E.wave == 2022].set_index("living_area")
        b = E[E.wave == 2024].set_index("living_area")
        idx = a.index
        truth = ((a.loc[idx, col] + b.loc[idx, col]) / 2).values
        sa, sb = a.loc[idx, se].values, b.loc[idx, se].values
        sims = []
        for _ in range(20000):
            x = rng.normal(truth, sa)
            y = rng.normal(truth, sb)
            sims.append(stats.spearmanr(x, y).statistic)
        sims = np.array(sims)
        obs = stats.spearmanr(a.loc[idx, col], b.loc[idx, col]).statistic
        tests.append(dict(
            test=f"rank_null_bootstrap_{lab}", statistic=round(float(obs), 4),
            p_value=round(float((sims <= obs).mean()), 4), df="", n=20000,
            note=(f"under H0 (true values unchanged) the rank correlation has median "
                  f"{np.median(sims):.2f} and 5th percentile "
                  f"{np.percentile(sims, 5):.2f}; p is the chance of a value as low "
                  f"as the observed one")))

    # Correlation between baseline level and change -- a regression-to-the-mean check
    for lab, col in [("crude", "crude"), ("std", "std"), ("unweighted", "unweighted")]:
        a = E[E.wave == 2022].set_index("living_area")[col]
        b = E[E.wave == 2024].set_index("living_area")[col]
        r, pv = stats.pearsonr(a, b - a)
        tests.append(dict(test=f"baseline_vs_change_{lab}", statistic=round(float(r), 4),
                          p_value=round(float(pv), 4), df="", n=len(a),
                          note="correlation between the 2022 level and the 2022-to-2024 "
                               "change; a strongly negative value suggests regression "
                               "to the mean"))

    T = pd.DataFrame(tests)
    E.to_csv(C.OUT / "official_living_area_estimates.csv", index=False)
    pd.concat([T, D.assign(test="wave_diff")], ignore_index=True).to_csv(
        C.OUT / "official_living_area_tests.csv", index=False)

    print("== total attendance rate by living area (crude / sex-age standardised, %) ==")
    out = E.pivot_table(index="living_area", values=["crude", "crude_se", "std", "std_se"],
                        columns="wave")
    print(out.round(2).to_string())
    print("\n== tests ==")
    print(T.to_string(index=False))
    print("\n== 2022-to-2024 change by living area (sex-age standardised) ==")
    print(D[D.basis == "std"][["living_area", "diff_pp", "se", "z", "p_raw",
                               "p_bonferroni", "sig_bonferroni"]].to_string(index=False))
    n_sig = int(D[(D.basis == "std")].sig_bonferroni.sum())
    print(f"\nliving areas significant after Bonferroni: {n_sig}/{len(piv)}")


if __name__ == "__main__":
    main()
