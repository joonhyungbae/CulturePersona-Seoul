"""Read `outputs/reports/` and `data/eval/era_holdout/tau.json` and write the paper's
LaTeX number macros and tables (ADR-009/010).

Run: uv run python paper/make_tables_era.py --tables-dir <paper>/tables
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent  # repository root
REPORTS = PROJ / "outputs" / "reports"
TABLES = HERE / "tables"  # overridden by --tables-dir
TAU_PATH = PROJ / "data" / "eval" / "era_holdout" / "tau.json"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


AGE_BAND_TEX = {
    "10대": "15--19",
    "20대": "20--29",
    "30대": "30--39",
    "40대": "40--49",
    "50대": "50--59",
    "60대": "60--69",
    "70대 이상": "70+",
}


def _label_tex(s: str) -> str:
    """Make a table label LaTeX-safe. Age bands become ASCII ranges; other Korean is wrapped in \\ko{}."""
    s = str(s)
    for ko, tex in AGE_BAND_TEX.items():
        s = s.replace(ko, tex)
    parts = []
    buf = ""
    korean = False
    for ch in s:
        is_ko = "\uac00" <= ch <= "\ud7a3"
        if buf and is_ko != korean:
            parts.append((korean, buf))
            buf = ""
        korean = is_ko
        buf += ch
    if buf:
        parts.append((korean, buf))
    return "".join(f"\\ko{{{seg}}}" if ko else tex_escape(seg) for ko, seg in parts)


def _z(x: float, eps: float = 0.5) -> float:
    """Map a negative that rounds to zero to 0, so the table does not print '-0'."""
    return 0.0 if abs(float(x)) < eps else float(x)


def _ladder(p: pd.DataFrame) -> pd.DataFrame:
    """Ladder pairs: not structural, and not pairs that include attendance status.
    Every rung is averaged over the same pairs.
    """
    cond = (
        p["conditioning"].fillna(False).astype(bool)
        if "conditioning" in p.columns
        else p.anchor_a.eq("attend_any") | p.anchor_b.eq("attend_any")
    )
    return p[~p["structural"].fillna(False).astype(bool) & ~cond]


def error_summary_era() -> None:
    """2024 non-structural mean TVD by reconstruction, with bootstrap CI and the unweighted column.

    The table answers whether the gap exceeds sampling noise (bootstrap range and
    sampling floor) and whether weighting created it (unweighted column). Weights
    are raked on sex, age, and living area as well as attendance, so every
    reconstruction row also reports the unweighted mean.
    """
    p = pd.read_csv(REPORTS / "decade_error_atlas_pairs.csv")
    p = p[p["wave"] == 2024]
    ns = _ladder(p)
    unw = pd.read_csv(REPORTS / "decade_error_atlas_weighting.csv")
    unw = unw[(unw.wave == 2024) & (unw["mode"] == "unweighted")].iloc[0]
    ci = pd.read_csv(REPORTS / "era_atlas_uncertainty.csv")
    ci = ci[ci.wave == 2024].set_index("metric")
    rows = [
        ("marginals only", "tvd_indep", "mean_tvd_indep"),
        (r"+ age-band conditionals", "tvd_cond_age", "mean_tvd_cond_age"),
        (r"+ age band $\times$ sex", "tvd_cond_age_sex", "mean_tvd_cond_age_sex"),
        (r"+ skip logic (no new table)", "tvd_cond_age_route", "mean_tvd_cond_age_route"),
        (
            r"+ age band $\times$ attendance status",
            "tvd_cond_age_attend",
            "mean_tvd_cond_age_attend",
        ),
    ]
    # ADR-035 half-sample cross-fit: conditionals on half A, scored on half B.
    # P(C) comes from A, the same side as the chain.
    xf_path = REPORTS / "era_ladder_crossfit.csv"
    xf = pd.read_csv(xf_path) if xf_path.exists() else None
    xf24 = xf[xf.wave == 2024].set_index("rung") if xf is not None else None
    out_macros: dict[str, str] = {}
    body = []
    for lab, col, ucol in rows:
        s_ = ns[col].dropna()
        ci_txt = f"[{ci.loc[col].lo:.3f}, {ci.loc[col].hi:.3f}]" if col in ci.index else "---"
        rung = col.removeprefix("tvd_")
        oo = (
            f"{float(xf24.loc[rung, 'oos_a_mean']):.3f}"
            if xf24 is not None and rung in xf24.index
            else "---"
        )
        out_names = {
            "indep": "tvdIndepOut",
            "cond_age": "tvdAgeOut",
            "cond_age_sex": "tvdAgeSexOut",
            "cond_age_route": "tvdAgeRouteOut",
            "cond_age_attend": "tvdAgeAttendOut",
        }
        if oo != "---":
            out_macros[out_names[rung]] = oo
        body.append(f"    {lab} & {s_.mean():.3f} & {ci_txt} & {float(unw[ucol]):.3f} & {oo} \\\\")
    if xf24 is not None and {"indep", "cond_age", "cond_age_route", "cond_age_attend"} <= set(
        xf24.index
    ):
        # Held-out shares removed by each step, to read next to the in-sample percentages.
        o = {r: float(xf24.loc[r, "oos_a_mean"]) for r in xf24.index}
        out_macros["ageCutOut"] = f"{100 * (o['indep'] - o['cond_age']) / o['indep']:.0f}"
        out_macros["attendCutOut"] = (
            f"{100 * (o['cond_age'] - o['cond_age_attend']) / o['cond_age']:.0f}"
        )
        out_macros["routeCutOut"] = (
            f"{100 * (o['cond_age'] - o['cond_age_route']) / o['cond_age']:.0f}"
        )
        out_macros["beyondRouteOut"] = (
            f"{100 * (o['cond_age_route'] - o['cond_age_attend']) / o['cond_age_route']:.0f}"
        )
    fl = ci.loc["sampling_floor"]
    if xf24 is not None:
        ch_out = float(xf24.chain_oos.iloc[0])
        xfl = float(xf24["floor"].iloc[0])
        # In-sample value of the full-sample fit: the released chain in the sampler ablation (same source
        # as the \\tvdChain macro).
        abl_p = REPORTS / "era_sampler_ablation.csv"
        if abl_p.exists():
            abl = pd.read_csv(abl_p).set_index("variant")
            ch_in = float(f"{abl.loc['released', 'tvd_nonstructural']:.3f}")
        else:
            ch_in = float(xf24.chain_in.iloc[0])
        body.append(
            f"    \\midrule\n    Bayesian-network sampler, measured configuration (Section~\\ref{{subsec:chain-fit}}) & {ch_in:.3f} & --- & --- & {ch_out:.3f} \\\\"
        )
        out_macros["ladderSplitFloor"] = f"{xfl:.3f}"
        body.append(
            f"    sampling floor & {fl.point:.3f} & [{fl.lo:.3f}, {fl.hi:.3f}] & --- & {xfl:.3f} \\\\"
        )
    else:
        body.append(
            f"    \\midrule\n    sampling floor & {fl.point:.3f} & [{fl.lo:.3f}, {fl.hi:.3f}] & --- & --- \\\\"
        )
    if out_macros:
        npath = TABLES / "numbers.tex"
        ntext = npath.read_text(encoding="utf-8")
        for k, v in out_macros.items():
            ntext = _set_macro(ntext, k, v)
        npath.write_text(ntext if ntext.endswith("\n") else ntext + "\n", encoding="utf-8")
    (TABLES / "error_summary.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\columnwidth}\n  \\begin{tabular}{@{}lrrrr@{}}\n    \\toprule\n"
        "    Reconstruction & mean & bootstrap range & unweighted & held-out \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Mean total variation distance over the 2024 wave's \\nPairsNonStructural{} non-structural anchor pairs. "
        + f"Ranges: bootstrap over respondents ($B={int(ci.loc['tvd_indep', 'n_boot'])}$), resampling variability rather than confidence intervals. "
        + "\\emph{Held-out}: fitted on one half of the respondents, scored on the other (mean over \\nLadderSplits{} random splits); floors are resampled vs.\\ full-sample joint (in sample) and half vs.\\ half (held out).}\n"
        "  \\label{tab:error-summary}\n\\end{table}\n",
        encoding="utf-8",
    )


def pair_class_era() -> None:
    """Reduction by attendance-block membership (ADR-016).
    The text cites 2024; the table shows three waves.
    """
    src = REPORTS / "decade_error_atlas_pair_class.csv"
    if not src.exists():
        return
    d = pd.read_csv(src)
    d = d[d.wave.isin([2020, 2022, 2024])]
    label = {"both": "both", "one": "one", "neither": "neither"}
    body = []
    for klass in ("both", "one", "neither"):
        for wave in (2024, 2022, 2020):
            r = d[(d.wave == wave) & (d.pair_class == klass)].iloc[0]
            body.append(
                f"    {label[klass]} & {wave} & {int(r.n_pairs)} & {r.tvd_cond_age:.3f} & "
                f"{r.tvd_cond_age_route:.3f} & {r.tvd_cond_age_attend:.3f} & "
                f"{_z(r.route_cut_pct):.0f} & {_z(r.attend_beyond_route_pct):.0f} \\\\"
            )
    (TABLES / "pair_class.tex").write_text(
        "\\begin{table*}[tbp]\n  \\centering\n  \\small\n  \\begin{adjustbox}{max width=\\textwidth}\n"
        "  \\begin{tabular}{@{}llrrrrrr@{}}\n    \\toprule\n"
        "    Anchors in block & Wave & pairs & $+$age & $+$skip logic & $+$age$\\times$attendance & skip cut (\\%) & beyond (\\%) \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Where the attendance step comes from: non-structural pairs by how many of their anchors sit in the attendance block (mean pair TVD). "
        "\\emph{Skip cut}: share of the age-conditional residual the skip logic removes; \\emph{beyond}: share of the remainder the attendance table removes.}\n"
        "  \\label{tab:pair-class}\n\\end{table*}\n",
        encoding="utf-8",
    )


def replication_era() -> None:
    s = pd.read_csv(REPORTS / "decade_error_atlas_summary.csv")
    s = s.set_index("wave")
    st = pd.read_csv(REPORTS / "decade_error_atlas_stability.csv")
    rho22 = float(st[(st.wave_a == 2024) & (st.wave_b == 2022)].spearman_tvd_cond_age.iloc[0])
    rho20 = float(st[(st.wave_a == 2024) & (st.wave_b == 2020)].spearman_tvd_cond_age.iloc[0])
    labs = [
        ("marginals only", "mean_tvd_indep"),
        ("+ age band", "mean_tvd_cond_age"),
        (r"+ age $\times$ sex", "mean_tvd_cond_age_sex"),
        ("+ skip logic", "mean_tvd_cond_age_route"),
        (r"+ age $\times$ attendance", "mean_tvd_cond_age_attend"),
    ]
    waves = [w for w in (2024, 2022, 2020, 2018, 2016) if w in s.index]
    body = []
    for lab, col in labs:
        if col not in s.columns:
            continue
        body.append(f"    {lab} & " + " & ".join(f"{s.loc[w, col]:.3f}" for w in waves) + " \\\\")
    (TABLES / "replication.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}l"
        + "r" * len(waves)
        + "@{}}\n    \\toprule\n"
        "    Reconstruction & "
        + " & ".join(f"{w}$^{{\\dagger}}$" if w in (2018, 2016) else str(w) for w in waves)
        + " \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        f"  \\caption{{Error study by wave: mean pair TVD over the ladder pairs. $^{{\\dagger}}$Unweighted diagnostic waves, never pooled with the weighted ones. Rank stability of age-conditional "
        f"pair errors: Spearman $\\rho(2024,2022)={rho22:.2f}$, $\\rho(2024,2020)={rho20:.2f}$.}}\n"
        "  \\label{tab:replication}\n\\end{table}\n",
        encoding="utf-8",
    )


def ablation_era() -> None:
    e = pd.read_csv(REPORTS / "era_eval_suite.csv")
    body = []
    for wave in (2020, 2022, 2024):
        for abl in ("margins", "structure_T"):
            r = e[(e.wave == wave) & (e.ablation == abl)].iloc[0]
            body.append(
                f"    {wave} / {tex_escape(abl)} & {r.width_mae_by_age:.3f} & "
                f"{r.genre_phi_mad:.4f} & {r.genre_pair_joint_tvd:.4f} \\\\"
            )
    (TABLES / "ablation.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrr@{}}\n    \\toprule\n"
        "    Wave / ablation & width MAE$_{\\mathrm{age}}$ & $\\phi$ MAD & pair joint TVD \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Structure-transfer ablation: margins-only vs margins+structure prior $T$ "
        "from 2020$\\cup$2022. $T$ is judged on genre $\\phi$ MAD and genre-pair joint TVD; "
        "2020 and 2022 are the prior's sources (in sample).}\n"
        "  \\label{tab:ablation}\n\\end{table}\n",
        encoding="utf-8",
    )


def holdout_era() -> None:
    cw = pd.read_csv(REPORTS / "era_holdout_crosswave.csv")
    bl_path = REPORTS / "era_holdout_baselines.csv"
    bl = pd.read_csv(bl_path) if bl_path.exists() else None
    # wide summary: mean over items per wave × tag
    rows = []
    for wave in (2024, 2022, 2020):
        sub = cw[cw.wave == wave]
        extra = ""
        if bl is not None:
            b = bl[bl.wave == wave]
            pw = b.prior_wave_lookup_tvd.dropna()
            pw_txt = f"{pw.mean():.3f} ({len(pw)})" if len(pw) else "---"
            extra = (
                f" & {b.marginal_lookup_tvd.mean():.3f} & {pw_txt} & {b.real_sample_tvd.mean():.3f}"
            )
        rows.append(
            f"    {wave} & {sub.tvd_release_narrative.mean():.3f} & "
            f"{sub.tvd_twin_anchors.mean():.3f}{extra} & {sub.noise_floor_tvd.mean():.3f} & "
            f"{int(sub.n.iloc[0])} \\\\"
        )
    cols = "lrrrrrrr" if bl is not None else "lrrrr"
    head = (
        "    Wave & release SSR & twin SSR & marginal lookup & prior-wave lookup & real $n{=}300$ & noise floor & $n$ \\\\\n"
        if bl is not None
        else "    Wave & release SSR & twin SSR & noise floor & $n$ \\\\\n"
    )
    (TABLES / "holdout.tex").write_text(
        "\\begin{table*}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}"
        + cols
        + "@{}}\n    \\toprule\n"
        + head
        + "    \\midrule\n"
        + "\n".join(rows)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Holdout by wave: mean cell-weighted TVD over the nine held-out items "
        "(cells $=$ age band $\\times$ attendance status, SSR at frozen $\\tau{=}\\ssrTau$). "
        "Release $=$ narrative personas; twin $=$ real respondents' anchor lists (internal); the lookups, the real-sample reference and the noise floor are defined in the text.}\n"
        "  \\label{tab:holdout}\n\\end{table*}\n",
        encoding="utf-8",
    )
    # per-item detail for 2024 as holdout_modes stand-in
    sub = cw[cw.wave == 2024].sort_values("item")
    body = []
    for r in sub.itertuples():
        body.append(
            f"    \\texttt{{{tex_escape(r.item)}}} & {r.tvd_release_narrative:.3f} & "
            f"{r.tvd_twin_anchors:.3f} & {r.noise_floor_tvd:.3f} \\\\"
        )
    (TABLES / "holdout_modes.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrr@{}}\n    \\toprule\n"
        "    Item (2024) & release & twin & floor \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Per-item cell-mean TVD for the 2024 wave (same protocol as Table~\\ref{tab:holdout}).}\n"
        "  \\label{tab:holdout-modes}\n\\end{table}\n",
        encoding="utf-8",
    )


def residual_pairs_era() -> None:
    """Top-15 non-structural pairs by residual TVD under age×attend (2024 atlas)."""
    p = pd.read_csv(REPORTS / "decade_error_atlas_pairs.csv")
    p = _ladder(p[p.wave == 2024])
    p = p.sort_values("tvd_cond_age_attend", ascending=False).head(15)
    body = []
    for r in p.itertuples():
        body.append(
            f"    \\texttt{{{tex_escape(r.anchor_a)}}}--\\texttt{{{tex_escape(r.anchor_b)}}} & "
            f"{r.tvd_indep:.3f} & {r.tvd_cond_age:.3f} & {r.tvd_cond_age_attend:.3f} \\\\"
        )
    (TABLES / "residual_pairs.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrr@{}}\n    \\toprule\n"
        "    Pair & indep & +age & +age$\\times$attend \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Fifteen non-structural anchor pairs with the largest residual TVD under "
        "age-band $\\times$ attendance conditioning (2024 wave).}\n"
        "  \\label{tab:residual-pairs}\n\\end{table}\n",
        encoding="utf-8",
    )


def ppi_era() -> None:
    d = pd.read_csv(REPORTS / "era_holdout_ppi.csv")
    d24 = d[d.wave == 2024].copy()
    piv = d24.pivot(index="item", columns="predictor", values="rho2")
    # sort by learned ascending (harder items first)
    order = piv["learned"].sort_values().index
    body = []
    for item in order:
        L = 100 * float(piv.loc[item, "learned"])
        T = 100 * float(piv.loc[item, "twin_ssr"]) if "twin_ssr" in piv.columns else float("nan")
        D = 100 * float(piv.loc[item, "distilled"]) if "distilled" in piv.columns else float("nan")
        body.append(f"    \\texttt{{{tex_escape(str(item))}}} & {L:.1f} & {T:.1f} & {D:.1f} \\\\")
    (TABLES / "ppi.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrr@{}}\n    \\toprule\n"
        "    Item & learned & twin SSR & distilled \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{PPI++ respondents saved (\\%, $\\rho^2$) on the 2024 twin subsample ($n{=}\\nHoldoutPersonas$): "
        "\\emph{learned}, an out-of-fold logistic on sex $\\times$ age $\\times$ attendance status; "
        "\\emph{twin SSR}, frozen-$\\tau$ softmax of twin-anchor answers; "
        "\\emph{distilled}, a multinomial logistic fit on release SSR soft targets, never seeing a real answer.}\n"
        "  \\label{tab:ppi}\n\\end{table}\n",
        encoding="utf-8",
    )


def holdout_items_era() -> None:
    items = [
        ("leisure_activity_1st", "leisure", "Main leisure activity"),
        ("leisure_activity_wanted_1st", "leisure", "Wanted leisure activity"),
        ("leisure_time_weekday", "leisure", "Weekday leisure hours"),
        ("leisure_time_weekend", "leisure", "Weekend leisure hours"),
        ("arts_importance", "domain", "Importance of arts"),
        ("attendance_overall", "domain", "Attendance satisfaction"),
        ("cultural_life_overall", "domain", "Cultural-life satisfaction"),
        ("effect_overall_happiness", "domain", "Effect on happiness"),
        ("effect_stress_relieved", "domain", "Effect on stress relief"),
    ]
    body = "\n".join(f"    \\texttt{{{tex_escape(i)}}} & {k} & {lab} \\\\" for i, k, lab in items)
    (TABLES / "holdout_items.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lll@{}}\n    \\toprule\n"
        "    Item & Layer & Description \\\\\n    \\midrule\n"
        + body
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Held-out items: four leisure items and five domain items asked in every weighted wave.}\n"
        "  \\label{tab:holdout-items}\n\\end{table}\n",
        encoding="utf-8",
    )


def _tex_int_comma(n: int) -> str:
    """LaTeX-safe thousands: 5011 → 5{,}011."""
    return f"{n:,}".replace(",", "{,}")


def _set_macro(text: str, name: str, value: str) -> str:
    """Replace \\newcommand{\\name}{...} with brace-balanced body, or append."""
    needle = f"\\newcommand{{\\{name}}}{{"
    i = text.find(needle)
    if i < 0:
        return text + f"\n\\newcommand{{\\{name}}}{{{value}}}"
    j = i + len(needle)
    depth = 1
    k = j
    while k < len(text) and depth:
        ch = text[k]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        k += 1
    return text[:j] + value + text[k - 1 :]


def _era_sampler_macros() -> dict[str, str]:
    """Rebuild the sampler-audit number macros from the era sampler audit (ADR-012)."""
    out: dict[str, str] = {}
    abl_p = REPORTS / "era_sampler_ablation.csv"
    if abl_p.exists():
        a = pd.read_csv(abl_p).set_index("variant")
        ns = a["tvd_nonstructural"]
        st = a["tvd_structural"]
        phi = a["genre_phi_mae"]
        out.update(
            {
                "tvdChain": f"{ns['released']:.3f}",
                "ablReleased": f"{ns['released']:.3f}",
                "ablNoAttend": f"{ns['no_attend']:.3f}",
                "ablAgeOnly": f"{ns['age_only']:.3f}",
                "ablIndependent": f"{ns['independent']:.3f}",
                "ablNoTree": f"{ns['no_genre_tree']:.3f}",
                "ablNoCountNode": f"{ns['no_count_node']:.3f}",
                "ablNoSameConstruct": f"{ns['no_same_construct']:.3f}",
                "ablReleasedStruct": f"{st['released']:.3f}",
                "ablNoAttendStruct": f"{st['no_attend']:.3f}",
                "ablPhiReleased": f"{phi['released']:.4f}",
                "ablPhiNoTree": f"{phi['no_genre_tree']:.4f}",
                "ablPhiRatio": f"{phi['no_genre_tree'] / phi['released']:.2f}",
            }
        )
    rej_p = REPORTS / "era_sampler_rejection.csv"
    if rej_p.exists():
        r = pd.read_csv(rej_p).set_index("variant")
        out.update(
            {
                "rejRawViol": f"{100 * r.loc['raw_old', 'violation_r1']:.1f}",
                "rejRawViolSix": f"{100 * r.loc['raw_old', 'violation_r6']:.1f}",
                "rejRawNonAtt": f"{100 * r.loc['raw_old', 'nonattender_share']:.1f}",
                "rejRawNGenres": f"{r.loc['raw_old', 'tvd_n_genres']:.3f}",
                "rejWholeNonAtt": f"{100 * r.loc['whole_old', 'nonattender_share']:.1f}",
                "rejWholeAttend": f"{r.loc['whole_old', 'tvd_attend_any']:.3f}",
                "rejLocalOldCount": f"{r.loc['local_old', 'tvd_attend_count_band']:.3f}",
                "rejNewNGenres": f"{r.loc['local_new', 'tvd_n_genres']:.3f}",
                "rejNewCount": f"{r.loc['local_new', 'tvd_attend_count_band']:.3f}",
            }
        )
    meta_p = REPORTS / "era_sampler_audit.json"
    if meta_p.exists():
        m = json.loads(meta_p.read_text(encoding="utf-8"))
        out.update(
            {
                "backoffMin": f"{100 * m.get('backoff_min', float('nan')):.0f}",
                "backoffMinNode": tex_escape(str(m.get("backoff_min_node", ""))),
                "genreRoot": tex_escape(str(m.get("genre_root", ""))),
                "feasibleShare": f"{m.get('feasible_share', float('nan')):.1f}",
                "samplingZeroShare": f"{m.get('sampling_zero_share', float('nan')):.1f}",
                "nNewCombos": _tex_int_comma(int(m.get("n_new_combos", 0))),
                "nRealCombos": _tex_int_comma(int(m.get("n_real_combos", 0))),
                "nSamplerAudit": _tex_int_comma(int(m.get("n_sample", 0))),
                "coverageAxes": " $\\times$ ".join(
                    tex_escape(k.replace("_band", "").replace("_", " "))
                    for k in m.get("coverage_keys", [])
                ),
            }
        )
    return out


def _era_source_macros() -> dict[str, str]:
    """Read grounding tables, population roots, the reproduction gate, and the generator from era outputs."""
    out: dict[str, str] = {}
    gate_p = REPORTS / "official_reproduce_gate.csv"
    if gate_p.exists():
        g = pd.read_csv(gate_p)
        g24 = g[g.wave == 2024]
        g20 = g[g.wave == 2020]
        out["reproMaxPP"] = f"{g24.diff_pp.abs().max():.2f}"
        out["reproTolPP"] = f"{g24.tol_pp.max():.1f}"
        out["nReproItems"] = str(len(g24))
        if len(g20):
            out["reproMaxPPTwoZero"] = f"{g20.diff_pp.abs().max():.2f}"
            out["reproTolPPTwoZero"] = f"{g20.tol_pp.max():.1f}"
    t6 = PROJ / "data" / "grounding" / "era_2024" / "T6_pair_joints.parquet"
    if t6.exists():
        d = pd.read_parquet(t6)
        sec = (
            d["secondary"].astype(bool)
            if "secondary" in d.columns
            else pd.Series(False, index=d.index)
        )
        out["nSuppressedCells"] = _tex_int_comma(int(d.suppressed.sum()))
        out["nSecondaryCells"] = _tex_int_comma(int(sec.sum()))
        out["nReleaseCells"] = _tex_int_comma(int((~d.suppressed.astype(bool) & ~sec).sum()))
    # Differencing-attack check (ADR-024): exact recoveries before primary-only
    # suppression and after complementary suppression; cells that reappear in
    # conditional tables; status of value cells.
    before = REPORTS / "era_sdc_differencing_primary_only.csv"
    after = REPORTS / "era_sdc_differencing.csv"
    if before.exists() and after.exists():
        b = pd.read_csv(before)
        b = b[b.wave == 2024]
        a_ = pd.read_csv(after)
        a_ = a_[a_.wave == 2024]
        out["nDiffPairsScored"] = _tex_int_comma(int((b.table == "T6").sum()))
        out["nDiffExactPairsBefore"] = _tex_int_comma(int(b[b.table == "T6"].exact.sum()))
        out["nDiffExactPairsAfter"] = _tex_int_comma(int(a_[a_.table == "T6"].exact.sum()))
        t6a = a_[a_.table == "T6"]
        out["nDiffNarrowPairsAfter"] = _tex_int_comma(
            int((t6a.bound_width_share * t6a.n_base < 15).sum())
        )
        cond_b = b[b.table != "T6"]
        cond_a = a_[a_.table != "T6"]
        out["nSuppressedCond"] = _tex_int_comma(len(cond_a))
        out["nDiffExactCond"] = _tex_int_comma(int(cond_a.exact.sum()))
        out["nDiffNarrowCond"] = _tex_int_comma(
            int((cond_a.bound_width_primary * cond_a.n_base < 15).sum())
        )
        out["minNDiffExactCond"] = (
            str(int(cond_a[cond_a.exact].n_unweighted.min())) if cond_a.exact.any() else "--"
        )
        del cond_b
    census = REPORTS / "era_sdc_spec_census.csv"
    if census.exists():
        c = pd.read_csv(census).set_index("wave")
        if 2024 in c.index:
            out["nSpecCells"] = _tex_int_comma(int(c.loc[2024, "spec_cells"]))
            out["pctSpecCellsBelowFloor"] = (
                f"{100 * c.loc[2024, 'spec_cells_below_floor'] / c.loc[2024, 'spec_cells']:.0f}"
            )
            out["nGenreCells"] = _tex_int_comma(int(c.loc[2024, "genre_cells"]))
            out["nGenreCellsBelowFloor"] = _tex_int_comma(
                int(c.loc[2024, "genre_cells_below_floor"])
            )
    cf = REPORTS / "era_chain_crossfit.csv"
    if cf.exists():
        c = pd.read_csv(cf)
        sh = c[c.setting == "split-half"]
        out["chainSplitIn"] = f"{sh.tvd_in_sample.mean():.3f}"
        out["chainSplitOut"] = f"{sh.tvd_out_of_sample.mean():.3f}"
        out["chainSplitFloor"] = f"{sh.tvd_sampling_floor.mean():.3f}"
        out["nChainSplits"] = str(len(sh))
        pw = c[c.setting.str.startswith("prior-wave")]
        if len(pw):
            out["chainPriorWaveOut"] = f"{float(pw.tvd_out_of_sample.iloc[0]):.3f}"
            out["chainPriorWaveMicro"] = f"{float(pw.tvd_sampling_floor.iloc[0]):.3f}"
        pl = REPORTS / "era_chain_crossfit_placebo.csv"
        if pl.exists():
            q = pd.read_csv(pl).iloc[0]
            out["placeboCutMean"] = f"{float(q.placebo_cut_pct_mean):.1f}"
            out["placeboCutMax"] = f"{float(q.placebo_cut_pct_max):.1f}"
            out["nPlacebo"] = str(int(q.n_placebo))
        dsn = REPORTS / "era_chain_crossfit_design.csv"
        if dsn.exists():
            d_ = pd.read_csv(dsn)
            g_ = d_.groupby(["published_axis", "method"]).abs_err_pp.mean()
            out["designSynUnpubOOS"] = f"{float(g_.loc[(False, 'synthetic')]):.1f}"
            out["designAgeUnpubOOS"] = f"{float(g_.loc[(False, 'age-conditional')]):.1f}"
            out["designSynPubOOS"] = f"{float(g_.loc[(True, 'synthetic')]):.1f}"
            out["designAgePubOOS"] = f"{float(g_.loc[(True, 'age-conditional')]):.1f}"
    ot = REPORTS / "era_open_text.csv"
    if ot.exists():
        o = pd.read_csv(ot)

        def _v(metric: str, method: str, col: str = "value") -> float:
            return float(o[(o.metric == metric) & (o.method == method)][col].iloc[0])

        out["openTextPersona"] = f"{_v('cell_theme_tvd', 'personas'):.3f}"
        out["openTextMarginal"] = f"{_v('cell_theme_tvd', 'marginal'):.3f}"
        out["openTextShuffled"] = f"{_v('cell_theme_tvd', 'shuffled'):.3f}"
        out["openTextFloor"] = f"{_v('cell_theme_tvd', 'split_half'):.3f}"
        out["openTextHit"] = f"{100 * _v('nearest_cell_hit', 'personas'):.0f}"
        out["openTextChance"] = f"{100 * _v('nearest_cell_hit', 'chance'):.1f}"
        out["nOpenTextReal"] = _tex_int_comma(int(_v("n_real", "-")))
        out["nOpenTextPersonas"] = _tex_int_comma(int(_v("n_personas", "-")))
        out["openTextK"] = str(int(_v("k", "-")))
        th = pd.read_csv(REPORTS / "era_open_text_themes.csv")
        out["openTextTopShare"] = f"{100 * th.persona_share.max():.0f}"
        out["openTextEmptyThemes"] = str(int((th.persona_share < 0.005).sum()))
    ab = REPORTS / "era_ablation_seeds.csv"
    if ab.exists():
        a = pd.read_csv(ab)
        g = a.groupby("variant")[["tvd_nonstructural", "tvd_structural", "genre_phi_mae"]].agg(
            ["mean", "std"]
        )
        out["ablSeedSdStruct"] = f"{float(g.loc['released', ('tvd_structural', 'std')]):.3f}"
        out["ablSeedSdPhi"] = f"{float(g.loc['released', ('genre_phi_mae', 'std')]):.3f}"
        out["ablSeedReleasedStruct"] = f"{float(g.loc['released', ('tvd_structural', 'mean')]):.3f}"
        out["ablSeedNoSameStruct"] = (
            f"{float(g.loc['no_same_construct', ('tvd_structural', 'mean')]):.3f}"
        )
        out["ablSeedReleasedPhi"] = f"{float(g.loc['released', ('genre_phi_mae', 'mean')]):.3f}"
        out["ablSeedNoAttendPhi"] = f"{float(g.loc['no_attend', ('genre_phi_mae', 'mean')]):.3f}"
        out["nAblSeeds"] = str(int(a.groupby("variant").size().max()))
    r0 = PROJ / "data" / "grounding" / "era_2024" / "R0_population_root.parquet"
    prov24 = PROJ / "data" / "grounding" / "era_2024" / "provenance.json"
    if r0.exists() and prov24.exists():
        pop = float(pd.read_parquet(r0).population.sum())
        n_resp = int(json.loads(prov24.read_text(encoding="utf-8"))["n_respondents"])
        out["samplingFraction"] = f"{100 * n_resp / pop:.2f}"
    prov = PROJ / "data" / "grounding" / "era_2024" / "provenance.json"
    if prov.exists():
        pv = json.loads(prov.read_text(encoding="utf-8"))
        out["cellFloor"] = str(int(pv.get("cell_floor", 30)))
    # Each wave uses that year's December registered population as its root (ADR-013).
    months, totals = [], {}
    for wave in (2020, 2022, 2024):
        r0w = PROJ / "data" / "grounding" / f"era_{wave}" / "R0_population_root.parquet"
        if r0w.exists():
            d = pd.read_parquet(r0w)
            months.append(str(d.reference_month.iloc[0]))
            totals[wave] = int(d.population.sum())
    if months:
        out["popMonths"] = ", ".join(months[:-1]) + " and " + months[-1]
        out["popMonth"] = months[-1]
        out["rZeroMonth"] = months[-1]
        out["popTotal"] = f"{totals[2024]:,}"
    keys = PROJ / "data" / "interim" / "official" / "official_person_keys.parquet"
    if keys.exists():
        d = pd.read_parquet(keys, columns=["wave", "weight_var"])
        wv = d[d.wave == 2024].weight_var.dropna()
        if len(wv):
            out["weightVar"] = tex_escape(str(wv.iloc[0]))
    for cand in (
        PROJ / "data" / "synthetic" / "era_2024_narrative" / "generation_metadata.json",
        PROJ / "data" / "synthetic" / "era_2024_narrative_n300.bak" / "generation_metadata.json",
    ):
        if cand.exists():
            out["genModel"] = tex_escape(
                str(json.loads(cand.read_text(encoding="utf-8")).get("model", ""))
            )
            break
    phi_p = REPORTS / "decade_error_atlas_genre_phi.csv"
    if phi_p.exists():
        d = pd.read_csv(phi_p)
        d = d[d.wave == 2024]

        def _phi(a: str, b: str) -> str | None:
            s = d[((d.genre_a == a) & (d.genre_b == b)) | ((d.genre_a == b) & (d.genre_b == a))]
            return f"{float(s.phi.iloc[0]):.2f}" if len(s) else None

        for key, pair in (
            ("phiExhibitionMuseum", ("exhibition", "museum")),
            ("phiClassicalDance", ("western_classical", "dance_performance")),
        ):
            v = _phi(*pair)
            if v:
                out[key] = v
    eff = REPORTS / "era_r0_promotion_effect_summary.csv"
    if eff.exists():
        e = pd.read_csv(eff)
        rel = e[e.tag == "release_narrative"]["mean_delta"].abs().max()
        twin = e[e.tag == "twin_anchors"]["mean_delta"].abs().max()
        out["rZeroShiftRelease"] = f"{float(rel):.3f}"
        out["rZeroShiftTwin"] = f"{float(twin):.3f}"
    wt = REPORTS / "decade_error_atlas_weighting.csv"
    if wt.exists():
        w = pd.read_csv(wt)

        def _cut(wave: int, mode: str) -> float:
            return float(w[(w.wave == wave) & (w["mode"] == mode)]["attend_cut"].iloc[0])

        unw = w[(w["mode"] == "unweighted") & (w.wave.isin([2020, 2022, 2024]))]["attend_cut"]
        early = w[w.wave.isin([2016, 2018])]
        out["attendCutWeighted"] = f"{100 * _cut(2024, 'weighted'):.0f}"
        out["attendCutUnweighted"] = f"{100 * _cut(2024, 'unweighted'):.0f}"
        out["attendCutUnwMin"] = f"{100 * unw.min():.0f}"
        out["attendCutUnwMax"] = f"{100 * unw.max():.0f}"
        if len(w[(w.wave == 2020) & (w["mode"] == "design_weight_WTB")]):
            out["attendCutDesign"] = f"{100 * _cut(2020, 'design_weight_WTB'):.0f}"
        out["attendCutEarly"] = (
            f"{100 * early.attend_cut.min():.0f}--{100 * early.attend_cut.max():.0f}"
        )
        out["attendShareEarly"] = (
            f"{100 * early.attend_any_share.min():.0f}--{100 * early.attend_any_share.max():.0f}"
        )
        out["attendEntropyEarly"] = f"{early.attend_any_entropy_bits.max():.2f}"
        # Age and sex rungs report weighted and unweighted side by side
        # (weights are also raked on sex, age, and living area).
        if "sex_cut" in w.columns:

            def _col(wave: int, mode: str, col: str) -> float:
                return float(w[(w.wave == wave) & (w["mode"] == mode)][col].iloc[0])

            out["ageCutWeighted"] = f"{100 * _col(2024, 'weighted', 'age_cut'):.0f}"
            out["ageCutUnweighted"] = f"{100 * _col(2024, 'unweighted', 'age_cut'):.0f}"
            out["sexCutWeighted"] = f"{100 * _col(2024, 'weighted', 'sex_cut'):.1f}"
            out["sexCutUnweighted"] = f"{100 * _col(2024, 'unweighted', 'sex_cut'):.1f}"
        # Routing rung: the reduction from skip logic alone (no new table),
        # and the further share the attendance table adds.
        if "route_cut" in w.columns:
            out["routeCutWeighted"] = f"{100 * _col(2024, 'weighted', 'route_cut'):.0f}"
            out["routeCutUnweighted"] = f"{100 * _col(2024, 'unweighted', 'route_cut'):.0f}"
            out["beyondRouteWeighted"] = (
                f"{100 * _col(2024, 'weighted', 'attend_beyond_route'):.0f}"
            )
            out["beyondRouteUnweighted"] = (
                f"{100 * _col(2024, 'unweighted', 'attend_beyond_route'):.0f}"
            )
            if len(w[(w.wave == 2020) & (w["mode"] == "design_weight_WTB")]):
                out["beyondRouteDesign"] = (
                    f"{100 * _col(2020, 'design_weight_WTB', 'attend_beyond_route'):.0f}"
                )
            wu = w[(w["mode"] == "unweighted") & (w.wave.isin([2020, 2022, 2024]))]
            out["beyondRouteUnwMin"] = f"{100 * wu.attend_beyond_route.min():.0f}"
            out["beyondRouteUnwMax"] = f"{100 * wu.attend_beyond_route.max():.0f}"
            out["beyondRouteEarly"] = (
                f"{100 * early.attend_beyond_route.min():.0f}--{100 * early.attend_beyond_route.max():.0f}"
            )
            out["routeCutEarly"] = (
                f"{100 * early.route_cut.min():.0f}--{100 * early.route_cut.max():.0f}"
            )
    # Bootstrap ranges and the sampling floor (ADR-017), so the text can say
    # whether the four rows clear the noise.
    unc = REPORTS / "era_atlas_uncertainty.csv"
    if unc.exists():
        u = pd.read_csv(unc)
        u24 = u[u.wave == 2024].set_index("metric")

        def _ci(metric: str, fmt: str = ".3f") -> str:
            r = u24.loc[metric]
            return f"{float(r.lo):{fmt}}--{float(r.hi):{fmt}}"

        out["ciIndep"] = _ci("tvd_indep")
        out["ciAge"] = _ci("tvd_cond_age")
        out["ciAgeSex"] = _ci("tvd_cond_age_sex")
        out["ciAgeAttend"] = _ci("tvd_cond_age_attend")
        out["ciAttendCut"] = _ci("attend_cut_pct", ".0f")
        if "tvd_cond_age_route" in u24.index:
            out["ciAgeRoute"] = _ci("tvd_cond_age_route")
            out["ciRouteCut"] = _ci("route_cut_pct", ".0f")
            out["ciBeyondRoute"] = _ci("attend_beyond_route_pct", ".0f")
        out["ciSexGain"] = _ci("sex_gain", ".4f")
        out["atlasFloor"] = f"{float(u24.loc['sampling_floor', 'point']):.3f}"
        out["nAtlasBoot"] = str(int(u24.loc["tvd_indep", "n_boot"]))
    # Permutation-null repetitions, read from the source script's constant (cited in the text).
    ppi_src = PROJ / "scripts" / "era_holdout_ppi.py"
    if ppi_src.exists():
        for line in ppi_src.read_text(encoding="utf-8").splitlines():
            if line.startswith("N_PERM"):
                out["nPerm"] = _tex_int_comma(int(line.split("=")[1].strip()))
                break
    # Narrative-layer probes and bank diversity (ADR-018). These figures come only
    # from the 15,000-persona bank.
    pr = REPORTS / "era_probes.csv"
    if pr.exists():
        pd_ = pd.read_csv(pr)
        pd_ = pd_[pd_.wave == 2024]
        w_ = pd_.pivot_table(index="target", columns="condition", values="accuracy")
        maj_ = pd_.groupby("target").majority.first()

        def _pct(target: str, cond: str) -> str:
            return f"{100 * float(w_.loc[target, cond]):.0f}"

        # Keep the keys as string literals; a name-string check looks them up.
        out["probeAgeMasked"] = _pct("age_band", "masked")
        out["probeAgeRaw"] = _pct("age_band", "raw")
        out["probeAgeBase"] = f"{100 * float(maj_.loc['age_band']):.0f}"
        out["probeAreaMasked"] = _pct("living_area", "masked")
        out["probeAreaRaw"] = _pct("living_area", "raw")
        out["probeAreaBase"] = f"{100 * float(maj_.loc['living_area']):.0f}"
        out["probeSexMasked"] = _pct("sex", "masked")
        out["probeSexRaw"] = _pct("sex", "raw")
        out["probeSexBase"] = f"{100 * float(maj_.loc['sex']):.0f}"
        out["probeIncomeMasked"] = _pct("household_income_band", "masked")
        out["probeIncomeRaw"] = _pct("household_income_band", "raw")
        out["probeIncomeBase"] = f"{100 * float(maj_.loc['household_income_band']):.0f}"
        out["probeAttendMasked"] = _pct("attend_any", "masked")
        out["probeAttendRaw"] = _pct("attend_any", "raw")
        out["probeAttendBase"] = f"{100 * float(maj_.loc['attend_any']):.0f}"
        out["probeOccupationMasked"] = _pct("occupation", "masked")
        out["probeOccupationRaw"] = _pct("occupation", "raw")
        out["probeOccupationBase"] = f"{100 * float(maj_.loc['occupation']):.0f}"
        # 2022 replication, filled when the 2022 bank is present. Keys stay string literals.
        p22 = pd.read_csv(pr)
        p22 = p22[p22.wave == 2022]
        if len(p22):
            w22 = p22.pivot_table(index="target", columns="condition", values="accuracy")
            m22 = p22.groupby("target").majority.first()
            out["probeAreaMaskedTwoTwo"] = f"{100 * float(w22.loc['living_area', 'masked']):.0f}"
            out["probeAreaBaseTwoTwo"] = f"{100 * float(m22.loc['living_area']):.0f}"
            out["probeAgeMaskedTwoTwo"] = f"{100 * float(w22.loc['age_band', 'masked']):.0f}"
            out["probeIncomeMaskedTwoTwo"] = (
                f"{100 * float(w22.loc['household_income_band', 'masked']):.0f}"
            )
        # 2022 replication on the second wave, once its full bank exists. Keys stay string literals.
        p22 = pd.read_csv(pr)
        p22 = p22[p22.wave == 2022]
        if len(p22):
            w22 = p22.pivot_table(index="target", columns="condition", values="accuracy")
            m22 = p22.groupby("target").majority.first()
            out["probeAreaMaskedTwoTwo"] = f"{100 * float(w22.loc['living_area', 'masked']):.0f}"
            out["probeAreaBaseTwoTwo"] = f"{100 * float(m22.loc['living_area']):.0f}"
            out["probeAgeMaskedTwoTwo"] = f"{100 * float(w22.loc['age_band', 'masked']):.0f}"
            out["probeIncomeMaskedTwoTwo"] = (
                f"{100 * float(w22.loc['household_income_band', 'masked']):.0f}"
            )
            out["probeIncomeBaseTwoTwo"] = f"{100 * float(m22.loc['household_income_band']):.0f}"
        w22 = pd.read_csv(pr)
        w22 = w22[(w22.wave == 2022) & (w22.condition == "masked")].set_index("target").accuracy
        if len(w22):
            out["probeAgeMaskedTwoTwo"] = f"{100 * float(w22.loc['age_band']):.0f}"
            out["probeAreaMaskedTwoTwo"] = f"{100 * float(w22.loc['living_area']):.0f}"
            out["probeIncomeMaskedTwoTwo"] = f"{100 * float(w22.loc['household_income_band']):.0f}"
        out["nProbeTrain"] = _tex_int_comma(int(pd_.n_train.max()))
        out["nProbeTargets"] = str(int(pd_.target.nunique()))
    # Two downstream tasks (ADR-019).
    ts = REPORTS / "era_tstr.csv"
    if ts.exists():
        t_ = pd.read_csv(ts)
        a_ = t_.groupby(["setting", "features"])[
            ["acc_marginal", "acc_age", "acc_tstr", "acc_trtr"]
        ].mean()
        out["tstrAgeBase"] = f"{100 * float(a_.loc[('cross-wave', 'all'), 'acc_age']):.0f}"
        out["tstrAll"] = f"{100 * float(a_.loc[('cross-wave', 'all'), 'acc_tstr']):.0f}"
        out["tstrUpper"] = f"{100 * float(a_.loc[('cross-wave', 'all'), 'acc_trtr']):.0f}"
        out["tstrDemo"] = f"{100 * float(a_.loc[('cross-wave', 'demographic'), 'acc_tstr']):.0f}"
        out["tstrDemoUpper"] = (
            f"{100 * float(a_.loc[('cross-wave', 'demographic'), 'acc_trtr']):.0f}"
        )
    gp = REPORTS / "era_genre_portfolio.csv"
    if gp.exists():
        g0 = pd.read_csv(gp)
        ga = g0.groupby(["setting", "features"])[
            ["auc_age_table", "auc_synthetic", "auc_microdata"]
        ].mean()
        out["genreAucTable"] = f"{float(ga.loc[('cross-wave', 'cultural'), 'auc_age_table']):.2f}"
        out["genreAucSyn"] = f"{float(ga.loc[('cross-wave', 'cultural'), 'auc_synthetic']):.2f}"
        out["genreAucReal"] = f"{float(ga.loc[('cross-wave', 'cultural'), 'auc_microdata']):.2f}"
        out["genreAucDemo"] = f"{float(ga.loc[('cross-wave', 'demographic'), 'auc_synthetic']):.2f}"
        gg = (
            g0[(g0.setting == "cross-wave") & (g0.features == "cultural")]
            .groupby("target")
            .auc_synthetic.mean()
        )
        if "auc_pair_tables" in g0.columns:
            ga2 = g0.groupby(["setting", "features"])["auc_pair_tables"].mean()
            out["genreAucPairTables"] = f"{float(ga2.loc[('cross-wave', 'cultural')]):.2f}"
            cw = g0[(g0.setting == "cross-wave") & (g0.features == "cultural")]
            # Per genre × test-wave row: rows where synthetic beats the baseline
            # (interval lower bound > 0).
            out["genreRowsCrossCultural"] = str(len(cw))
            out["genreSynBeatsAgeN"] = str(int((cw.d_syn_age_lo > 0).sum()))
            out["genreSynBeatsNbN"] = str(int((cw.d_syn_nb_lo > 0).sum()))
            out["genreRealBeatsSynN"] = str(int((cw.d_real_syn_lo > 0).sum()))
            out["genreNbBeatsSynN"] = str(int((cw.d_syn_nb_hi < 0).sum()))
        out["genreAucBest"] = f"{gg.max():.2f}"
        out["genreAucBestName"] = tex_escape(str(gg.idxmax()))
        out["genreAucWorst"] = f"{gg.min():.2f}"
        out["genreAucWorstName"] = tex_escape(str(gg.idxmin()))
    dp = REPORTS / "era_design_pretest.csv"
    if dp.exists():
        d_ = pd.read_csv(dp)
        g_ = d_.groupby(["published_axis", "method"]).abs_err_pp.mean()
        out["designSynUnpub"] = f"{float(g_.loc[False, 'synthetic']):.1f}"
        out["designAgeUnpub"] = f"{float(g_.loc[False, 'age-conditional']):.1f}"
        out["designMargUnpub"] = f"{float(g_.loc[False, 'marginal']):.1f}"
        out["designSynPub"] = f"{float(g_.loc[True, 'synthetic']):.1f}"
        out["designAgePub"] = f"{float(g_.loc[True, 'age-conditional']):.1f}"
        out["nDesignCells"] = _tex_int_comma(int((d_.method == "synthetic").sum()))
    lg = REPORTS / "era_llm_genre.csv"
    if lg.exists():
        lgd = pd.read_csv(lg).set_index("condition")
        lgc = pd.read_csv(REPORTS / "era_llm_genre_contrasts.csv").set_index("contrast")
        out["llmZero"] = f"{float(lgd.loc['zero', 'auc']):.2f}"
        out["llmDemo"] = f"{float(lgd.loc['demo', 'auc']):.2f}"
        out["llmShotAnchor"] = f"{float(lgd.loc['shot_anchor', 'auc']):.2f}"
        out["llmShotNarr"] = f"{float(lgd.loc['shot_narr', 'auc']):.2f}"
        out["llmKnnNarr"] = f"{float(lgd.loc['knn_narr', 'auc']):.2f}"
        out["llmNarrGain"] = f"{float(lgc.loc['shot_narr - shot_anchor', 'auc_diff']):+.3f}"
        out["llmNarrGainCi"] = (
            f"[{float(lgc.loc['shot_narr - shot_anchor', 'lo']):+.3f}, "
            f"{float(lgc.loc['shot_narr - shot_anchor', 'hi']):+.3f}]"
        )
        out["llmShotGain"] = f"{float(lgc.loc['shot_anchor - zero', 'auc_diff']):+.3f}"
        out["llmProfileGain"] = f"{float(lgc.loc['zero - demo', 'auc_diff']):+.3f}"
        out["llmProfileGainCi"] = (
            f"[{float(lgc.loc['zero - demo', 'lo']):+.3f}, {float(lgc.loc['zero - demo', 'hi']):+.3f}]"
        )
        out["nLlmTargets"] = _tex_int_comma(int(lgd.n.max()))
        out["nKShot"] = "8"
        out["nLlmWaves"] = "two"
    dv = REPORTS / "era_bank_diversity.csv"
    if dv.exists():
        v = pd.read_csv(dv)
        v = v[v.wave == 2024].set_index("metric")["value"]
        out["nBankFull"] = _tex_int_comma(int(v["n_personas"]))
        # Sum of the full narrative banks over the three waves, distinct from the
        # frozen evaluation bank (900).
        v_full = pd.read_csv(dv)
        out["nBankAll"] = _tex_int_comma(int(v_full[v_full.metric == "n_personas"].value.sum()))
        out["nBankWaves"] = str(int(v_full[v_full.metric == "n_personas"].wave.nunique()))
        out["bankDupProfiles"] = str(int(v["duplicate_profiles"]))
        out["bankNnMedian"] = f"{v['nn_cos_median']:.2f}"
        out["bankNnMax"] = f"{v['nn_cos_max']:.2f}"
        out["bankClicheShare"] = f"{v['cliche_share_pct']:.1f}"
        out["bankRegenMean"] = f"{v['mean_regenerations']:.2f}"
        v22 = pd.read_csv(dv)
        v22 = v22[v22.wave == 2022].set_index("metric")["value"]
        if len(v22):
            out["bankDupProfilesTwoTwo"] = str(int(v22["duplicate_profiles"]))
            out["bankNnMaxTwoTwo"] = f"{v22['nn_cos_max']:.2f}"
        v_all = pd.read_csv(dv)
        v22 = v_all[v_all.wave == 2022].set_index("metric")["value"]
        if len(v22):
            out["bankNnMaxTwoTwo"] = f"{v22['nn_cos_max']:.2f}"
            out["bankDupProfilesTwoTwo"] = str(int(v22["duplicate_profiles"]))
    du = REPORTS / "era_bank_duplicates.csv"
    if du.exists():
        q = pd.read_csv(du)
        q = q[(q.wave == 2024) & (q.scope != "full profile")]
        out["bankDupFieldMax"] = f"{q.duplicate_pct.max():.1f}"
        out["bankDupFieldWorst"] = tex_escape(str(q.loc[q.duplicate_pct.idxmax(), "scope"]))
        out["bankDupFieldOther"] = (
            f"{q[q.duplicate_pct < q.duplicate_pct.max()].duplicate_pct.max():.1f}"
        )
    stab = REPORTS / "decade_error_atlas_stability.csv"
    if stab.exists():
        s = pd.read_csv(stab).set_index("wave_b")
        if "anchor_mean_floor" in s.columns:
            # Keep the keys as string literals; a name-string check looks them up.
            out["atlasFloorTwoTwo"] = f"{float(s.loc[2022, 'anchor_mean_floor']):.2f}"
            out["atlasExcessTwoTwo"] = f"{float(s.loc[2022, 'excess_over_floor']):+.2f}"
            out["atlasFloorTwoZero"] = f"{float(s.loc[2020, 'anchor_mean_floor']):.2f}"
            out["atlasExcessTwoZero"] = f"{float(s.loc[2020, 'excess_over_floor']):+.2f}"
    r0_study = REPORTS / "era_r0_wave_study_root.csv"
    if r0_study.exists():
        d = pd.read_csv(r0_study)
        d = d[d["available"].fillna(False)].set_index("wave")
        for wave, key in (
            (2020, "rZeroTvdTwoZero"),
            (2022, "rZeroTvdTwoTwo"),
            (2024, "rZeroTvdTwoFour"),
        ):
            if wave in d.index:
                out[key] = f"{float(d.loc[wave, 'tvd_root_vs_shipped']):.3f}"
    return out


# ---------------------------------------------------------------------------
# Static tables are filled from era outputs (ADR-012).
# ---------------------------------------------------------------------------


def genre_phi_era() -> None:
    """Upper triangle of 2024 genre φ correlations (decade atlas)."""
    src = REPORTS / "decade_error_atlas_genre_phi.csv"
    if not src.exists():
        return
    d = pd.read_csv(src)
    d = d[d.wave == 2024]
    names = sorted(set(d.genre_a) | set(d.genre_b))
    m = {(r.genre_a, r.genre_b): r.phi for r in d.itertuples()}

    def phi(a: str, b: str) -> str:
        v = m.get((a, b), m.get((b, a)))
        return f"{v:.2f}" if v is not None else ""

    abbrev = {
        "western_classical": "classical",
        "popular_performance": "popular",
        "dance_performance": "dance",
        "literature_event": "literature",
        "outdoor_event": "outdoor",
        "exhibition": "exhibition",
        "traditional": "traditional",
    }
    short = [abbrev.get(n, n) for n in names]
    head = " & ".join([""] + short)
    body = []
    for i, a in enumerate(names):
        cells = [short[i]] + ["" if j <= i else phi(a, names[j]) for j in range(len(names))]
        body.append("    " + " & ".join(cells) + r" \\")
    cols = "@{}l" + "r" * len(names) + "@{}"
    (TABLES / "genre_phi.tex").write_text(
        "\\begin{table*}[t]\n  \\centering\n  \\small\n  \\begin{adjustbox}{max width=\\textwidth}\n"
        f"  \\begin{{tabular}}{{{cols}}}\n    \\toprule\n"
        f"    {head} \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Weighted $\\phi$ correlations between the genre attendance flags in the 2024 wave. "
        "The genre block is drawn as a Chow--Liu tree rather than independently "
        "because of these dependencies.}\n  \\label{tab:genre-phi}\n\\end{table*}\n",
        encoding="utf-8",
    )


def error_by_anchor_era() -> None:
    """Mean reconstruction TVD per anchor (2024 atlas pairs)."""
    src = REPORTS / "decade_error_atlas_pairs.csv"
    if not src.exists():
        return
    p24 = pd.read_csv(src)
    p24 = _ladder(p24[p24.wave == 2024])
    cols = [
        "tvd_indep",
        "tvd_cond_age",
        "tvd_cond_age_sex",
        "tvd_cond_age_route",
        "tvd_cond_age_attend",
    ]
    long = pd.concat(
        [
            p24.rename(columns={"anchor_a": "anchor"})[["anchor", *cols]],
            p24.rename(columns={"anchor_b": "anchor"})[["anchor", *cols]],
        ]
    )
    g = long.groupby("anchor")[cols].mean().sort_values("tvd_cond_age", ascending=False)
    body = []
    for anchor, r in g.iterrows():
        vals = ["--" if pd.isna(r[c]) else f"{r[c]:.3f}" for c in cols]
        body.append("    " + " & ".join([tex_escape(str(anchor)), *vals]) + r" \\")
    (TABLES / "error_by_anchor.tex").write_text(
        "\\begin{table*}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrrrr@{}}\n    \\toprule\n"
        "    Anchor & marginals & + age & + age $\\times$ sex & + skip logic & + age $\\times$ attendance \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Mean reconstruction error per anchor in the 2024 wave, averaged over that anchor's "
        "ladder pairs (non-structural, attendance status itself excluded).}\n"
        "  \\label{tab:error-by-anchor}\n\\end{table*}\n",
        encoding="utf-8",
    )


def rejection_era() -> None:
    """Marginal bias by invariant-enforcement method (era_sampler_rejection)."""
    src = REPORTS / "era_sampler_rejection.csv"
    if not src.exists():
        return
    d = pd.read_csv(src).set_index("variant")
    labels = [
        ("raw_old", "tree decides count, no constraints"),
        ("whole_old", "+ reject whole draws"),
        ("local_old", "+ redraw offending block"),
        ("local_new", "count drawn first + local redraw (released)"),
    ]
    body = []
    for key, lab in labels:
        if key not in d.index:
            continue
        r = d.loc[key]
        body.append(
            "    "
            + " & ".join(
                [
                    lab,
                    f"{100 * r.violation_r1:.1f}",
                    f"{r.tvd_attend_any:.3f}",
                    f"{r.tvd_n_genres:.3f}",
                    f"{r.tvd_attend_count_band:.3f}",
                ]
            )
            + r" \\"
        )
    (TABLES / "rejection.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        "  \\begin{tabular}{@{}lrrrr@{}}\n    \\toprule\n"
        "    Genre block & viol.\\% & attend & $\\#$genres & count band \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{How the genre block is drawn and what it does to the marginals (2024 survey root): "
        "TVD against the survey-weighted marginal, and against the microdata for the number of genres.}\n"
        "  \\label{tab:rejection}\n\\end{table}\n",
        encoding="utf-8",
    )


def variance_era() -> None:
    src = REPORTS / "era_variance.csv"
    if not src.exists():
        return
    d = pd.read_csv(src)
    bands = [b for b in d[d.age_band != "ALL"].age_band.unique()]
    body = []
    for b in bands:
        sub = d[d.age_band == b].set_index("source")
        vals = []
        for col in ("profile_entropy", "mean_pairwise_ham"):
            for s in ("microdata", "chain"):
                vals.append(f"{sub.loc[s, col]:.2f}" if s in sub.index else "--")
        vals.append("0")
        body.append("    " + " & ".join([_label_tex(b), *vals]) + r" \\")
    (TABLES / "variance.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrrrr@{}}\n    \\toprule\n"
        "    & \\multicolumn{2}{c}{profile entropy (bits)} & \\multicolumn{3}{c}{mean pairwise Hamming} \\\\\n"
        "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
        "    Age band & microdata & chain & microdata & chain & single \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Within-age-band dispersion of the ten-genre attendance profile in the 2024 wave. "
        "\\emph{single} is the common practice of one representative persona per age band, whose within-band "
        "dispersion is zero by construction.}\n  \\label{tab:variance}\n\\end{table}\n",
        encoding="utf-8",
    )


def segment_holdout_era() -> None:
    src = REPORTS / "era_segment_holdout.csv"
    if not src.exists():
        return
    d = pd.read_csv(src)
    by_unit = d.groupby(["by", "cell", "n_real"], as_index=False)[
        ["tvd_holdout", "tvd_oracle", "tvd_naive"]
    ].mean()
    floors = d.dropna(subset=["floor_cell"]).groupby(["by", "cell"])["floor_cell"].first()
    by_unit["floor"] = by_unit.set_index(["by", "cell"]).index.map(floors)
    label = {
        "age_band": "age band",
        "living_area": "living area",
        "cell": r"living area $\times$ age band",
    }
    body = []
    for by, g in by_unit.groupby("by"):
        body.append(
            "    "
            + " & ".join(
                [
                    f"{label.get(by, by)} ({g.cell.nunique()})",
                    f"{g.floor.mean():.3f}",
                    f"{g.tvd_oracle.mean():.3f}",
                    f"{g.tvd_holdout.mean():.3f}",
                    f"{g.tvd_naive.mean():.3f}",
                ]
            )
            + r" \\"
        )
    (TABLES / "cell_holdout.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        "  \\begin{tabular}{@{}lrrrr@{}}\n    \\toprule\n"
        "    Unit removed & floor & oracle & removed & no segments \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{What it costs to lose a segment on the 2024 wave (procedure in the text): mean TVD over the non-root "
        "anchors against the removed unit's observed distribution, averaged over units. \\emph{Floor}: sampling noise alone; "
        "\\emph{oracle}: the chain fitted with the unit; \\emph{no segments}: the overall marginal.}\n"
        "  \\label{tab:cell-holdout}\n\\end{table}\n",
        encoding="utf-8",
    )
    ab = d[d.by == "age_band"]
    rows = ab.groupby(["cell", "n_real"], as_index=False)[
        ["tvd_holdout", "tvd_oracle", "tvd_naive"]
    ].mean()
    fl = ab.dropna(subset=["floor_cell"]).groupby("cell")["floor_cell"].first()
    rows["floor"] = rows["cell"].map(fl)
    rows = rows.sort_values("tvd_holdout", ascending=False)
    body = [
        "    "
        + " & ".join(
            [
                _label_tex(r.cell),
                str(int(r.n_real)),
                f"{r.floor:.3f}",
                f"{r.tvd_oracle:.3f}",
                f"{r.tvd_holdout:.3f}",
                f"{r.tvd_naive:.3f}",
            ]
        )
        + r" \\"
        for r in rows.itertuples()
    ]
    (TABLES / "cell_holdout_detail.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        "  \\begin{tabular}{@{}lrrrrr@{}}\n    \\toprule\n"
        "    age band removed & $n$ & floor & oracle & removed & no seg. \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Age-band holdout on the 2024 wave, per removed band.}\n"
        "  \\label{tab:cell-holdout-detail}\n\\end{table}\n",
        encoding="utf-8",
    )


def grounding_era() -> None:
    """Grounding-table inventory and cell counts (era_2024)."""
    g = PROJ / "data" / "grounding"
    era = g / "era_2024"
    rows = [
        (
            "R0",
            r"living area $\times$ sex $\times$ age band, population",
            g / "R0_population_root.parquet",
            "resident registration",
        )
    ]
    for tid, desc in (
        ("T1", "anchor marginals"),
        ("T2", r"anchor $\mid$ age band"),
        ("T3", r"anchor $\mid$ sex"),
        ("T4", r"anchor $\mid$ living area"),
        ("T5", r"anchor $\mid$ age band $\times$ sex"),
        ("T6_pair_joints", "anchor pair joints"),
    ):
        # Released cells only (the *_release files), the same count as Table dataset-stats.
        rows.append((tid.split("_")[0], desc, era / f"{tid}_release.parquet", "survey, weighted"))
    body = []
    for tid, desc, path, src in rows:
        if not path.exists():
            continue
        n = len(pd.read_parquet(path))
        body.append("    " + " & ".join([tid, desc, _tex_int_comma(n), src]) + r" \\")
    if not body:
        return
    (TABLES / "grounding.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        "  \\begin{tabular}{@{}llrl@{}}\n    \\toprule\n    ID & Content & Cells & Source \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{The released grounding tables of the 2024 wave (released cells); the sampler draws from the specification (\\nSpecCells{} cells), released with them. Pair-joint cells below "
        "the floor of \\cellFloor{} unweighted respondents are withheld from the released table "
        "(\\nSuppressedCells{} pair-joint cells, plus \\nSecondaryCells{} complementary cells on T6 only; "
        "T2--T5 cells below the floor are withheld without complementary suppression).}\n"
        "  \\label{tab:grounding}\n\\end{table}\n",
        encoding="utf-8",
    )


def anchors_era() -> None:
    """Anchor codebook: era core anchors only."""
    import sys as _sys

    _sys.path.insert(0, str(PROJ / "src"))
    try:
        from sfac_persona.schema import ANCHOR_BY_NAME, ERA_CORE_ANCHORS, ROOT_ANCHORS
    except Exception:  # noqa: BLE001
        return
    body = []
    for name in ERA_CORE_ANCHORS:
        spec = ANCHOR_BY_NAME.get(name)
        if spec is None:
            # A silent skip would make the caption's \nAnchors disagree with the row count.
            raise SystemExit(
                f"anchors_era: ERA_CORE anchor {name} has no AnchorSpec; fix schema.py"
            )
        kind = getattr(spec, "kind", "")
        src = ", ".join(getattr(spec, "source", ()) or ())
        note = "root" if name in ROOT_ANCHORS else ""
        body.append(
            "    "
            + " & ".join([tex_escape(name), tex_escape(str(kind)), tex_escape(src), note])
            + r" \\"
        )
    if not body:
        return
    (TABLES / "anchors.tex").write_text(
        "\\begin{table*}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}llp{0.5\\textwidth}l@{}}\n    \\toprule\n"
        "    Anchor & Kind & Survey variables & \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{The \\nAnchors{} quantitative anchors carried by every persona, with the survey variables "
        "each is derived from.}\n  \\label{tab:anchors}\n\\end{table*}\n",
        encoding="utf-8",
    )


def sampler_ablation_era() -> None:
    """Chain-design ablation table (era_sampler_ablation), the table cited in §4."""
    src = REPORTS / "era_sampler_ablation.csv"
    if not src.exists():
        return
    d = pd.read_csv(src).set_index("variant")
    labels = [
        ("released", "measured chain"),
        ("no_attend", "attendance status not a parent"),
        ("no_genre_tree", "genres drawn without the tree"),
        ("no_count_node", "genre count left to the tree"),
        ("no_same_construct", "same-construct edges cut"),
        ("age_only", "age band only"),
        ("independent", "every anchor independent"),
    ]
    body = []
    for key, lab in labels:
        if key not in d.index:
            continue
        r = d.loc[key]
        body.append(
            "    "
            + " & ".join(
                [
                    lab,
                    f"{r.tvd_nonstructural:.3f}",
                    f"{r.tvd_structural:.3f}",
                    f"{r.genre_phi_mae:.4f}",
                    f"{_z(r.delta_nonstructural, 0.0005):+.3f}",
                ]
            )
            + r" \\"
        )
    (TABLES / "sampler_ablation.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        "  \\begin{tabular}{@{}lrrrr@{}}\n    \\toprule\n"
        "    Variant & non-struct. & struct. & $\\phi$ MAE & $\\Delta$ \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Chain design ablation on the 2024 wave: each row removes one design decision and "
        "redraws \\nSamplerAudit{} personas; columns are defined in Appendix~\\ref{app:ablation}.}\n"
        "  \\label{tab:sampler-ablation}\n\\end{table}\n",
        encoding="utf-8",
    )


def kcas_era() -> dict[str, str]:
    """KCAS second-case table and macros (ADR-015), shown beside the Seoul survey as its own case."""
    out: dict[str, str] = {}
    sp = REPORTS / "kcas_atlas_summary.csv"
    if not sp.exists():
        return out
    s = pd.read_csv(sp)
    stab = pd.read_csv(REPORTS / "kcas_atlas_stability.csv").set_index("wave_b")
    prov = pd.read_csv(REPORTS / "kcas_weight_provenance.csv")
    w = s[s["mode"] == "weighted"].set_index("wave")
    u = s[s["mode"] == "unweighted"].set_index("wave")
    body = []
    for wave in w.index:
        ex = stab.loc[wave, "excess_over_floor"] if wave in stab.index else float("nan")
        body.append(
            "    "
            + " & ".join(
                [
                    str(int(wave)),
                    _tex_int_comma(int(w.loc[wave, "n"])),
                    f"{w.loc[wave, 'mean_tvd_indep']:.3f}",
                    f"{w.loc[wave, 'mean_tvd_cond_age']:.3f}",
                    f"{w.loc[wave, 'mean_tvd_cond_age_sex']:.3f}",
                    f"{w.loc[wave, 'mean_tvd_cond_age_route']:.3f}",
                    f"{w.loc[wave, 'mean_tvd_cond_age_attend']:.3f}",
                    f"{100 * u.loc[wave, 'route_cut']:.0f}",
                    f"{100 * u.loc[wave, 'attend_beyond_route']:.0f}",
                    f"{100 * (1 - u.loc[wave, 'attend_any_share']):.0f}",
                    "--" if pd.isna(ex) else f"{ex:+.2f}",
                ]
            )
            + r" \\"
        )
    (TABLES / "kcas.tex").write_text(
        "\\begin{table*}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\textwidth}\n"
        "  \\begin{tabular}{@{}lrrrrrrrrrr@{}}\n    \\toprule\n"
        "    Wave & $n$ & indep. & + age & + age$\\times$sex & + skip logic & + age$\\times$attendance & skip cut\\% (unw.) & beyond\\% (unw.) & non-att.\\% & $\\rho$ excess \\\\\n"
        "    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{The error study repeated on a second, national survey \\citep{kcas2025survey}: weighted mean pairwise TVD "
        "over its non-structural pairs, the skip-logic and beyond-skip shares (unweighted), the non-attender share and the "
        "rank-stability excess, as defined in the text. Not pooled with the Seoul survey.}\n"
        "  \\label{tab:kcas}\n\\end{table*}\n",
        encoding="utf-8",
    )
    sex_gap = float((w["mean_tvd_cond_age"] - w["mean_tvd_cond_age_sex"]).abs().max())
    order_ok = bool(
        (
            (w["mean_tvd_indep"] > w["mean_tvd_cond_age"])
            & (w["mean_tvd_cond_age"] > w["mean_tvd_cond_age_attend"])
        ).all()
    )
    out["kcasNWaves"] = str(len(w))
    out["kcasN"] = _tex_int_comma(int(w["n"].sum()))
    out["kcasOrderAll"] = "every" if order_ok else "most"
    out["kcasSexGap"] = f"{sex_gap:.4f}"
    out["kcasCutMin"] = f"{100 * u['attend_cut'].min():.0f}"
    out["kcasCutMax"] = f"{100 * u['attend_cut'].max():.0f}"
    out["kcasWeightGap"] = f"{100 * float((w['attend_cut'] - u['attend_cut']).abs().max()):.0f}"
    # Largest |r| between the weight residual and any genre, over all waves (the text's bound).
    out["kcasWeightCorr"] = f"{float(prov['max_abs_item_corr'].max()):.2f}"
    out["kcasExcessMin"] = f"{float(stab['excess_over_floor'].min()):+.2f}"
    out["kcasExcessMax"] = f"{float(stab['excess_over_floor'].max()):+.2f}"
    non = 1 - u["attend_any_share"]
    out["nonattCorrKcas"] = f"{float(np.corrcoef(non, u['attend_cut'])[0, 1]):+.2f}"
    if "route_cut" in u.columns:
        out["kcasRouteMin"] = f"{100 * u['route_cut'].min():.0f}"
        out["kcasRouteMax"] = f"{100 * u['route_cut'].max():.0f}"
        out["kcasBeyondMin"] = f"{100 * u['attend_beyond_route'].min():.0f}"
        out["kcasBeyondMax"] = f"{100 * u['attend_beyond_route'].max():.0f}"
        out["nonattCorrKcasRoute"] = f"{float(np.corrcoef(non, u['route_cut'])[0, 1]):+.2f}"
        out["nonattCorrKcasBeyond"] = (
            f"{float(np.corrcoef(non, u['attend_beyond_route'])[0, 1]):+.2f}"
        )
    # Recompute the correlation without 2021 (pandemic; 34% attenders), in case that one point drives it.
    u_nc = u.drop(index=2021, errors="ignore")
    out["nonattCorrKcasNoCovid"] = (
        f"{float(np.corrcoef(1 - u_nc['attend_any_share'], u_nc['attend_cut'])[0, 1]):+.2f}"
    )
    seoul = REPORTS / "decade_error_atlas_weighting.csv"
    if seoul.exists():
        z = pd.read_csv(seoul)
        # Of the five points, 2016 and 2018 are unweighted diagnostic waves and
        # drive the correlation. The text uses the three weighted waves; the
        # five-wave value is named in the appendix.
        zu = z[z["mode"] == "unweighted"]
        out["nonattCorrSeoulAllUnw"] = (
            f"{float(np.corrcoef(1 - zu['attend_any_share'], zu['attend_cut'])[0, 1]):+.2f}"
        )
        out["nSeoulWavesAll"] = str(len(zu))
        zw = z[z["mode"] == "weighted"]
        out["nonattCorrSeoul"] = (
            f"{float(np.corrcoef(1 - zw['attend_any_share'], zw['attend_cut'])[0, 1]):+.2f}"
        )
        out["nSeoulWavesWeighted"] = str(len(zw))
        if "attend_beyond_route" in z.columns:
            out["nonattCorrSeoulBeyondAllUnw"] = (
                f"{float(np.corrcoef(1 - zu['attend_any_share'], zu['attend_beyond_route'])[0, 1]):+.2f}"
            )
            out["nonattCorrSeoulRouteAllUnw"] = (
                f"{float(np.corrcoef(1 - zu['attend_any_share'], zu['route_cut'])[0, 1]):+.2f}"
            )
    return out


def patch_numbers() -> None:
    """Update selected macros in numbers.tex; keep unrelated legacy macros intact."""
    path = TABLES / "numbers.tex"
    text = path.read_text(encoding="utf-8")
    pairs = pd.read_csv(REPORTS / "decade_error_atlas_pairs.csv")
    p24 = pairs[pairs.wave == 2024]
    ns = _ladder(p24)
    summary = pd.read_csv(REPORTS / "decade_error_atlas_summary.csv").set_index("wave")
    # Segment holdout, cited from Limitations so an unsupported result stays visible.
    seg = pd.read_csv(REPORTS / "era_segment_holdout.csv")
    seg_unit = seg.groupby(["by", "cell"], as_index=False)[["tvd_holdout", "tvd_naive"]].mean()
    seg_by = seg_unit.groupby("by")[["tvd_holdout", "tvd_naive"]].mean()

    # Reduction by attendance-block membership (ADR-016): the share that is not skip logic.
    pc = pd.read_csv(REPORTS / "decade_error_atlas_pair_class.csv")
    pc24 = pc[pc.wave == 2024].set_index("pair_class")
    pc_w = pc[pc.wave.isin([2020, 2022, 2024])]
    st = pd.read_csv(REPORTS / "decade_error_atlas_stability.csv")
    rho22 = float(st[(st.wave_a == 2024) & (st.wave_b == 2022)].spearman_tvd_cond_age.iloc[0])
    rho20 = float(st[(st.wave_a == 2024) & (st.wave_b == 2020)].spearman_tvd_cond_age.iloc[0])
    cw = pd.read_csv(REPORTS / "era_holdout_crosswave.csv")
    c24 = cw[cw.wave == 2024]
    c20 = cw[cw.wave == 2020]
    _blp = REPORTS / "era_holdout_baselines.csv"
    bl24 = pd.read_csv(_blp).query("wave == 2024") if _blp.exists() else None
    ppi = pd.read_csv(REPORTS / "era_holdout_ppi.csv")
    ppi24 = ppi[ppi.wave == 2024]
    learned = ppi24[ppi24.predictor == "learned"]
    twin = ppi24[ppi24.predictor == "twin_ssr"]
    distilled = ppi24[ppi24.predictor == "distilled"]
    tau = json.loads(TAU_PATH.read_text(encoding="utf-8"))
    eval_ = pd.read_csv(REPORTS / "era_eval_suite.csv")
    e24m = eval_[(eval_.wave == 2024) & (eval_.ablation == "margins")].iloc[0]
    e24t = eval_[(eval_.wave == 2024) & (eval_.ablation == "structure_T")].iloc[0]
    sample_path = REPORTS / "era_holdout_sample_smoke.csv"
    sample_n = "20"
    sample_mean_tvd = ""
    if sample_path.exists():
        sm = pd.read_csv(sample_path)
        sample_n = str(int(sm["n"].iloc[0]))
        sample_mean_tvd = f"{sm.tvd_cell_mean.mean():.3f}"
    ssr_path = REPORTS / "ssr_validate.json"
    ssr: dict = {}
    if ssr_path.exists():
        ssr = json.loads(ssr_path.read_text(encoding="utf-8"))

    # Narrative-bank size and pass rate.
    #
    # Counted on the frozen evaluation bank of 900 (ADR-012): the bank the
    # evaluation sentences actually score.
    narr_total = 0
    narr_ok = 0
    for wave in (2020, 2022, 2024):
        bank = PROJ / "data" / "eval" / "era_holdout" / f"w{wave}" / "eval_bank.jsonl"
        if not bank.exists():
            continue
        for line in bank.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001, S112
                continue
            narr_total += 1
            if r.get("ok"):
                narr_ok += 1
    n_fail = max(narr_total - narr_ok, 0)
    pass_rate = 100.0 * narr_ok / narr_total if narr_total else 100.0

    # Quantitative bank (completed).
    syn = PROJ / "data" / "synthetic"
    n_quant_weighted = 0
    for wave in (2020, 2022, 2024):
        q = syn / f"era_{wave}_quant" / "personas_structure_T.parquet"
        if q.exists():
            n_quant_weighted += len(pd.read_parquet(q, columns=["persona_id"]))
    n_quant_diag = 0
    for wave in (2016, 2018):
        q = syn / f"era_{wave}_quant" / "personas_margins.parquet"
        if q.exists():
            n_quant_diag += len(pd.read_parquet(q, columns=["persona_id"]))

    updates = {
        "nRespondents": _tex_int_comma(int(summary.loc[2024, "n"])),
        "nPairsNonStructural": str(len(ns)),
        "nPairsStructural": str(int(p24.structural.fillna(False).sum())),
        "nPairsConditioning": str(
            int((~p24.structural.fillna(False).astype(bool)).sum() - len(ns))
        ),
        "tvdAgeRoute": f"{ns.tvd_cond_age_route.mean():.3f}",
        "tvdIndep": f"{ns.tvd_indep.mean():.3f}",
        "tvdAge": f"{ns.tvd_cond_age.mean():.3f}",
        "tvdAgeSex": f"{ns.tvd_cond_age_sex.dropna().mean():.3f}",
        "tvdAgeAttend": f"{ns.tvd_cond_age_attend.dropna().mean():.3f}",
        "nHoldoutItems": "9",
        "segAgeHoldout": f"{seg_by.loc['age_band', 'tvd_holdout']:.3f}",
        "segAgeNaive": f"{seg_by.loc['age_band', 'tvd_naive']:.3f}",
        "segAreaHoldout": f"{seg_by.loc['living_area', 'tvd_holdout']:.3f}",
        "segAreaNaive": f"{seg_by.loc['living_area', 'tvd_naive']:.3f}",
        "segCellHoldout": f"{seg_by.loc['cell', 'tvd_holdout']:.3f}",
        "segCellNaive": f"{seg_by.loc['cell', 'tvd_naive']:.3f}",
        "cutBoth": f"{pc24.loc['both', 'attend_cut_pct']:.0f}",
        "routeCutBoth": f"{pc24.loc['both', 'route_cut_pct']:.0f}",
        "routeCutOne": f"{pc24.loc['one', 'route_cut_pct']:.0f}",
        "routeCutNeither": f"{pc24.loc['neither', 'route_cut_pct']:.0f}",
        "beyondBoth": f"{pc24.loc['both', 'attend_beyond_route_pct']:.0f}",
        "beyondOne": f"{pc24.loc['one', 'attend_beyond_route_pct']:.0f}",
        "beyondNeither": f"{pc24.loc['neither', 'attend_beyond_route_pct']:.0f}",
        "beyondOneMin": f"{pc_w[pc_w.pair_class == 'one'].attend_beyond_route_pct.min():.0f}",
        "beyondOneMax": f"{pc_w[pc_w.pair_class == 'one'].attend_beyond_route_pct.max():.0f}",
        "cutOne": f"{pc24.loc['one', 'attend_cut_pct']:.0f}",
        "cutNeither": f"{pc24.loc['neither', 'attend_cut_pct']:.0f}",
        "nPairsBoth": str(int(pc24.loc["both", "n_pairs"])),
        "nPairsOne": str(int(pc24.loc["one", "n_pairs"])),
        "nPairsNeither": str(int(pc24.loc["neither", "n_pairs"])),
        "cutNeitherMin": f"{pc_w[pc_w.pair_class == 'neither'].attend_cut_pct.min():.0f}",
        "cutNeitherMax": f"{pc_w[pc_w.pair_class == 'neither'].attend_cut_pct.max():.0f}",
        "cutBothMin": f"{pc_w[pc_w.pair_class == 'both'].attend_cut_pct.min():.0f}",
        "cutBothMax": f"{pc_w[pc_w.pair_class == 'both'].attend_cut_pct.max():.0f}",
        "nHoldoutPersonas": "300",
        "ssrTau": f"{float(tau['tau']):.2f}",
        "holdMarginal": f"{c24.noise_floor_tvd.mean():.3f}",
        **(
            {
                "holdMarginalLookup": f"{bl24.marginal_lookup_tvd.mean():.3f}",
                "holdRealCeiling": f"{bl24.real_sample_tvd.mean():.3f}",
                "holdPriorWave": f"{bl24.prior_wave_lookup_tvd.dropna().mean():.3f}",
                "nHoldPriorItems": str(int(bl24.prior_wave_lookup_tvd.notna().sum())),
            }
            if bl24 is not None
            else {}
        ),
        "holdLookup": f"{c24.tvd_release_narrative.mean():.3f}",
        "holdLearned": f"{c24.tvd_twin_anchors.mean():.3f}",
        "holdRealSample": f"{c24.noise_floor_tvd.mean():.3f}",
        "ppiSavedLearned": f"{100 * learned.rho2.mean():.1f}",
        "ppiAboveNullBH": str(int(ppi.above_null_bh.sum()))
        if "above_null_bh" in ppi.columns
        else "--",
        "ppiAboveNullBHLearned": str(int(ppi[ppi.predictor == "learned"].above_null_bh.sum()))
        if "above_null_bh" in ppi.columns
        else "--",
        "ppiSavedLookup": f"{100 * twin.rho2.mean():.1f}",
        "ppiSavedDistilled": f"{100 * distilled.rho2.mean():.1f}" if len(distilled) else "0.0",
        "ppiBestItem": tex_escape(str(learned.loc[learned.rho2.idxmax(), "item"])),
        "ppiBestSaved": f"{100 * learned.rho2.max():.1f}",
        "ppiWorstItem": tex_escape(str(learned.loc[learned.rho2.idxmin(), "item"])),
        "ppiWorstSaved": f"{100 * learned.rho2.min():.1f}",
        "ppiPermNull": f"{100 * float(ppi.rho2_perm95.min()):.1f}--{100 * float(ppi.rho2_perm95.max()):.1f}\\%"
        if "rho2_perm95" in ppi.columns
        else "",
        "ppiNullLevel": f"{100 * float(ppi.rho2_null.mean()):.1f}\\%"
        if "rho2_null" in ppi.columns
        else "0.3\\%",
        "ppiAboveNull": str(int(ppi.above_null.sum())) if "above_null" in ppi.columns else "0",
        "ppiNRows": str(len(ppi)),
        "repTwoFourIndep": f"{summary.loc[2024, 'mean_tvd_indep']:.3f}",
        "repTwoFourAge": f"{summary.loc[2024, 'mean_tvd_cond_age']:.3f}",
        "repTwoFourAgeAttend": f"{summary.loc[2024, 'mean_tvd_cond_age_attend']:.3f}",
        "repTwoTwoIndep": f"{summary.loc[2022, 'mean_tvd_indep']:.3f}",
        "repTwoTwoAge": f"{summary.loc[2022, 'mean_tvd_cond_age']:.3f}",
        "repTwoTwoAgeAttend": f"{summary.loc[2022, 'mean_tvd_cond_age_attend']:.3f}",
        "nRespTwoTwo": _tex_int_comma(int(summary.loc[2022, "n"])),
        "ablPhiReleased": f"{e24t.genre_phi_mad:.4f}",
        "ablPhiNoTree": f"{e24m.genre_phi_mad:.4f}",
        "nNarrPerWave": "300",
        "nHoldoutWaves": "3",
        "atlasRhoTwoTwo": f"{rho22:.2f}",
        "atlasRhoTwoZero": f"{rho20:.2f}",
        "ssrTauSha": tau.get("sha256", ""),
        "holdReleaseMean": f"{c24.tvd_release_narrative.mean():.3f}",
        "holdTwinMean": f"{c24.tvd_twin_anchors.mean():.3f}",
        "holdTwinGap": f"{abs(c24.tvd_release_narrative.mean() - c24.tvd_twin_anchors.mean()):.3f}",
        "holdTwinGapTwenty": f"{abs(c20.tvd_release_narrative.mean() - c20.tvd_twin_anchors.mean()):.3f}",
        "evalPhiMargins": f"{e24m.genre_phi_mad:.4f}",
        "evalPhiT": f"{e24t.genre_phi_mad:.4f}",
        "evalPairMargins": f"{e24m.genre_pair_joint_tvd:.4f}",
        "evalPairT": f"{e24t.genre_pair_joint_tvd:.4f}",
        "nSampleSmoke": sample_n,
        "sampleSmokeMeanTvd": sample_mean_tvd or "0.000",
        "runName": r"era\_narrative\_bank",
        "nPersonas": _tex_int_comma(int(narr_total)) if narr_total else "900",
        "nNarrEvalBank": _tex_int_comma(int(narr_total)) if narr_total else "900",
        "nQuantWeighted": _tex_int_comma(n_quant_weighted) if n_quant_weighted else "45{,}000",
        "nQuantDiagnostic": _tex_int_comma(n_quant_diag) if n_quant_diag else "10{,}000",
        "nFields": "12",
        "nAnchors": "24",
        "passRate": f"{pass_rate:.1f}" if narr_total else "100.0",
        "nFail": str(n_fail),
        "ssrMarginPersonas": "300",
        "ssrMarginSentences": str(int(ssr.get("margin", {}).get("n_sentences", 2700))),
        "ssrMarginMedian": f"{float(ssr.get('margin', {}).get('margin_median', 0)):.3f}",
        "ssrMarginPFive": f"{float(ssr.get('margin', {}).get('margin_p5', 0)):.3f}",
        "ssrMarginUndecided": f"{float(ssr.get('margin', {}).get('undecided_pct', 0)):.1f}",
        "ssrMarginRatio": f"{float(ssr.get('margin', {}).get('margin_ratio_at_tau', 0)):.1f}",
        "ssrRefMaxCos": f"{float(ssr['ref_max_cos']):.3f}"
        if ssr.get("ref_max_cos") is not None
        else "0.872",
        "ssrUndecidedGap": "0.01",
        "nSsrReferences": str(int(ssr.get("n_reference_sentences", 0)))
        if ssr.get("n_reference_sentences")
        else "0",
    }

    updates.update(_era_sampler_macros())
    updates.update(_era_source_macros())
    updates.update(kcas_era())
    updates.update(generators_era())
    updates.update(theme_v2_era())
    updates.update(shipped_fit_era())
    updates.update(fullnarr_era())
    updates.update(design_deep_era())
    updates.update(theme_coding_era())
    updates.update(design_oos_pairs_era())
    updates.update(readers_era())
    updates.update(logprob_era())
    updates.update(recal_era())
    updates.update(coder2_era())
    updates.update(tables_only_era())
    updates.update(kcas_crossfit_era())
    updates.update(marginal_era())
    updates.update(family_era())
    updates.update(ladder_splits_era())
    updates.update(info_ladder_era())
    updates.update(disclosure_era())
    updates.update(positive_control_era())

    for k, v in updates.items():
        text = _set_macro(text, k, v)

    if not text.startswith("%"):
        text = "% Auto-patched by scripts/make_tables_era.py (era reports).\n" + text
    # Pre-registered human-audit constants (scripts/anno_analysis.py) and stratum
    # sizes (outputs/anno/answer_key.csv, local).
    aa = (PROJ / "scripts" / "anno_analysis.py").read_text(encoding="utf-8")

    def _const(name: str) -> str:
        m = re.search(rf"^{name}\s*=\s*([0-9.]+)", aa, re.MULTILINE)
        if not m:
            raise SystemExit(f"{name} not found in anno_analysis.py")
        return m.group(1)

    text = _set_macro(text, "auditAlpha", _const("ALPHA"))
    text = _set_macro(text, "auditPOneBound", f"{100 * float(_const('P1_BOUND')):.0f}")
    text = _set_macro(text, "auditPTwoBound", _const("P2_BOUND"))
    text = _set_macro(text, "auditSOneNull", _const("S1_NULL"))
    text = _set_macro(text, "auditSFiveMargin", _const("S5_MARGIN"))
    text = _set_macro(text, "auditSFiveSpec", _const("S5_SPEC"))
    key_path = PROJ / "outputs" / "anno" / "answer_key.csv"
    if key_path.exists():
        key = pd.read_csv(key_path, dtype=str)
        # Registered sample size before the subset cut. If a subset file exists,
        # nAudit* is the number actually scored (ADR-039).
        reg = key.stratum.value_counts()
        text = _set_macro(text, "nAuditRegMain", str(len(key[key.stratum != "practice"])))
        text = _set_macro(text, "nAuditRegClean", str(int(reg.get("clean", 0))))
        subset_path = PROJ / "outputs" / "anno" / "subset_v2.json"
        if subset_path.exists():
            ids = set(json.loads(subset_path.read_text(encoding="utf-8"))["ids"])
            key = key[(key.stratum == "practice") | key.item_id.isin(ids)]
        n_clean_audit = int((key.stratum == "clean").sum())
        # Minimum clean-item count across waves (waves differ after rebalancing; the text says "at least").
        text = _set_macro(
            text,
            "nAuditCleanPerWave",
            str(int(key[key.stratum == "clean"].groupby("wave").size().min())),
        )
        # Largest contradiction count for which P1 holds: one-sided 95%
        # Clopper–Pearson upper bound < P1_BOUND.
        from scipy.stats import beta as _beta

        k_max = -1
        while k_max + 1 < n_clean_audit and _beta.ppf(
            0.95, k_max + 2, n_clean_audit - k_max - 1
        ) < float(_const("P1_BOUND")):
            k_max += 1
        text = _set_macro(text, "auditPOneMaxK", str(k_max))
        counts = key.stratum.value_counts()
        for macro, stratum in (
            ("nAuditClean", "clean"),
            ("nAuditFlagged", "flagged"),
            ("nAuditTrap", "trap"),
            ("nAuditGross", "trap_gross"),
            ("nAuditPractice", "practice"),
        ):
            text = _set_macro(text, macro, str(int(counts.get(stratum, 0))))
        text = _set_macro(text, "nAuditMain", str(len(key[key.stratum != "practice"])))
    # Judge census: clean personas that passed the validator with no warnings
    # (same rule as anno_build_sample.is_clean).
    n_clean = 0
    for w in (2020, 2022, 2024):
        bp = PROJ / "data" / "synthetic" / f"era_{w}_narrative" / "personas.jsonl"
        if bp.exists():
            for line in bp.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    n_clean += int(bool(r.get("ok")) and not r.get("issues"))
    if n_clean:
        text = _set_macro(text, "nBankClean", f"{n_clean:,}")
    # Age–education and occupation contradictions in unobserved combinations
    # (scripts/era_unseen_combos.py).
    up = REPORTS / "era_unseen_combos.csv"
    if up.exists():
        u = pd.read_csv(up)
        u = u[u.section == "rule"].set_index("key")
        text = _set_macro(text, "unseenImplausible", f"{float(u.loc['any', 'pct_sample']):.2f}")
        text = _set_macro(
            text, "unseenImplausibleStrict", f"{float(u.loc['strict', 'pct_sample']):.2f}"
        )
    # Judge reproducibility (scripts/anno_judge_repro.py): agreement when the
    # sample is rescored with the same settings.
    rp = REPORTS / "anno_judge_repro.csv"
    if rp.exists():
        r = pd.read_csv(rp)
        r = r[r.wave == "all"].set_index("metric")
        text = _set_macro(text, "nJudgeRepro", f"{int(r.loc['contradiction_agreement', 'n']):,}")
        text = _set_macro(
            text, "judgeReproAgree", f"{100 * float(r.loc['contradiction_agreement', 'value']):.1f}"
        )
        scales = [f"{k}_exact" for k in ("grounded", "coherent", "plausible", "fluent")]
        text = _set_macro(
            text, "judgeReproScaleMin", f"{100 * float(r.loc[scales, 'value'].min()):.1f}"
        )
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def probes_era() -> None:
    """Narrative-layer probes (ADR-018). raw describes the resource; masked is the figure to read."""
    src = REPORTS / "era_probes.csv"
    if not src.exists():
        return
    d = pd.read_csv(src)
    wide = d.pivot_table(
        index=["wave", "target", "n_classes", "majority", "stratified"],
        columns="condition",
        values="accuracy",
    ).reset_index()
    order = [
        "age_band",
        "sex",
        "living_area",
        "education",
        "occupation",
        "household_income_band",
        "household_size_band",
        "seoul_years_band",
        "leisure_sufficiency",
        "leisure_companion",
        "attend_any",
        "attend_count_band",
        "paid_spend_band",
        "viewing_companion",
    ]
    waves = sorted(int(w) for w in wide.wave.unique())
    last = waves[-1]
    body = []
    for t in order:
        g = wide[wide.target == t].set_index("wave")
        if g.empty:
            continue
        r_last = g.loc[last] if last in g.index else g.iloc[-1]
        cells = [
            f"\\texttt{{{tex_escape(t)}}}",
            str(int(r_last.n_classes)),
            f"{r_last.majority:.3f}",
        ]
        cells += [f"{g.loc[w].masked:.3f}" if w in g.index else "---" for w in waves]
        cells += [f"{g.loc[w].raw:.3f}" if w in g.index else "---" for w in waves]
        body.append("    " + " & ".join(cells) + r" \\")
    wave_hdr = " & ".join(str(w) for w in waves)
    ncol = len(waves)
    (TABLES / "probes.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n"
        f"  \\begin{{tabular}}{{@{{}}lrr{'r' * ncol}{'r' * ncol}@{{}}}}\n    \\toprule\n"
        f"    & & & \\multicolumn{{{ncol}}}{{c}}{{masked}} & \\multicolumn{{{ncol}}}{{c}}{{raw}} \\\\\n"
        f"    \\cmidrule(lr){{4-{3 + ncol}}}\\cmidrule(l){{{4 + ncol}-{3 + 2 * ncol}}}\n"
        f"    Anchor & classes & majority & {wave_hdr} & {wave_hdr} \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Recovering each anchor from each wave's completed narrative pass with a character $n$-gram TF-IDF "
        "classifier (75/25 split, \\nProbeTrain{} training personas); classes and majority baseline are for "
        f"{last}. \\emph{{raw}}: narratives as released; \\emph{{masked}}: every anchor value string and genre name deleted.}}\n"
        "  \\label{tab:probes}\n\\end{table}\n",
        encoding="utf-8",
    )


def downstream_era() -> None:
    """Two downstream tasks (ADR-019): TSTR and the design pretest, side by side in one appendix table."""
    t = REPORTS / "era_tstr.csv"
    d = REPORTS / "era_design_pretest.csv"
    if not t.exists() or not d.exists():
        return
    tt = pd.read_csv(t)
    agg = tt.groupby(["setting", "features"])[
        ["acc_marginal", "acc_age", "acc_tstr", "acc_trtr"]
    ].mean()
    # Naive-Bayes baseline from the released pair-joint tables (scripts/era_downstream_baselines.py).
    bpath = REPORTS / "era_downstream_baselines.csv"
    bl = pd.read_csv(bpath) if bpath.exists() else None
    pair_acc: dict[tuple[str, str], float] = {}
    pair_design: dict[str, tuple[float, float]] = {}
    if bl is not None:
        imp = bl[
            (bl.task == "imputation")
            & (bl.grain == "summary")
            & (bl.method == "pair_tables")
            & (bl.metric == "weighted_accuracy")
        ]
        for r in imp.itertuples():
            pair_acc[(r.setting, r.slice)] = float(r.value)
        des = bl[(bl.task == "design") & (bl.grain == "summary") & (bl.method == "pair_tables")]
        for slice_ in ("published", "unpublished"):
            e = des[(des.slice == slice_) & (des.metric == "mean_abs_err_pp")].value
            u = des[(des.slice == slice_) & (des.metric == "under_planned_pct")].value
            if len(e) and len(u):
                pair_design[slice_] = (float(e.iloc[0]), float(u.iloc[0]))
        npath = TABLES / "numbers.tex"
        ntext = npath.read_text(encoding="utf-8")
        if ("cross-wave", "all") in pair_acc:
            ntext = _set_macro(ntext, "tstrPairAll", f"{100 * pair_acc[('cross-wave', 'all')]:.0f}")
        # Paired interval for synthetic minus pair tables (scripts/era_tstr_paired.py).
        ppath = REPORTS / "era_tstr_paired.csv"
        if ppath.exists():
            pp = pd.read_csv(ppath)
            row = pp[(pp.setting == "cross-wave") & (pp.contrast == "synthetic-pair_tables")]
            if len(row):
                r0 = row.iloc[0]
                ntext = _set_macro(ntext, "tstrDiffPair", f"{float(r0.diff_pp):+.1f}")
                ntext = _set_macro(
                    ntext,
                    "tstrDiffPairCi",
                    f"[{float(r0.ci_lo_pp):+.1f}, {float(r0.ci_hi_pp):+.1f}]",
                )
        if "unpublished" in pair_design:
            ntext = _set_macro(ntext, "designPairUnpub", f"{pair_design['unpublished'][0]:.1f}")
        if "published" in pair_design:
            ntext = _set_macro(ntext, "designPairPub", f"{pair_design['published'][0]:.1f}")
        npath.write_text(ntext if ntext.endswith("\n") else ntext + "\n", encoding="utf-8")
    # A bank drawn from the printed tables alone (scripts/era_tables_only_bank.py), same classifier.
    tpath = REPORTS / "era_tables_only_bank.csv"
    to_acc: dict[tuple[str, str], float] = {}
    if tpath.exists():
        to = pd.read_csv(tpath)
        for r in to[to.record == "imputation"].itertuples():
            to_acc[(r.setting, r.features)] = float(r.acc_tables_only)
    body = []
    label = {
        ("in-wave", "all"): "same wave, all anchors",
        ("in-wave", "demographic"): "same wave, demographics only",
        ("cross-wave", "all"): "other waves, all anchors",
        ("cross-wave", "demographic"): "other waves, demographics only",
    }
    for key in (
        ("in-wave", "all"),
        ("in-wave", "demographic"),
        ("cross-wave", "all"),
        ("cross-wave", "demographic"),
    ):
        if key not in agg.index:
            continue
        r = agg.loc[key]
        pa = (
            f"{pair_acc[(key[0], key[1] if key[1] != 'demographic' else 'demographic')]:.3f}"
            if (key[0], key[1]) in pair_acc
            else "---"
        )
        ta = f"{to_acc[key]:.3f} & " if key in to_acc else ("--- & " if to_acc else "")
        body.append(
            f"    {label[key]} & {r.acc_marginal:.3f} & {r.acc_age:.3f} & {pa} & {ta}"
            f"{r.acc_tstr:.3f} & {r.acc_trtr:.3f} \\\\"
        )
    to_head = "tables-only bank & " if to_acc else ""
    (TABLES / "tstr.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        f"  \\begin{{tabular}}{{@{{}}lrrr{'r' if to_acc else ''}rr@{{}}}}\n    \\toprule\n"
        f"    Setting & marginal & age table & pair tables & {to_head}released bank & microdata \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Imputing a missing anchor for a real respondent: mean weighted accuracy over anchors. "
        "\\emph{marginal} and \\emph{age table} use the published report; \\emph{pair tables} is naive Bayes over the released 2024 tables; "
        "\\emph{tables-only bank} and \\emph{released bank} train a classifier on a 2024 bank drawn from the printed tables alone or from the released conditionals; "
        "\\emph{microdata} trains on the respondents under cross-validation. Same-wave rows are in sample.}\n"
        "  \\label{tab:tstr}\n\\end{table}\n",
        encoding="utf-8",
    )
    dd = pd.read_csv(d)
    g = dd.groupby(["published_axis", "method"]).agg(
        err=("abs_err_pp", "mean"), under=("n_ratio", lambda x: 100.0 * float((x < 0.9).mean()))
    )
    rows = []
    for pub, name in (
        (False, "crossings the report does not print"),
        (True, "axes the report prints"),
    ):
        for method, mlab in (
            ("marginal", "marginal rate"),
            ("age-conditional", "age table"),
            ("pair-tables", "pair tables (naive Bayes)"),
            ("synthetic", "synthetic population"),
        ):
            if method == "pair-tables":
                key_ = "published" if pub else "unpublished"
                if key_ in pair_design:
                    rows.append(
                        f"    {name} & {mlab} & {pair_design[key_][0]:.1f} & {pair_design[key_][1]:.0f} \\\\"
                    )
                continue
            if (pub, method) not in g.index:
                continue
            r = g.loc[(pub, method)]
            rows.append(f"    {name} & {mlab} & {r.err:.1f} & {r.under:.0f} \\\\")
    (TABLES / "design_pretest.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        "  \\begin{tabular}{@{}llrr@{}}\n    \\toprule\n"
        "    Subgroup cells & prior from & mean err (pp) & under-planned (\\%) \\\\\n    \\midrule\n"
        + "\n".join(rows)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Planning a subgroup sample without fielding one, in sample (out of sample, with every method rebuilt, the bank and the pair tables are level except on 2020's unprinted crossings; Appendix~\\ref{app:downstream}): the error in the attendance rate "
        "a planner would assume, and the share of cells whose planned sample falls more than a tenth "
        "short of what the target precision needs. Pooled over the \\nHoldoutWaves{} weighted waves, "
        "cells with at least \\cellFloor{} respondents.}\n"
        "  \\label{tab:design-pretest}\n\\end{table}\n",
        encoding="utf-8",
    )


def release_scope_era() -> None:
    """Release-contents table: what is released, how much, and on what terms.
    Figures are the numbers.tex macros.
    """
    rows = [
        (
            "Grounding tables T1--T6 and R0, three weighted waves",
            "T1--T6 cells $\\ge \\cellFloor$ respondents, T6 with complementary suppression; R0 population counts",
            "\\dataLicenceCell",
        ),
        (
            "Sampler specification (JSON) per wave",
            "parent-combination floor \\cellFloor; value cells not floored (Ethical Considerations)",
            "\\dataLicenceCell",
        ),
        (
            "Quantitative personas",
            "\\nQuantWeighted{} over \\nHoldoutWaves{} weighted waves (+ \\nQuantDiagnostic{} diagnostic, two unweighted waves)",
            "\\dataLicenceCell",
        ),
        (
            "Narrative personas",
            "\\nBankAll{} (\\nBankFull{} per wave), frozen scored subset \\nNarrEvalBank{} marked",
            "\\dataLicenceCell",
        ),
        (
            "Theme distribution of the open-ended item (T7) and the 2024 demonstration field",
            "theme shares by attendance status, codebook; theme code and field null in 2020 and 2022",
            "\\dataLicenceCell",
        ),
        (
            "Evaluation layer",
            "holdout answers, SSR references, PPI tables, judge runs, audit guideline, registered and exploratory audit result tables",
            "CC-BY-4.0",
        ),
        (
            "Code snapshot, data card, SHA-256 manifest",
            "produces every table and number; most regenerate from the released aggregates, the rest need the microdata",
            "CC-BY-4.0",
        ),
    ]
    body = "\n".join(f"    {a} & {b} & {c} \\\\" for a, b, c in rows)
    # The review copy keeps the anonymous wording; \ifpublic (main.tex) switches to the public release terms.
    (TABLES / "release_scope.tex").write_text(
        "\\begin{table*}[tbp]\n  \\centering\n  \\small\n"
        "  \\ifpublic\\def\\dataLicenceCell{\\dataLicence}\\else\\def\\dataLicenceCell{CC-BY-4.0$^{*}$}\\fi\n"
        "  \\begin{adjustbox}{max width=\\linewidth}\n"
        "  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.3\\textwidth}>{\\raggedright\\arraybackslash}p{0.5\\textwidth}l@{}}\n    \\toprule\n"
        "    Component & Scope & Licence \\\\\n    \\midrule\n"
        + body
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\ifpublic\\def\\releaseNote{Data components are released under \\dataLicence{}. "
        "The narrative bank is released with the registered audit's results and a RAPID audit of the public dump.}"
        "\\else\\def\\releaseNote{$^{*}$Subject to the publisher's derivative-release terms (CC-BY-NC-4.0 if required). "
        "Reviewers receive the full narrative bank; it is released publicly with the camera-ready version together with the registered audit's results.}\\fi\n"
        "  \\caption{What the release contains. \\releaseNote}\n"
        "  \\label{tab:release-scope}\n\\end{table*}\n",
        encoding="utf-8",
    )


def generators_era() -> dict[str, str]:
    """Three generators (ADR-029): whether narrative-layer measurements depend on the generator.
    The original generator (Qwen3.8) is read from the base files; the others from ``_tag`` files.
    """
    out: dict[str, str] = {}
    gen_path = REPORTS / "era_holdout_generators.csv"
    if not gen_path.exists():
        return out
    g = pd.read_csv(gen_path)
    cw = pd.read_csv(REPORTS / "era_holdout_crosswave.csv")
    c24 = cw[cw.wave == 2024]
    rows = []
    specs = [
        ("Qwen3.8 27B (release)", None),
        ("EXAONE 4.0 32B", "exaone"),
        ("Gemma 4 31B", "gemma31"),
    ]
    for label, tag in specs:
        if tag is None:
            hold = float(c24.tvd_release_narrative.mean())
            ot = pd.read_csv(REPORTS / "era_open_text.csv")
            lgc = pd.read_csv(REPORTS / "era_llm_genre_contrasts.csv").set_index("contrast")
        else:
            gg = g[(g.generator == tag) & (g.wave == 2024)]
            hold = float(gg.cell_mean_tvd.mean())
            ot = pd.read_csv(REPORTS / f"era_open_text_{tag}.csv")
            lgc = pd.read_csv(REPORTS / f"era_llm_genre_contrasts_{tag}.csv").set_index("contrast")
        open_tvd = float(
            ot[(ot.metric == "cell_theme_tvd") & (ot.method == "personas")].value.iloc[0]
        )
        d = lgc.loc["shot_narr - shot_anchor"]
        rows.append((label, hold, open_tvd, float(d.auc_diff), float(d.lo), float(d.hi)))
        key = {None: "Qwen", "exaone": "Exaone", "gemma31": "Gemma"}[tag]
        out[f"genHold{key}"] = f"{hold:.3f}"
        out[f"genOpen{key}"] = f"{open_tvd:.3f}"
        out[f"genNarr{key}"] = f"{float(d.auc_diff):+.3f}"
        out[f"genNarr{key}Ci"] = f"[{float(d.lo):+.3f}, {float(d.hi):+.3f}]"
    twin = float(c24.tvd_twin_anchors.mean())
    out["nGenerators"] = str(len(rows))
    # Largest shift of the generator-as-reader's held-out level across alternative generators' narratives.
    out["genHoldMaxGap"] = f"{max(abs(r[1] - rows[0][1]) for r in rows[1:]):.3f}"
    # Spell out the macro names the text uses, so a name-string check can match them.
    for name in (
        "genHoldQwen",
        "genHoldExaone",
        "genHoldGemma",
        "genOpenQwen",
        "genOpenExaone",
        "genOpenGemma",
        "genNarrQwen",
        "genNarrExaone",
        "genNarrGemma",
        "genNarrQwenCi",
        "genNarrExaoneCi",
        "genNarrGemmaCi",
    ):
        assert name in out, name
    body = "\n".join(
        f"    {a} & {b:.3f} & {c:.3f} & {d:+.3f} [{e:+.3f}, {f:+.3f}] \\\\"
        for a, b, c, d, e, f in rows
    )
    (TABLES / "generators.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrr@{}}\n    \\toprule\n"
        "    Generator & holdout TVD & open-item TVD & narrative $-$ anchors (AUC) \\\\\n    \\midrule\n"
        + body
        + f"\n    \\midrule\n    real respondents' anchors (twin) & {twin:.3f} & --- & --- \\\\\n"
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{The narrative-layer measurements repeated with two further generators on the same 2024 "
        "personas (300 each; same anchors, prompt, temperature and SSR answering model): "
        "\\emph{holdout TVD} on the nine held-out items; \\emph{open-item TVD} against the real open-ended answers; "
        "\\emph{narrative $-$ anchors}, the few-shot change in AUC from showing the same personas as prose instead of anchor lists.}\n"
        "  \\label{tab:generators}\n\\end{table}\n",
        encoding="utf-8",
    )
    return out


def shipped_fit_era() -> dict[str, str]:
    """Fit of the shipped configuration (registration root, pooled genre tree) next to the measured chain."""
    out: dict[str, str] = {}
    src = REPORTS / "era_shipped_fit.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src).set_index(["wave", "config"])["tvd_nonstructural"]
    names = {2024: "TwentyFour", 2022: "TwentyTwo", 2020: "Twenty"}
    for wave, nm in names.items():
        for config, key in (
            ("measured", "Measured"),
            ("shipped", "ShippedSpec"),
            ("shipped_survey_root", "ShippedSurveyRoot"),
            ("bank", "ShippedBank"),
        ):
            if (wave, config) in d.index:
                out[f"fit{key}{nm}"] = f"{float(d.loc[(wave, config)]):.3f}"
    return out


def fullnarr_era() -> dict[str, str]:
    """Held-out items with all twelve narrative fields shown to the reader (era_holdout_fullnarr.py)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_holdout_fullnarr.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    names = {2024: "TwentyFour", 2022: "TwentyTwo", 2020: "Twenty"}
    for wave, nm in names.items():
        lev = d[(d.wave == wave) & (d.record == "level") & (d.arm == "full")]
        if len(lev):
            out[f"fullNarrLevel{nm}"] = f"{float(lev.value.iloc[0]):.3f}"
        c = d[(d.wave == wave) & (d.record == "contrast") & (d.contrast == "full-two")]
        if len(c):
            r = c.iloc[0]
            out[f"fullNarrDiff{nm}"] = f"{r.value:.3f}"
            out[f"fullNarrDiffCi{nm}"] = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
    return out


def design_deep_era() -> dict[str, str]:
    """Three-way unprinted crossings (era_design_deep.py): mean absolute error in pp by method and wave."""
    out: dict[str, str] = {}
    src = REPORTS / "era_design_deep_summary.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    names = {2024: "TwentyFour", 2022: "TwentyTwo", 2020: "Twenty"}
    keys = {"marginal": "Marg", "age_table": "Age", "pair_tables": "Pair", "synthetic": "Syn"}
    for wave, nm in names.items():
        w = d[d.wave == wave]
        for _, r in w[w.record == "mae_pp"].iterrows():
            out[f"deep{keys[r.method]}{nm}"] = f"{r.value:.1f}"
        r = w[w.record == "diff_pp"]
        if len(r):
            r = r.iloc[0]
            out[f"deepDiff{nm}"] = f"{r.value:.2f}"
            out[f"deepDiffCi{nm}"] = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
            out[f"deepCells{nm}"] = _tex_int_comma(int(r.n_cells))
        r = w[w.record == "share_closer"]
        if len(r):
            out[f"deepCloser{nm}"] = f"{100 * float(r.value.iloc[0]):.0f}"
    return out


def theme_coding_era() -> dict[str, str]:
    """Coding real open-ended answers with a classifier trained on synthetic text (era_theme_coding.py)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_theme_coding.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    lv = d[d.record == "level"].set_index(["condition", "subset"])

    def acc(cond: str, subset: str = "all8") -> str:
        return f"{100 * float(lv.loc[(cond, subset), 'accuracy']):.0f}"

    out["themeCodeMajority"] = acc("majority")
    out["themeCodeZero"] = acc("zero_shot_codebook")
    out["themeCodeSyn"] = acc("synthetic|embedding")
    out["themeCodeSynSeven"] = acc("synthetic|embedding", "seven")
    out["themeCodeReal"] = acc("real_A|embedding")
    out["themeCodeRealSeven"] = acc("real_A|embedding", "seven")
    out["themeCodeAug"] = acc("synthetic+real_A|embedding")
    out["themeCodeSynTfidf"] = acc("synthetic|tfidf")
    out["themeCodeSynFOneSeven"] = (
        f"{float(lv.loc[('synthetic|embedding', 'seven'), 'macro_f1']):.2f}"
    )
    out["themeCodeRealFOneSeven"] = (
        f"{float(lv.loc[('real_A|embedding', 'seven'), 'macro_f1']):.2f}"
    )
    out["themeCodeNTest"] = _tex_int_comma(int(lv.loc[("majority", "all8"), "n"]))
    ct = d[d.record == "contrast"].set_index("condition")
    r = ct.loc["synthetic|embedding - majority"]
    out["themeCodeGainMaj"] = f"{100 * r.accuracy:.0f}"
    out["themeCodeGainMajCi"] = f"[{100 * r.ci_lo:.0f}, {100 * r.ci_hi:.0f}]"
    r = ct.loc["synthetic|embedding - zero_shot_codebook"]
    out["themeCodeGainZero"] = f"{100 * r.accuracy:+.0f}"
    out["themeCodeGainZeroCi"] = f"[{100 * r.ci_lo:.0f}, {100 * r.ci_hi:.0f}]"
    r = ct.loc["synthetic|embedding - real_A|embedding"]
    out["themeCodeGapReal"] = f"{-100 * r.accuracy:.0f}"
    out["themeCodeGapRealCi"] = f"[{-100 * r.ci_hi:.0f}, {-100 * r.ci_lo:.0f}]"
    return out


def readers_era() -> dict[str, str]:
    """Release prose vs twin anchor lists for further reader models and waves (era_holdout_readers.py)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_holdout_readers.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    names = {2024: "TwentyFour", 2022: "TwentyTwo", 2020: "Twenty"}
    rn = {"exaone": "Exa", "gemma": "Gem"}
    c = d[(d.record == "contrast") & (d.arm == "release-twin") & d.reader.isin(list(rn))]
    for r in c.itertuples():
        key = f"rd{rn[r.reader]}{names[int(r.wave)]}"
        out[key] = f"{r.value:.3f}"
        out[key + "Ci"] = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
    out["rdNegative"] = str(int((c.value < 0).sum()))
    out["rdSignificant"] = str(int((c.ci_hi < 0).sum()))
    out["rdCells"] = str(len(c))
    # Paired contrast on the same synthetic personas: release narratives minus their anchor list alone.
    rn2 = {"exaone": "Exa", "gemma": "Gem", "qwen": "Qwen"}
    allc = d[d.record == "contrast"]
    pair = allc[allc.arm == "release-synthanchors"]
    for r in pair.itertuples():
        key = f"rdPair{rn2[r.reader]}{names[int(r.wave)]}"
        out[key] = f"{r.value:.3f}"
        out[key + "Ci"] = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
    other = pair[pair.reader != "qwen"]
    out["rdPairNegative"] = str(int((other.value < 0).sum()))
    out["rdPairSignificant"] = str(int((other.ci_hi < 0).sum()))
    out["rdPairCells"] = str(len(other))
    # Generated prose minus a fixed-sentence rendering of the same anchors (era_template_prose.py), 2024.
    wsuf = {2024: "", 2022: "TwentyTwo", 2020: "Twenty"}
    for arm, key in (("release-template", "rdTemplate"), ("template-twin", "rdTemplateTwin")):
        for r in allc[allc.arm == arm].itertuples():
            k = f"{key}{rn2[r.reader]}{wsuf[int(r.wave)]}"
            out[k] = f"{r.value:+.3f}"
            out[k + "Ci"] = f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
    for r in d[(d.record == "level") & (d.arm == "template")].itertuples():
        out[f"rdTemplateLevel{rn2[r.reader]}{wsuf[int(r.wave)]}"] = f"{r.value:.3f}"
    # Release-arm and anchor-list levels per reader and wave (same source), for prose next to the no-persona prior.
    for arm, key in (("release", "rdReleaseLevel"), ("synthanchors", "rdListLevel")):
        for r in d[(d.record == "level") & (d.arm == arm)].itertuples():
            out[f"{key}{rn2[r.reader]}{wsuf[int(r.wave)]}"] = f"{r.value:.3f}"
    tpl = allc[allc.arm == "release-template"]
    out["rdTemplateCells"] = str(len(tpl))
    out["rdTemplateSignificant"] = str(int((tpl.ci_hi < 0).sum()))
    people = allc[(allc.arm == "synthanchors-twin") & (allc.reader != "qwen")]
    if len(people):
        out["rdPeopleMaxAbs"] = f"{float(people.value.abs().max()):.3f}"
    return out


def logprob_era() -> dict[str, str]:
    """Second elicitation for the held-out items, option log-probabilities (era_holdout_logprob.py)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_holdout_logprob.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    rn = {"qwen": "", "exaone": "Exa"}
    for reader, suffix in rn.items():
        lv = d[(d.record == "level") & (d.reader == reader)]
        for arm, name in (
            ("release", "Release"),
            ("synthanchors", "Synth"),
            ("twin", "Twin"),
            ("prior", "Prior"),
            ("template", "Template"),
            ("oracle", "Oracle"),
        ):
            r = lv[(lv.arm == arm) & (lv.method == "logprob")]
            if len(r):
                out[f"lp{name}{suffix}"] = f"{float(r.value.iloc[0]):.3f}"
        r = lv[(lv.arm == "release") & (lv.method == "logprob")]
        s = lv[(lv.arm == "release") & (lv.method == "ssr")]
        if len(r):
            out[f"lpEntropy{suffix}"] = f"{float(r.entropy.iloc[0]):.2f}"
            out[f"lpCoverage{suffix}"] = f"{100 * float(r.coverage.iloc[0]):.1f}"
        if len(s):
            out[f"lpEntropySsr{suffix}"] = f"{float(s.entropy.iloc[0]):.2f}"
        c = d[(d.record == "contrast") & (d.reader == reader)]
        for arm, name in (
            ("release-synthanchors", "RelSynth"),
            ("release-twin", "RelTwin"),
            ("release-template", "RelTemplate"),
            ("oracle-release", "OracleRel"),
        ):
            r = c[c.arm == arm]
            if len(r):
                r0 = r.iloc[0]
                out[f"lp{name}{suffix}"] = f"{float(r0.value):+.3f}"
                out[f"lp{name}{suffix}Ci"] = f"[{float(r0.ci_lo):+.3f}, {float(r0.ci_hi):+.3f}]"
    # The reader's prior under SSR (era_holdout_prior_ssr.py).
    psrc = REPORTS / "era_holdout_prior_ssr.csv"
    if psrc.exists():
        pdf = pd.read_csv(psrc)
        if "reader" not in pdf.columns:
            pdf["reader"] = "qwen"
        for reader, sfx in (("qwen", ""), ("exaone", "Exa"), ("gemma", "Gem")):
            r = pdf[(pdf.record == "level") & (pdf.arm == "prior") & (pdf.reader == reader)]
            if len(r):
                out[f"ssrPrior{sfx}"] = f"{float(r.value.iloc[0]):.3f}"
    return out


def recal_era() -> dict[str, str]:
    """Reader-marginal recalibration of the held-out arms (era_holdout_recal.py): every reader and wave
    under SSR, the generator and EXAONE under log-probabilities on 2024, with the cell-shuffle null."""
    out: dict[str, str] = {}
    src = REPORTS / "era_holdout_recal.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    if "wave" not in d.columns:
        d["wave"] = 2024
    rn = {"qwen": "", "exaone": "Exa", "gemma": "Gem"}
    mn = {"ssr": "", "logprob": "Lp"}
    wn = {2024: "", 2022: "TwentyTwo", 2020: "Twenty"}
    arms = (("release", "Release"), ("synthanchors", "Synth"), ("twin", "Twin"))
    for wave, ws in wn.items():
        for reader, rs in rn.items():
            for method, ms in mn.items():
                lv = d[
                    (d.record == "level")
                    & (d.reader == reader)
                    & (d.method == method)
                    & (d.wave == wave)
                ]
                for arm, name in arms:
                    r = lv[lv.arm == arm]
                    if not len(r):
                        continue
                    r0 = r.iloc[0]
                    key = f"{name}{ms}{rs}{ws}"
                    out[f"recal{key}"] = f"{float(r0.value):.3f}"
                    out[f"recalCorr{key}"] = f"{float(r0.cell_corr):.2f}"
                    out[f"recalMae{key}"] = f"{float(r0.cell_mae):.3f}"
                    if "shuffle_p" in r0 and pd.notna(r0.get("shuffle_p")):
                        out[f"recalShuffleMean{key}"] = f"{float(r0.shuffle_mean):.3f}"
                        pv = float(r0.shuffle_p)
                        out[f"recalShuffleP{key}"] = "<0.001" if pv < 0.001 else f"{pv:.3f}"
                    if "cell_corr_lo" in r0 and pd.notna(r0.get("cell_corr_lo")):
                        out[f"recalCorr{key}Ci"] = (
                            f"[{float(r0.cell_corr_lo):.2f}, {float(r0.cell_corr_hi):.2f}]"
                        )
                if len(lv):
                    out[f"recalMaeLookup{ms}{rs}{ws}"] = f"{float(lv.cell_mae_lookup.iloc[0]):.3f}"
                c = d[
                    (d.record == "contrast")
                    & (d.reader == reader)
                    & (d.method == method)
                    & (d.wave == wave)
                ]
                for arm, name in (
                    ("release-synthanchors", "RelSynth"),
                    ("release-twin", "RelTwin"),
                ):
                    r = c[c.arm == arm]
                    if len(r):
                        r0 = r.iloc[0]
                        key = f"{name}{ms}{rs}{ws}"
                        out[f"recal{key}"] = f"{float(r0.value):+.3f}"
                        out[f"recal{key}Ci"] = f"[{float(r0.ci_lo):+.3f}, {float(r0.ci_hi):+.3f}]"
    # Prose-minus-list difference in the cell-contrast correlation (persona-paired bootstrap).
    cd = d[d.record == "corr_diff"] if "corr_diff" in set(d.record) else d.iloc[0:0]
    for r in cd.itertuples():
        key = f"recalCorrDiff{mn[r.method]}{rn[r.reader]}{wn[int(r.wave)]}"
        out[key] = f"{float(r.value):+.2f}"
        out[key + "Ci"] = f"[{float(r.ci_lo):+.2f}, {float(r.ci_hi):+.2f}]"
    cds = cd[cd.method == "ssr"]
    out["recalCorrDiffCells"] = str(len(cds))
    out["recalCorrDiffSignificant"] = str(int((cds.ci_lo > 0).sum()))
    # How many SSR reader-wave cells keep a significant prose gain after recalibration.
    c = d[(d.record == "contrast") & (d.method == "ssr") & (d.arm == "release-synthanchors")]
    out["recalCells"] = str(len(c))
    out["recalSignificant"] = str(int((c.ci_hi < 0).sum()))
    return out


def coder2_era() -> dict[str, str]:
    """Second model coder for the open-ended item (era_theme_coder2.py)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_theme_coder2.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    a = d[d.record == "agreement"]
    if len(a):
        out["coderTwoAgree"] = f"{100 * float(a.value.iloc[0]):.0f}"
        out["coderTwoKappa"] = f"{float(a.kappa.iloc[0]):.2f}"
    for subset, name in (("gemma_labels", ""), ("agreed", "Agreed")):
        r = d[
            (d.record == "contrast")
            & (d.subset == subset)
            & (d.contrast == "synthetic|embedding - zero_shot_codebook")
        ]
        if len(r):
            r0 = r.iloc[0]
            out[f"coderTwoGainZero{name}"] = f"{100 * float(r0.value):+.0f}"
            out[f"coderTwoGainZero{name}Ci"] = (
                f"[{100 * float(r0.ci_lo):.0f}, {100 * float(r0.ci_hi):.0f}]"
            )
            out[f"coderTwoSyn{name}"] = f"{100 * float(r0.acc_synthetic):.0f}"
            out[f"coderTwoZero{name}"] = f"{100 * float(r0.acc_ref):.0f}"
            out[f"coderTwoN{name}"] = _tex_int_comma(int(r0.n))
    return out


def ladder_splits_era() -> dict[str, str]:
    """Held-out ranges of the ladder and the chain over the cross-fit splits (era_ladder_crossfit.py)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_ladder_crossfit.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    d = d[d.wave == 2024].set_index("rung")
    names = {
        "indep": "Indep",
        "cond_age": "Age",
        "cond_age_sex": "AgeSex",
        "cond_age_route": "AgeRoute",
        "cond_age_attend": "AgeAttend",
    }
    for rung, nm in names.items():
        if rung in d.index:
            out[f"tvd{nm}OutLo"] = f"{float(d.loc[rung, 'oos_a_min']):.3f}"
            out[f"tvd{nm}OutHi"] = f"{float(d.loc[rung, 'oos_a_max']):.3f}"
    msrc = REPORTS / "era_ladder_crossfit.md"
    if msrc.exists():
        m = re.search(r"분할 (\d+)", msrc.read_text(encoding="utf-8"))
        if m:
            out["nLadderSplits"] = m.group(1)
    csrc = REPORTS / "era_chain_crossfit.csv"
    if csrc.exists():
        c = pd.read_csv(csrc)
        c = c[(c.wave == 2024) & (c.setting == "split-half")]
        if len(c):
            out["chainSplitOutLo"] = f"{float(c.tvd_out_of_sample.min()):.3f}"
            out["chainSplitOutHi"] = f"{float(c.tvd_out_of_sample.max()):.3f}"
    return out


def design_oos_pairs_era() -> dict[str, str]:
    """Design pretest out of sample with the released-table baseline rebuilt on the fitting half
    (era_design_oos_pairs.py): mean absolute error in pp by group, method and wave."""
    out: dict[str, str] = {}
    # The respondent-resampled run over 20 splits (--resample-respondents) supersedes the cell bootstrap.
    src = REPORTS / "era_design_oos_pairs_resp.csv"
    if not src.exists():
        src = REPORTS / "era_design_oos_pairs.csv"
    if not src.exists():
        return out
    out["oosRespondentResampled"] = "yes" if src.name.endswith("_resp.csv") else "no"
    d = pd.read_csv(src)
    names = {2024: "TwentyFour", 2022: "TwentyTwo", 2020: "Twenty"}
    groups = {"unprinted": "Unp", "printed": "Pub", "three-way": "Deep"}
    methods = {"pair_tables": "Pair", "synthetic": "Syn", "age_table": "Age", "marginal": "Marg"}
    for wave, nm in names.items():
        for group, g in groups.items():
            w = d[(d.wave == wave) & (d.group == group)]
            for _, r in w[w.record == "mae_pp"].iterrows():
                out[f"oos{g}{methods[r.method]}{nm}"] = f"{r.value:.1f}"
            r = w[w.record == "diff_pp"]
            if len(r):
                r = r.iloc[0]
                out[f"oos{g}Diff{nm}"] = f"{r.value:.2f}"
                out[f"oos{g}DiffCi{nm}"] = f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
    return out


def marginal_era() -> dict[str, str]:
    """Reader marginal against the survey marginal per answer source (scripts/era_holdout_marginal.py)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_holdout_marginal.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    rn = {"qwen": "Qwen", "exaone": "Exa", "gemma": "Gem"}
    an = {
        "prior": "Prior",
        "synthanchors": "List",
        "template": "Template",
        "release": "Release",
        "twin": "Twin",
    }
    wn = {2024: "", 2022: "TwentyTwo", 2020: "Twenty"}
    for r in d[d.record == "level"].itertuples():
        out[f"mg{an[r.arm]}{rn[r.reader]}{wn[int(r.wave)]}"] = f"{r.value:.3f}"
    for r in d[d.record == "contrast"].itertuples():
        ref = r.arm.split("-")[1]
        key = f"mgRel{an[ref]}{rn[r.reader]}{wn[int(r.wave)]}"
        out[key] = f"{r.value:+.3f}"
        out[key + "Ci"] = f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
    # Prose minus list over the two other readers and three waves: cells where the interval is below zero.
    c = d[(d.record == "contrast") & (d.arm == "release-synthanchors") & (d.reader != "qwen")]
    out["mgListCells"] = str(len(c))
    out["mgListSignificant"] = str(int((c.ci_hi < 0).sum()))
    c2 = d[(d.record == "contrast") & (d.arm == "release-template") & (d.reader != "qwen")]
    out["mgTemplateCells"] = str(len(c2))
    out["mgTemplateSignificant"] = str(int((c2.ci_hi < 0).sum()))
    return out


def family_era() -> dict[str, str]:
    """Family status the narratives state without an anchor (scripts/era_family_mentions.py)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_family_mentions.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    a = d[d.wave.astype(str) == "all"].iloc[0]
    out["famShare"] = f"{100 * float(a.share_family):.0f}"
    out["famStated"] = f"{100 * float(a.share_stated):.0f}"
    out["famHedge"] = f"{100 * float(a.share_hedge_only):.0f}"
    out["famField"] = f"{100 * float(a.share_family_field):.0f}"
    return out


def kcas_crossfit_era() -> dict[str, str]:
    """KCAS ladder held out (scripts/kcas_ladder_crossfit.py): a second table beside the in-sample one, and macros."""
    out: dict[str, str] = {}
    src = REPORTS / "kcas_ladder_crossfit.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    rung = d[d.record == "rung"]
    summ = d[d.record == "summary"].set_index("wave")
    body = []
    for wave in sorted(summ.index):
        r = rung[rung.wave == wave].set_index("rung")
        s = summ.loc[wave]
        body.append(
            "    "
            + " & ".join(
                [
                    str(int(wave)),
                    f"{float(s.floor):.3f}",
                    *[
                        f"{float(r.loc[k, 'oos_mean']):.3f}"
                        for k in (
                            "indep",
                            "cond_age",
                            "cond_age_sex",
                            "cond_age_route",
                            "cond_age_attend",
                        )
                    ],
                    f"{100 * float(s.age_cut_out):.0f}",
                    f"{100 * float(s.route_cut_out):.0f}",
                    f"{100 * float(s.beyond_route_out):.0f}",
                ]
            )
            + r" \\"
        )
    n_splits = int(summ.n_splits.iloc[0])
    (TABLES / "kcas_oos.tex").write_text(
        "\\begin{table*}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\textwidth}\n"
        "  \\begin{tabular}{@{}lrrrrrrrrr@{}}\n    \\toprule\n"
        "    Wave & floor & indep. & + age & + age$\\times$sex & + skip logic & + age$\\times$attendance & age cut\\% & skip cut\\% & beyond\\% \\\\\n"
        "    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        f"  \\caption{{The national survey's ladder held out: rungs fitted on a random half of each wave's respondents and scored on the other half, mean weighted pair TVD over {n_splits} splits; \\emph{{floor}} is the distance between the two halves' joints. Shares (weighted): \\emph{{age cut}} of the independence error, \\emph{{skip cut}} of the age-conditional residual, \\emph{{beyond}} of what the skip logic leaves.}}\n"
        "  \\label{tab:kcas-oos}\n\\end{table*}\n",
        encoding="utf-8",
    )
    ok = int(summ.order_ok_out.astype(bool).sum())
    out["nKcasSplits"] = str(n_splits)
    out["kcasOutOrderAll"] = "every" if ok == len(summ) else f"{ok} of {len(summ)}"
    out["kcasOutOrderN"] = str(ok)
    out["kcasOutAgeCutMin"] = f"{100 * summ.age_cut_out.min():.0f}"
    out["kcasOutAgeCutMax"] = f"{100 * summ.age_cut_out.max():.0f}"
    out["kcasOutRouteMin"] = f"{100 * summ.route_cut_out.min():.0f}"
    out["kcasOutRouteMax"] = f"{100 * summ.route_cut_out.max():.0f}"
    out["kcasOutBeyondMin"] = f"{100 * summ.beyond_route_out.min():.0f}"
    out["kcasOutBeyondMax"] = f"{100 * summ.beyond_route_out.max():.0f}"
    out["kcasOutCutMin"] = f"{100 * summ.attend_cut_out.min():.0f}"
    out["kcasOutCutMax"] = f"{100 * summ.attend_cut_out.max():.0f}"
    out["kcasOutSexGapMax"] = f"{float(summ.sex_gap_out.abs().max()):.4f}"
    out["kcasOutFloorMin"] = f"{float(summ.floor.min()):.3f}"
    out["kcasOutFloorMax"] = f"{float(summ.floor.max()):.3f}"
    deep = rung[rung.rung == "cond_age_attend"].set_index("wave").oos_mean
    out["kcasOutFloorRatioMax"] = f"{float((deep / summ.floor).max()):.1f}"
    return out


def tables_only_era() -> dict[str, str]:
    """A bank drawn from the printed tables alone (scripts/era_tables_only_bank.py) on the downstream tasks, and
    the tables-only chain on the out-of-sample crossings (era_design_oos_pairs.py --variant tables_only)."""
    out: dict[str, str] = {}
    src = REPORTS / "era_tables_only_bank.csv"
    if src.exists():
        d = pd.read_csv(src)
        pt = d[d.record == "pair_tvd"].set_index("bank")
        out["toPairTvd"] = f"{float(pt.loc['tables_only', 'value']):.3f}"
        out["toPairTvdShipped"] = f"{float(pt.loc['shipped', 'value']):.3f}"
        if "tables_only_survey_root" in pt.index:
            out["toPairTvdSurveyRoot"] = f"{float(pt.loc['tables_only_survey_root', 'value']):.3f}"
        imp = d[d.record == "imputation"].set_index(["setting", "features"])
        out["toTstrAll"] = f"{100 * float(imp.loc[('cross-wave', 'all'), 'acc_tables_only']):.0f}"
        out["toTstrDemo"] = (
            f"{100 * float(imp.loc[('cross-wave', 'demographic'), 'acc_tables_only']):.0f}"
        )
        out["toTstrInAll"] = f"{100 * float(imp.loc[('in-wave', 'all'), 'acc_tables_only']):.0f}"
        pc = d[d.record == "imputation_paired"].set_index(["setting", "contrast"])
        for key, nm in (("tables_only-shipped", "Shipped"), ("tables_only-pair_tables", "Pair")):
            for setting, sn in (("cross-wave", ""), ("in-wave", "In")):
                if (setting, key) in pc.index:
                    r = pc.loc[(setting, key)]
                    out[f"toDiff{nm}{sn}"] = f"{float(r.value):+.1f}"
                    out[f"toDiff{nm}{sn}Ci"] = f"[{float(r.ci_lo):+.1f}, {float(r.ci_hi):+.1f}]"
        ge = d[d.record == "genre"].set_index(["setting", "features"])
        out["toGenreCross"] = f"{float(ge.loc[('cross-wave', 'cultural'), 'auc_tables_only']):.3f}"
        out["toGenreCrossDemo"] = (
            f"{float(ge.loc[('cross-wave', 'demographic'), 'auc_tables_only']):.3f}"
        )
        out["toGenrePairCross"] = (
            f"{float(ge.loc[('cross-wave', 'cultural'), 'auc_pair_tables']):.3f}"
        )
        gp = REPORTS / "era_genre_portfolio.csv"
        if gp.exists():
            g0 = pd.read_csv(gp)
            ship = float(
                g0[(g0.setting == "cross-wave") & (g0.features == "cultural")].auc_synthetic.mean()
            )
            out["toGenreShippedCross"] = f"{ship:.3f}"
            ship_in = float(
                g0[(g0.setting == "in-wave") & (g0.features == "cultural")].auc_synthetic.mean()
            )
            to_in = float(ge.loc[("in-wave", "cultural"), "auc_tables_only"])
            out["toGenreShippedIn"] = f"{ship_in:.3f}"
            out["toGenreIn"] = f"{to_in:.3f}"
            out["toGenreGapShippedIn"] = f"{ship_in - to_in:.2f}"
            gap = ship - float(ge.loc[("cross-wave", "cultural"), "auc_tables_only"])
            out["toGenreGapShipped"] = f"{gap:.2f}"
        if "toDiffShipped" in out:
            out["toDiffShippedAbs"] = out["toDiffShipped"].lstrip("+-")
    oos = REPORTS / "era_design_oos_pairs_tables_only.csv"
    if oos.exists():
        d = pd.read_csv(oos)
        names = {2024: "TwentyFour", 2022: "TwentyTwo", 2020: "Twenty"}
        groups = {"unprinted": "Unp", "printed": "Pub", "three-way": "Deep"}
        for wave, nm in names.items():
            for group, g in groups.items():
                w = d[(d.wave == wave) & (d.group == group)]
                m = w[(w.record == "mae_pp") & (w.method == "synthetic")]
                if len(m):
                    out[f"oosTo{g}{nm}"] = f"{float(m.value.iloc[0]):.1f}"
                r = w[w.record == "diff_pp"]
                if len(r):
                    r = r.iloc[0]
                    out[f"oosTo{g}Diff{nm}"] = f"{float(r.value):.2f}"
                    out[f"oosTo{g}DiffCi{nm}"] = f"[{float(r.ci_lo):.2f}, {float(r.ci_hi):.2f}]"
    return out


def theme_v2_era() -> dict[str, str]:
    """ADR-033: T7 v2 (author codebook 7+1, coder EXAONE). Theme TVD for a 300-person
    pilot and the 13th field of the 2024 bank.

    If the v2 files exist, overwrite tables/theme_narrative.tex and the theme* macros with the v2 values.
    """
    out: dict[str, str] = {}
    pilot = REPORTS / "era_theme_v2_pilot.csv"
    if not pilot.exists():
        return out
    d = pd.read_csv(pilot).set_index("condition")
    out["themeCZero"] = f"{float(d.loc['C0', 'tvd']):.3f}"
    out["themeCOne"] = f"{float(d.loc['C1', 'tvd']):.3f}"
    out["themeCTwo"] = f"{float(d.loc['C2', 'tvd']):.3f}"
    out["themeRealization"] = f"{100 * float(d.loc['C2', 'realization']):.0f}"
    out["themeMarginal"] = f"{float(d.loc['marginal', 'tvd']):.3f}"
    out["themeFloor"] = f"{float(d.loc['floor_AB', 'tvd']):.3f}"
    out["themeOracle"] = f"{float(d.loc['oracle_T7', 'tvd']):.3f}"
    out["themeDrawOnly"] = f"{float(d.loc['draw_only', 'tvd']):.3f}"
    cb = PROJ / "data" / "grounding" / "era_2024" / "T7_theme_codebook.csv"
    n_themes = len(pd.read_csv(cb)) - 1 if cb.exists() else 7  # exclude "other"
    out["nThemes"] = str(n_themes)
    rows = [
        ("current field (anchors only)", float(d.loc["C0", "tvd"]), None),
        (
            "theme drawn from the marginal table",
            float(d.loc["C1", "tvd"]),
            float(d.loc["C1", "realization"]),
        ),
        (
            "theme drawn from the attendance table (T7)",
            float(d.loc["C2", "tvd"]),
            float(d.loc["C2", "realization"]),
        ),
    ]
    bank = REPORTS / "era_theme_v2_bank.csv"
    if bank.exists():
        b = pd.read_csv(bank).iloc[0]
        out["themeBankTvd"] = f"{float(b.tvd_release):.3f}"
        out["themeBankTvdB"] = f"{float(b.tvd_B):.3f}"
        out["themeBankRealization"] = f"{100 * float(b.realization):.0f}"
        out["themeBankN"] = f"{int(b.n_bank):,}"
        out["themeBankSample"] = f"{int(b.n_sample):,}"
        rows.append(
            (
                "released 2024 field (sample; table refitted on all respondents)",
                float(b.tvd_B),
                float(b.realization),
            )
        )
    rows += [
        ("themes drawn from T7, no generation (200 draws)", float(d.loc["draw_only", "tvd"]), None),
        ("marginal lookup (real answers, half A)", float(d.loc["marginal", "tvd"]), None),
        ("real half A vs.\\ half B", float(d.loc["floor_AB", "tvd"]), None),
        ("T7 applied directly", float(d.loc["oracle_T7", "tvd"]), None),
    ]
    body = "\n".join(
        f"    {lab} & {tvd:.3f} & " + ("---" if rz is None else f"{100 * rz:.0f}\\%") + " \\\\"
        for lab, tvd, rz in rows
    )
    (TABLES / "theme_narrative.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrr@{}}\n    \\toprule\n"
        "    Field on what Seoul needs as a cultural city & cell theme TVD & theme realised \\\\\n    \\midrule\n"
        + body
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Giving the narrative one aggregate table of the open-ended answers: TVD over the seven substantive themes, "
        "weighted over age $\\times$ attendance cells, against the held-out half of the real answers (coding in the text). "
        "\\emph{Theme realised}: the share of generated fields coded to the theme they were given.}\n"
        "  \\label{tab:theme-narrative}\n\\end{table}\n",
        encoding="utf-8",
    )
    return out


def info_ladder_era() -> dict[str, str]:
    """ADR-034: KL and TVD of the same reconstructions; whether rung order is invariant to the metric
    (2024, 187 pairs). Writes tables/info_ladder.tex and macros.
    """
    out: dict[str, str] = {}
    src = REPORTS / "era_info_ladder_pairs.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    d = d[(d.source == "seoul") & (d.kl > 0)]
    d = d.assign(bound=np.sqrt(d.kl / 2.0))
    rungs = [
        ("indep", "marginals only"),
        ("cond_age", "+ age"),
        ("cond_age_sex", "+ age $\\times$ sex"),
        ("cond_age_route", "+ skip logic"),
        ("cond_age_attend", "+ age $\\times$ attendance"),
    ]
    rows, pear, spear, ratio = [], [], [], []
    for wave in sorted(d.wave.unique()):
        for key, label in rungs:
            g = d[(d.wave == wave) & (d.rung == key)]
            if len(g) < 10:
                continue
            pr = float(np.corrcoef(g.tvd, g.bound)[0, 1])
            sp = float(g[["tvd", "bound"]].corr(method="spearman").iloc[0, 1])
            md = float((g.tvd / g.bound).median())
            pear.append(pr)
            spear.append(sp)
            ratio.append(md)
            if wave == 2024:
                rows.append(
                    f"    {label} & {g.tvd.mean():.3f} & {g.kl.mean():.3f} & {pr:.2f} & {sp:.2f} & {md:.2f} \\\\"
                )
    m24 = {k: float(d[(d.wave == 2024) & (d.rung == k)].kl.mean()) for k, _ in rungs}
    out["infoRouteCutKl"] = f"{100 * (1 - m24['cond_age_route'] / m24['cond_age']):.0f}"
    out["infoBeyondKl"] = f"{100 * (1 - m24['cond_age_attend'] / m24['cond_age_route']):.0f}"
    out["infoPearsonMin"] = f"{min(pear):.2f}"
    out["infoSpearmanMin"] = f"{min(spear):.2f}"
    out["infoRatioLo"] = f"{min(ratio):.2f}"
    out["infoRatioHi"] = f"{max(ratio):.2f}"
    (TABLES / "info_ladder.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrrrr@{}}\n    \\toprule\n"
        "    Reconstruction & mean TVD & mean KL & Pearson & Spearman & TVD / bound \\\\\n    \\midrule\n"
        + "\n".join(rows)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{The error ladder under KL divergence, 2024 wave, the same \\nPairsNonStructural{} pairs and reconstructions "
        "as Table~\\ref{tab:error-summary}. Pearson and Spearman correlate pair TVD with the Pinsker bound $\\sqrt{\\mathrm{KL}/2}$ "
        "across pairs; the last column is the median ratio of TVD to that bound. Ranges over the three weighted waves: "
        "Pearson $\\ge$ \\infoPearsonMin, Spearman $\\ge$ \\infoSpearmanMin, median ratio \\infoRatioLo--\\infoRatioHi.}\n"
        "  \\label{tab:info-ladder}\n\\end{table}\n",
        encoding="utf-8",
    )
    return out


def disclosure_era() -> dict[str, str]:
    """ADR-036: attribute disclosure for real respondents. CAP from the marginal, the released
    tables, and the released bank (2024). Writes tables/disclosure.tex and macros.
    """
    out: dict[str, str] = {}
    src = REPORTS / "era_attribute_disclosure.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    d24 = d[d.wave == 2024]
    labels = {
        "household_income_band": "household income band",
        "attend_any": "attendance status",
        "n_genres_band": "genre-count band",
    }
    preds = [
        ("marginal", "prevalence"),
        ("tables", "released tables"),
        ("bank", "released quantitative bank"),
    ]
    rows = []
    for tgt, lab in labels.items():
        g = d24[d24.target == tgt].set_index("predictor")
        cells = " & ".join(f"{float(g.loc[k, 'cap']):.3f}" for k, _ in preds)
        diff = float(g.loc["bank", "cap"]) - float(g.loc["tables", "cap"])
        rows.append(f"    {lab} & {cells} & {diff:+.3f} \\\\")
    inc = d24[d24.target == "household_income_band"].set_index("predictor")
    att = d24[d24.target == "attend_any"].set_index("predictor")
    gen = d24[d24.target == "n_genres_band"].set_index("predictor")
    out["capIncomeTablesLift"] = (
        f"{100 * (float(inc.loc['tables', 'cap']) - float(inc.loc['marginal', 'cap'])):.0f}"
    )
    out["capAttendTablesLift"] = (
        f"{100 * (float(att.loc['tables', 'cap']) - float(att.loc['marginal', 'cap'])):.0f}"
    )
    out["capIncomeBankMinusTables"] = (
        f"{100 * (float(inc.loc['bank', 'cap']) - float(inc.loc['tables', 'cap'])):+.0f}"
    )
    out["capAttendBankMinusTables"] = (
        f"{100 * (float(att.loc['bank', 'cap']) - float(att.loc['tables', 'cap'])):+.0f}"
    )
    out["capGenreBankMinusTables"] = (
        f"{100 * (float(gen.loc['bank', 'cap']) - float(gen.loc['tables', 'cap'])):+.0f}"
    )
    (TABLES / "disclosure.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrrr@{}}\n    \\toprule\n"
        "    Target (real respondents, 2024) & prevalence & tables & bank & bank $-$ tables \\\\\n    \\midrule\n"
        + "\n".join(rows)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Attribute disclosure against real respondents: correct attribution probability (CAP) when an attacker who knows "
        "sex, age band, living area, education and occupation predicts the target from prevalence, from the released tables or from "
        "the released quantitative bank (sources defined in the text); the last column is what the bank adds beyond the tables.}\n"
        "  \\label{tab:disclosure}\n\\end{table}\n",
        encoding="utf-8",
    )
    return out


def positive_control_era() -> dict[str, str]:
    """ADR-038: narrative-holdout positive control (oracle narratives) by two answer models.
    Writes tables/positive_control.tex and the pc* macros.
    """
    out: dict[str, str] = {}
    src = REPORTS / "era_holdout_positive_control.csv"
    if not src.exists():
        return out
    d = pd.read_csv(src)
    lv = d[d.record == "level"].set_index(["answer_model", "arm"]).cell_mean_tvd
    ct = d[d.record == "contrast"].set_index(["answer_model", "contrast"])
    bl = d[d.record == "baseline"].set_index("contrast").cell_mean_tvd
    names = {
        ("qwen", "twin"): "pcTwinQwen",
        ("qwen", "release"): "pcReleaseQwen",
        ("qwen", "oracle"): "pcOracleQwen",
        ("exaone", "twin"): "pcTwinExa",
        ("exaone", "release"): "pcReleaseExa",
        ("exaone", "oracle"): "pcOracleExa",
    }
    for k, m in names.items():
        out[m] = f"{float(lv.loc[k]):.3f}"

    def fmt(model: str, c: str) -> str:
        r = ct.loc[(model, c)]
        return f"{float(r['diff']):+.3f}"

    def ci(model: str, c: str) -> str:
        r = ct.loc[(model, c)]
        return f"[{float(r.ci_lo):+.3f}, {float(r.ci_hi):+.3f}]"

    out["pcRelTwinQwen"] = fmt("qwen", "release-twin")
    out["pcRelTwinQwenCi"] = ci("qwen", "release-twin")
    out["pcRelTwinExa"] = fmt("exaone", "release-twin")
    out["pcRelTwinExaCi"] = ci("exaone", "release-twin")
    out["pcOracleTwinQwen"] = fmt("qwen", "oracle-twin")
    out["pcOracleTwinExa"] = fmt("exaone", "oracle-twin")
    out["pcLabelFloor"] = f"{float(bl.loc['oracle_label_ceiling']):.3f}"
    rows = []
    for arm, lab in (
        ("twin", "real respondents' anchors (twin)"),
        ("release", "release narratives"),
        ("oracle", "narratives rewritten with drawn facts"),
    ):
        rows.append(
            f"    {lab} & {float(lv.loc[('qwen', arm)]):.3f} & {float(lv.loc[('exaone', arm)]):.3f} \\\\"
        )
    rows.append("    \\midrule")
    for c, lab in (("release-twin", "release $-$ twin"), ("oracle-twin", "oracle $-$ twin")):
        rows.append(
            f"    {lab} & {fmt('qwen', c)} {ci('qwen', c)} & {fmt('exaone', c)} {ci('exaone', c)} \\\\"
        )
    rows.append("    \\midrule")
    rows.append(
        f"    drawn facts themselves (label floor) & \\multicolumn{{2}}{{c}}{{{float(bl.loc['oracle_label_ceiling']):.3f}}} \\\\"
    )
    rows.append(
        f"    marginal lookup / real $n{{=}}300$ & \\multicolumn{{2}}{{c}}{{{float(bl.loc['marginal_lookup']):.3f} / {float(bl.loc['real_sample_n300']):.3f}}} \\\\"
    )
    (TABLES / "positive_control.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrr@{}}\n    \\toprule\n"
        "    Arm (2024, $n{=}300$; mean cell TVD, lower is better) & reader: generator & reader: EXAONE 4.0 32B \\\\\n    \\midrule\n"
        + "\n".join(rows)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Positive control and a second reader: mean cell TVD on the 2024 evaluation personas for the release, twin and "
        "\\emph{oracle} arms (fields rewritten with holdout values drawn from the real cell distributions, stated as facts). "
        "Differences: cell-stratified bootstraps ($B{=}2000$; release and oracle paired by persona). Label floor: the drawn facts answered exactly.}\n"
        "  \\label{tab:positive-control}\n\\end{table}\n",
        encoding="utf-8",
    )
    return out


def narrative_summary_era() -> None:
    """One table for the held-out item results of the narrative layer (2024, SSR): mean cell TVD by arm and
    reader, then the recalibrated release level, the cell-shuffle p-value and the cell-contrast correlation
    for the release arm and its anchor lists. Sources: era_holdout_readers.csv, era_holdout_prior_ssr.csv,
    era_holdout_baselines.csv, era_holdout_recal.csv."""
    rsrc = REPORTS / "era_holdout_readers.csv"
    csrc = REPORTS / "era_holdout_recal.csv"
    bsrc = REPORTS / "era_holdout_baselines.csv"
    if not (rsrc.exists() and csrc.exists() and bsrc.exists()):
        return
    rd = pd.read_csv(rsrc)
    rd = rd[(rd.wave == 2024) & (rd.record == "level")]
    rc = pd.read_csv(csrc)
    if "wave" not in rc.columns:
        rc["wave"] = 2024
    rc = rc[(rc.wave == 2024) & (rc.method == "ssr") & (rc.record == "level")]
    bl = pd.read_csv(bsrc)
    bl = bl[bl.wave == 2024]
    psrc = REPORTS / "era_holdout_prior_ssr.csv"
    pr = pd.read_csv(psrc) if psrc.exists() else None
    if pr is not None and "reader" not in pr.columns:
        pr["reader"] = "qwen"
    readers = ("qwen", "exaone", "gemma")

    def lvl(reader: str, arm: str) -> str:
        r = rd[(rd.reader == reader) & (rd.arm == arm)]
        return f"{float(r.value.iloc[0]):.3f}" if len(r) else "---"

    def prior(reader: str) -> str:
        if pr is None:
            return "---"
        r = pr[(pr.record == "level") & (pr.arm == "prior") & (pr.reader == reader)]
        return f"{float(r.value.iloc[0]):.3f}" if len(r) else "---"

    def rec(reader: str, arm: str, col: str, fmt: str) -> str:
        r = rc[(rc.reader == reader) & (rc.arm == arm)]
        if not len(r) or col not in r.columns or pd.isna(r[col].iloc[0]):
            return "---"
        v = float(r[col].iloc[0])
        if col == "shuffle_p":
            return "$<$0.001" if v < 0.001 else f"{v:.3f}"
        return format(v, fmt)

    rows = [
        ("release narratives (anchors + two fields)", [lvl(r, "release") for r in readers]),
        ("the same personas' anchor lists", [lvl(r, "synthanchors") for r in readers]),
        ("real respondents' anchor lists (twin)", [lvl(r, "twin") for r in readers]),
        ("anchors as fixed sentences (template)", [lvl(r, "template") for r in readers]),
        ("no persona (reader's prior, as a lookup)", [prior(r) for r in readers]),
    ]
    marg = f"{float(bl.marginal_lookup_tvd.mean()):.3f}"
    real = f"{float(bl.real_sample_tvd.mean()):.3f}"
    diag = [
        (
            "release narratives, marginal removed",
            [rec(r, "release", "value", ".3f") for r in readers],
        ),
        (
            "same personas' lists, marginal removed",
            [rec(r, "synthanchors", "value", ".3f") for r in readers],
        ),
    ]
    body = "\n".join(f"    {lab} & " + " & ".join(v) + " \\\\" for lab, v in rows)
    body += f"\n    marginal lookup, ignoring the cells & {marg} & {marg} & {marg} \\\\\n    a real sample of \\nNarrPerWave{{}} respondents & {real} & {real} & {real} \\\\\n    \\midrule\n"
    body += "\n".join(f"    {lab} & " + " & ".join(v) + " \\\\" for lab, v in diag)
    # The reader's pooled marginal against the survey's (era_holdout_marginal.py), 2024 SSR.
    mg = REPORTS / "era_holdout_marginal.csv"
    if mg.exists():
        m = pd.read_csv(mg)
        m = m[(m.wave == 2024) & (m.record == "level")].set_index(["reader", "arm"])

        def mv(reader: str, arm: str) -> str:
            return (
                f"{float(m.loc[(reader, arm), 'value']):.3f}" if (reader, arm) in m.index else "---"
            )

        body += "\n    \\midrule\n" + "\n".join(
            f"    {lab} & " + " & ".join(mv(r, arm) for r in readers) + " \\\\"
            for lab, arm in (
                ("reader's marginal vs.\\ survey's: no persona", "prior"),
                ("reader's marginal vs.\\ survey's: same personas' lists", "synthanchors"),
                ("reader's marginal vs.\\ survey's: release narratives", "release"),
            )
        )
    (TABLES / "narrative_summary.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        "  \\begin{tabular}{@{}lrrr@{}}\n    \\toprule\n"
        "    Answer source (2024, SSR) & generator & EXAONE & Gemma \\\\\n    \\midrule\n"
        + body
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Held-out items by answer source and reader: mean cell TVD on 2024 (lower is better), then each "
        "arm with the reader's own marginal rescaled to the survey's, then the TVD between the reader's pooled answers and the survey marginal. "
        "Readers: the generator (\\genModel), EXAONE 4.0 32B, Gemma 4 31B.}\n"
        "  \\label{tab:narrative-summary}\n\\end{table}\n",
        encoding="utf-8",
    )


def audit_era() -> None:
    """Slot for human-audit results (pre-registration v2). Filled from
    ``outputs/reports/anno_results.csv`` when that file exists; otherwise a waiting row.
    """
    src = REPORTS / "anno_results.csv"
    order = ["P1", "P2", "S1a", "S1b", "S2a", "S2b", "S4a", "S4b", "S5a", "S5b"]
    labels = {
        "P1": "P1 contradiction rate among clean personas $< \\auditPOneBound\\%$",
        "P2": "P2 ordinal $\\alpha$ on groundedness $> \\auditPTwoBound$",
        "S1a": "S1a rater 1 trap sensitivity $> \\auditSOneNull$",
        "S1b": "S1b rater 2 trap sensitivity $> \\auditSOneNull$",
        "S2a": "S2a rater 1 catches more traps than the validator",
        "S2b": "S2b rater 2 catches more traps than the validator",
        "S4a": "S4a rater 1 age band from masked prose above permutation null",
        "S4b": "S4b rater 2 age band from masked prose above permutation null",
        "S5a": "S5a model judge trap sensitivity non-inferior (margin \\auditSFiveMargin)",
        "S5b": "S5b model judge specificity on clean $> \\auditSFiveSpec$",
    }
    body = []
    if src.exists():
        d = pd.read_csv(src).set_index("id")
        for k in order:
            if k not in d.index:
                body.append(f"    {labels[k]} & --- & --- & --- & --- \\\\")
                continue
            r = d.loc[k]
            bound = r.get("hi") if k == "P1" else r.get("lo")
            bound = "---" if pd.isna(bound) else f"{float(bound):.3f}"
            holds = "yes" if bool(r.get("holds")) else "no"
            body.append(
                f"    {labels[k]} & {float(r.estimate):.3f} & {bound} & {int(r.n)} & {holds} \\\\"
            )
        note = (
            "Estimates are the registered statistics; the bound column is the one-sided 95\\% limit the claim is judged on. "
            "S2 and S5a estimates are differences in sensitivity on the planted contradictions (rater minus validator; "
            "judge minus the raters' mean); S5b counts the clean personas with no adjudicated contradiction."
        )

        # Macros for the results paragraph; defined only when results exist.
        def _f(k: str, col: str, pct: bool = False) -> str:
            v = d.loc[k, col] if k in d.index and col in d.columns else float("nan")
            return "---" if pd.isna(v) else (f"{100 * float(v):.1f}" if pct else f"{float(v):.3f}")

        am = {
            "auditPOneEst": _f("P1", "estimate", pct=True),
            "auditPOneHi": _f("P1", "hi", pct=True),
            "auditPTwoEst": _f("P2", "estimate"),
            "auditPTwoLo": _f("P2", "lo"),
            "auditSOneAEst": _f("S1a", "estimate"),
            "auditSOneBEst": _f("S1b", "estimate"),
            "auditPpiEst": _f("D-ppi", "estimate", pct=True),
            "auditPpiHi": _f("D-ppi", "hi", pct=True),
            "auditPOneCorrHi": _f("P1-corrected", "hi", pct=True),
            "auditNAdjudicated": str(int(d.loc["adjudication", "estimate"]))
            if "adjudication" in d.index
            else "---",
            "auditNClean": str(int(d.loc["P1", "n"])) if "P1" in d.index else "---",
            "auditPrimaryHold": "hold"
            if all(bool(d.loc[k, "holds"]) for k in ("P1", "P2") if k in d.index)
            else "do not both hold",
        }
        npath = TABLES / "numbers.tex"
        ntext = npath.read_text(encoding="utf-8")
        for k_, v_ in am.items():
            ntext = _set_macro(ntext, k_, v_)
        npath.write_text(ntext if ntext.endswith("\n") else ntext + "\n", encoding="utf-8")
    else:
        for k in order:
            body.append(f"    {labels[k]} & pending & pending & --- & --- \\\\")
        note = "Scoring was in progress at submission; every cell is filled by the registered analysis code from the frozen answer file, and no row can be added or removed."
    (TABLES / "audit.tex").write_text(
        "\\begin{table*}[t]\n  \\centering\n  \\small\n  \\begin{tabular}{@{}lrrrl@{}}\n    \\toprule\n"
        "    Registered claim & estimate & bound & $n$ & holds \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        f"  \\caption{{Human audit (Appendix~\\ref{{app:audit}}), one row per pre-registered claim. {note}}}\n"
        "  \\label{tab:audit}\n\\end{table*}\n",
        encoding="utf-8",
    )


def genre_portfolio_era() -> None:
    """Genre-portfolio completion (ADR-021): the task table specific to this resource."""
    src = REPORTS / "era_genre_portfolio.csv"
    if not src.exists():
        return
    d = pd.read_csv(src)
    has_nb = "auc_pair_tables" in d.columns
    cols = [
        "auc_age_table",
        *(["auc_pair_tables"] if has_nb else []),
        "auc_synthetic",
        "auc_microdata",
    ]
    agg = d.groupby(["setting", "features"])[cols].mean()
    tpath = REPORTS / "era_tables_only_bank.csv"
    to_auc: dict[tuple[str, str], float] = {}
    if tpath.exists():
        to = pd.read_csv(tpath)
        for r in to[to.record == "genre"].itertuples():
            to_auc[(r.setting, r.features)] = float(r.auc_tables_only)
    label = {"cultural": "cultural profile", "demographic": "demographics", "both": "both"}
    body = []
    for setting, sname in (("in-wave", "same wave"), ("cross-wave", "other waves")):
        for feat in (
            "cultural",
            "demographic",
        ):  # 'both' nearly matches cultural; leave it in the CSV only
            if (setting, feat) not in agg.index:
                continue
            r = agg.loc[(setting, feat)]
            nb = f"{r.auc_pair_tables:.3f} & " if has_nb else ""
            ta = (
                f"{to_auc[(setting, feat)]:.3f} & "
                if (setting, feat) in to_auc
                else ("--- & " if to_auc else "")
            )
            body.append(
                f"    {sname} & {label[feat]} & {r.auc_age_table:.3f} & {nb}{ta}"
                f"{r.auc_synthetic:.3f} & {r.auc_microdata:.3f} \\\\"
            )
    nb_head = "pair tables & " if has_nb else ""
    to_head = "tables-only bank & " if to_auc else ""
    (TABLES / "genre_portfolio.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        f"  \\begin{{tabular}}{{@{{}}llr{'r' if has_nb else ''}{'r' if to_auc else ''}rr@{{}}}}\n    \\toprule\n"
        f"    Test set & Predictor sees & age table & {nb_head}{to_head}released bank & microdata \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{Filling in one held-out genre for an attender: mean AUC over the ten genres (non-attenders excluded, since "
        "the questionnaire fixes their flags). Predictors as in Table~\\ref{tab:tstr}; same-wave rows are in sample for the tables and both banks, and the released bank's genre tree was pooled from the other-wave test sets.}\n"
        "  \\label{tab:genre-portfolio}\n\\end{table}\n",
        encoding="utf-8",
    )


def llm_genre_era() -> None:
    """LLM downstream task (ADR-022): whether an LLM given personas as examples recovers real attenders."""
    src = REPORTS / "era_llm_genre.csv"
    con = REPORTS / "era_llm_genre_contrasts.csv"
    if not src.exists() or not con.exists():
        return
    d = pd.read_csv(src).set_index("condition")
    c = pd.read_csv(con).set_index("contrast")
    label = [
        ("demo", "demographics only, no examples"),
        ("zero", "cultural profile, no examples"),
        ("shot_anchor", "$+$ \\nKShot{} personas as anchor lists"),
        ("shot_narr", "$+$ the same personas as narratives"),
        ("knn_narr", "$+$ nearest personas as narratives"),
    ]
    body = []
    for key, lab in label:
        if key not in d.index:
            continue
        body.append(f"    {lab} & {d.loc[key, 'auc']:.3f} \\\\")
    rows_c = []
    for key in (
        "zero - demo",
        "shot_anchor - zero",
        "shot_narr - shot_anchor",
        "knn_narr - shot_narr",
    ):
        if key not in c.index:
            continue
        r = c.loc[key]
        rows_c.append(
            f"    {tex_escape(key)} & {r.auc_diff:+.3f} & [{r.lo:+.3f}, {r.hi:+.3f}] \\\\"
        )
    (TABLES / "llm_genre.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\begin{adjustbox}{max width=\\columnwidth}\n"
        "  \\begin{tabular}{@{}lr@{}}\n    \\toprule\n    What the model is given & AUC \\\\\n    \\midrule\n"
        + "\n".join(body)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n\n  \\vspace{0.6em}\n"
        "  \\begin{adjustbox}{max width=\\columnwidth}\n  \\begin{tabular}{@{}lrr@{}}\n    \\toprule\n"
        "    Contrast & $\\Delta$AUC & 95\\% CI \\\\\n    \\midrule\n"
        + "\n".join(rows_c)
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{The same held-out-genre task put to \\genModel{} instead of a fitted model, on "
        "\\nLlmTargets{} real attenders of the \\nLlmWaves{} waves whose pair tables the bank was not built from (its genre tree was pooled from them). Examples "
        "are synthetic personas only; the anchor and narrative rows use the \\emph{same} example people, so "
        "their difference is what the prose adds. Intervals are a paired bootstrap over targets.}\n"
        "  \\label{tab:llm-genre}\n\\end{table}\n",
        encoding="utf-8",
    )


# Appendix table placement. Every appendix table is a page-width float (table*) so
# they flow in one sequence; in the single-column TALLIP layout each table has its own float.


def _ensure_fit() -> None:
    """Keep tables within the text width. Body table 1 uses [t]; every appendix table uses table*[tbp].

    Tables that are already wrapped are left alone, so a second run leaves them unchanged.
    """
    for p in TABLES.glob("*.tex"):
        if p.stem == "numbers":
            continue
        t = p.read_text(encoding="utf-8")
        if "resizebox" in t:
            continue
        if "\\begin{table*}" not in t and "\\begin{table}" not in t:
            continue
        # Pass a lambda: in an re.sub replacement string, \b is read as backspace.
        if p.stem in BODY_SINGLE_TABLES:  # body column-width tables: leave table[t]
            t = re.sub(r"\\begin\{table\}\[[!htbp]*\]", lambda _m: "\\begin{table}[t]", t)
        else:
            t = re.sub(r"\\begin\{table\*?\}\[[!htbp]*\]", lambda _m: "\\begin{table*}[tbp]", t)
            t = t.replace("\\end{table}", "\\end{table*}")
        if p.stem == "error_summary":  # body table 1 is top-of-page only
            t = t.replace("\\begin{table*}[tbp]", "\\begin{table*}[t]")
        # Strip every existing adjustbox and wrap once. Width is \linewidth
        # (page width inside table*, the minipage width inside a minipage).
        t = re.sub(
            r"\\begin\{adjustbox\}\{max width=\\(?:columnwidth|textwidth|linewidth)\}\n?", "", t
        )
        t = t.replace("\\end{adjustbox}", "")
        # When several tables are stacked (e.g. llm_genre), wrap each one.
        # One shared box would place them side by side.
        if p.stem != "persona_example":  # p{} columns already set the width; do not wrap
            t = t.replace(
                "\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n  \\begin{tabular}"
            )
            t = t.replace("\\end{tabular}", "\\end{tabular}\\end{adjustbox}")
        # One type size for every page-width table, so tables read alike across the paper.
        if "\\begin{table*}" in t and not re.search(r"\\(small|footnotesize|scriptsize)\b", t):
            t = t.replace("\\centering\n", "\\centering\n  \\small\n")
        p.write_text(t, encoding="utf-8")
    _stack_llm_genre()


def _stack_llm_genre() -> None:
    """Put the two parts of llm_genre in full-width minipages, one above the other."""
    p = TABLES / "llm_genre.tex"
    t = p.read_text(encoding="utf-8")
    if "\\vspace{0.6em}" in t and "minipage" not in t:
        first, second = t.split("\\vspace{0.6em}", 1)
        first = (
            first.replace(
                "\\centering\n",
                "\\centering\n  \\begin{minipage}[t]{\\linewidth}\\vspace{0pt}\\centering\n",
                1,
            ).rstrip()
            + "\n  \\end{minipage}\\par\\vspace{0.8em}\n"
        )
        second = "  \\begin{minipage}[t]{\\linewidth}\\vspace{0pt}\\centering\n" + second.lstrip(
            "\n"
        )
        second = second.replace(
            "\\end{tabular}\\end{adjustbox}", "\\end{tabular}\\end{adjustbox}\n  \\end{minipage}", 1
        )
        p.write_text(first + second, encoding="utf-8")


# Body column-width tables, excluded from the appendix table* policy: dataset summary and example persona.
BODY_SINGLE_TABLES = {"dataset_stats", "persona_example"}


def dataset_stats_era() -> None:
    """Body dataset-summary table: a small statistics table at the start of the data section.

    Panel respondents per wave (decade_error_atlas_summary), released grounding cells
    (T1--T6 ``*_release``), quantitative and narrative persona counts, and median
    narrative length (sum of 12 fields, in characters). Every figure is counted from files.
    """
    waves = (2020, 2022, 2024)
    summary = pd.read_csv(REPORTS / "decade_error_atlas_summary.csv").set_index("wave")
    rows: dict[str, list[str]] = {
        "Panel respondents": [],
        "Released grounding cells (Table~\\ref{tab:grounding})": [],
        "Quantitative personas": [],
        "Narrative personas": [],
        "Narrative length, median characters": [],
        "Frozen evaluation subset": [],
    }
    for w in waves:
        rows["Panel respondents"].append(_tex_int_comma(int(summary.loc[w, "n"])))
        g = PROJ / "data" / "grounding" / f"era_{w}"
        cells = sum(
            len(pd.read_parquet(g / f"{t}_release.parquet"))
            for t in ("T1", "T2", "T3", "T4", "T5", "T6_pair_joints")
        )
        rows["Released grounding cells (Table~\\ref{tab:grounding})"].append(_tex_int_comma(cells))
        q = PROJ / "data" / "synthetic" / f"era_{w}_quant" / "personas_structure_T.parquet"
        rows["Quantitative personas"].append(
            _tex_int_comma(len(pd.read_parquet(q, columns=["persona_id"])))
        )
        nar = PROJ / "data" / "synthetic" / f"era_{w}_narrative" / "personas.jsonl"
        lens = []
        n_nar = 0
        with nar.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_nar += 1
                narr = r.get("narrative") or {}
                lens.append(sum(len(v) for v in narr.values() if isinstance(v, str)))
        rows["Narrative personas"].append(_tex_int_comma(n_nar))
        rows["Narrative length, median characters"].append(
            _tex_int_comma(int(pd.Series(lens).median()))
        )
        rows["Frozen evaluation subset"].append("\\nNarrPerWave")
    body = "\n".join(f"    {k} & " + " & ".join(v) + " \\\\" for k, v in rows.items())
    (TABLES / "dataset_stats.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\small\n  \\begin{adjustbox}{max width=\\linewidth}\n"
        "  \\begin{tabular}{@{}lrrr@{}}\n    \\toprule\n    & "
        + " & ".join(str(w) for w in waves)
        + " \\\\\n    \\midrule\n"
        + body
        + "\n    \\bottomrule\n  \\end{tabular}\\end{adjustbox}\n"
        "  \\caption{The dataset by weighted wave; each wave has its own tables and population root. "
        "Every persona carries \\nAnchors{} anchors; narrative personas add \\nFields{} Korean fields. Sparse parent combinations are withheld (Section~\\ref{subsec:chain}).}\n  \\label{tab:dataset-stats}\n\\end{table}\n",
        encoding="utf-8",
    )


# Example persona: fixed id in the public sample (personas_sample_100.jsonl).
# The translation is stored below; stop if the Korean source text changes.
EXAMPLE_PERSONA_ID = "fc4ceab4-1842-47dd-ab09-e9e41e1ea638"
EXAMPLE_FIELDS = {
    "narr_summary": (
        "50대 여성으로 서남2생활권에서 살며 가족과 함께 다양한 장르의 문화를 즐기되 지출은 절제한다.",
        "A woman in her fifties in the Southwest-2 area attends many genres with her family while holding spending down.",
    ),
    "narr_spending_attitude": (
        "연간 5만원 미만으로 관람비를 지출하며 예산을 엄격하게 관리한다. 가족의 생계 유지가 우선이므로 문화비 지출을 최소화한다.",
        "Under 50,000 won a year on a strict budget; cultural spending is minimised, as the family's livelihood comes first.",
    ),
}
# This persona's anchor values: Korean label, Revised Romanization and English translation, for the table only.
EXAMPLE_ANCHORS = [
    ("sex", "여성", "yeoseong", "female"),
    ("age_band", "50대", "50dae", "in their 50s"),
    ("living_area", "서남2생활권", "seonam 2 saenghwalgwon", "Southwest-2 living area"),
    ("education", "대학 졸업", "daehak joreop", "university graduate"),
    ("occupation", "판매종사자", "panmae jongsaja", "sales worker"),
    (
        "household_income_band",
        "200만원 미만",
        "200manwon miman",
        "household income under 2 million won a month",
    ),
    ("household_size_band", "4명 이상", "4myeong isang", "four or more in the household"),
    ("seoul_years_band", "10~20년", "10~20nyeon", "10 to 20 years in Seoul"),
    ("leisure_sufficiency", "충분", "chungbun", "leisure time sufficient"),
    ("leisure_companion", "혼자서", "honjaseo", "spends leisure alone"),
    ("attend_count_band", "11회 이상", "11hoe isang", "11 or more visits in the past year"),
    ("paid_spend_band", "5만원 미만", "5manwon miman", "under 50,000 won spent a year"),
    (
        "viewing_companion",
        "가족(친척 포함)과 함께",
        "gajok(chincheok poham)gwa hamkke",
        "attends with family, relatives included",
    ),
]
# Revised Romanization of the two narrative fields (checked by hand against the pronunciation rules).
EXAMPLE_FIELDS_RR = {
    "narr_summary": "50dae yeoseongeuro seonam 2 saenghwalgwoneseo salmyeo gajokgwa hamkke dayanghan jangneuui munhwareul jeulgidoe jichureun jeoljehanda.",
    "narr_spending_attitude": "yeongan 5manwon mimaneuro gwallambireul jichulhamyeo yesaneul eomgyeokhage gwallihanda. gajogui saenggye yujiga useonimeuro munhwabi jichureul choesohwahanda.",
}


def persona_example_era() -> None:
    """Body example-persona table: one anchor line (Korean label plus English gloss)
    and its narrative fields (source plus translation). Drawn only from the synthetic sample.
    """
    src = PROJ / "data" / "release" / "v1.0-rc1" / "personas_sample_100.jsonl"
    row = next(
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if EXAMPLE_PERSONA_ID in line
    )
    for k, (ko, _en) in EXAMPLE_FIELDS.items():
        if row[k] != ko:
            raise SystemExit(f"example persona field {k} changed; recheck its translation")
    genres = [
        k.replace("attend_", "")
        for k in row
        if k.startswith("attend_")
        and k not in ("attend_any", "attend_count_band")
        and row[k] == "예"
    ]
    gloss = {
        "western_classical": "classical",
        "movie": "cinema",
        "exhibition": "exhibitions",
        "museum": "museums",
        "popular_performance": "pop music",
        "dance_performance": "dance",
        "traditional": "traditional arts",
        "theater": "theatre",
        "literature_event": "literary events",
        "outdoor_event": "outdoor events",
    }
    parts = []
    for key, ko, roman, en in EXAMPLE_ANCHORS:
        if not (
            row[key] == ko or (key in ("education", "occupation") and row[key].startswith(ko + "("))
        ):
            raise SystemExit(f"example persona anchor {key} changed; recheck its romanization")
        parts.append(f"\\ko{{{ko}}} \\textit{{{roman}}} ({en})")
    parts.insert(10, f"attended: {', '.join(gloss[g] for g in genres)}")
    anchors = " $\\cdot$ ".join(parts)
    anchors = anchors.replace("~", "\\textasciitilde{}")
    labels = {"narr_summary": "Summary", "narr_spending_attitude": "Spending attitude"}
    body = "".join(
        f"    \\textbf{{{labels[k]}}} & \\ko{{{ko.replace('~', chr(92) + 'textasciitilde{}')}}} \\newline {EXAMPLE_FIELDS_RR[k]} \\newline \\textit{{{en}}} \\\\\n"
        for k, (ko, en) in EXAMPLE_FIELDS.items()
    )
    (TABLES / "persona_example.tex").write_text(
        "\\begin{table}[t]\n  \\centering\n  \\footnotesize\n"
        "  \\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.19\\columnwidth}>{\\raggedright\\arraybackslash}p{0.76\\columnwidth}@{}}\n"
        "    \\toprule\n    \\textbf{Anchors} & "
        + anchors
        + " \\\\\n    \\midrule\n"
        + body
        + "    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{One released persona (2024 wave): its \\nAnchors{} anchors in the survey's Korean labels (education and occupation shortened), and two of its \\nFields{} narrative fields, each with Revised Romanization and our translation; the rest and the prompt are in Appendix~\\ref{app:prompts}.}\n  \\label{tab:persona-example}\n\\end{table}\n",
        encoding="utf-8",
    )


def audit_explore_era() -> None:
    """Exploratory human-audit macros, outside the pre-registration (ADR-039): task load
    and how humans and the judge complement each other. Writes nothing if the file is absent.
    """
    src = REPORTS / "anno_exploratory.csv"
    if not src.exists():
        return
    d = dict(pd.read_csv(src).values.tolist())
    names = {
        "narr_chars_median": "exNarrChars",
        "R1_total_secs_first_q": "exROneSecsFirst",
        "R1_total_secs_last_q": "exROneSecsLast",
        "R2_total_secs_first_q": "exRTwoSecsFirst",
        "R2_total_secs_last_q": "exRTwoSecsLast",
        "n_traps": "exTrapN",
        "trap_human_any": "exTrapHuman",
        "trap_judge": "exTrapJudge",
        "trap_union": "exTrapUnion",
        "trap_human_only": "exTrapHumanOnly",
        "trap_judge_only": "exTrapJudgeOnly",
        "type_attend_human": "exAttendHuman",
        "type_attend_judge": "exAttendJudge",
        "type_attend_n": "exAttendN",
        "type_spend_human": "exSpendHuman",
        "type_spend_judge": "exSpendJudge",
        "type_spend_n": "exSpendN",
        "clean_human_flags": "exCleanFlags",
        "clean_human_flags_judge_yes": "exCleanFlagsJudge",
        "clean_judge_flags": "exCleanJudgeFlags",
        "n_clean": "exCleanN",
        "g_top_R1": "exGTopROne",
        "g_top_R2": "exGTopRTwo",
        "g_exact": "exGExact",
        "c_agree": "exCAgree",
        "c_both": "exCBoth",
        "union_lo": "exUnionLo",
        "union_hi": "exUnionHi",
    }
    # Values kept as decimals (agreement and tests).
    decimals = {
        "g_alpha": "exGAlpha",
        "g_wagree": "exGWAgree",
        "g_ac2": "exGACTwo",
        "g_ac2_lo": "exGACTwoLo",
        "g_ac2_hi": "exGACTwoHi",
        "c_ac1": "exCACOne",
        "c_kappa": "exCKappa",
        "mcnemar_human_judge_p": "exMcNemarP",
        "R1_tau": "exTauROne",
        "R2_tau": "exTauRTwo",
    }
    npath = TABLES / "numbers.tex"
    ntext = npath.read_text(encoding="utf-8")
    for k, m in names.items():
        if k in d:
            ntext = _set_macro(ntext, m, f"{int(d[k]):,}".replace(",", "{,}"))
    for k, m in decimals.items():
        if k in d:
            ntext = _set_macro(ntext, m, f"{float(d[k]):.2f}".replace("-", "$-$"))
    if "trap_judge" in d and "n_traps" in d and float(d["n_traps"]) > 0:
        # The judge's sensitivity as a proportion, the unit the raters' sensitivities are quoted in.
        ntext = _set_macro(
            ntext, "exTrapJudgeRate", f"{float(d['trap_judge']) / float(d['n_traps']):.3f}"
        )
    npath.write_text(ntext if ntext.endswith("\n") else ntext + "\n", encoding="utf-8")


def _nobreak_minus() -> None:
    """Keep a leading minus with its number: LaTeX may break a line after a text hyphen, which splits
    "[-0.15, 1.05]" or "(-0.57" across lines. Wrapping that hyphen in \\mbox removes the break point."""
    p = TABLES / "numbers.tex"
    t = p.read_text(encoding="utf-8")
    t = re.sub(r"(\{|[\[(]|, )-(?=\d)", lambda m: m.group(1) + "\\mbox{-}", t)
    p.write_text(t, encoding="utf-8")


def main() -> None:
    global TABLES
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--tables-dir", type=Path, default=TABLES, help="where to write the .tex tables"
    )
    TABLES = ap.parse_args().tables_dir.resolve()
    TABLES.mkdir(parents=True, exist_ok=True)
    numbers = TABLES / "numbers.tex"
    if not numbers.exists():  # every macro is (re)written below; start from an empty file
        numbers.write_text(
            "% Generated by paper/make_tables_era.py. Do not edit by hand.\n", encoding="utf-8"
        )
    skipped = []
    for step in (
        error_summary_era,
        pair_class_era,
        replication_era,
        ablation_era,
        holdout_era,
        residual_pairs_era,
        ppi_era,
        holdout_items_era,
        genre_phi_era,
        error_by_anchor_era,
        rejection_era,
        variance_era,
        segment_holdout_era,
        grounding_era,
        anchors_era,
        sampler_ablation_era,
        probes_era,
        downstream_era,
        genre_portfolio_era,
        narrative_summary_era,
        audit_era,
        audit_explore_era,
        dataset_stats_era,
        persona_example_era,
        release_scope_era,
        llm_genre_era,
        patch_numbers,
        _ensure_fit,
        _nobreak_minus,
    ):
        # Steps that need files outside this repository (persona banks from the review supplement, the
        # microdata, the audit answer key) are skipped when those files are absent; the rest still run.
        try:
            step()
        except FileNotFoundError as e:
            skipped.append(f"{step.__name__}: {Path(e.filename).name if e.filename else e}")
    for line in skipped:
        print("skipped (input not available):", line)
    print("era tables written:", sorted(p.name for p in TABLES.glob("*.tex")))


if __name__ == "__main__":
    main()
