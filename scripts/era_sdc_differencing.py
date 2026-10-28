"""Differencing attack on public grounding tables — can a suppressed cell be recovered
as a sum of public cells.

The public rule was only "drop unweighted cells < cell_floor". But conditional
tables (T2–T5) sum shares to 1 within a condition row, and the pair-joint table
(T6) has row and column sums equal to the T1 margin. If a row (column) has only
one suppressed cell, that value comes out exactly by subtraction — the classic
defect of primary suppression with no complementary (secondary) suppression.

Two checks.
  exact   Is a suppressed cell **uniquely** determined by linear constraints
          (row sums, column sums, margins) alone — a rank check. If e_i lies in
          the row space of constraint matrix A, cell i is recovered exactly.
  bounds  Even if not unique, the width of the feasible interval from unknown
          cells ≥ 0 and the fact that "suppressed cells are below floor".
          (LP; if scipy is present. A width narrower than floor/n is recovery in practice.)

Only public files are read — the same input an attacker sees. The internal
tables (`suppressed` flag) are used only to score the truth.

Run: uv run python scripts/era_sdc_differencing.py [--wave 2024]
Outputs: outputs/reports/era_sdc_differencing.csv · .md (aggregates only)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.sdc import COND_BY, _build_anchor_system, determined

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sdc_differencing")


def _determined(a: np.ndarray, n_unknown: int) -> np.ndarray:
    """Whether each unknown is uniquely determined in constraint matrix a (m × n_unknown) — e_i ∈ rowspace(a)."""
    if n_unknown == 0 or a.size == 0:
        return np.zeros(n_unknown, dtype=bool)
    r = np.linalg.matrix_rank(a)
    out = np.zeros(n_unknown, dtype=bool)
    for i in range(n_unknown):
        e = np.zeros((1, n_unknown))
        e[0, i] = 1.0
        out[i] = np.linalg.matrix_rank(np.vstack([a, e])) == r
    return out


def _bounds(a: np.ndarray, b: np.ndarray, upper: np.ndarray) -> np.ndarray | None:
    """Feasible interval width of each unknown (0 ≤ x ≤ upper, a x = b). None if scipy is absent."""
    try:
        from scipy.optimize import linprog
    except ImportError:  # pragma: no cover
        return None
    n = a.shape[1]
    widths = np.full(n, np.nan)
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1.0
        lo = linprog(c, A_eq=a, b_eq=b, bounds=list(zip([0.0] * n, upper, strict=True)))
        hi = linprog(-c, A_eq=a, b_eq=b, bounds=list(zip([0.0] * n, upper, strict=True)))
        if lo.success and hi.success:
            widths[i] = float(-hi.fun - lo.fun)
    return widths


def check_conditional(wave: int, floor: int) -> list[dict]:
    """Conditional tables T1–T5 as a per-anchor linear system (row sums + cross-table aggregation) — `sdc._build_anchor_system`.

    Attacker view: the value list (survey categories) is known, and every cell
    missing from the public table is an unknown. Intervals come from two models —
    `bound_width_share` uses only unknowns ≥ 0; `bound_width_primary` also uses
    "every cell absent from the public table is below floor" (true under primary
    suppression alone).
    """
    era = settings.era_grounding_dir(wave)
    full = {t: pd.read_parquet(era / f"{t}.parquet") for t in COND_BY}
    pub = {t: pd.read_parquet(era / f"{t}_release.parquet") for t in COND_BY}
    rows = []
    anchors = sorted(set(full["T1"].anchor.astype(str)))
    for anchor in anchors:
        hidden: set[tuple] = set()
        meta: dict[tuple, tuple[int, int, bool]] = {}
        for t, by in COND_BY.items():
            d = full[t][full[t].anchor == anchor]
            p_ = pub[t][pub[t].anchor == anchor]
            seen = {(tuple(str(getattr(r, c)) for c in by), str(r.value)) for r in p_.itertuples()}
            for r in d.itertuples():
                cond = tuple(str(getattr(r, c)) for c in by)
                if (cond, str(r.value)) not in seen:
                    k = (t, cond, str(r.value))
                    hidden.add(k)
                    meta[k] = (int(r.n_unweighted), int(r.n_base), bool(r.suppressed))
        if not hidden:
            continue
        sys_ = _build_anchor_system(full, anchor, hidden)
        a = sys_.matrix()
        det = determined(a)
        # Interval: the right-hand side is built from actual values (public share and internal share)
        share = {}
        for t, by in COND_BY.items():
            for r in full[t][full[t].anchor == anchor].itertuples():
                share[(t, tuple(str(getattr(r, c)) for c in by), str(r.value))] = float(r.share)
        x_true = np.array([share[k] for k in sys_.unknowns])
        b = a @ x_true
        w_free = _bounds(a, b, np.ones(len(sys_.unknowns)))
        w_prim = _bounds(a, b, np.array([floor / meta[k][1] for k in sys_.unknowns]))
        for k in hidden:
            if not meta[k][2]:
                continue  # unknowns that are not primary suppression (complementary, if any) are not scored
            j = sys_.index.get(k)
            rows.append(
                {
                    "wave": wave,
                    "table": k[0],
                    "anchor": anchor,
                    "condition": "|".join(k[1]),
                    "value": k[2],
                    "n_unweighted": meta[k][0],
                    "exact": bool(det[j]) if j is not None else False,
                    "bound_width_share": float(w_free[j])
                    if w_free is not None and j is not None
                    else np.nan,
                    "bound_width_primary": float(w_prim[j])
                    if w_prim is not None and j is not None
                    else np.nan,
                    "n_base": meta[k][1],
                }
            )
    return rows


def check_pairs(wave: int, floor: int, primary_only: bool = False) -> list[dict]:
    """T6 pair joint: unknowns = (value_a, value_b) cells present internally and absent from the public table (zero combinations are treated as known).
    Constraints = each value_a row sum equals the T1 margin, each value_b column sum equals the T1 margin (published values only).
    """
    era = settings.era_grounding_dir(wave)
    full = pd.read_parquet(era / "T6_pair_joints.parquet")
    pub = pd.read_parquet(era / "T6_pair_joints_release.parquet")
    if primary_only:
        # The table as it would be published with primary suppression alone (cells below the floor dropped,
        # no complementary suppression): the "before" condition of the check.
        pub = full[~full["suppressed"].astype(bool)][pub.columns.intersection(full.columns)]
    t1 = pd.read_parquet(era / "T1_release.parquet")
    marg = {(r.anchor, r.value): float(r.share) for r in t1.itertuples()}
    n_resp = int(t1.n_base.iloc[0])
    rows = []
    for (a_name, b_name), g in full.groupby(["anchor_a", "anchor_b"], observed=True):
        gp = pub[(pub.anchor_a == a_name) & (pub.anchor_b == b_name)]
        vals_a = [v for v in t1[t1.anchor == a_name].value.tolist()]
        vals_b = [v for v in t1[t1.anchor == b_name].value.tolist()]
        known = {(r.value_a, r.value_b): float(r.joint) for r in gp.itertuples()}
        # Conservative attacker: combinations with zero respondents are known from the instrument → unknowns are cells in the internal table and absent from the public table
        present = {(r.value_a, r.value_b) for r in g.itertuples()}
        unknown = [
            (va, vb)
            for va in vals_a
            for vb in vals_b
            if (va, vb) not in known and (va, vb) in present
        ]
        if not unknown:
            continue
        idx = {c: i for i, c in enumerate(unknown)}
        cons, rhs = [], []
        for va in vals_a:
            row = np.zeros(len(unknown))
            for vb in vals_b:
                if (va, vb) in idx:
                    row[idx[(va, vb)]] = 1.0
            if row.any():
                cons.append(row)
                rhs.append(marg[(a_name, va)] - sum(known.get((va, vb), 0.0) for vb in vals_b))
        for vb in vals_b:
            row = np.zeros(len(unknown))
            for va in vals_a:
                if (va, vb) in idx:
                    row[idx[(va, vb)]] = 1.0
            if row.any():
                cons.append(row)
                rhs.append(marg[(b_name, vb)] - sum(known.get((va, vb), 0.0) for va in vals_a))
        a = np.array(cons)
        b = np.array(rhs)
        det = _determined(a, len(unknown))
        w_free = _bounds(a, b, np.ones(len(unknown)))
        w_prim = _bounds(a, b, np.full(len(unknown), floor / n_resp))
        sup = g[g.suppressed.astype(bool)].set_index(["value_a", "value_b"])
        for cell, j in idx.items():
            if cell not in sup.index:
                continue  # actually a zero combination, not a suppression — nothing to recover
            r = sup.loc[cell]
            rows.append(
                {
                    "wave": wave,
                    "table": "T6",
                    "anchor": f"{a_name}×{b_name}",
                    "condition": str(cell[0]),
                    "value": str(cell[1]),
                    "n_unweighted": int(r.n_unweighted),
                    "exact": bool(det[j]),
                    "bound_width_share": float(w_free[j]) if w_free is not None else np.nan,
                    "bound_width_primary": float(w_prim[j]) if w_prim is not None else np.nan,
                    "n_base": n_resp,
                }
            )
    return rows


def spec_census(waves: list[int], floor: int) -> list[str]:
    """Value cells in the sampler spec: the parent-combination floor (n ≥ floor) holds, but a value cell can be under 30.
    Recount unweighted (parent combination, value) on the frame and report the share of spec value cells below floor."""
    from sfac_persona.grounding.chain import add_derived_columns
    from sfac_persona.grounding.official import build_official_anchor_frame

    lines = [
        "## 샘플러 스펙 값 셀 (부모 조합 하한만 적용됨)",
        "",
        "| wave | 값 셀 | floor 미만 | 그 질량 합 | 2×2 장르 조건부 셀 | 그중 floor 미만 |",
        "|---|---|---|---|---|---|",
    ]
    rows_out = []
    for wave in waves:
        spec_path = settings.era_grounding_dir(wave) / "sampler_specs_structure_T.json"
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        frame = add_derived_columns(build_official_anchor_frame(wave))
        total = below = 0
        mass_below = 0.0
        g_total = g_below = 0
        genre_names = {n["genre"] for n in spec.get("genre_tree", [])}
        for node in spec["nodes"]:
            if "derived" in node:
                continue
            for level in node["levels"]:
                ps = level["parents"]
                if not ps:
                    continue
                sub = frame.dropna(subset=[node["name"]])
                for key, entry in level["table"].items():
                    vals = key.split("|")
                    m = np.ones(len(sub), dtype=bool)
                    for p_, v in zip(ps, vals, strict=True):
                        m &= sub[p_].astype(str).to_numpy() == v
                    grp = sub[m]
                    counts = grp.groupby(node["name"], observed=True).size()
                    for v, w_ in zip(entry["values"], entry["weights"], strict=True):
                        n = int(counts.get(v, 0))
                        total += 1
                        is_genre = node["name"] in genre_names
                        if is_genre:
                            g_total += 1
                        if n < floor:
                            below += 1
                            mass_below += float(w_) / max(len(level["table"]), 1)
                            if is_genre:
                                g_below += 1
        rows_out.append((wave, total, below, g_total, g_below))
        lines.append(
            f"| {wave} | {total} | {below} ({100 * below / max(total, 1):.0f}%) | — | {g_total} | {g_below} ({100 * g_below / max(g_total, 1):.0f}%) |"
        )
    pd.DataFrame(
        rows_out,
        columns=[
            "wave",
            "spec_cells",
            "spec_cells_below_floor",
            "genre_cells",
            "genre_cells_below_floor",
        ],
    ).to_csv(settings.outputs_dir / "reports" / "era_sdc_spec_census.csv", index=False)
    lines.append("")
    lines.append(
        "값 셀 = 스펙의 (부모 조합, 값) 항목. 부모 조합의 n 은 floor 이상이지만 값 하나의 비가중 수는 그보다 작을 수 있다."
    )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, action="append", default=None)
    ap.add_argument("--tag", default="", help="output filename suffix (e.g. primary_only)")
    ap.add_argument(
        "--primary-only",
        action="store_true",
        help="attack the pair joint as published without complementary suppression (use with --tag primary_only)",
    )
    args = ap.parse_args()
    waves = args.wave or [2024, 2022, 2020]
    floor = settings.cell_floor
    rows: list[dict] = []
    for wave in waves:
        rows += check_conditional(wave, floor)
        rows += check_pairs(wave, floor, primary_only=args.primary_only)
        log.info("wave %s done", wave)
    d = pd.DataFrame(rows)
    out = settings.outputs_dir / "reports"
    tag = f"_{args.tag}" if args.tag else ""
    d.to_csv(out / f"era_sdc_differencing{tag}.csv", index=False)
    # Interval width below half of floor (in respondent counts): unknowns ≥ 0 only (free) / primary suppression assumed (primary)
    d["narrow"] = d.bound_width_share * d.n_base < floor / 2
    d["narrow_primary"] = d.bound_width_primary * d.n_base < floor / 2
    summ = (
        d.groupby(["wave", "table"])
        .agg(
            suppressed=("exact", "size"),
            exact=("exact", "sum"),
            narrow=("narrow", "sum"),
            narrow_primary=("narrow_primary", "sum"),
            min_n_exact=(
                "n_unweighted",
                lambda s: (
                    int(s[d.loc[s.index, "exact"]].min()) if d.loc[s.index, "exact"].any() else -1
                ),
            ),
        )
        .reset_index()
    )
    lines = [
        "# 공개 근거표 차분 공격 점검",
        "",
        f"> 자동 생성: `scripts/era_sdc_differencing.py`. cell_floor={floor}. 공개판 파일만 입력으로 쓴다.",
        "",
        "| wave | table | 억제 셀 | 정확 복원 | 폭 < floor/2 (≥0 만) | 폭 < floor/2 (1 차만 가정) | 복원된 최소 n |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summ.itertuples(index=False):
        lines.append(
            f"| {r.wave} | {r.table} | {r.suppressed} | {int(r.exact)} | {int(r.narrow)} | "
            f"{int(r.narrow_primary)} | {'-' if r.min_n_exact < 0 else r.min_n_exact} |"
        )
    lines += [
        "",
        "`정확 복원` 은 행합·열합·주변분포의 선형 제약만으로 억제 셀 값이 유일하게 결정되는 수 (계수 검사).",
        "`구간 폭` 은 미지 셀 ≥ 0, 억제 셀 < floor 를 더한 LP 실현 가능 구간의 폭이 응답자 수로 floor/2 미만인 수 —",
        "정확하지 않아도 사실상 알려지는 셀이다. 보완 억제(2 차 억제)가 필요한지는 이 표가 결정한다.",
    ]
    spec_lines = spec_census(waves, floor)
    lines += ["", *spec_lines]
    (out / f"era_sdc_differencing{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
