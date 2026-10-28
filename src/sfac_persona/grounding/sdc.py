"""Statistical disclosure control — secondary (complementary) suppression of public
grounding tables.

Primary suppression (drop unweighted cells < cell_floor) is not enough: a row of a
conditional table sums to share 1, conditional tables are tied by aggregation
identities such as Σ_a P(a) P(v|a) = P(v), and row and column sums of a pair-joint
equal the marginals. If a suppressed cell is uniquely determined by those linear
constraints, deleting it did nothing. Treat the tables as a linear system:

  determination   if e_i is in the row space of constraint matrix A, unknown i is
                  determined (a rank check).
  secondary       until no primary-suppressed cell is determined, also delete, among
  suppression     the published cells that share a constraint with it, the one with
                  the smallest unweighted n. Deleted cells are marked `secondary=True`.

Split into two stages (the global system is too large).
  1) Per-anchor system: every cell of that anchor in T1 · T2 (age) · T3 (sex) ·
     T4 (living area) · T5 (age × sex), plus row-sum and aggregation constraints.
  2) Per-pair system: one anchor pair in T6 plus row and column sums of the T1
     marginal (published values only).
Take the attacker's view: a combination absent from the public release might be
suppressed or zero, so both are unknowns. Approximate P(a), P(s), P(a,s), P(area)
are treated as known (they are almost read off T1 and the root joint — a conservative
assumption).

`scripts/era_sdc_differencing.py` scores the public release with the same check.

**Scope (ADR-024, 2026-09-23).** The public path actually uses only
`secondary_suppress_pairs` (T6). `secondary_suppress_conditionals` and
`secondary_suppress_root_joint` are diagnostics of how secondary suppression
cascades on linked tables (2024: T2 529→192 cells, T4 687→219) — that cost is why
the T1–T5 floor stays a reliability rule, and the number of cells that come back is
reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COND_BY: dict[str, list[str]] = {
    "T1": [],
    "T2": ["age_band"],
    "T3": ["sex"],
    "T4": ["living_area"],
    "T5": ["age_band", "sex"],
}


@dataclass
class LinearSystem:
    """Unknowns (cells) and linear constraints. Only coefficients are kept, so the
    right-hand side is not carried.
    """

    unknowns: list[tuple] = field(default_factory=list)
    index: dict[tuple, int] = field(default_factory=dict)
    rows: list[dict[int, float]] = field(default_factory=list)

    def var(self, key: tuple) -> int:
        if key not in self.index:
            self.index[key] = len(self.unknowns)
            self.unknowns.append(key)
        return self.index[key]

    def matrix(self) -> np.ndarray:
        a = np.zeros((len(self.rows), len(self.unknowns)))
        for r, row in enumerate(self.rows):
            for j, c in row.items():
                a[r, j] = c
        return a


def determined(a: np.ndarray) -> np.ndarray:
    """Whether each column (unknown) is uniquely determined by the row space.
    An empty system is all False.
    """
    n = a.shape[1] if a.ndim == 2 else 0
    if n == 0 or a.size == 0:
        return np.zeros(n, dtype=bool)
    # Orthogonal projection: e_i ∈ rowspace(A) ⇔ ‖(I − A⁺A) e_i‖ = 0
    proj = np.linalg.pinv(a) @ a
    resid = np.abs(np.diag(proj) - 1.0)
    return resid < 1e-8


# ----------------------------------------------------------------------------
# 1) Per-anchor system (T1–T5)
# ----------------------------------------------------------------------------


def _cond_key(table: str, cond: tuple, value: str) -> tuple:
    return (table, cond, value)


def _build_anchor_system(
    tables: dict[str, pd.DataFrame], anchor: str, hidden: set[tuple]
) -> LinearSystem:
    """Hidden cells are unknowns. Build every row-sum and aggregation constraint for that anchor."""
    sys_ = LinearSystem()
    cells: dict[str, pd.DataFrame] = {}
    for t, by in COND_BY.items():
        if t not in tables:
            continue
        d = tables[t]
        d = d[d.anchor == anchor]
        cells[t] = d

    # Row sum: values inside the same (table, condition) sum to 1
    for t, d in cells.items():
        by = COND_BY[t]
        for cond, g in d.groupby(by, observed=True) if by else [((), d)]:
            cond = cond if isinstance(cond, tuple) else (cond,)
            row = {}
            for v in g.value.astype(str):
                k = _cond_key(t, cond, v)
                if k in hidden:
                    row[sys_.var(k)] = 1.0
            if row:
                sys_.rows.append(row)

    # Aggregation: collapsing a fine table's conditions onto a coarse table's
    # conditions yields the same value — Σ_fine P(fine|coarse) P(v|fine) = P(v|coarse).
    # The coefficient P(fine|coarse) is treated as approximately known. Only unknowns
    # remain, so the coefficient values do not affect determination (unless they are 0);
    # only the structure of the system matters — here the unweighted n_base shares are used.
    def base(t: str, cond: tuple) -> float:
        d = cells[t]
        by = COND_BY[t]
        if not by:
            return float(d.n_base.iloc[0])
        m = np.ones(len(d), dtype=bool)
        for c, v in zip(by, cond, strict=True):
            m &= d[c].astype(str).to_numpy() == v
        return float(d[m].n_base.iloc[0]) if m.any() else 0.0

    links = [("T2", "T1"), ("T3", "T1"), ("T4", "T1"), ("T5", "T1"), ("T5", "T2"), ("T5", "T3")]
    for fine, coarse in links:
        if fine not in cells or coarse not in cells:
            continue
        fby, cby = COND_BY[fine], COND_BY[coarse]
        values = sorted(set(cells[fine].value.astype(str)) | set(cells[coarse].value.astype(str)))
        fine_conds = {
            (c if isinstance(c, tuple) else (c,))
            for c, _ in cells[fine].groupby(fby, observed=True)
        }
        coarse_conds = (
            {
                (c if isinstance(c, tuple) else (c,))
                for c, _ in cells[coarse].groupby(cby, observed=True)
            }
            if cby
            else {()}
        )
        for cc in coarse_conds:
            cbase = base(coarse, cc)
            if cbase <= 0:
                continue
            fine_in = [
                fc
                for fc in fine_conds
                if all(fc[fby.index(c)] == v for c, v in zip(cby, cc, strict=True))
            ]
            for v in values:
                row = {}
                for fc in fine_in:
                    k = _cond_key(fine, fc, v)
                    if k in hidden:
                        row[sys_.var(k)] = base(fine, fc) / cbase
                kc = _cond_key(coarse, cc, v)
                if kc in hidden:
                    row[sys_.var(kc)] = row.get(sys_.var(kc), 0.0) - 1.0
                if row:
                    sys_.rows.append(row)
    return sys_


def secondary_suppress_conditionals(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return T1–T5 with a `secondary` column. Delete until no primary cell is determined."""
    out = {t: d.copy() for t, d in tables.items()}
    for d in out.values():
        d["secondary"] = False
    anchors = sorted(set().union(*(set(d.anchor.astype(str)) for d in out.values())))
    for anchor in anchors:
        primary: set[tuple] = set()
        released: dict[tuple, int] = {}
        for t, by in COND_BY.items():
            if t not in out:
                continue
            d = out[t][out[t].anchor == anchor]
            for r in d.itertuples(index=False):
                cond = tuple(str(getattr(r, c)) for c in by)
                k = _cond_key(t, cond, str(r.value))
                if bool(r.suppressed):
                    primary.add(k)
                else:
                    released[k] = int(r.n_unweighted)
        if not primary:
            continue
        hidden = set(primary)
        for _ in range(10_000):
            sys_ = _build_anchor_system(out, anchor, hidden)
            det = determined(sys_.matrix())
            bad = [k for k in primary if k in sys_.index and det[sys_.index[k]]]
            if not bad:
                break
            k = bad[0]
            # Published-cell candidates that share a constraint with this cell: same row
            # (table, condition) first, else the same value in another table tied by aggregation
            same_row = [c for c in released if c[0] == k[0] and c[1] == k[1] and c not in hidden]
            linked = [c for c in released if c[2] == k[2] and c not in hidden and c[0] != k[0]]
            pool = same_row or linked
            if not pool:
                logger.warning("%s %s: no secondary-suppression candidate — left as is", anchor, k)
                primary.discard(k)
                continue
            pick = min(pool, key=lambda c: released[c])
            hidden.add(pick)
        for k in hidden - primary:
            t, cond, v = k
            d = out[t]
            m = (d.anchor == anchor) & (d.value.astype(str) == v)
            for c, cv in zip(COND_BY[t], cond, strict=True):
                m &= d[c].astype(str) == cv
            out[t].loc[m, "secondary"] = True
    return out


def _root_parts(value: str) -> tuple[str, str, str]:
    sex, age, area = str(value).split("|")
    return sex, age, area


def secondary_suppress_root_joint(rj: pd.DataFrame) -> pd.DataFrame:
    """Root joint (sex|age|living area). Published n_base makes the (age × sex) row
    sums and the living-area column sums known (T5, T4).
    """
    out = rj.copy()
    out["secondary"] = False
    cells = {str(r.value): (bool(r.suppressed), int(r.n_unweighted)) for r in out.itertuples()}
    primary = {v for v, (sup, _) in cells.items() if sup}
    if not primary:
        return out
    released = {v: n for v, (sup, n) in cells.items() if not sup}
    hidden = set(primary)
    for _ in range(10_000):
        sys_ = LinearSystem()
        groups: dict[tuple, dict[int, float]] = {}
        for v in hidden:
            sex, age, area = _root_parts(v)
            groups.setdefault(("row", age, sex), {})[sys_.var(v)] = 1.0
            groups.setdefault(("col", area), {})[sys_.var(v)] = 1.0
        sys_.rows.extend(groups.values())
        det = determined(sys_.matrix())
        bad = [v for v in primary if v in sys_.index and det[sys_.index[v]]]
        if not bad:
            break
        sex, age, area = _root_parts(bad[0])
        pool = [
            c
            for c in released
            if c not in hidden
            and (
                _root_parts(c)[1:] == (age, area)
                or _root_parts(c)[:2] == (sex, age)
                or _root_parts(c)[2] == area
            )
        ]
        if not pool:
            logger.warning("root %s: no secondary-suppression candidate", bad[0])
            primary.discard(bad[0])
            continue
        hidden.add(min(pool, key=lambda c: released[c]))
    out["secondary"] = out.value.astype(str).isin(hidden - primary)
    return out


# ----------------------------------------------------------------------------
# 2) Per-pair system (T6)
# ----------------------------------------------------------------------------


def _pair_system(hidden: set[tuple], va_all: list[str], vb_all: list[str]) -> LinearSystem:
    sys_ = LinearSystem()
    for va in va_all:
        row = {sys_.var((va, vb)): 1.0 for vb in vb_all if (va, vb) in hidden}
        if row:
            sys_.rows.append(row)
    for vb in vb_all:
        row = {sys_.var((va, vb)): 1.0 for va in va_all if (va, vb) in hidden}
        if row:
            sys_.rows.append(row)
    return sys_


def _n_determined(hidden: set[tuple], primary: set[tuple], va: list[str], vb: list[str]) -> int:
    sys_ = _pair_system(hidden, va, vb)
    det = determined(sys_.matrix())
    return sum(1 for k in primary if k in sys_.index and det[sys_.index[k]])


def secondary_suppress_pairs(pairs: pd.DataFrame, t1_release: pd.DataFrame) -> pd.DataFrame:
    """Return T6 with a `secondary` column. Constraints are only row and column sums
    of published marginal values.

    Attacker model (conservative): combinations with zero respondents are treated as
    known from the survey structure and removed from the unknowns — unknowns are only
    cells that are in the internal table and absent from the public release. Among
    published cells that share a row or column with any hidden cell, the candidate is
    the one that, after deletion, leaves the **fewest determined primary cells**
    (ties broken by smaller n) — this blocks the greedy path that deletes one row at
    a time until column sums restore everything.
    """
    out = pairs.copy()
    out["secondary"] = False
    vals = {a: g.value.astype(str).tolist() for a, g in t1_release.groupby("anchor")}
    for (a_name, b_name), g in out.groupby(["anchor_a", "anchor_b"], observed=True):
        va_all, vb_all = vals.get(a_name, []), vals.get(b_name, [])
        primary = {(str(r.value_a), str(r.value_b)) for r in g.itertuples() if bool(r.suppressed)}
        if not primary:
            continue
        released = {
            (str(r.value_a), str(r.value_b)): int(r.n_unweighted)
            for r in g.itertuples()
            if not bool(r.suppressed)
        }
        hidden = set(primary)
        for _ in range(10_000):
            sys_ = _pair_system(hidden, va_all, vb_all)
            det = determined(sys_.matrix())
            bad = [k for k in primary if k in sys_.index and det[sys_.index[k]]]
            if not bad:
                break
            va, vb = bad[0]
            # Candidates are published cells that share a row or column with any hidden
            # cell — a cycle closes through the hidden cells, so looking only at the
            # determined cell's row and column never closes on a 2×k table.
            pool = [
                c
                for c in released
                if c not in hidden and any(c[0] == h[0] or c[1] == h[1] for h in hidden)
            ]
            if not pool:
                logger.warning(
                    "%s×%s %s: no secondary-suppression candidate", a_name, b_name, (va, vb)
                )
                primary.discard((va, vb))
                continue
            scored = sorted(
                (_n_determined(hidden | {c}, primary, va_all, vb_all), released[c], c) for c in pool
            )
            hidden.add(scored[0][2])
        sec = [c for c in hidden if c in released]
        if sec:
            m = (out.anchor_a == a_name) & (out.anchor_b == b_name)
            key = list(zip(out.loc[m, "value_a"].astype(str), out.loc[m, "value_b"].astype(str)))
            out.loc[m, "secondary"] = [k in sec for k in key]
    return out
