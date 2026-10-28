"""Survey-design pretest out of sample, with the released-table baseline rebuilt on the fitting half.

era_design_pretest.py and era_design_deep.py score every method in sample, and era_chain_crossfit.py scores
the bank and the age table out of sample but not the pair-table naive Bayes, whose tables were built from all
respondents. This script scores all four methods out of sample on the same cells: respondents are split at
random into halves A and B (the same seeds as era_chain_crossfit.py), the released tables (T1--T4, T6 with
the cell floor and complementary suppression) and the chain are built from A alone, and every method predicts
the attendance rate of B's cells.

Design fixed before running (2026-10-09):
  cells       the two-way and one-way crossings of era_design_pretest.py (printed and unprinted) and the
              three-way crossings of era_design_deep.py, keeping cells with at least 30 respondents in B
  methods     marginal rate (A); age-band table (A); naive Bayes over A's released tables; the chain built on A
  primary     2024, unprinted crossings: mean absolute error (pp) of the chain minus that of the pair tables,
              averaged over three splits, with a bootstrap over cells within each split (2,000 draws)
  secondary   printed axes and three-way crossings; 2022 and 2020
Addition fixed before running (2026-10-10): --resample-respondents R draws R bootstrap samples of the scoring
  half within every split and recomputes every cell's observed rate and each method's error on them; the
  interval of the chain-minus-pair-tables difference then pools respondent resampling with the split-to-split
  spread (default 20 splits), instead of resampling cells within three splits.
Addition fixed before running (2026-10-10, second): --variant tables_only builds, on the fitting half, the chain a
  builder holding only the printed report could build (era_sampler_audit.variant_nodes: age-band parents, the
  attendance block on attendance status and age band, no genre tree) in place of the released chain; the other
  three methods and every seed are unchanged, so the two runs are compared on the same cells and draws.
Every result is reported, whatever its sign.

Run: uv run python scripts/era_design_oos_pairs.py [--wave 2024] [--splits 3]
Output: outputs/reports/era_design_oos_pairs.csv (summary) · era_design_oos_pairs_cells.csv (cell errors)
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era_chain_crossfit import SEED
from era_design_deep import ANCHORS as DEEP_ANCHORS
from era_design_pretest import CELL_KEYS, MIN_CELL, PUBLISHED_KEYS, cell_rates
from era_downstream_baselines import attendance_probability
from era_genre_portfolio import ROOT_COND
from era_sampler_audit import build_variant

from sfac_persona.config import settings
from sfac_persona.grounding import chain as C
from sfac_persona.grounding.era_tables import _build_survey_tables_core
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.grounding.sdc import secondary_suppress_pairs
from sfac_persona.grounding.tables import build_pair_joints
from sfac_persona.schema import ERA_CORE_ANCHORS, ROOT_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("design_oos_pairs")

OUTCOME = "attend_any"
N_DRAW = 30000
N_BOOT = 2000
THREE_WAY = tuple(itertools.combinations(DEEP_ANCHORS, 3))


@dataclass
class HalfTables:
    """The released tables of one half of the respondents, in the form the naive-Bayes baseline reads."""

    marg: dict = field(default_factory=dict)
    joint: dict = field(default_factory=dict)
    cond: dict = field(default_factory=dict)


def released_tables(frame: pd.DataFrame) -> HalfTables:
    anchors = [a for a in ERA_CORE_ANCHORS if a in frame.columns]
    core = _build_survey_tables_core(frame, anchors)
    t1 = core["T1"][~core["T1"].suppressed.astype(bool)]
    t6 = build_pair_joints(frame, [a for a in anchors if a not in ROOT_ANCHORS])
    t6 = secondary_suppress_pairs(t6, t1)
    t6 = t6[~(t6.suppressed.astype(bool) | t6.secondary.astype(bool))]
    ht = HalfTables()
    ht.marg = {(r.anchor, str(r.value)): float(r.share) for r in t1.itertuples()}
    for r in t6.itertuples():
        ht.joint[(r.anchor_a, str(r.value_a), r.anchor_b, str(r.value_b))] = float(r.joint)
        ht.joint[(r.anchor_b, str(r.value_b), r.anchor_a, str(r.value_a))] = float(r.joint)
    for root, t in ROOT_COND.items():
        d = core[t][~core[t].suppressed.astype(bool)]
        for r in d.itertuples():
            ht.cond[(r.anchor, str(r.value), root, str(getattr(r, root)))] = float(r.share)
    return ht


def score_split(
    frame: pd.DataFrame, k: int, wave: int, resample: int = 0, variant: str = "released"
) -> list[dict]:
    rng = np.random.default_rng(SEED + k)
    mask = rng.random(len(frame)) < 0.5
    fa, fb = frame[mask].reset_index(drop=True), frame[~mask].reset_index(drop=True)
    tables = released_tables(fa)
    spec = build_variant(fa, variant)
    syn = C.ChainSampler(spec).sample(N_DRAW, seed=SEED + k)
    marginal = float((fa[OUTCOME].astype(str) == "예").astype(float).mul(fa.w).sum() / fa.w.sum())
    by_age = cell_rates(fa, ["age_band"], OUTCOME, "w")
    rows = []
    groups = [(k_, "printed" if k_ in PUBLISHED_KEYS else "unprinted") for k_ in CELL_KEYS]
    groups += [(k_, "three-way") for k_ in THREE_WAY]
    # draw 0 is the observed scoring half; draws 1..resample are bootstrap samples of its respondents
    brng = np.random.default_rng(SEED + 1000 + k)
    draws = [fb] + [
        fb.iloc[brng.integers(0, len(fb), len(fb))].reset_index(drop=True) for _ in range(resample)
    ]
    for keys, group in groups:
        kk = list(keys)
        syn_p = cell_rates(syn, kk, OUTCOME, None)
        counts0 = fb.groupby(kk).size()
        keep = set(counts0[counts0 >= MIN_CELL].index)
        for d, fb_d in enumerate(draws):
            truth = cell_rates(fb_d, kk, OUTCOME, "w")
            truth = truth[[c in keep for c in truth.index]]
            rows += _score_cells(truth, kk, group, wave, k, d, syn_p, marginal, by_age, tables)
    log.info("wave %d split %d: %d cell rows (%d draws)", wave, k, len(rows), len(draws))
    return rows


def _score_cells(truth, kk, group, wave, k, draw, syn_p, marginal, by_age, tables) -> list[dict]:
    rows = []
    if True:
        for cell, p_true in truth.items():
            cell_t = cell if isinstance(cell, tuple) else (cell,)
            values = [str(v) for v in cell_t]
            p_age = (
                float(by_age.get(values[kk.index("age_band")], marginal))
                if "age_band" in kk
                else marginal
            )
            est = {
                "marginal": marginal,
                "age_table": p_age,
                "pair_tables": attendance_probability(tables, kk, values),
                "synthetic": float(syn_p.get(cell, np.nan)),
            }
            for method, p_hat in est.items():
                if not np.isfinite(p_hat):
                    continue
                rows.append(
                    {
                        "wave": wave,
                        "split": k,
                        "draw": draw,
                        "group": group,
                        "cell_key": "+".join(kk),
                        "cell": " / ".join(values),
                        "method": method,
                        "abs_err_pp": 100 * abs(float(p_hat) - float(p_true)),
                    }
                )
    return rows


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    resampled = "draw" in cells.columns and cells.draw.max() > 0
    for (wave, group), part in cells.groupby(["wave", "group"]):
        if resampled:
            # Respondent resampling pooled with the split spread: one mean difference per (split, draw).
            w_all = part.pivot_table(
                index=["split", "draw", "cell_key", "cell"], columns="method", values="abs_err_pp"
            ).dropna(subset=["synthetic", "pair_tables"])
            point = w_all.xs(0, level="draw")
            for m in ("marginal", "age_table", "pair_tables", "synthetic"):
                out.append(
                    {
                        "wave": wave,
                        "group": group,
                        "record": "mae_pp",
                        "method": m,
                        "value": float(point[m].mean()),
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "n_cells": len(point),
                    }
                )
            per = (
                (w_all["synthetic"] - w_all["pair_tables"]).groupby(level=["split", "draw"]).mean()
            )
            boot = per[per.index.get_level_values("draw") > 0]
            lo, hi = np.percentile(boot.to_numpy(), [2.5, 97.5])
            out.append(
                {
                    "wave": wave,
                    "group": group,
                    "record": "diff_pp",
                    "method": "synthetic-pair_tables",
                    "value": float((point["synthetic"] - point["pair_tables"]).mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "n_cells": len(point),
                }
            )
            continue
        part = part[part.draw == 0] if "draw" in part.columns else part
        wide = part.pivot_table(
            index=["split", "cell_key", "cell"], columns="method", values="abs_err_pp"
        )
        wide = wide.dropna(subset=["synthetic", "pair_tables"])
        for m in ("marginal", "age_table", "pair_tables", "synthetic"):
            out.append(
                {
                    "wave": wave,
                    "group": group,
                    "record": "mae_pp",
                    "method": m,
                    "value": float(wide[m].mean()),
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "n_cells": len(wide),
                }
            )
        diff = wide["synthetic"] - wide["pair_tables"]
        boots = []
        for _ in range(N_BOOT):
            vals = []
            for _, w_ in diff.groupby(level="split"):
                arr = w_.to_numpy()
                vals.append(float(arr[rng.integers(0, len(arr), len(arr))].mean()))
            boots.append(float(np.mean(vals)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(
            {
                "wave": wave,
                "group": group,
                "record": "diff_pp",
                "method": "synthetic-pair_tables",
                "value": float(diff.mean()),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "n_cells": len(wide),
            }
        )
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wave", type=int, action="append", default=None)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--resample-respondents", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--variant", default="released", choices=["released", "tables_only"])
    args = ap.parse_args()
    waves = args.wave or [2024, 2022, 2020]
    rows: list[dict] = []
    for wave in waves:
        frame = C.add_derived_columns(build_official_anchor_frame(wave))
        for k in range(args.splits):
            rows += score_split(frame, k, wave, args.resample_respondents, args.variant)
    cells = pd.DataFrame(rows)
    reports = settings.outputs_dir / "reports"
    cells.to_csv(reports / f"era_design_oos_pairs_cells{args.out_suffix}.csv", index=False)
    summary = summarise(cells)
    summary.to_csv(reports / f"era_design_oos_pairs{args.out_suffix}.csv", index=False)
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
