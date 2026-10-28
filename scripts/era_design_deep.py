"""Survey-design pretest on three-way crossings that no table prints.

era_design_pretest.py asks how far a planner is off when setting a subgroup's attendance rate (and hence its
sample size) before fielding a survey, on one- and two-way crossings. This script applies the same procedure,
truth and baselines to three-way crossings, where the released tables say the least.

Design fixed before running (2026-10-09):
  cells       every three-way crossing of seven demographic and household anchors (age band, sex, living area,
              education, household income, household size, occupation; 35 combinations), keeping cells with at
              least 30 unweighted respondents
  outcome     attendance status (attend_any); truth = the cell's weighted rate in the same wave's microdata
  methods     marginal rate; age-band table (when age band is in the crossing, otherwise the marginal);
              naive Bayes over the released marginal, conditional and pair-joint tables (the strongest
              table-only baseline of era_downstream_baselines.py); the shipped persona bank
  primary     2024: mean absolute error (pp) of the bank minus that of the pair tables, 95% interval from a
              bootstrap over cells (2,000 draws)
  secondary   2022 and 2020; share of cells where the bank is closer
Every result is reported, whatever its sign.

Run: uv run python scripts/era_design_deep.py
Outputs: outputs/reports/era_design_deep.csv (cell rows: rates and errors only) · era_design_deep_summary.csv
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era_design_pretest import MIN_CELL, cell_rates, synthetic_bank
from era_downstream_baselines import attendance_probability
from era_genre_portfolio import PairTables

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("design_deep")

ANCHORS = (
    "age_band",
    "sex",
    "living_area",
    "education",
    "household_income_band",
    "household_size_band",
    "occupation",
)
WAVES = (2024, 2022, 2020)
OUTCOME = "attend_any"
N_BOOT = 2000
SEED = 20261009


def run(wave: int) -> pd.DataFrame:
    real = build_official_anchor_frame(wave)
    syn = synthetic_bank(wave)
    tables = PairTables(wave)
    y = (real[OUTCOME].astype(str) == "예").astype(float)
    marginal = float((y * real.w).sum() / real.w.sum())
    by_age = cell_rates(real, ["age_band"], OUTCOME, "w")
    rows = []
    for keys in itertools.combinations(ANCHORS, 3):
        k = list(keys)
        counts = real.groupby(k).size()
        truth = cell_rates(real, k, OUTCOME, "w")
        truth = truth[counts.reindex(truth.index).fillna(0) >= MIN_CELL]
        syn_p = cell_rates(syn, k, OUTCOME, None)
        for cell, p_true in truth.items():
            values = [str(v) for v in cell]
            p_age = (
                float(by_age.get(values[k.index("age_band")], marginal))
                if "age_band" in k
                else marginal
            )
            estimates = {
                "marginal": marginal,
                "age_table": p_age,
                "pair_tables": attendance_probability(tables, k, values),
                "synthetic": float(syn_p.get(cell, np.nan)),
            }
            for method, p_hat in estimates.items():
                if not np.isfinite(p_hat):
                    continue
                rows.append(
                    {
                        "wave": wave,
                        "cell_key": "+".join(k),
                        "cell": " / ".join(values),
                        "method": method,
                        "p_true": float(p_true),
                        "p_hat": float(p_hat),
                        "abs_err_pp": 100 * abs(float(p_hat) - float(p_true)),
                    }
                )
    out = pd.DataFrame(rows)
    log.info("wave %d: %d cells", wave, out[["cell_key", "cell"]].drop_duplicates().shape[0])
    return out


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for wave, part in cells.groupby("wave"):
        wide = part.pivot_table(index=["cell_key", "cell"], columns="method", values="abs_err_pp")
        wide = wide.dropna(subset=["synthetic", "pair_tables"])
        for method in ("marginal", "age_table", "pair_tables", "synthetic"):
            rows.append(
                {
                    "wave": wave,
                    "record": "mae_pp",
                    "method": method,
                    "value": float(wide[method].mean()),
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "n_cells": len(wide),
                }
            )
        diff = (wide["synthetic"] - wide["pair_tables"]).to_numpy()
        boot = [float(diff[rng.integers(0, len(diff), len(diff))].mean()) for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(
            {
                "wave": wave,
                "record": "diff_pp",
                "method": "synthetic-pair_tables",
                "value": float(diff.mean()),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "n_cells": len(wide),
            }
        )
        rows.append(
            {
                "wave": wave,
                "record": "share_closer",
                "method": "synthetic",
                "value": float((diff < 0).mean()),
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "n_cells": len(wide),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    cells = pd.concat([run(w) for w in WAVES], ignore_index=True)
    reports = settings.outputs_dir / "reports"
    cells.to_csv(reports / "era_design_deep.csv", index=False)
    summary = summarise(cells)
    summary.to_csv(reports / "era_design_deep_summary.csv", index=False)
    log.info("wrote %s", reports / "era_design_deep_summary.csv")
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
