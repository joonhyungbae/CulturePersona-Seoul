"""Bias-corrected intervals for the cell-contrast correlations of era_holdout_recal.py.

The recalibration reports, for each arm, the correlation between its cell deviations and the real ones, with a
persona-bootstrap percentile interval. Resampling personas with replacement adds sampling noise to every cell mean,
which attenuates the correlation, so the bootstrap distribution sits below the point estimate and the percentile
interval ends at or near the point (e.g. 0.38 [0.32, 0.38]). This script recomputes those intervals with the
bias-corrected (BC) percentile method, which shifts the percentiles by the share of draws below the point estimate,
and writes them back into outputs/reports/era_holdout_recal.csv. Point estimates are unchanged.

Correction decided after inspecting the published intervals (2026-10-13), recorded here before rerunning:
  scope     every row of era_holdout_recal.csv with a cell-contrast interval (record "level", arms release,
            synthanchors, twin, every reader, wave and method that the recalibration covered) and every
            "corr_diff" row (persona-paired difference, release minus synthanchors)
  draws     500 for single-arm correlations and 200 for paired differences, as in the original script, same seed
  method    BC percentile: z0 = Phi^-1(share of draws below the point); bounds at Phi(2 z0 -/+ 1.96)
Change after the first BC run (2026-10-13, recorded before rerunning): where nearly every draw falls below the
  point (the generator's 2024 twin arm: 0.390 with BC [0.392, 0.401]) the BC interval excludes the point, so the
  reported interval is the normal approximation, point -/+ 1.96 x bootstrap standard deviation, which keeps the
  point at the centre; BC bounds and all draws are saved to outputs/logs/corr_bc_draws.json for inspection.
Every interval is reported, whatever it shows.

Run: uv run python scripts/era_holdout_corr_bc.py
Output: updates cell_corr_lo/hi and corr_diff ci_lo/hi in outputs/reports/era_holdout_recal.csv (aggregates only)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_holdout_recal as RC
import era_holdout_run as H

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("era_holdout_corr_bc")
SEED = 20261010


DRAWS: dict[str, dict] = {}


def normal(point: float, draws: list[float]) -> tuple[float, float]:
    d = np.array([v for v in draws if np.isfinite(v)])
    sd = float(d.std(ddof=1))
    return point - 1.96 * sd, point + 1.96 * sd


def bc(point: float, draws: list[float]) -> tuple[float, float]:
    d = np.array([v for v in draws if np.isfinite(v)])
    share = np.clip((d < point).mean(), 1e-3, 1 - 1e-3)
    z0 = norm.ppf(share)
    lo_q, hi_q = norm.cdf(2 * z0 - 1.96), norm.cdf(2 * z0 + 1.96)
    return float(np.quantile(d, lo_q)), float(np.quantile(d, hi_q))


def main() -> int:
    items = list(H.QUESTIONS)
    tau, _ = H.load_frozen_tau()
    df = pd.read_csv(RC.REPORT)
    for wave in sorted(df.wave.dropna().unique()):
        wave = int(wave)
        refs = json.loads((H.wave_dir(wave) / "ssr_refs.json").read_text(encoding="utf-8"))
        refs = {
            i: {str(k): (v if isinstance(v, list) else [str(v)]) for k, v in o.items()}
            for i, o in refs.items()
        }
        order = {i: list(o) for i, o in refs.items()}
        real = H.real_item_frame(wave)
        for (reader, method), _g in df[(df.wave == wave) & df.reader.notna()].groupby(
            ["reader", "method"]
        ):
            arms = {}
            for arm in ("release", "synthanchors", "twin"):
                recs = (
                    RC.load_ssr(reader, arm, wave, refs, tau)
                    if method == "ssr"
                    else RC.load_lp(reader, arm)
                )
                if recs:
                    arms[arm] = recs
            for arm, recs in arms.items():
                mask = (
                    (df.wave == wave)
                    & (df.reader == reader)
                    & (df.method == method)
                    & (df.record == "level")
                    & (df.arm == arm)
                )
                if not mask.any() or pd.isna(df.loc[mask, "cell_corr_lo"]).all():
                    continue
                point = RC.cell_contrast(recs, real, items, order)[0]
                rng = np.random.default_rng(SEED)
                draws = [
                    RC.cell_contrast(
                        [recs[k] for k in rng.integers(0, len(recs), len(recs))], real, items, order
                    )[0]
                    for _ in range(500)
                ]
                lo, hi = normal(point, draws)
                DRAWS[f"{wave}|{reader}|{method}|{arm}"] = {
                    "point": point,
                    "bc": bc(point, draws),
                    "draws": draws,
                }
                df.loc[mask, ["cell_corr_lo", "cell_corr_hi"]] = [lo, hi]
                log.info(
                    "%d %s %s %s corr %.3f normal [%.3f, %.3f]",
                    wave,
                    reader,
                    method,
                    arm,
                    point,
                    lo,
                    hi,
                )
            mask = (
                (df.wave == wave)
                & (df.reader == reader)
                & (df.method == method)
                & (df.record == "corr_diff")
            )
            if mask.any() and {"release", "synthanchors"} <= set(arms):
                bi = {r["persona_id"]: r for r in arms["synthanchors"]}
                pairs = [(r, bi[r["persona_id"]]) for r in arms["release"] if r["persona_id"] in bi]
                point = (
                    RC.cell_contrast([a for a, _ in pairs], real, items, order)[0]
                    - RC.cell_contrast([b for _, b in pairs], real, items, order)[0]
                )
                rng = np.random.default_rng(SEED)
                draws = []
                for _ in range(200):
                    idx = rng.integers(0, len(pairs), len(pairs))
                    draws.append(
                        RC.cell_contrast([pairs[k][0] for k in idx], real, items, order)[0]
                        - RC.cell_contrast([pairs[k][1] for k in idx], real, items, order)[0]
                    )
                lo, hi = normal(point, draws)
                DRAWS[f"{wave}|{reader}|{method}|corr_diff"] = {
                    "point": point,
                    "bc": bc(point, draws),
                    "draws": draws,
                }
                df.loc[mask, ["ci_lo", "ci_hi"]] = [lo, hi]
                log.info(
                    "%d %s %s corr_diff %.3f normal [%.3f, %.3f]",
                    wave,
                    reader,
                    method,
                    point,
                    lo,
                    hi,
                )
    df.to_csv(RC.REPORT, index=False)
    (ROOT / "outputs" / "logs" / "corr_bc_draws.json").write_text(
        json.dumps(DRAWS), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
