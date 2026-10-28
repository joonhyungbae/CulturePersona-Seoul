"""Non-LLM baselines and a ceiling for the narrative holdout, on the same metric
(weighted mean of per-cell TVD) (re-review 2026-09-24, MAJOR 1).

The §6.3 holdout table broke the §2 rule (every figure gets a baseline and a ceiling).
Three values use the **same cells (age band × attendance) and the same weights** as the
SSR results.

  marginal_lookup   that wave's weighted marginal on every cell — when there is no cell
                    information at all.
  prior_wave_lookup the same cell's actual distribution from the immediately previous
                    weighted wave — "when last survey's table is used as-is" (2020 has no
                    prior wave, so 2022 is used).
  real_sample       cell distributions from drawing n real respondents (same n=300 as the
                    public narrative sample) — the ceiling at that sample size
                    (mean of 15 draws). It cannot fall below this.

Run: uv run python scripts/era_holdout_baselines.py
Output: outputs/reports/era_holdout_baselines.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_holdout_run as H

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("holdout_baselines")

WAVES = (2024, 2022, 2020)
PRIOR = {2024: 2022, 2022: 2020, 2020: 2022}
N_REP = 15


def _cell_dists(real: pd.DataFrame, item: str) -> dict[tuple, tuple[pd.Series, int]]:
    out = {}
    for cell, g in real[real[item].notna()].groupby(H.CELL):
        if len(g) < 5:
            continue
        d = g.groupby(item)["w"].sum()
        out[tuple(cell)] = (d / d.sum(), len(g))
    return out


def _cell_mean(truth: dict, pred: dict) -> float:
    tvs, ws = [], []
    for cell, (d, n) in truth.items():
        q = pred.get(cell)
        if q is None:
            continue
        tvs.append(H._tvd(d, q))
        ws.append(n)
    return float(np.average(tvs, weights=ws)) if tvs else float("nan")


def main() -> int:
    refs = json.loads((H.OUT / "ssr_refs.json").read_text(encoding="utf-8"))
    items = list(refs.keys())
    frames = {w: H.real_item_frame(w) for w in WAVES}
    rng = np.random.default_rng(H.SEED)
    rows = []
    for wave in WAVES:
        real = frames[wave]
        prior = frames[PRIOR[wave]]
        for item in items:
            if item not in real.columns:
                continue
            truth = _cell_dists(real, item)
            ok = real[real[item].notna()]
            marg = ok.groupby(item)["w"].sum()
            marg = marg / marg.sum()
            marginal = _cell_mean(truth, {c: marg for c in truth})
            # A prior-wave lookup means something only when the option sets match (leisure
            # items have different options by wave, and the 2020 scales are 5- and 4-point).
            # If they differ, leave NaN and state the applicable-item count in the table.
            same_opts = item in prior.columns and set(real[item].dropna().astype(str)) == set(
                prior[item].dropna().astype(str)
            )
            prior_wave = (
                _cell_mean(truth, {c: d for c, (d, _) in _cell_dists(prior, item).items()})
                if same_opts
                else float("nan")
            )
            samples = []
            for _ in range(N_REP):
                idx = rng.choice(len(ok), size=min(H.N_RELEASE, len(ok)), replace=False)
                sub = ok.iloc[idx]
                samples.append(
                    _cell_mean(truth, {c: d for c, (d, _) in _cell_dists(sub, item).items()})
                )
            rows.append(
                {
                    "wave": wave,
                    "item": item,
                    "marginal_lookup_tvd": marginal,
                    "prior_wave": PRIOR[wave],
                    "prior_wave_lookup_tvd": prior_wave,
                    "prior_same_options": bool(same_opts),
                    "real_sample_tvd": float(np.nanmean(samples)),
                    "n_real": len(ok),
                }
            )
        log.info("wave %s done", wave)
    d = pd.DataFrame(rows)
    d.to_csv(settings.outputs_dir / "reports" / "era_holdout_baselines.csv", index=False)
    log.info(
        "\n%s",
        d.groupby("wave")[["marginal_lookup_tvd", "prior_wave_lookup_tvd", "real_sample_tvd"]]
        .mean()
        .round(4),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
