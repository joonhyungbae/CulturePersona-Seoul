"""Held-out (split-half) remeasurement of the error ladder on the national survey (KCAS).

kcas_atlas.py repeats the ladder on KCAS in sample: conditionals estimated on all respondents of a wave and
scored on the same respondents. The Seoul ladder is also reported held out (era_ladder_crossfit.py). This
script gives KCAS the same held-out column, so the replication claim rests on the same test in both surveys.

Design fixed before running (2026-10-11):
  data        the cleaned KCAS person file read in memory (kcas_atlas.kcas_path), every wave, anchors and
              frame exactly as kcas_atlas.anchor_frame / non_root_anchors (education, income, nine genres,
              attendance status, count band; root sex, age band, province); pair set = kcas_atlas's ladder
              pairs (ladder_mask: no structural pair, no pair with attendance status)
  splits      20 random half splits per wave with era_ladder_crossfit.split_halves (same seeds as Seoul);
              rungs estimated on half A, scored on half B's weighted joint with P_B(C) mixture weights
              (era_ladder_crossfit.score_split, the "oos" column); floor = TVD between the halves' joints
  summary     per wave and rung: in-sample mean (kcas_atlas_summary.csv, weighted), held-out mean, min and max
              over splits, floor; per wave: age cut, skip cut, beyond-skip share and the sex gap, held out;
              ordering (indep > age > age x attendance) checked on the held-out means
Every result is reported, whatever its sign.

Run: uv run python scripts/kcas_ladder_crossfit.py [--splits 20]
Output: outputs/reports/kcas_ladder_crossfit.csv (aggregates only; no person row leaves memory)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era_ladder_crossfit import score_split, split_halves
from kcas_atlas import anchor_frame, kcas_path, non_root_anchors

from sfac_persona.config import settings
from sfac_persona.grounding.error_study import RECON

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("kcas_ladder_crossfit")
REPORTS = settings.outputs_dir / "reports"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", type=int, default=20)
    args = ap.parse_args()
    raw_all = pd.read_parquet(kcas_path())
    insample = pd.read_csv(REPORTS / "kcas_atlas_summary.csv")
    insample = insample[insample["mode"] == "weighted"].set_index("wave")
    rows: list[dict] = []
    for wave, raw in raw_all.groupby("wave"):
        wave = int(wave)
        frame = anchor_frame(raw.reset_index(drop=True))
        anchors = non_root_anchors(frame)
        parts = []
        for k in range(args.splits):
            fit, score = split_halves(frame, k)
            parts.append(score_split(fit, score, anchors))
        n_pairs = {p.n_pairs for p in parts}
        assert len(n_pairs) == 1, n_pairs
        oos = {r: [p.tvd_b[r] for p in parts] for r in RECON}
        floor = sum(p.floor for p in parts) / len(parts)
        for r in RECON:
            rows.append(
                {
                    "wave": wave,
                    "record": "rung",
                    "rung": r,
                    "in_sample": float(insample.loc[wave, f"mean_tvd_{r}"]),
                    "oos_mean": sum(oos[r]) / len(oos[r]),
                    "oos_min": min(oos[r]),
                    "oos_max": max(oos[r]),
                    "floor": floor,
                    "n_pairs": n_pairs.pop() if n_pairs else None,
                    "n_splits": args.splits,
                }
            )
        m = {r: sum(v) / len(v) for r, v in oos.items()}
        rows.append(
            {
                "wave": wave,
                "record": "summary",
                "age_cut_out": 1 - m["cond_age"] / m["indep"],
                "sex_gap_out": m["cond_age"] - m["cond_age_sex"],
                "route_cut_out": 1 - m["cond_age_route"] / m["cond_age"],
                "attend_cut_out": 1 - m["cond_age_attend"] / m["cond_age"],
                "beyond_route_out": 1 - m["cond_age_attend"] / m["cond_age_route"],
                "order_ok_out": bool(m["indep"] > m["cond_age"] > m["cond_age_attend"]),
                "floor": floor,
                "n_splits": args.splits,
            }
        )
        log.info(
            "KCAS %d held out: %s | floor %.4f | order %s",
            wave,
            " ".join(f"{r}={m[r]:.4f}" for r in RECON),
            floor,
            m["indep"] > m["cond_age"] > m["cond_age_attend"],
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "kcas_ladder_crossfit.csv", index=False)
    print(out[out.record == "summary"].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
