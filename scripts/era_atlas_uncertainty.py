"""Uncertainty of atlas figures — bootstrap intervals and a sampling floor (ADR-017).

The manuscript tables 0.063 → 0.044 → 0.026 without saying whether the gaps exceed
sampling noise. In particular, the claim that "the sex conditional changes almost nothing"
is 0.044 versus 0.044, a sentence that cannot be checked without an interval. Two
quantities are reported here.

  bootstrap   Resample respondents with replacement and recompute mean TVD under the four
              reconstruction assumptions → percentile intervals. Intervals for the
              reduction rates (attendance · sex) come from the same draws (pairwise
              correlation is carried along as-is).
  floor       TVD between the **observed** joint of that same resample and the observed
              joint of the full sample. The distance comes from sampling alone, not from
              reconstruction error — the floor is this value, not 0.

Run: uv run python scripts/era_atlas_uncertainty.py --wave 2024 --boot 200
Outputs: outputs/reports/era_atlas_uncertainty.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.error_study import _joint, ladder_mask, pair_errors
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import ERA_CORE_ANCHORS, ROOT_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _n in ("sfac_persona.grounding.error_study", "sfac_persona.grounding.official"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("atlas_uncertainty")

RECON_COLS = (
    "tvd_indep",
    "tvd_cond_age",
    "tvd_cond_age_sex",
    "tvd_cond_age_route",
    "tvd_cond_age_attend",
)

_FRAME: pd.DataFrame | None = None
_ANCHORS: list[str] = []
_BASE_JOINTS: dict[tuple[str, str], pd.DataFrame] = {}
_PAIRS: list[tuple[str, str]] = []


def _core_anchors(frame: pd.DataFrame) -> list[str]:
    return [a for a in ERA_CORE_ANCHORS if a in frame.columns and a not in ROOT_ANCHORS]


def _tvd(a: pd.DataFrame, b: pd.DataFrame) -> float:
    a2, b2 = a.align(b, fill_value=0.0)
    return float(0.5 * np.abs(a2.to_numpy() - b2.to_numpy()).sum())


def _init(frame: pd.DataFrame, anchors: list[str], pairs: list[tuple[str, str]]) -> None:
    """Worker init. Build the frame and the full-sample observed joints once."""
    global _FRAME, _ANCHORS, _BASE_JOINTS, _PAIRS
    _FRAME, _ANCHORS, _PAIRS = frame, anchors, pairs
    _BASE_JOINTS = {(a, b): _joint(frame, a, b) for a, b in pairs}


def _derived(out: dict[str, float]) -> None:
    out["attend_cut_pct"] = 100.0 * (1.0 - out["tvd_cond_age_attend"] / out["tvd_cond_age"])
    out["sex_gain"] = out["tvd_cond_age"] - out["tvd_cond_age_sex"]
    out["route_cut_pct"] = 100.0 * (1.0 - out["tvd_cond_age_route"] / out["tvd_cond_age"])
    out["attend_beyond_route_pct"] = 100.0 * (
        1.0 - out["tvd_cond_age_attend"] / out["tvd_cond_age_route"]
    )


DERIVED = ("attend_cut_pct", "sex_gain", "route_cut_pct", "attend_beyond_route_pct")


def _one_resample(seed: int) -> dict[str, float]:
    assert _FRAME is not None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(_FRAME), len(_FRAME))
    boot = _FRAME.iloc[idx].reset_index(drop=True)
    p = pair_errors(boot, anchors=_ANCHORS)
    ns = p[ladder_mask(p)]
    out = {c: float(ns[c].dropna().mean()) for c in RECON_COLS}
    _derived(out)
    # Sampling floor: observed joint of this resample vs the full-sample observed joint.
    out["sampling_floor"] = float(
        np.mean([_tvd(_joint(boot, a, b), _BASE_JOINTS[(a, b)]) for a, b in _PAIRS])
    )
    return out


def run_wave(wave: int, boot: int, workers: int) -> pd.DataFrame:
    frame = build_official_anchor_frame(wave)
    anchors = _core_anchors(frame)
    point = pair_errors(frame, anchors=anchors)
    ns = point[ladder_mask(point)]
    pairs = list(zip(ns.anchor_a, ns.anchor_b, strict=True))
    log.info(
        "wave=%d n=%d anchors=%d pairs=%d boot=%d", wave, len(frame), len(anchors), len(pairs), boot
    )

    base = {c: float(ns[c].dropna().mean()) for c in RECON_COLS}
    _derived(base)

    seeds = [20260920 + wave * 1000 + i for i in range(boot)]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(frame, anchors, pairs)
    ) as ex:
        draws = list(ex.map(_one_resample, seeds, chunksize=1))
    d = pd.DataFrame(draws)

    rows = []
    for metric in [*RECON_COLS, *DERIVED]:
        v = d[metric].to_numpy()
        rows.append(
            {
                "wave": wave,
                "metric": metric,
                "point": base[metric],
                "boot_mean": float(v.mean()),
                "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5)),
                "n_boot": boot,
            }
        )
    fl = d["sampling_floor"].to_numpy()
    rows.append(
        {
            "wave": wave,
            "metric": "sampling_floor",
            "point": float(fl.mean()),
            "boot_mean": float(fl.mean()),
            "lo": float(np.percentile(fl, 2.5)),
            "hi": float(np.percentile(fl, 97.5)),
            "n_boot": boot,
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, action="append", default=None)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    waves = args.wave or [2024, 2022, 2020]

    out_dir = settings.outputs_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "era_atlas_uncertainty.csv"
    # Write after each wave — if a long run stops midway, finished waves remain.
    for w in waves:
        res = run_wave(w, args.boot, args.workers)
        if path.exists():
            prev = pd.read_csv(path)
            res = pd.concat([prev[prev.wave != w], res], ignore_index=True)
        res.sort_values(["wave", "metric"]).to_csv(path, index=False)
        log.info("wave=%d written", w)
    res = pd.read_csv(path)
    log.info("wrote %s", path)
    print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
