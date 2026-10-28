"""Per-era quantitative persona samples (ADR-007 E).

Target size: weighted 15k / unweighted 5k.
Run: uv run python scripts/era_sample_quant.py
     uv run python scripts/era_sample_quant.py --wave 2024 --n 1000 --ablation structure_T
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.chain import ChainSampler
from sfac_persona.schema import (
    ERA_WAVES,
    UNWEIGHTED_WAVES,
    WAVE_REGIME,
    WEIGHTED_WAVES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_sample")

IDEAL_N = {**{w: 15_000 for w in WEIGHTED_WAVES}, **{w: 5_000 for w in UNWEIGHTED_WAVES}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="override ideal N")
    ap.add_argument(
        "--ablation",
        choices=["margins", "structure_T"],
        default="structure_T",
    )
    ap.add_argument("--seed", type=int, default=20240910)
    ap.add_argument(
        "--out-suffix",
        default="",
        help="Output filename suffix. Use when writing a comparison sample without overwriting the parquet an in-progress generation is reading.",
    )
    args = ap.parse_args()

    waves = [args.wave] if args.wave else list(ERA_WAVES)
    spec_name = (
        "sampler_specs_structure_T.json"
        if args.ablation == "structure_T"
        else "sampler_specs_margins.json"
    )

    for wave in waves:
        # unweighted eras: margins only (no T transfer required)
        sn = spec_name
        if wave in UNWEIGHTED_WAVES:
            sn = "sampler_specs_margins.json"
        path = settings.era_grounding_dir(wave) / sn
        if not path.exists():
            # fall back
            path = settings.era_grounding_dir(wave) / "sampler_specs_margins.json"
        if not path.exists():
            log.error(
                "missing spec for wave %s — run structure_priors.py / era grounding first", wave
            )
            return 1
        spec = json.loads(path.read_text(encoding="utf-8"))
        n = args.n if args.n is not None else IDEAL_N[wave]
        sampler = ChainSampler(spec)
        df = sampler.sample(n, seed=args.seed + wave)
        df["persona_id"] = [f"{wave}-{args.ablation}-{i:05d}" for i in range(len(df))]
        df["survey_year"] = wave
        df["survey_era"] = WAVE_REGIME.get(wave, "unknown")
        df["weighting"] = "none" if wave in UNWEIGHTED_WAVES else "survey"
        df["instrument_regime"] = WAVE_REGIME.get(wave, "unknown")
        df["ablation"] = args.ablation if wave in WEIGHTED_WAVES else "margins"
        out_dir = settings.synthetic_dir / f"era_{wave}_quant"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"personas_{args.ablation}{args.out_suffix}.parquet"
        df.to_parquet(out_path, index=False)
        meta = {
            "n": n,
            "wave": wave,
            "ablation": args.ablation,
            "spec": str(path),
            "rejections": sampler.rejections,
            "genre_fallbacks": sampler.genre_fallbacks,
        }
        (out_dir / f"meta_{args.ablation}{args.out_suffix}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("wrote %s n=%d rejections=%d", out_path, n, sampler.rejections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
