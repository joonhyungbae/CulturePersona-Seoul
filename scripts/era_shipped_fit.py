"""Fit of the shipped sampler configuration next to the configuration used for the fidelity measurements.

The sampler-fidelity numbers (ablations, held-out fit) are measured on the released chain structure with a
survey-weighted root and each wave's own genre tree, so that they isolate what the tables carry. The shipped
bank differs in two inputs: the root is the resident-registration population (ADR-011) and the genre tree is
the structure prior pooled from 2020 and 2022 (structure_priors.py). This script scores, against each wave's
weighted microdata frame and on the same pair set as the ladder:

  measured   the chain as measured in era_sampler_audit.py ("released" variant)
  shipped    a fresh draw from the shipped specification (sampler_specs_structure_T.json)
  shipped_survey_root  the shipped specification with the measured chain's survey-weighted root, which
             separates the effect of the root from that of the pooled genre tree
  bank       the shipped 15,000-persona quantitative bank

Run: uv run python scripts/era_shipped_fit.py
Output: outputs/reports/era_shipped_fit.csv (aggregates only)
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import era_sampler_audit as A

from sfac_persona.config import settings
from sfac_persona.grounding import chain as C
from sfac_persona.grounding.official import build_official_anchor_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_shipped_fit")

WAVES = (2024, 2022, 2020)
N_DRAW = 30000  # same draw size as era_sampler_audit.py


def main() -> None:
    rows = []
    for wave in WAVES:
        frame = C.add_derived_columns(build_official_anchor_frame(wave))
        anchors = A._non_root(frame)
        spec_measured = A.build_variant(frame, "released")
        spec_shipped = json.loads(
            (settings.era_grounding_dir(wave) / "sampler_specs_structure_T.json").read_text(
                encoding="utf-8"
            )
        )
        bank = pd.read_parquet(
            settings.data_root / "synthetic" / f"era_{wave}_quant" / "personas_structure_T.parquet"
        )
        spec_survey_root = copy.deepcopy(spec_shipped)
        spec_survey_root["root"] = spec_measured["root"]
        draws = {
            "measured": C.ChainSampler(spec_measured).sample(N_DRAW, seed=A.SEED),
            "shipped": C.ChainSampler(spec_shipped).sample(N_DRAW, seed=A.SEED),
            "shipped_survey_root": C.ChainSampler(spec_survey_root).sample(N_DRAW, seed=A.SEED),
            "bank": bank,
        }
        for config, syn in draws.items():
            ns, st, n_ns, _n_st = A.pair_tvd(frame, syn, anchors)
            rows.append(
                {
                    "wave": wave,
                    "config": config,
                    "n": len(syn),
                    "tvd_nonstructural": ns,
                    "tvd_structural": st,
                    "genre_phi_mae": A.genre_phi_mae(frame, syn),
                    "n_pairs_nonstructural": n_ns,
                }
            )
            log.info("wave=%d %-20s ns=%.4f st=%.4f", wave, config, ns, st)
    out = settings.outputs_dir / "reports" / "era_shipped_fit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
