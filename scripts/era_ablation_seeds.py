"""Seed noise of ablation variants — whether a variant "better than the released
chain" is inside sampling noise.

In `era_sampler_ablation.csv`, no_attend's genre φ MAE (0.042 < 0.046) and no_same_construct's
structural-pair TVD (0.014 < 0.017) are lower than the released chain. Those figures come from
one seed's sample of 30,000, so the same spec is redrawn with only the seed changed and the
between-seed standard deviation of the metrics is reported. If the gap falls inside that,
there is no basis for adopting the variant.

Run: uv run python scripts/era_ablation_seeds.py [--seeds 5] [--sample 30000]
Outputs: outputs/reports/era_ablation_seeds.csv · .md
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

from era_sampler_audit import SEED, _non_root, build_variant, genre_phi_mae, pair_tvd

from sfac_persona.config import settings
from sfac_persona.grounding import chain as C
from sfac_persona.grounding.official import build_official_anchor_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("ablation_seeds")

VARIANTS = ("released", "no_attend", "no_same_construct")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2024)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sample", type=int, default=30000)
    args = ap.parse_args()
    frame = build_official_anchor_frame(args.wave)
    anchors = _non_root(frame)
    rows = []
    for v in VARIANTS:
        spec = build_variant(frame, v)
        for k in range(args.seeds):
            syn = C.ChainSampler(spec).sample(args.sample, seed=SEED + 1000 * (k + 1))
            ns, st, _, _ = pair_tvd(frame, syn, anchors)
            rows.append(
                {
                    "wave": args.wave,
                    "variant": v,
                    "seed_idx": k,
                    "tvd_nonstructural": ns,
                    "tvd_structural": st,
                    "genre_phi_mae": genre_phi_mae(frame, syn),
                }
            )
            log.info(
                "%s seed %d: ns %.4f st %.4f phi %.4f", v, k, ns, st, rows[-1]["genre_phi_mae"]
            )
    d = pd.DataFrame(rows)
    out = settings.outputs_dir / "reports"
    d.to_csv(out / "era_ablation_seeds.csv", index=False)
    g = d.groupby("variant")[["tvd_nonstructural", "tvd_structural", "genre_phi_mae"]].agg(
        ["mean", "std"]
    )
    lines = [
        "# 소거 변형의 시드 잡음",
        "",
        f"> 자동 생성: `scripts/era_ablation_seeds.py` (wave {args.wave}, 시드 {args.seeds}, 표본 {args.sample:,}).",
        "",
        g.round(4).to_markdown(),
        "",
        "차이가 시드 간 표준편차의 2 배 안이면 변형을 채택할 근거가 없다 (원고 부록 소거 표의 각주).",
    ]
    (out / "era_ablation_seeds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
