"""Per-era official reproduction gate plus an anchor-frame smoke check.

Run: uv run python scripts/official_reproduce_gate.py
     uv run python scripts/official_reproduce_gate.py --wave 2024 --smoke-frame
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.official import (
    build_official_anchor_frame,
    gate_all_eras,
    reproduce_gate,
)
from sfac_persona.schema import ERA_WAVES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("official_gate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--smoke-frame", action="store_true")
    args = ap.parse_args()

    out_dir = settings.project_root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wave is not None:
        table = reproduce_gate(args.wave)
        table.insert(0, "wave", args.wave)
        waves = [args.wave]
    else:
        table = gate_all_eras()
        waves = list(ERA_WAVES)

    csv_path = out_dir / "official_reproduce_gate.csv"
    table.to_csv(csv_path, index=False)

    if args.smoke_frame:
        for w in waves:
            fr = build_official_anchor_frame(w)
            log.info(
                "frame wave=%s n=%d living_na=%d edu_na=%d attend=%s",
                w,
                len(fr),
                int(fr["living_area"].isna().sum()),
                int(fr["education"].isna().sum()),
                fr["attend_any"].value_counts().to_dict(),
            )

    check = table
    if "note" in check.columns:
        check = check[check["note"].isna() | (check["note"] != "unweighted_diagnostic_skip_pub")]
    n_fail = int((~check["ok"]).sum()) if "ok" in check.columns and len(check) else 0
    verdict = "PASS" if n_fail == 0 else "FAIL"
    md_path = out_dir / "official_reproduce_gate.md"
    md_path.write_text(
        f"# official 재현 게이트\n\n- 판정: **{verdict}** (실패 {n_fail})\n- `{csv_path.name}`\n",
        encoding="utf-8",
    )
    log.info("%s → %s", verdict, csv_path)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
