"""Per-era grounding build (ADR-007 E).

Run: uv run python scripts/era_grounding.py
     uv run python scripts/era_grounding.py --wave 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.grounding.era_tables import write_all_eras, write_era_grounding
from sfac_persona.schema import ERA_WAVES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_grounding")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=None)
    args = ap.parse_args()
    if args.wave is not None:
        paths = write_era_grounding(args.wave)
        log.info("era_%s: %s", args.wave, sorted(p.name for p in paths.values()))
    else:
        all_paths = write_all_eras(waves=ERA_WAVES)
        for w, paths in all_paths.items():
            log.info("era_%s: %d files", w, len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
