"""Strip direct identifiers from interim free-text responses (ADR-020).

`data/interim/official/official_person_text.parquet` is a copy of the preprocessing
repository's cleaned file, and no pipeline reads it. It nevertheless contains **raw
addresses** (SQ1_OP) and **dong names** (SQ2_NAME). Dong name is not read
(release geography stops at the living area) and is not kept even as a local copy.

  drop     whole identifier variables (address, dong name, contact-like fields)
  redact   in opinion free text, replace only the body of rows that match an address or phone pattern (the row stays)

**Rerun after regenerating.** Copying again from the preprocessing repository brings the raw text
back — the interim file is a copy of the upstream cleaned file. This script is idempotent.

Run: uv run python scripts/redact_interim_text.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("redact_text")

TARGET = "official_person_text.parquet"
# Variables that are identifiers by design — drop the whole row.
DROP_VARIABLES = {"SQ1_OP", "SQ2_NAME"}
REDACTED = "[redacted]"
ADDRESS_RE = re.compile(r"(?:시|도|구|군|읍|면|동|로|길)\s*\d|아파트|번지|\d{2,4}-\d{2,4}")
PHONE_RE = re.compile(r"01[016-9][- ]?\d{3,4}[- ]?\d{4}")


def redact(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    kept = df[~df.variable.isin(DROP_VARIABLES)].copy()
    dropped = before - len(kept)
    text = kept.text.astype(str)
    hit = text.str.contains(ADDRESS_RE, na=False) | text.str.contains(PHONE_RE, na=False)
    kept.loc[hit, "text"] = REDACTED
    return kept, {"dropped_rows": dropped, "redacted_rows": int(hit.sum()), "kept_rows": len(kept)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = settings.data_root / "interim" / "official" / TARGET
    if not path.exists():
        log.info("no %s — nothing to redact", path)
        return 0
    df = pd.read_parquet(path)
    out, stats = redact(df)
    log.info("%s: %s", TARGET, stats)
    if args.dry_run:
        return 0
    out.to_parquet(path, index=False)
    log.info("wrote redacted %s (%d rows)", path, len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
