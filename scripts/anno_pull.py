"""Download human-audit scores accumulated in the spreadsheet (ADR-023).

Scores accumulate in a Google spreadsheet (Apps Script web app). The preregistered analysis
(`anno_analysis.py`) reads local files, so this downloads them **in that same format** — the
analysis code is not changed at all.

  outputs/anno/annotations/R1.jsonl · R2.jsonl   independent scores
  outputs/anno/adjudication.jsonl                consensus decisions

An administrator (OWNER) code is required. Pass the code and URL on the command line, or set
ANNO_ENDPOINT and ANNO_OWNER_CODE. Neither is written into the repository.

Run: uv run python scripts/anno_pull.py --endpoint <web app URL> --code <OWNER code>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_pull")


def fetch(endpoint: str, code: str) -> dict:
    # The Apps Script web app redirects POST once to googleusercontent — urllib follows it.
    body = json.dumps({"action": "export", "code": code}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "text/plain"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if not out.get("ok"):
        raise SystemExit(f"download failed: {out.get('error')}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("ANNO_ENDPOINT"))
    ap.add_argument("--code", default=os.environ.get("ANNO_OWNER_CODE"))
    ap.add_argument("--store", type=Path, default=settings.outputs_dir / "anno")
    ap.add_argument(
        "--progress",
        action="store_true",
        help="show progress counts only (read-only; allowed during scoring)",
    )
    args = ap.parse_args()
    if not args.endpoint or not args.code:
        raise SystemExit("--endpoint and --code (OWNER) are required")

    if args.progress:
        body = json.dumps({"action": "progress", "code": args.code}).encode("utf-8")
        req = urllib.request.Request(
            args.endpoint,
            data=body,
            headers={"Content-Type": "text/plain;charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            prog = json.loads(r.read().decode("utf-8"))
        log.info("progress: %s", json.dumps(prog, ensure_ascii=False))
        return 0

    out = fetch(args.endpoint, args.code)
    ann_dir = args.store / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for rater in ("R1", "R2"):
        rows = [a for a in out["annotations"] if a.get("rater") == rater]
        rows.sort(key=lambda a: a.get("saved_at", ""))
        with (ann_dir / f"{rater}.jsonl").open("w", encoding="utf-8") as f:
            for a in rows:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        counts[rater] = len(rows)
    with (args.store / "adjudication.jsonl").open("w", encoding="utf-8") as f:
        for a in sorted(out.get("adjudication", []), key=lambda a: a.get("saved_at", "")):
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    log.info(
        "downloaded: %s · adjudication %d → %s",
        counts,
        len(out.get("adjudication", [])),
        args.store,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
