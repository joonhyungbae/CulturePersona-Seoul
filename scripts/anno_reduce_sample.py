"""Shrink the human-review sample to a shared half-size subset for both raters (ADR-039).

The registered design had both raters score all 343 main items, each in a different random
order. That could not be finished inside the submission window, so the shared subset is fixed
**before looking at any scores**. This script does not read scores — inputs are the answer
key's stratum fields, the registered order files, and the per-rater completion counts the
server reported.

  Rule    The UI opens the first unsaved item in the order, so a rater's finished items are
          the prefix of that rater's order. The union of both raters' prefixes (completed +
          buffer) **must** enter the subset (the order was random, so those items are a
          random sample too). The remainder is filled up to the per-cell target by a
          seeded random draw.
  Target  40 clean per wave · 24 flagged (proportional to waves) · 12 planted contradictions
          per difficulty · 3 manipulation checks.
          Cells whose forced inclusion already exceeds the target are left as they are
          (scores are not discarded).

Outputs: outputs/anno/subset_v2.json · annotation/data_private/main_subset.json ·
         annotation/public/data/order_R{1,2}.json (registered orders kept in
         outputs/anno/order_registered_R*.json)

Run: uv run python scripts/anno_reduce_sample.py --done-r1 17 --done-r2 59
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_reduce")

SEED = 20261004
RATERS = ("R1", "R2")
CLEAN_PER_WAVE = 40
FLAGGED_TOTAL = 24
TRAP_PER_TIER = 12
GROSS_TOTAL = 3


def largest_remainder(shares: dict[str, int], total: int) -> dict[str, int]:
    """Split total in proportion to the counts (largest remainder; ties follow key order)."""
    s = sum(shares.values())
    raw = {k: total * v / s for k, v in shares.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:rest]:
        out[k] += 1
    return out


def cell_of(row: pd.Series) -> str:
    if row.stratum == "clean":
        return f"clean|{int(row.wave)}"
    if row.stratum == "flagged":
        return f"flagged|{int(row.wave)}"
    if row.stratum == "trap":
        return f"trap|{row.tier}"
    return str(row.stratum)


def targets_for(key: pd.DataFrame) -> dict[str, int]:
    t: dict[str, int] = {}
    for w in sorted(key[key.stratum == "clean"].wave.unique()):
        t[f"clean|{int(w)}"] = CLEAN_PER_WAVE
    fl = key[key.stratum == "flagged"].groupby("wave").size()
    for w, n in largest_remainder(
        {str(int(w)): int(n) for w, n in fl.items()}, FLAGGED_TOTAL
    ).items():
        t[f"flagged|{w}"] = n
    for tier in sorted(key[key.stratum == "trap"].tier.unique()):
        t[f"trap|{tier}"] = TRAP_PER_TIER
    t["trap_gross"] = GROSS_TOTAL
    return t


def build(
    key: pd.DataFrame,
    orders: dict[str, list[str]],
    done: dict[str, int],
    buffer: dict[str, int],
    seed: int,
) -> tuple[list[str], dict[str, list[str]], dict]:
    """(subset ids, new orders, summary). Scores are not read anywhere."""
    key = key.copy()
    practice = set(key[key.stratum == "practice"].item_id)
    main = key[key.stratum != "practice"].copy()
    main["cell"] = main.apply(cell_of, axis=1)
    known = set(main.item_id)

    prefix: dict[str, list[str]] = {}
    for r in RATERS:
        m = [i for i in orders[r] if i not in practice]
        if set(m) != known or len(m) != len(known):
            raise ValueError(f"{r}: main items in the registered order differ from the answer key")
        prefix[r] = m[: done[r] + buffer[r]]
    forced = set().union(*prefix.values())

    targets = targets_for(main)
    rng = np.random.default_rng(seed)
    subset = set(forced)
    for cell in sorted(targets):
        ids = sorted(main[main.cell == cell].item_id)
        have = [i for i in ids if i in forced]
        need = targets[cell] - len(have)
        if need > 0:
            pool = [i for i in ids if i not in forced]
            subset.update(pool[j] for j in rng.permutation(len(pool))[:need])

    new_orders: dict[str, list[str]] = {}
    for j, r in enumerate(RATERS):
        head = [i for i in orders[r] if i in practice]
        rest = sorted(subset - set(prefix[r]))
        r_rng = np.random.default_rng(seed + 100 + j)
        new_orders[r] = head + prefix[r] + [rest[k] for k in r_rng.permutation(len(rest))]
        # Invariant: 12 practice + the subset as-is, no duplicates, finished prefix matches
        # the registered order.
        assert len(new_orders[r]) == len(set(new_orders[r])) == len(head) + len(subset)
        assert set(new_orders[r]) - practice == subset
        old_main = [i for i in orders[r] if i not in practice]
        assert [i for i in new_orders[r] if i not in practice][: len(prefix[r])] == old_main[
            : len(prefix[r])
        ]
    assert not (subset & practice) and subset <= known

    sub = main[main.item_id.isin(subset)]
    summary = {
        "seed": seed,
        "created": datetime.now().astimezone().date().isoformat(),
        "done": done,
        "buffer": buffer,
        "targets": targets,
        "n_registered": len(main),
        "n_total": len(subset),
        "n_forced": len(forced),
        "counts": {
            "by_stratum": {k: int(v) for k, v in sub.groupby("stratum").size().items()},
            "by_cell": {k: int(v) for k, v in sub.groupby("cell").size().items()},
        },
        "ids": sorted(subset),
    }
    return sorted(subset), new_orders, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--done-r1", type=int, required=True)
    ap.add_argument("--done-r2", type=int, required=True)
    ap.add_argument("--buffer-r1", type=int, default=2)
    ap.add_argument("--buffer-r2", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--store", type=Path, default=settings.outputs_dir / "anno")
    ap.add_argument("--web", type=Path, default=ROOT / "annotation")
    args = ap.parse_args()

    web_data = args.web / "public" / "data"
    orders: dict[str, list[str]] = {}
    for r in RATERS:
        backup = args.store / f"order_registered_{r}.json"
        if not backup.exists():
            if args.dry_run:
                backup = web_data / f"order_{r}.json"
            else:
                shutil.copyfile(web_data / f"order_{r}.json", backup)
                log.info("kept registered order: %s", backup.name)
        orders[r] = json.loads(backup.read_text(encoding="utf-8"))

    key = pd.read_csv(args.store / "answer_key.csv", dtype={"item_id": str})
    done = {"R1": args.done_r1, "R2": args.done_r2}
    buffer = {"R1": args.buffer_r1, "R2": args.buffer_r2}
    ids, new_orders, summary = build(key, orders, done, buffer, args.seed)
    log.info(
        "subset %d / registered %d (forced inclusion %d)",
        summary["n_total"],
        summary["n_registered"],
        summary["n_forced"],
    )
    log.info("by stratum: %s", summary["counts"]["by_stratum"])
    log.info("by cell: %s", summary["counts"]["by_cell"])
    if args.dry_run:
        return 0

    (args.store / "subset_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    private = args.web / "data_private"
    private.mkdir(exist_ok=True)
    (private / "main_subset.json").write_text(json.dumps(ids), encoding="utf-8")
    for r in RATERS:
        (web_data / f"order_{r}.json").write_text(json.dumps(new_orders[r]), encoding="utf-8")
    log.info("wrote: subset_v2.json · data_private/main_subset.json · order_R1/R2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
