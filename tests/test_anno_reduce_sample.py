"""anno_reduce_sample: check invariants on a synthetic key (reads no private files)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "anno_reduce_sample", Path(__file__).resolve().parents[1] / "scripts" / "anno_reduce_sample.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _fake() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = []
    for w in (2020, 2022, 2024):
        rows += [
            {"item_id": f"C{w}{i:03d}", "stratum": "clean", "wave": w, "tier": None}
            for i in range(81)
        ]
    for w, n in ((2020, 10), (2022, 15), (2024, 21)):
        rows += [
            {"item_id": f"F{w}{i:03d}", "stratum": "flagged", "wave": w, "tier": None}
            for i in range(n)
        ]
    for t in ("T1", "T2", "T3"):
        rows += [
            {"item_id": f"{t}{i:03d}", "stratum": "trap", "wave": 2024, "tier": t}
            for i in range(16)
        ]
    rows += [
        {"item_id": f"G{i:03d}", "stratum": "trap_gross", "wave": 2024, "tier": None}
        for i in range(6)
    ]
    rows += [
        {"item_id": f"P{i:03d}", "stratum": "practice", "wave": 2024, "tier": None}
        for i in range(12)
    ]
    key = pd.DataFrame(rows)
    practice = [r["item_id"] for r in rows if r["stratum"] == "practice"]
    main = [r["item_id"] for r in rows if r["stratum"] != "practice"]
    orders = {
        r: practice + [main[i] for i in np.random.default_rng(7 + j).permutation(len(main))]
        for j, r in enumerate(mod.RATERS)
    }
    return key, orders


def test_reduce_invariants() -> None:
    key, orders = _fake()
    done, buf = {"R1": 17, "R2": 59}, {"R1": 2, "R2": 5}
    ids, new, summ = mod.build(key, orders, done, buf, 1)
    ids2, new2, _ = mod.build(key, orders, done, buf, 1)
    assert ids == ids2 and new == new2  # same input, same result
    by_cell = summ["counts"]["by_cell"]
    for cell, t in summ["targets"].items():
        assert by_cell[cell] >= t  # the target is a floor
    assert (
        sum(v for k, v in summ["targets"].items() if k.startswith("flagged")) == mod.FLAGGED_TOTAL
    )
    for r in mod.RATERS:
        old_main, new_main = orders[r][12:], new[r][12:]
        k = done[r] + buf[r]
        assert new_main[:k] == old_main[:k]  # the finished prefix is unchanged
        assert set(old_main[:k]) <= set(ids)  # every finished item is included
        assert new[r][:12] == orders[r][:12] and set(new_main) == set(ids)
    assert 183 <= summ["n_total"] <= 183 + summ["n_forced"]
