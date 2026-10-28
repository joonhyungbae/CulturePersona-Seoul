"""Judge-reproducibility aggregates and stratified samples. Does not call a model."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import anno_judge_repro as repro


def _row(
    item_id: str,
    contradiction: str | None,
    fields: list[str] | None,
    scales: tuple[int, int, int, int] | None,
) -> dict:
    row: dict = {
        "item_id": item_id,
        "contradiction": contradiction,
        "contra_fields": fields,
    }
    names = ("grounded", "coherent", "plausible", "fluent")
    for name, value in zip(names, scales or (None, None, None, None), strict=True):
        row[name] = value
    return row


def test_stratified_sample_equal_and_stable() -> None:
    ids = {wave: [f"{wave}-{i}" for i in range(400)] for wave in (2020, 2022, 2024)}
    first = repro.stratified_sample(ids, 1000, 20260926)
    second = repro.stratified_sample(ids, 1000, 20260926)
    assert first == second
    assert len(first) == 999  # 1000 is not divisible by 3, so 333 per wave
    assert sum(item.startswith("2020-") for item in first) == 333
    assert sum(item.startswith("2022-") for item in first) == 333
    assert sum(item.startswith("2024-") for item in first) == 333
    small = repro.stratified_sample(ids, 3, 20260926)
    assert len(small) == 3
    assert len({item.split("-", 1)[0] for item in small}) == 3


def test_aggregate_agreement_kappa_and_failures() -> None:
    first = {
        "a": _row("a", "yes", ["x"], (5, 5, 5, 5)),
        "b": _row("b", "no", [], (4, 4, 4, 4)),
        "c": _row("c", "yes", ["y"], (3, 3, 3, 3)),
        "d": _row("d", "no", [], (2, 2, 2, 2)),
    }
    second = {
        "a": _row("a", "yes", ["x"], (5, 4, 5, 5)),
        "b": _row("b", "no", [], (4, 4, 4, 4)),
        "c": _row("c", "no", [], (1, 3, 3, 3)),
        "d": _row("d", None, None, None),
    }
    wave = {"a": 2020, "b": 2020, "c": 2022, "d": 2022}
    summary = repro.aggregate(first, second, wave, ["a", "b", "c", "d"])
    assert summary["n_sample"] == 4
    assert summary["n_fail"] == 1
    binary = summary["binary"]
    assert binary["n"] == 3
    assert binary["n_agree"] == 2
    assert abs(binary["agreement"] - 2 / 3) < 1e-9
    assert abs(binary["kappa"] - 0.4) < 1e-9
    assert abs(binary["yes_rate_first"] - 2 / 3) < 1e-9
    assert abs(binary["yes_rate_repro"] - 1 / 3) < 1e-9
    assert summary["fields_n"] == 2
    assert summary["fields_exact"] == 1
    scales = {row["name"]: row for row in summary["scales"]}
    assert scales["grounded"]["n_exact"] == 2
    assert scales["grounded"]["n_within1"] == 2
    assert scales["coherent"]["n_exact"] == 2
    assert scales["coherent"]["n_within1"] == 3
    assert summary["by_wave"][2020]["n"] == 2
    assert summary["by_wave"][2020]["agreement"] == 1.0
    assert summary["by_wave"][2022]["n"] == 1
    assert summary["by_wave"][2022]["agreement"] == 0.0


def test_kappa_undefined_when_no_variation() -> None:
    assert math.isnan(repro.cohen_kappa(["yes", "yes"], ["yes", "yes"]))
    both = repro.cohen_kappa(["yes", "no"], ["yes", "no"])
    assert abs(both - 1.0) < 1e-9
