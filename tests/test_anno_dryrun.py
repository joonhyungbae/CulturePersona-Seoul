"""Synthetic scoring records pass load() of the pre-registered analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import anno_analysis
import anno_dryrun as dry


def test_synthetic_record_passes_load(tmp_path: Path) -> None:
    key_path = tmp_path / "answer_key.csv"
    pd.DataFrame(
        [
            {
                "item_id": "CLEAN1",
                "stratum": "clean",
                "true_age_band": "30대",
                "true_attend": "아니오",
            },
            {
                "item_id": "TRAP1",
                "stratum": "trap",
                "true_age_band": "40대",
                "true_attend": "예",
            },
            {
                "item_id": "PRAC1",
                "stratum": "practice",
                "true_age_band": "20대",
                "true_attend": "예",
            },
        ]
    ).to_csv(key_path, index=False)

    store = tmp_path / "store"
    stage1 = {"age_guess": "30대", "attend_guess": "아니오", "secs": 9}
    early = dry.annotation_record(
        "R1",
        "CLEAN1",
        stage1,
        dry._stage2(False, {sc: 5 for sc in anno_analysis.SCALES}, np.random.default_rng(0)),
        "2026-09-26T00:00:01.000Z",
    )
    late = dry.annotation_record(
        "R1",
        "CLEAN1",
        stage1,
        dry._stage2(True, {sc: 4 for sc in anno_analysis.SCALES}, np.random.default_rng(1)),
        "2026-09-26T00:00:02.000Z",
    )
    other = dry.annotation_record(
        "R2",
        "CLEAN1",
        {"age_guess": "20대", "attend_guess": "예", "secs": 8},
        dry._stage2(False, {sc: 5 for sc in anno_analysis.SCALES}, np.random.default_rng(2)),
        "2026-09-26T00:00:03.000Z",
    )
    practice = dry.annotation_record(
        "R1",
        "PRAC1",
        stage1,
        dry._stage2(True, {sc: 1 for sc in anno_analysis.SCALES}, np.random.default_rng(3)),
        "2026-09-26T00:00:04.000Z",
    )
    adj = dry.adjudication_record(
        "CLEAN1",
        {
            "contradiction": "no",
            "contra_fields": [],
            "contra_quote": "",
            "grounded": 4,
            "coherent": 4,
            "plausible": 4,
            "fluent": 4,
            "note": "",
        },
        "R1",
        "예행연습 합의",
        "R2",
        "2026-09-26T00:00:05.000Z",
    )
    dry.write_jsonl(store / "annotations" / "R1.jsonl", [early, practice, late])
    dry.write_jsonl(store / "annotations" / "R2.jsonl", [other])
    dry.write_jsonl(store / "adjudication.jsonl", [adj])

    df, got_adj, llm, bank = anno_analysis.load(store, key_path, subset_path=None)
    assert llm is None and bank is None
    assert set(df.item_id) == {"CLEAN1", "TRAP1"}
    row = df.set_index("item_id").loc["CLEAN1"]
    assert int(row.R1_contra) == 1
    assert int(row.R1_grounded) == 4
    assert row.R1_age == "30대"
    assert row.R1_attend == "아니오"
    assert int(row.R2_contra) == 0
    assert row.R2_age == "20대"
    assert got_adj["CLEAN1"]["consensus"]["contradiction"] == "no"
    assert got_adj["CLEAN1"]["changed_by"] == "R1"
    assert (
        json.loads(
            (store / "annotations" / "R1.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )["stage2"]["contradiction"]
        == "no"
    )

    bundle = dry.generate(
        pd.read_csv(key_path, dtype=str).fillna(""),
        dry.SCENARIOS["good"],
        np.random.default_rng(0),
    )
    gen_store = tmp_path / "generated"
    for rater, rows in bundle.annotations.items():
        dry.write_jsonl(gen_store / "annotations" / f"{rater}.jsonl", rows)
    dry.write_jsonl(gen_store / "adjudication.jsonl", bundle.adjudication)
    generated, _adj2, _llm2, _bank2 = anno_analysis.load(gen_store, key_path, subset_path=None)
    assert set(generated.item_id) == {"CLEAN1", "TRAP1"}
    assert generated.R1_contra.notna().all()
    assert generated.R2_grounded.notna().all()
    assert set(generated.R1_age).issubset(set(dry.AGES))
    assert set(generated.R1_attend).issubset(set(dry.ATTEND))
