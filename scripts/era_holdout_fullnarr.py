"""Held-out items with all twelve narrative fields shown to the answering model.

The release arm of the held-out test (era_holdout_run.py) shows the answering model the anchor list and two
narrative fields (summary, weekend routine). This script adds a third arm that shows the anchor list and all
twelve fields, each under its Korean field label, and changes nothing else: the same frozen evaluation subset
(300 per weighted wave), answering model, prompt, temperature, nine held-out items, frozen tau and cell
definition.

Design fixed before any answer was generated (2026-10-08):
  primary     2024, mean cell TVD over the nine items: (twelve fields) - (two fields), with a 95% interval from
              the cell-stratified paired persona bootstrap of era_holdout_positive_control.py (2,000 draws)
  secondary   the same contrast in 2022 and 2020; (twelve fields) - (twin anchors) in every wave
Every result is reported, whatever its sign.

Stages (each resumable):
  answer   uv run python scripts/era_holdout_fullnarr.py answer --wave 2024
  embed    uv run python scripts/era_holdout_fullnarr.py embed --wave 2024      (CPU)
  report   uv run python scripts/era_holdout_fullnarr.py report
Output: outputs/reports/era_holdout_fullnarr.csv (aggregates only)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import era_holdout_positive_control as P
import era_holdout_run as H

from sfac_persona.config import settings
from sfac_persona.narrative import FIELDS, anchor_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_fullnarr")

WAVES = (2024, 2022, 2020)
FULL_NAME = "answers_release_full_ssr.jsonl"
TWO_NAME = "answers_release_narrative_ssr.jsonl"
TWIN_NAME = "answers_twin_anchors_ssr.jsonl"


def full_blob(quant: dict, narrative: dict | None) -> str:
    """Anchor list plus every narrative field under its Korean label (the two-field arm shows two)."""
    bits = [anchor_block(quant)]
    if narrative:
        for f in FIELDS:
            text = narrative.get(f.key)
            if text:
                bits.append(f"{f.label_ko}: {text}")
    return "\n".join(bits)


def answer(wave: int) -> Path:
    recs = H.load_eval_bank(wave, H.N_RELEASE)
    # The answering call builds its persona text through H._persona_blob; swap in the twelve-field version.
    H._persona_blob = full_blob
    path = H.wave_dir(wave) / FULL_NAME
    return H._run_answer_file(
        path, [(r["persona_id"], r["quant"], r.get("narrative")) for r in recs]
    )


def embed(wave: int) -> None:
    H._embed(wave, pattern=FULL_NAME)


def report() -> None:
    tau, _meta = H.load_frozen_tau()
    rows: list[dict] = []
    for wave in WAVES:
        paths = {
            arm: H.wave_dir(wave) / name
            for arm, name in (("full", FULL_NAME), ("two", TWO_NAME), ("twin", TWIN_NAME))
        }
        if not all(H._cos_path_for(p).exists() for p in paths.values()):
            log.warning("wave %d: embeddings missing, skipped", wave)
            continue
        refs = json.loads((H.wave_dir(wave) / "ssr_refs.json").read_text(encoding="utf-8"))
        refs = {
            item: {str(k): (v if isinstance(v, list) else [str(v)]) for k, v in o.items()}
            for item, o in refs.items()
        }
        order = {item: list(opts) for item, opts in refs.items()}
        real = H.real_item_frame(wave)
        items = list(H.QUESTIONS)
        arms = {arm: H.load_cos_probs(p, refs, tau)[0] for arm, p in paths.items()}
        for arm, ans in arms.items():
            per_item = [H.cell_mean_tvd(ans, real, item, order[item]) for item in items]
            rows.append(
                {
                    "wave": wave,
                    "record": "level",
                    "arm": arm,
                    "contrast": "",
                    "value": float(np.nanmean(per_item)),
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "n": len(ans),
                }
            )
        # The positive-control bootstrap takes arms named release / oracle / twin; here release = two fields and
        # oracle = twelve fields, so "oracle-release" is (twelve - two) and "oracle-twin" is (twelve - twin).
        boot = P.bootstrap_diffs(
            {"release": arms["two"], "oracle": arms["full"], "twin": arms["twin"]},
            real,
            items,
            order,
        )
        names = {
            "oracle-release": "full-two",
            "oracle-twin": "full-twin",
            "release-twin": "two-twin",
        }
        for b in boot:
            rows.append(
                {
                    "wave": wave,
                    "record": "contrast",
                    "arm": "",
                    "contrast": names[b["contrast"]],
                    "value": b["diff"],
                    "ci_lo": b["ci_lo"],
                    "ci_hi": b["ci_hi"],
                    "n": b.get("n_boot", P.N_BOOT),
                }
            )
        log.info("wave %d done", wave)
    out = settings.outputs_dir / "reports" / "era_holdout_fullnarr.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s", out)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["answer", "embed", "report"])
    ap.add_argument("--wave", type=int, default=2024)
    args = ap.parse_args()
    if args.stage == "answer":
        answer(args.wave)
    elif args.stage == "embed":
        embed(args.wave)
    else:
        report()


if __name__ == "__main__":
    main()
