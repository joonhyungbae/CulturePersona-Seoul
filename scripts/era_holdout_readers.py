"""Does prose read more easily than an anchor list? Further reader models and waves.

The positive control (era_holdout_positive_control.py) found that a second reader, EXAONE 4.0 32B, answers the
held-out items better from the release prose (anchors plus two fields) than from the twin arm's anchor lists,
on 2024 only. This script repeats that comparison with a third reader, Gemma 4 31B, and with EXAONE on 2022 and
2020, changing nothing else: same frozen evaluation subset and twin respondents, same prompt, temperature,
items, reference statements and frozen tau.

Design fixed before running (2026-10-09):
  primary     2024, Gemma reader: mean cell TVD of the release arm minus the twin arm, 95% interval from the
              cell-stratified paired persona bootstrap (2,000 draws); negative = prose read more easily
  secondary   EXAONE on 2022 and 2020; Gemma on 2022 and 2020 if time allows
Every result is reported, whatever its sign.

Addition fixed before running (2026-10-09): a `synthanchors` arm shows each reader the release personas'
anchor lists with no narrative, so that release minus synthanchors isolates the prose on the same people
(paired by persona), and synthanchors minus twin isolates the people (sampled vs real anchors). Readers:
the generator (qwen), EXAONE and Gemma on 2024 (primary: EXAONE and Gemma, release minus synthanchors);
EXAONE and Gemma on 2022 and 2020 as secondary.

Stages:
  answer   uv run python scripts/era_holdout_readers.py answer --reader gemma --wave 2024
  embed    uv run python scripts/era_holdout_readers.py embed --reader gemma --wave 2024   (CPU)
  report   uv run python scripts/era_holdout_readers.py report
Output: outputs/reports/era_holdout_readers.csv (aggregates only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_holdout_positive_control as P
import era_holdout_run as H
from era_template_prose import template_fields

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_readers")

READERS = {
    "qwen": settings.llm_default_model,
    "exaone": P.EXAONE_ID,
    "gemma": "hf.co/bartowski/google_gemma-4-31b-it-GGUF:IQ4_XS",
}
WAVES = (2024, 2022, 2020)
CONCURRENCY = 2


def answers_path(wave: int, arm: str, reader: str) -> Path:
    # 2024 EXAONE release/twin answers come from the positive control; the generator's 2024 release/twin
    # answers are the main held-out files. Everything else is written by this script.
    if arm in ("release", "twin"):
        if wave == 2024 and reader == "exaone":
            return H.wave_dir(wave) / f"answers_{arm}_exaone_ssr.jsonl"
        if reader == "qwen":
            name = (
                "answers_release_narrative_ssr.jsonl"
                if arm == "release"
                else "answers_twin_anchors_ssr.jsonl"
            )
            return H.wave_dir(wave) / name
    return H.wave_dir(wave) / f"answers_{arm}_reader_{reader}_ssr.jsonl"


def twin_units(wave: int) -> list[tuple[str, dict, None]]:
    """The wave's frozen twin respondents (ids from the published twin answer file), anchors reattached."""
    frozen = H.wave_dir(wave) / "answers_twin_anchors_ssr.jsonl"
    seen: list[str] = []
    have: set[str] = set()
    for rec in H._read_jsonl(frozen):
        pid = str(rec["persona_id"])
        if pid not in have:
            have.add(pid)
            seen.append(pid)
    frame = H._attach_row_uid(build_official_anchor_frame(wave), wave).copy()
    frame["_pid"] = "twin-" + frame["row_uid"].astype(str)
    frame = frame.drop_duplicates("_pid").set_index("_pid")
    missing = [pid for pid in seen if pid not in frame.index]
    if missing:
        raise SystemExit(f"wave {wave}: twin ids missing from the anchor frame: {len(missing)}")
    ban = set(H.ANCHOR_SKIP) | set(H.QUESTIONS)
    units = []
    for pid in seen:
        row = frame.loc[pid]
        units.append(
            (pid, {k: str(v) for k, v in row.items() if k not in ban and pd.notna(v)}, None)
        )
    return units


def unload_all_but(keep: str) -> None:
    for name in P.loaded_models():
        if not P._same_model(name, keep):
            P.unload(name)


def answer(reader: str, wave: int, arms: tuple[str, ...] = ()) -> None:
    os.environ["SFP_LLM_NUM_CTX"] = "4096"
    mid = READERS[reader]
    unload_all_but(mid)
    bank = H.load_eval_bank(wave)
    units = {
        "release": [(r["persona_id"], r["quant"], r.get("narrative")) for r in bank],
        "twin": twin_units(wave),
        "synthanchors": [(r["persona_id"], r["quant"], None) for r in bank],
        # Anchors as fixed sentences in the two slots the reader sees (era_template_prose.py, design there).
        "template": [(r["persona_id"], r["quant"], template_fields(r["quant"])) for r in bank],
    }
    if arms:
        units = {k: v for k, v in units.items() if k in arms}
    prev, prev_conc = settings.llm_default_model, H.CONCURRENCY
    settings.llm_default_model, H.CONCURRENCY = mid, CONCURRENCY
    try:
        for arm, us in units.items():
            path = answers_path(wave, arm, reader)
            log.info("answer reader=%s wave=%d arm=%s -> %s", reader, wave, arm, path.name)
            H._run_answer_file(path, us)
    finally:
        settings.llm_default_model, H.CONCURRENCY = prev, prev_conc


def embed(reader: str, wave: int, arms: tuple[str, ...] = ("release", "twin")) -> None:
    py = ROOT / ".venv-embed" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for arm in arms:
        path = answers_path(wave, arm, reader)
        subprocess.check_call(
            [
                str(py),
                str(ROOT / "scripts" / "ssr_embed.py"),
                "--dir",
                str(H.wave_dir(wave)),
                "--pattern",
                path.name,
            ],
            cwd=str(ROOT),
            env=env,
        )


def report() -> None:
    tau, _ = H.load_frozen_tau()
    rows: list[dict] = []

    def row(wave, reader, record, arm, value, lo=np.nan, hi=np.nan, n=0):
        rows.append(
            {
                "wave": wave,
                "reader": reader,
                "record": record,
                "arm": arm,
                "value": value,
                "ci_lo": lo,
                "ci_hi": hi,
                "n": n,
            }
        )

    for wave in WAVES:
        refs = json.loads((H.wave_dir(wave) / "ssr_refs.json").read_text(encoding="utf-8"))
        refs = {
            i: {str(k): (v if isinstance(v, list) else [str(v)]) for k, v in o.items()}
            for i, o in refs.items()
        }
        order = {i: list(o) for i, o in refs.items()}
        real = H.real_item_frame(wave)
        items = list(H.QUESTIONS)
        for reader in READERS:
            paths = {
                arm: answers_path(wave, arm, reader)
                for arm in ("release", "twin", "synthanchors", "template")
            }
            have = {arm: p for arm, p in paths.items() if H._cos_path_for(p).exists()}
            if not {"release", "twin"} <= set(have):
                continue
            arms = {arm: H.load_cos_probs(p, refs, tau)[0] for arm, p in have.items()}
            for arm, ans in arms.items():
                lvl = float(np.nanmean([H.cell_mean_tvd(ans, real, it, order[it]) for it in items]))
                row(wave, reader, "level", arm, lvl, n=len(ans))
            boot = P.bootstrap_diffs(
                {"release": arms["release"], "oracle": arms["release"], "twin": arms["twin"]},
                real,
                items,
                order,
            )
            b = next(x for x in boot if x["contrast"] == "release-twin")
            row(
                wave,
                reader,
                "contrast",
                "release-twin",
                b["diff"],
                b["ci_lo"],
                b["ci_hi"],
                b.get("n_boot", P.N_BOOT),
            )
            log.info(
                "wave=%d reader=%s release-twin=%.4f [%.4f, %.4f]",
                wave,
                reader,
                b["diff"],
                b["ci_lo"],
                b["ci_hi"],
            )
            if "synthanchors" in arms:
                # bootstrap_diffs pairs its "release" and "oracle" arms by persona: with release = synthanchors and
                # oracle = release narratives, "oracle-release" is release minus synthanchors on the same people.
                boot = P.bootstrap_diffs(
                    {
                        "release": arms["synthanchors"],
                        "oracle": arms["release"],
                        "twin": arms["twin"],
                    },
                    real,
                    items,
                    order,
                )
                names = {
                    "oracle-release": "release-synthanchors",
                    "release-twin": "synthanchors-twin",
                }
                for x in boot:
                    if x["contrast"] in names:
                        row(
                            wave,
                            reader,
                            "contrast",
                            names[x["contrast"]],
                            x["diff"],
                            x["ci_lo"],
                            x["ci_hi"],
                            x.get("n_boot", P.N_BOOT),
                        )
                        log.info(
                            "wave=%d reader=%s %s=%.4f [%.4f, %.4f]",
                            wave,
                            reader,
                            names[x["contrast"]],
                            x["diff"],
                            x["ci_lo"],
                            x["ci_hi"],
                        )
            if "template" in arms:
                # release minus template on the same people: what generated prose adds beyond sentence form.
                boot = P.bootstrap_diffs(
                    {"release": arms["template"], "oracle": arms["release"], "twin": arms["twin"]},
                    real,
                    items,
                    order,
                )
                x = next(b for b in boot if b["contrast"] == "oracle-release")
                row(
                    wave,
                    reader,
                    "contrast",
                    "release-template",
                    x["diff"],
                    x["ci_lo"],
                    x["ci_hi"],
                    x.get("n_boot", P.N_BOOT),
                )
                x = next(b for b in boot if b["contrast"] == "release-twin")
                row(
                    wave,
                    reader,
                    "contrast",
                    "template-twin",
                    x["diff"],
                    x["ci_lo"],
                    x["ci_hi"],
                    x.get("n_boot", P.N_BOOT),
                )
    out = settings.outputs_dir / "reports" / "era_holdout_readers.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["answer", "embed", "report"])
    ap.add_argument("--reader", choices=list(READERS), default="gemma")
    ap.add_argument("--wave", type=int, default=2024)
    ap.add_argument(
        "--arms", default="", help="comma-separated subset of release,twin,synthanchors"
    )
    a = ap.parse_args()
    arms = tuple(x for x in a.arms.split(",") if x)
    if a.stage == "answer":
        answer(a.reader, a.wave, arms)
    elif a.stage == "embed":
        embed(a.reader, a.wave, arms or ("release", "twin"))
    else:
        report()


if __name__ == "__main__":
    main()
