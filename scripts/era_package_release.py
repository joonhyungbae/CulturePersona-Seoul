"""Era multi-wave local release package (B2). HF/Zenodo is era_push_release.py.

Output data/release/<version>/:
  personas_{2020,2022,2024}.parquet
  personas_all.parquet
  personas_sample_100.jsonl
  grounding/era_{wave}/… + structure_priors.json
  eval/era_holdout_registry.json · tau.json · w*/eval_bank.jsonl · w*/ssr_refs.json
  (person-level distilled prediction npz and internal twin answers are **not included**)
  code_snapshot/
  DATASET_CARD.md
  LICENSE
  MANIFEST.sha256

Run:
  uv run python scripts/era_package_release.py --version v0.1-era
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sfac_persona.config import settings
from sfac_persona.generate import load_run
from sfac_persona.narrative import FIELD_KEYS, WISH_FIELD
from sfac_persona.schema import ERA_CORE_ANCHORS, WEIGHTED_WAVES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("era_package")

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_wish(wave: int) -> dict[str, dict]:
    """Table-grounded slot (ADR-033): 2024 only. persona_id → {wish_theme, text}. Empty dict if absent."""
    p = settings.synthetic_dir / f"era_{wave}_narrative" / "wish_for_seoul.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["persona_id"]] = r
    return out


def records_to_frame(recs: list[dict], wave: int) -> pd.DataFrame:
    rows = []
    wish = load_wish(wave)
    for r in recs:
        row: dict = {
            "persona_id": r["persona_id"],
            "survey_year": wave,
            "index": r.get("index"),
        }
        row.update({a: r["quant"].get(a) for a in ERA_CORE_ANCHORS})
        row["n_genres_band"] = r["quant"].get("n_genres_band")
        narr = r.get("narrative") or {}
        row.update({f"narr_{k}": narr.get(k, "") for k in FIELD_KEYS})
        wr = wish.get(r["persona_id"])
        row["wish_theme"] = None if wr is None else int(wr["wish_theme"])
        row[f"narr_{WISH_FIELD.key}"] = None if wr is None else wr["text"]
        row["branches"] = "|".join(r.get("branches") or [])
        row["validator_pass"] = bool(r.get("ok"))
        issues = r.get("issues") or []
        row["validator_errors"] = "|".join(
            f"{i['check']}:{i['field']}" for i in issues if i.get("severity") == "error"
        )
        row["validator_warnings"] = "|".join(
            f"{i['check']}:{i['field']}" for i in issues if i.get("severity") == "warning"
        )
        row["cliche_hits"] = len(r.get("cliches") or [])
        row["generator"] = r.get("model")
        rows.append(row)
    return pd.DataFrame(rows)


def copy_era_grounding(dst: Path) -> None:
    """Export **only the public** grounding tables.

    Fix 2026-09-11: the previous version copied the whole `era_{wave}` directory.
    That included the internal `T6_pair_joints.parquet` and `T1`–`T5`·`T1_root_joint`,
    where cells with n<30 were present with only a `suppressed` flag. The release
    was therefore shipping unweighted cells under 30 (public grounding tables
    keep unweighted cells ≥ 30). Now only
    `*_release.parquet` plus specs and metadata are copied.
    """
    dst.mkdir(parents=True, exist_ok=True)
    # R0 is the resident-registration population (public aggregate), not survey data.
    allowed_exact = {"provenance.json", "reproduce_gate.csv", "R0_population_root.parquet"}
    for wave in WEIGHTED_WAVES:
        src = settings.grounding_dir / f"era_{wave}"
        if not src.is_dir():
            log.warning("missing grounding %s", src)
            continue
        out = dst / f"era_{wave}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        n_pub = 0
        for p in sorted(src.iterdir()):
            if not p.is_file():
                continue
            keep = (
                p.name in allowed_exact
                or p.name.startswith("sampler_specs_")
                or p.name == "T7_theme_codebook.csv"
                or (p.suffix == ".parquet" and p.stem.endswith("_release"))
            )
            if not keep:
                continue
            shutil.copy(p, out / p.name)
            n_pub += p.suffix == ".parquet"
        log.info("grounding era_%s: %d public tables", wave, n_pub)
        if n_pub == 0:
            raise SystemExit(
                f"era_{wave}: no public grounding tables. "
                "Run `write_release_variants` first (era_tables)."
            )
    for name in ("structure_priors.json",):
        p = settings.grounding_dir / name
        if p.exists():
            shutil.copy(p, dst / name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0.1-era")
    ap.add_argument("--min-n", type=int, default=15000, help="refuse if any wave below this")
    args = ap.parse_args()

    frames: list[pd.DataFrame] = []
    for wave in WEIGHTED_WAVES:
        run_name = f"era_{wave}_narrative"
        path = settings.synthetic_dir / run_name / "personas.jsonl"
        if not path.exists():
            raise SystemExit(f"missing {path} — finish B1 first")
        recs = load_run(run_name)
        if len(recs) < args.min_n:
            raise SystemExit(f"{run_name} n={len(recs)} < {args.min_n}")
        frames.append(records_to_frame(recs, wave))
        log.info(
            "%s n=%d pass=%.1f%%", run_name, len(recs), 100 * sum(r["ok"] for r in recs) / len(recs)
        )

    rel = settings.data_root / "release" / args.version
    if rel.exists():
        shutil.rmtree(rel)
    rel.mkdir(parents=True)

    # Mark the frozen evaluation subset (300 per wave) that the paper's narrative-layer results are scored on.
    for wave, df in zip(WEIGHTED_WAVES, frames, strict=True):
        bank_path = settings.data_root / "eval" / "era_holdout" / f"w{wave}" / "eval_bank.jsonl"
        ids = set()
        if bank_path.exists():
            ids = {
                json.loads(line)["persona_id"]
                for line in bank_path.open(encoding="utf-8")
                if line.strip()
            }
        df["eval_subset"] = df["persona_id"].isin(ids)
        log.info(
            "wave %s: %d personas in the frozen evaluation subset",
            wave,
            int(df["eval_subset"].sum()),
        )
    for wave, df in zip(WEIGHTED_WAVES, frames, strict=True):
        df.to_parquet(rel / f"personas_{wave}.parquet", index=False)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_parquet(rel / "personas_all.parquet", index=False)
    with open(rel / "personas_sample_100.jsonl", "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(rec, ensure_ascii=False, default=str) + "\n"
            for rec in all_df.sample(n=min(100, len(all_df)), random_state=20260910).to_dict(
                "records"
            )
        )

    # Diagnostic quantitative personas for the two unweighted waves (margins specification only).
    diag = rel / "diagnostic"
    for wave in (2016, 2018):
        src = settings.synthetic_dir / f"era_{wave}_quant" / "personas_margins.parquet"
        if src.exists():
            diag.mkdir(exist_ok=True)
            shutil.copy(src, diag / f"personas_{wave}_quant.parquet")

    copy_era_grounding(rel / "grounding")
    eval_dst = rel / "eval"
    eval_dst.mkdir(parents=True)
    for src in (
        settings.data_root / "eval" / "era_holdout_registry.json",
        settings.data_root / "eval" / "era_holdout" / "tau.json",
    ):
        if src.exists():
            shutil.copy(src, eval_dst / src.name)
    # Frozen eval bank and SSR reference sentences — needed to reproduce the paper's holdout numbers (ADR-012).
    # Person-level (`pred_real_distilled_ssr.npz`) and internal twin answers are not included.
    # Persona answers to the held-out items (release arm, every generator) and their cosines. Twin answers are
    # excluded: the twin arm reads real respondents' anchors.
    for wave in (2020, 2022, 2024):
        wdir = settings.data_root / "eval" / "era_holdout" / f"w{wave}"
        names = ["eval_bank.jsonl", "ssr_refs.json"]
        names += sorted(p.name for p in wdir.glob("answers_release_*_ssr.jsonl"))
        names += sorted(p.name for p in wdir.glob("cos_release_*.jsonl"))
        # Option log-probability answers (second elicitation); synthetic personas only, twins excluded.
        names += sorted(p.name for p in wdir.glob("answers_release_*_logprob.jsonl"))
        names += sorted(p.name for p in wdir.glob("answers_synthanchors_*_logprob.jsonl"))
        names += sorted(p.name for p in wdir.glob("answers_prior_*_logprob.jsonl"))
        # Template arm (anchors as fixed sentences) and its cosines; synthetic personas only.
        names += sorted(p.name for p in wdir.glob("answers_template_*_ssr.jsonl"))
        names += sorted(p.name for p in wdir.glob("cos_template_*.jsonl"))
        names += sorted(p.name for p in wdir.glob("answers_prior_*_ssr.jsonl"))
        names += sorted(p.name for p in wdir.glob("cos_prior_*.jsonl"))
        for name in names:
            src = wdir / name
            if src.exists():
                dst = eval_dst / f"w{wave}"
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst / name)
    # Aggregate evaluation tables (PPI and held-out scores), the human audit's registered and exploratory
    # result tables (aggregates only), model-judge runs on synthetic personas, and the rater guideline.
    reports = settings.outputs_dir / "reports"
    for name in (
        "era_holdout_ppi.csv",
        "era_ppi_wave_concentration.csv",
        "era_holdout_scores.csv",
        "era_holdout_baselines.csv",
        "era_holdout_crosswave.csv",
        "anno_results.csv",
        "anno_exploratory.csv",
        "anno_exploratory_tests.csv",
        "era_tables_only_bank.csv",
        "era_tstr_tables_only.csv",
        "era_genre_portfolio_tables_only.csv",
        "era_design_oos_pairs_tables_only.csv",
        "era_holdout_marginal.csv",
        "era_family_mentions.csv",
        "era_rapid.csv",
        "era_rapid.md",
    ):
        if (reports / name).exists():
            shutil.copy(reports / name, eval_dst / name)
    judge_dst = eval_dst / "model_judge"
    for name in ("llm_judge.jsonl", "llm_bank.jsonl", "llm_bank_repro.jsonl"):
        src = settings.outputs_dir / "anno" / name
        if src.exists():
            judge_dst.mkdir(exist_ok=True)
            shutil.copy(src, judge_dst / name)
    guide = ROOT / "annotation" / "src" / "guide.md"
    if guide.exists():
        shutil.copy(guide, eval_dst / "audit_guideline_ko.md")

    # Code snapshot: every tracked file outside data/ (library, scripts, tests, microdata preparation, the
    # audit tool, the paper-table script, the registered analysis), so that the archive carries all study code.
    snap = rel / "code_snapshot"
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for f_ in tracked:
        if f_.startswith("data/"):
            continue
        dst = snap / f_
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f_, dst)

    n_pass = int(all_df.validator_pass.sum())
    models = sorted({str(m) for m in all_df.generator.dropna().unique()})
    card = f"""# CulturePersona-Seoul — {args.version}

**What.** {len(all_df):,} synthetic personas across weighted survey waves {list(WEIGHTED_WAVES)}
({", ".join(f"{w}: {len(f):,}" for w, f in zip(WEIGHTED_WAVES, frames, strict=True))}).
Each row has {len(ERA_CORE_ANCHORS)} quantitative anchors from aggregate tables only and {len(FIELD_KEYS)} Korean narrative fields.
The 2024 rows also carry, as a demonstration, a theme code `wish_theme` (drawn from the released table of the survey's own
open-ended answers, `grounding/era_2024/T7_theme_*`, codebook in `T7_theme_codebook.csv`), and a field
`narr_wish_for_seoul` written from it; both are null in 2020 and 2022, whose questionnaires had no such item.
This field is outside the registered human audit (which covers the twelve anchor-only fields).

**Grounded and invented content.** The anchors are the grounded layer; the narratives elaborate them, and the
prompt tells the model to invent what the anchors do not give. Three fields ask for information no anchor carries:
`narr_family_context` (marriage, co-resident children), `narr_participation_history` (playing, creating, clubs) and
`narr_view_on_seoul` (satisfaction, status); `eval/era_family_mentions.csv` counts how often family status is stated.
Treat these three fields as invented text, not survey evidence, and drop them where family status or participation matters.

**Grounding target.** Weighted estimates of an online panel of Seoul residents (not the Seoul population), from the
Seoul Citizens' Cultural Participation Survey. These are cultural-participation personas from one survey; they do not
represent other cities, populations or survey domains, and must not be read as evidence about cultural participation itself.
No synthetic individual derives from an individual response record. Cells below the project cell floor are withheld; geography stops at living area.

**Generator.** {", ".join(models) or "local open model"}. Validator pass: {n_pass:,}/{len(all_df):,}.

**Evaluation layer.** `eval/`: held-out item registry, frozen SSR temperature (`tau.json`), per wave the frozen
evaluation subset (`eval_bank.jsonl`, also marked by `eval_subset` in the persona files), SSR reference statements, the
personas' answers and cosines (release arm), PPI and held-out score tables, model-judge runs and the rater guideline
(Korean). Twin-arm answers are not included because they read real respondents' anchors.

**Code.** `code_snapshot/` holds all study code, from microdata preparation to the script that writes the paper's numbers.

**Disclosure.** No individual record is released. `eval/era_rapid.md` reports a RAPID audit (Templ, Thees and Mueller
2026, arXiv:2602.09235) of this dump against the survey's respondents: an attacker who knows a respondent's sex, age
band, living area, education and occupation and trains on these personas is compared with one who uses the released
tables alone and with the reference of releasing the records themselves.

**Source and citation.** Seoul Citizens' Cultural Participation Survey, 2020, 2022 and 2024 waves (Seoul Foundation for
Arts and Culture; reports published 2021, 2023 and 2025), used under the publisher's data-use terms. Cite the survey
together with this dataset. This dataset is not an official statistic or product of the Seoul Foundation for Arts and
Culture and implies no approval or endorsement by it.

**Licence.** Data (grounding tables, sampler specifications, theme tables, personas and narratives): CC BY-NC 4.0.
Code and evaluation layer: CC BY 4.0; persona records and table values inside `eval/` remain CC BY-NC 4.0. See `LICENSE`.
"""
    (rel / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    shutil.copy(ROOT / "LICENSE", rel / "LICENSE")

    lines = []
    for p in sorted(rel.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            lines.append(f"{sha(p)}  {p.relative_to(rel).as_posix()}")
    (rel / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s (%d files in manifest)", rel, len(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
