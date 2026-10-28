"""Rewrite the frozen eval bank (300 per wave) with a second and third generator and rerun the same experiments (re-review DA, 2026-09-24).

The manuscript's narrative-layer conclusions (holdout matches anchors · AUC loss
when given as examples · cannot answer the open item) are results of one
generator (Qwen3.8 27B). A same-class model from another family writes the
narratives with the same anchors, the same prompt, and the same temperature,
and the **answer model (SSR) is left unchanged** so only the narrative
generator changes.

  generate  --tag exaone --model hf.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF:IQ4_XS
            Dump the frozen bank's quant to parquet and pass it to `generate.run`. quant_id = frozen-bank persona_id (alignment key).
            Output data/synthetic/era_{wave}_narrative_{tag}/personas.jsonl
  holdout   --tag exaone
            SSR answers on the new narratives (answers_release_{tag}_ssr.jsonl; the answer model stays as before) → embedding is a separate command →
            evaluate reports per-cell TVD.
  evaluate  --tag exaone   → append rows to outputs/reports/era_holdout_generators.csv (wave, item, generator, cell_mean_tvd)

scripts/era_second_generator.sh ties the run order together.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_holdout_run as H

from sfac_persona.config import settings
from sfac_persona.generate import GenerateConfig, load_run, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("second_generator")

WAVES = (2024, 2022, 2020)


def run_name(wave: int, tag: str) -> str:
    return f"era_{wave}_narrative_{tag}"


def generate(
    tag: str, model: str, waves: tuple[int, ...], concurrency: int | None, limit: int = 0
) -> None:
    for wave in waves:
        bank = H.load_eval_bank(wave)
        if limit:
            bank = bank[:limit]  # smoke test
        out_dir = settings.synthetic_dir / run_name(wave, tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in bank:
            q = dict(r["quant"])
            q["persona_id"] = r["persona_id"]  # recorded as quant_id → aligns with the frozen bank
            rows.append(q)
        qp = out_dir / "quant_from_eval_bank.parquet"
        pd.DataFrame(rows).to_parquet(qp, index=False)
        cfg = GenerateConfig(
            run_name=run_name(wave, tag),
            n=len(rows),
            seed=20260924 + wave,
            model=model,
            concurrency=concurrency,
            quant_parquet=qp,
        )
        run(cfg)
        recs = load_run(run_name(wave, tag))
        ok = sum(1 for x in recs if x.get("ok"))
        log.info(
            "%s: n=%d ok=%d (%.1f%%)",
            run_name(wave, tag),
            len(recs),
            ok,
            100 * ok / max(len(recs), 1),
        )


def _narratives(wave: int, tag: str) -> dict[str, dict]:
    return {str(r.get("quant_id")): r.get("narrative") for r in load_run(run_name(wave, tag))}


def holdout(tag: str, waves: tuple[int, ...]) -> None:
    for wave in waves:
        bank = H.load_eval_bank(wave)
        narr = _narratives(wave, tag)
        units = []
        for r in bank:
            n = narr.get(r["persona_id"])
            if n:
                units.append((r["persona_id"], r["quant"], n))
        log.info("wave %d: %d/%d personas have a %s narrative", wave, len(units), len(bank), tag)
        path = H.wave_dir(wave) / f"answers_release_{tag}_ssr.jsonl"
        H._run_answer_file(path, units)


def evaluate(tag: str, waves: tuple[int, ...]) -> Path:
    tau, _ = H.load_frozen_tau()
    out = settings.outputs_dir / "reports" / "era_holdout_generators.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = []
    for wave in waves:
        refs = H.write_refs(wave)
        real = H.real_item_frame(wave)
        path = H.wave_dir(wave) / f"answers_release_{tag}_ssr.jsonl"
        if not H._cos_path_for(path).exists():
            raise SystemExit(f"embedding missing: {H._cos_path_for(path)} — run ssr_embed first")
        ans, opt_order = H.load_cos_probs(path, refs, tau)
        for item in H.QUESTIONS:
            rows.append(
                {
                    "wave": wave,
                    "item": item,
                    "generator": tag,
                    "cell_mean_tvd": H.cell_mean_tvd(ans, real, item, opt_order[item]),
                    "n": len(ans),
                }
            )
    new = pd.DataFrame(rows)
    if len(prev):
        prev = prev[prev.generator != tag]
    pd.concat([prev, new], ignore_index=True).to_csv(out, index=False)
    log.info("\n%s", new.groupby("wave").cell_mean_tvd.mean().round(4))
    return out


def report(tags: list[str]) -> None:
    """Per-generator summary markdown — the original generator (Qwen3.8) values come from era_holdout_crosswave.csv."""
    cw = pd.read_csv(settings.outputs_dir / "reports" / "era_holdout_crosswave.csv")
    g = pd.read_csv(settings.outputs_dir / "reports" / "era_holdout_generators.csv")
    lines = [
        "# 생성기별 holdout (셀별 TVD 가중 평균, 문항 평균)",
        "",
        "| wave | Qwen3.8 27B (release) | twin (anchors) | " + " | ".join(tags) + " |",
        "|---|---|---|" + "---|" * len(tags),
    ]
    for wave in WAVES:
        c = cw[cw.wave == wave]
        cells = [f"{c.tvd_release_narrative.mean():.3f}", f"{c.tvd_twin_anchors.mean():.3f}"]
        for t in tags:
            s = g[(g.wave == wave) & (g.generator == t)]
            cells.append(f"{s.cell_mean_tvd.mean():.3f}" if len(s) else "—")
        lines.append(f"| {wave} | " + " | ".join(cells) + " |")
    p = settings.outputs_dir / "reports" / "era_holdout_generators.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "holdout", "evaluate", "report"])
    ap.add_argument("--tag", required=False)
    ap.add_argument("--tags", default="exaone,gemma31")
    ap.add_argument("--model", default=None)
    ap.add_argument("--wave", type=int, action="append", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first n people per wave")
    args = ap.parse_args()
    waves = tuple(args.wave) if args.wave else WAVES
    if args.cmd == "generate":
        if not args.model or not args.tag:
            raise SystemExit("--tag and --model are required")
        generate(args.tag, args.model, waves, args.concurrency, args.limit)
    elif args.cmd == "holdout":
        holdout(args.tag, waves)
    elif args.cmd == "evaluate":
        evaluate(args.tag, waves)
    else:
        report([t.strip() for t in args.tags.split(",") if t.strip()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
