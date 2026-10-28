"""Per-era narrative generation — era quant parquet → LLM (ADR-007).

Run:
  uv run python scripts/era_generate_narrative.py --wave 2024 --n 100
  uv run python scripts/era_generate_narrative.py --wave 2024 --n 0   # entire quant
  uv run python scripts/era_generate_narrative.py --all-weighted --n 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sfac_persona.config import settings
from sfac_persona.generate import GenerateConfig, load_run, run
from sfac_persona.narrative import FIELD_KEYS
from sfac_persona.schema import WEIGHTED_WAVES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_narrative")


def _report(run_name: str) -> str:
    recs = load_run(run_name)
    n = len(recs)
    ok = sum(r["ok"] for r in recs)
    errors = Counter(i["check"] for r in recs for i in r["issues"] if i["severity"] == "error")
    lines = [
        f"# Era narrative — {run_name}",
        "",
        f"- n={n}, pass={ok} ({100 * ok / max(n, 1):.1f}%)",
        f"- errors: {dict(errors)}",
        "- mean field chars: "
        + ", ".join(
            f"{k}={pd.Series([len(r['narrative'][k]) for r in recs]).mean():.0f}"
            for k in FIELD_KEYS[:4]
        ),
        "",
    ]
    return "\n".join(lines)


def generate_wave(
    wave: int, n: int, ablation: str, seed: int, model: str | None, concurrency: int | None
) -> Path:
    quant = settings.synthetic_dir / f"era_{wave}_quant" / f"personas_{ablation}.parquet"
    if not quant.exists():
        raise FileNotFoundError(quant)
    run_name = f"era_{wave}_narrative"
    if n <= 0:
        n = len(pd.read_parquet(quant))
    cfg = GenerateConfig(
        run_name=run_name,
        n=n,
        seed=seed + wave,
        model=model,
        concurrency=concurrency,
        quant_parquet=quant,
        ablation=ablation,
    )
    path = run(cfg)
    md = _report(run_name)
    out = settings.outputs_dir / "reports" / f"era_{wave}_narrative.md"
    out.write_text(md, encoding="utf-8")
    (settings.synthetic_dir / run_name / "README.md").write_text(
        f"# era_{wave} narrative\n\n정량: `{quant}`\n산출: `personas.jsonl`\n보고: `{out.name}`\n",
        encoding="utf-8",
    )
    log.info("%s", md)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=None)
    ap.add_argument("--all-weighted", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ablation", default="structure_T")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args()

    waves = list(WEIGHTED_WAVES) if args.all_weighted else [args.wave or 2024]
    for w in waves:
        generate_wave(w, args.n, args.ablation, args.seed, args.model, args.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
