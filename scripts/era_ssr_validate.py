"""Era SSR transfer checks from existing cos (no re-embed of answers).

  uv run / .venv-embed optional for ref self-cosine:
  .venv-embed/bin/python scripts/era_ssr_validate.py

Outputs: outputs/reports/ssr_validate.json (era keys; appendix macros via make_tables_era)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "era_holdout"
REPORTS = ROOT / "outputs" / "reports"
TAU = float(json.loads((OUT / "tau.json").read_text(encoding="utf-8"))["tau"])  # frozen on 2024
UNDECIDED = 0.01

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("era_ssr_validate")


def margin_from_cos(cos_path: Path) -> dict:
    gaps: list[float] = []
    n_und = 0
    n = 0
    for line in cos_path.open(encoding="utf-8"):
        r = json.loads(line)
        for c in (r.get("cos") or {}).values():
            if not c or len(c) < 2:
                continue
            a = np.asarray(c, dtype=float)
            s = np.sort(a)
            d = float(s[-1] - s[-2])
            gaps.append(d)
            n += 1
            if d < UNDECIDED:
                n_und += 1
    g = np.asarray(gaps, dtype=float)
    return {
        "n_sentences": int(n),
        "n_personas": sum(1 for _ in cos_path.open(encoding="utf-8")),
        "margin_median": float(np.median(g)) if len(g) else float("nan"),
        "margin_p5": float(np.percentile(g, 5)) if len(g) else float("nan"),
        "undecided_pct": float(100.0 * n_und / n) if n else float("nan"),
        "margin_ratio_at_tau": float(np.exp(np.median(g) / TAU)) if len(g) else float("nan"),
    }


def ref_max_cos(refs: dict) -> float:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util

    sp = importlib.util.spec_from_file_location("se", ROOT / "scripts" / "ssr_embed.py")
    se = importlib.util.module_from_spec(sp)
    assert sp.loader is not None
    sp.loader.exec_module(se)
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(se.DEFAULT_MODEL, device="cpu")
    m.max_seq_length = min(m.max_seq_length, 256)
    mx = 0.0
    n_ref = 0
    for opts in refs.values():
        E = se.option_embeddings(m, opts, 32)
        C = E @ E.T
        iu = np.triu_indices(len(opts), 1)
        if len(iu[0]):
            mx = max(mx, float(C[iu].max()))
        n_ref += sum(len(x) for x in opts.values())
    return mx, n_ref, len(refs), sum(len(o) for o in refs.values())


def _resolve(wave: int, name: str) -> Path:
    """Prefer w{wave}/; otherwise the flat path (2024 legacy)."""
    per_wave = OUT / f"w{wave}" / name
    if per_wave.exists():
        return per_wave
    return OUT / name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2024)
    args = ap.parse_args()
    refs_path = _resolve(args.wave, "ssr_refs.json")
    if not refs_path.exists():
        raise SystemExit(f"refs missing: {refs_path}")
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    cos_path = _resolve(args.wave, "cos_release_narrative.jsonl")
    out: dict = {
        "source": "era_holdout",
        "refs": str(refs_path),
        "cos": str(cos_path),
        "tau": TAU,
        "adr": "ADR-010",
    }
    try:
        mx, n_ref, n_items, n_opts = ref_max_cos(refs)
        out.update(
            ref_max_cos=mx,
            n_reference_sentences=n_ref,
            n_items=n_items,
            n_options=n_opts,
        )
        log.info("ref_max_cos=%.3f n_ref=%d", mx, n_ref)
    except Exception as exc:  # noqa: BLE001
        log.warning("ref self-cosine skipped: %s", exc)
        out["ref_max_cos"] = None

    # The manuscript appendix reads SSR statistics from this file. If they cannot
    # be obtained, **do not write them quietly** — the previous version wrote an
    # empty result, so appendix numbers could fall to 0 or a default (2026-09-11).
    if not cos_path.exists():
        raise SystemExit(
            f"cos missing: {cos_path}. Run the holdout first — partial results are not written."
        )
    out["margin"] = margin_from_cos(cos_path)
    out["wave"] = args.wave
    log.info("margin %s", out["margin"])
    if out.get("ref_max_cos") is None:
        log.warning(
            "ref self-cosine missing — run `.venv-embed/bin/python scripts/era_ssr_validate.py`"
        )
    REPORTS.mkdir(parents=True, exist_ok=True)
    dst = REPORTS / "ssr_validate.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("wrote %s", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
