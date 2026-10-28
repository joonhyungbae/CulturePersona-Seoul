"""A second model coder for the open-ended item: does the synthetic-trained classifier's margin survive it?

era_theme_coding.py scores the classifier trained on synthetic fields against real answers whose labels come
from one model coder (EXAONE 4.0 32B). This script codes the same held-out real answers with a second model
of a different family, reports the agreement between the two coders, and repeats the two contrasts the paper
quotes (synthetic-trained classifier minus majority, minus nearest-codebook matching) under the second
coder's labels and on the answers both coders label the same way.

Design fixed before running (2026-10-10):
  coder 2     Gemma 4 31B (same codebook prompt, temperature 0, think off), on the held-out half's real answers
  agreement   exact agreement and Cohen's kappa between the two coders over eight codes
  contrasts   accuracy of the synthetic-trained embedding classifier minus the majority label and minus
              nearest-codebook matching, under Gemma labels and on the agreed subset; 95% bootstrap over
              answers (2,000 draws); the classifier and embeddings are those of era_theme_coding.py (unchanged)
Every result is reported, whatever its sign.

Run:  SFP_THEME_CODER='hf.co/bartowski/google_gemma-4-31b-it-GGUF:IQ4_XS' uv run python scripts/era_theme_coder2.py code
      uv run python scripts/era_theme_coder2.py eval
Output: outputs/reports/era_theme_coder2.csv (aggregates only)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_theme_coding as TC
import era_theme_v2 as T

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("theme_coder2")

NAME = "real_gemma"
REPORT = settings.outputs_dir / "reports" / "era_theme_coder2.csv"
N_BOOT = 2000
SEED = 20261010


def stage_code() -> None:
    coder = os.environ.get("SFP_THEME_CODER")
    if not coder:
        raise SystemExit("set SFP_THEME_CODER to the second coder's model id")
    T.CODER = coder
    d = pd.read_parquet(TC.DATA)
    b = d[(d.source == "real") & (d.split == "B")]
    texts = dict(zip(b.id.astype(str), b.text, strict=True))
    codes = T.code_all(NAME, texts)
    log.info("coded %d held-out answers with %s", len(codes), coder)


def _kappa(a: np.ndarray, b: np.ndarray) -> float:
    labs = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == x).mean()) * float((b == x).mean()) for x in labs)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def stage_eval() -> None:
    d = pd.read_parquet(TC.DATA)
    e = np.load(TC.EMB)
    codes2 = T.code_all(NAME, {})
    b = (d.source == "real") & (d.split == "B")
    ids_b = d.id[b].astype(str).to_numpy()
    y1 = d.label[b].to_numpy()
    y2 = np.array([codes2.get(i, -1) for i in ids_b])
    ok = y2 >= 0
    rows: list[dict] = []
    rows.append(
        {
            "record": "agreement",
            "subset": "all8",
            "n": int(ok.sum()),
            "value": float((y1[ok] == y2[ok]).mean()),
            "kappa": _kappa(y1[ok], y2[ok]),
        }
    )
    # The paper's classifier: synthetic-trained logistic regression on sentence embeddings.
    syn = d.source == "syn"
    cb = d.source == "codebook"
    pred_syn = TC._fit_predict(e[syn.to_numpy()], d.label[syn].to_numpy(), e[b.to_numpy()])
    sims = e[b.to_numpy()] @ e[cb.to_numpy()].T
    pred_zero = d.label[cb].to_numpy()[sims.argmax(axis=1)]
    majority = int(pd.Series(d.label[syn]).mode().iloc[0])
    rng = np.random.default_rng(SEED)
    for subset, mask in (("gemma_labels", ok), ("agreed", ok & (y1 == y2))):
        y = y2[mask]
        for ref_name, ref in (
            ("majority", np.full(mask.sum(), majority)),
            ("zero_shot_codebook", pred_zero[mask]),
        ):
            hit_c = (pred_syn[mask] == y).astype(float)
            hit_r = (ref == y).astype(float)
            idx = np.arange(len(y))
            boots = [
                float(hit_c[s].mean() - hit_r[s].mean())
                for s in (rng.choice(idx, len(idx)) for _ in range(N_BOOT))
            ]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(
                {
                    "record": "contrast",
                    "subset": subset,
                    "n": int(mask.sum()),
                    "contrast": f"synthetic|embedding - {ref_name}",
                    "acc_synthetic": float(hit_c.mean()),
                    "acc_ref": float(hit_r.mean()),
                    "value": float(hit_c.mean() - hit_r.mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT, index=False)
    print(out.round(4).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["code", "eval"])
    a = ap.parse_args()
    {"code": stage_code, "eval": stage_eval}[a.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
