"""How far each answer source sits from the survey marginal: does the prose calibrate the reader?

The held-out test scores answers inside age x attendance cells. The recalibration (era_holdout_recal.py) showed
that most of what the narratives gain over anchor lists for two readers is a shift of the reader's pooled
answer distribution, its marginal, toward the survey's. This script measures that shift directly, on the
answers that already exist, so that the prose's effect on the reader's marginal is a number of its own rather
than a residual of the cell score.

Design fixed before running (2026-10-11):
  data        the 2024 SSR answers of every reader (generator, EXAONE, Gemma) for the arms that exist: no persona
              (prior), the same personas' anchor lists (synthanchors), anchors as fixed sentences (template),
              released narratives (release), real respondents' anchor lists (twin); 2022 and 2020 where files exist
  measure     per item, TVD between the arm's pooled answer distribution (mean of the personas' SSR probability
              vectors) and the weighted survey marginal; the marginal TVD of an arm is the mean over the nine items
  intervals   persona bootstrap of the pooled distribution (200 draws); contrasts release minus synthanchors,
              release minus template and release minus prior by the same persona draws where the arms share
              personas (prior answers are independent, so its draw is unpaired)
  reading     a lower marginal TVD = the reader's answers, pooled, sit closer to the survey; the question is
              whether prose moves it where the list does not. Reported for every reader and wave, whatever the sign.

Run: uv run python scripts/era_holdout_marginal.py
Output: outputs/reports/era_holdout_marginal.csv (aggregates only)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_holdout_readers as R
import era_holdout_run as H
from era_holdout_recal import survey_marginal

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_marginal")
REPORT = settings.outputs_dir / "reports" / "era_holdout_marginal.csv"
N_BOOT = 200
SEED = 20261011
ARMS = ("prior", "synthanchors", "template", "release", "twin")


def arm_path(wave: int, arm: str, reader: str) -> Path:
    if arm == "prior":
        return H.wave_dir(wave) / f"answers_prior_{reader}_ssr.jsonl"
    return R.answers_path(wave, arm, reader)


def load(wave: int, arm: str, reader: str, refs: dict, tau: float) -> list[dict] | None:
    p = arm_path(wave, arm, reader)
    if not p.exists() or not H._cos_path_for(p).exists():
        return None
    ans, _ = H.load_cos_probs(p, refs, tau)
    return ans


def vectors(recs: list[dict], items: list[str], order: dict) -> dict[str, np.ndarray]:
    """Per item, the matrix of persona probability vectors (rows = personas with an answer)."""
    out = {}
    for item in items:
        rows = [
            np.array([float(r["probs"][item].get(o, 0.0)) for o in order[item]])
            for r in recs
            if (r.get("probs") or {}).get(item)
        ]
        out[item] = np.vstack(rows) if rows else np.zeros((0, len(order[item])))
    return out


def marginal_tvd(
    vecs: dict[str, np.ndarray], marg: dict[str, np.ndarray], idx: dict[str, np.ndarray] | None
) -> float:
    vals = []
    for item, mat in vecs.items():
        if mat.shape[0] == 0:
            continue
        sel = mat if idx is None else mat[idx[item]]
        pooled = sel.mean(axis=0)
        vals.append(0.5 * float(np.abs(pooled - marg[item]).sum()))
    return float(np.mean(vals))


def main() -> int:
    items = list(H.QUESTIONS)
    tau, _ = H.load_frozen_tau()
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for wave in (2024, 2022, 2020):
        refs = json.loads((H.wave_dir(wave) / "ssr_refs.json").read_text(encoding="utf-8"))
        refs = {
            i: {str(k): (v if isinstance(v, list) else [str(v)]) for k, v in o.items()}
            for i, o in refs.items()
        }
        order = {i: list(o) for i, o in refs.items()}
        real = H.real_item_frame(wave)
        marg = {item: survey_marginal(real, item, order[item]) for item in items}
        for reader in ("qwen", "exaone", "gemma"):
            arms: dict[str, dict[str, np.ndarray]] = {}
            ids: dict[str, list] = {}
            for arm in ARMS:
                recs = load(wave, arm, reader, refs, tau)
                if recs is None:
                    continue
                arms[arm] = vectors(recs, items, order)
                ids[arm] = [
                    r.get("persona_id") for r in recs if (r.get("probs") or {}).get(items[0])
                ]
                point = marginal_tvd(arms[arm], marg, None)
                draws = []
                for _ in range(N_BOOT):
                    idx = {
                        it: rng.integers(0, m.shape[0], m.shape[0])
                        for it, m in arms[arm].items()
                        if m.shape[0]
                    }
                    draws.append(marginal_tvd(arms[arm], marg, idx))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                n = int(np.median([m.shape[0] for m in arms[arm].values()]))
                rows.append(
                    {
                        "wave": wave,
                        "reader": reader,
                        "record": "level",
                        "arm": arm,
                        "value": point,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n": n,
                    }
                )
                log.info(
                    "wave %d %s %-12s marginal TVD %.3f [%.3f, %.3f] n=%d",
                    wave,
                    reader,
                    arm,
                    point,
                    lo,
                    hi,
                    n,
                )
            if "release" not in arms:
                continue
            for ref in ("synthanchors", "template", "twin", "prior"):
                if ref not in arms:
                    continue
                a, b = arms["release"], arms[ref]
                paired = ref != "prior" and all(a[it].shape[0] == b[it].shape[0] for it in items)
                point = marginal_tvd(a, marg, None) - marginal_tvd(b, marg, None)
                draws = []
                for _ in range(N_BOOT):
                    ia = {
                        it: rng.integers(0, m.shape[0], m.shape[0])
                        for it, m in a.items()
                        if m.shape[0]
                    }
                    ib = (
                        ia
                        if paired
                        else {
                            it: rng.integers(0, m.shape[0], m.shape[0])
                            for it, m in b.items()
                            if m.shape[0]
                        }
                    )
                    draws.append(marginal_tvd(a, marg, ia) - marginal_tvd(b, marg, ib))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                rows.append(
                    {
                        "wave": wave,
                        "reader": reader,
                        "record": "contrast",
                        "arm": f"release-{ref}",
                        "value": point,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "paired": paired,
                    }
                )
                log.info(
                    "wave %d %s release-%s %.3f [%.3f, %.3f] paired=%s",
                    wave,
                    reader,
                    ref,
                    point,
                    lo,
                    hi,
                    paired,
                )
    pd.DataFrame(rows).to_csv(REPORT, index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
