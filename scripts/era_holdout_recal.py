"""Do the personas carry cell information once the reader's own marginal is removed?

Every arm of the held-out test sits far above the marginal lookup, and the reader shown no persona at all
sits close to the persona arms. That level mixes two things: the reader getting the survey marginal wrong,
and how well the arms tell the cells apart. This script separates them on the answers that already exist:
each person's answer distribution is recalibrated by the ratio of the survey marginal to the arm's own pooled
distribution (the same correction for every person in the arm, so it cannot add cell information), and the
cell TVD is recomputed. A lookup that ignores the cells becomes the marginal lookup under this correction, so
anything the recalibrated arms gain over it is cell information the reader took from the persona.

Design fixed before running (2026-10-10):
  arms        release, synthanchors, twin under SSR and under option log-probabilities; prior (log-probabilities)
  reader      the generator (qwen); EXAONE where its files exist
  correction  per arm and item: p'(o) = p(o) * m(o) / q(o), renormalised, with m the weighted survey marginal
              of the item and q the arm's mean answer distribution over its personas
  primary     recalibrated mean cell TVD of the release arm (2024, SSR, generator) read against the marginal
              lookup and the real-sample ceiling: below the ceiling = the personas carry cell information this
              instrument detects; at the lookup = they carry none it detects
  secondary   release minus synthanchors and release minus twin after recalibration (cell-stratified paired
              bootstrap, 2,000 draws); the same under log-probabilities; EXAONE
Addition fixed before running (2026-10-10, second pass): the SSR recalibration is repeated for every reader
  (generator, EXAONE, Gemma) and wave (2024, 2022, 2020) whose answer files exist; a cell-shuffle null
  (persona-to-cell labels permuted within the arm, 1,000 draws) gives the chance level of each arm's raw cell
  TVD and a one-sided p-value; the cell-contrast correlation carries a persona bootstrap interval (200 draws), and the prose-minus-list
  correlation difference a persona-paired bootstrap interval (200 draws).
Every result is reported, whatever its sign.

Run: uv run python scripts/era_holdout_recal.py
Output: outputs/reports/era_holdout_recal.csv (aggregates only)
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

import era_holdout_logprob as L
import era_holdout_positive_control as P
import era_holdout_readers as R
import era_holdout_run as H

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_recal")

REPORT = settings.outputs_dir / "reports" / "era_holdout_recal.csv"
SSR_FILES = {
    "qwen": {
        "release": "release_narrative",
        "twin": "twin_anchors",
        "synthanchors": "synthanchors_reader_qwen",
    },
    "exaone": {
        "release": "release_exaone",
        "twin": "twin_exaone",
        "synthanchors": "synthanchors_reader_exaone",
    },
}


def survey_marginal(real: pd.DataFrame, item: str, order: list[str]) -> np.ndarray:
    rw = real[real[item].notna()]
    d = rw.groupby(item)["w"].sum()
    m = np.array([float(d.get(o, 0.0)) for o in order])
    return m / m.sum()


def recalibrate(recs: list[dict], real: pd.DataFrame, items: list[str], order: dict) -> list[dict]:
    out = []
    pooled: dict[str, np.ndarray] = {}
    for item in items:
        ps = [
            np.array([float(r["probs"][item].get(o, 0.0)) for o in order[item]])
            for r in recs
            if (r.get("probs") or {}).get(item)
        ]
        pooled[item] = np.mean(ps, axis=0) if ps else None
    for r in recs:
        probs = {}
        for item in items:
            p = (r.get("probs") or {}).get(item)
            if not p or pooled[item] is None:
                probs[item] = None
                continue
            pv = np.array([float(p.get(o, 0.0)) for o in order[item]])
            m = survey_marginal(real, item, order[item])
            q = pooled[item]
            w = np.where(q > 0, m / np.where(q > 0, q, 1.0), 0.0)
            pv = pv * w
            s = pv.sum()
            pv = pv / s if s > 0 else np.full(len(pv), 1.0 / len(pv))
            probs[item] = dict(zip(order[item], pv.tolist(), strict=True))
        out.append({"persona_id": r["persona_id"], "quant": r["quant"], "probs": probs})
    return out


def cell_contrast(
    recs: list[dict], real: pd.DataFrame, items: list[str], order: dict
) -> tuple[float, float, float]:
    """Cell deviations from the arm's own mean against the real deviations from the survey marginal.

    Returns the weighted correlation over (cell, option) pairs pooled across items, the weighted MAE of the
    arm's deviations against the real ones, and the same MAE for zero deviations (the lookup)."""
    xs, ys, ws = [], [], []
    for item in items:
        o = order[item]
        pm = {}
        for r in recs:
            p = (r.get("probs") or {}).get(item)
            if not p:
                continue
            cell = tuple(r["quant"].get(c, "") for c in H.CELL)
            pm.setdefault(cell, []).append(np.array([float(p.get(k, 0.0)) for k in o]))
        if not pm:
            continue
        arm_mean = np.mean([v for vs in pm.values() for v in vs], axis=0)
        m = survey_marginal(real, item, o)
        for cell, vs in pm.items():
            rw = real[
                (real["age_band"] == cell[0]) & (real["attend_any"] == cell[1]) & real[item].notna()
            ]
            if len(rw) < 5:
                continue
            d = rw.groupby(item)["w"].sum()
            rv = np.array([float(d.get(k, 0.0)) for k in o])
            rv = rv / rv.sum()
            syn_dev = np.mean(vs, axis=0) - arm_mean
            real_dev = rv - m
            xs.extend(syn_dev.tolist())
            ys.extend(real_dev.tolist())
            ws.extend([float(len(rw))] * len(o))
    x, y, w = np.array(xs), np.array(ys), np.array(ws)
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    corr = cov / np.sqrt(
        np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)
    )
    return (
        float(corr),
        float(np.average(np.abs(x - y), weights=w)),
        float(np.average(np.abs(y), weights=w)),
    )


def load_ssr(reader: str, arm: str, wave: int, refs: dict, tau: float) -> list[dict] | None:
    path = R.answers_path(wave, arm, reader)
    if not path.exists() or not H._cos_path_for(path).exists():
        return None
    ans, _ = H.load_cos_probs(path, refs, tau)
    return ans


def load_lp(reader: str, arm: str) -> list[dict] | None:
    path = L.answers_path(arm, reader)
    return H._read_jsonl(path) if path.exists() else None


def shuffle_null(
    recs: list[dict],
    real: pd.DataFrame,
    items: list[str],
    order: dict,
    n: int = 1000,
    seed: int = 20261010,
) -> tuple[float, float, float]:
    """Mean cell TVD when persona-to-cell labels are permuted within the arm; returns the null mean, the
    null 5th percentile and the one-sided p-value of the observed value."""
    rng = np.random.default_rng(seed)
    index = P._CellTvdIndex(
        real, items, order
    )  # the fast path of the paired bootstrap, same statistic
    observed = index.mean(recs)
    quants = [r["quant"] for r in recs]
    vals = []
    for _ in range(n):
        perm = rng.permutation(len(recs))
        shuffled = [{**r, "quant": quants[k]} for r, k in zip(recs, perm, strict=True)]
        vals.append(index.mean(shuffled))
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 5)), float((v <= observed).mean())


def corr_ci(
    recs: list[dict],
    real: pd.DataFrame,
    items: list[str],
    order: dict,
    n: int = 500,
    seed: int = 20261010,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(recs), len(recs))
        vals.append(cell_contrast([recs[k] for k in idx], real, items, order)[0])
    lo, hi = np.percentile([v for v in vals if np.isfinite(v)], [2.5, 97.5])
    return float(lo), float(hi)


def corr_diff_ci(
    a: list[dict],
    b: list[dict],
    real: pd.DataFrame,
    items: list[str],
    order: dict,
    n: int = 200,
    seed: int = 20261010,
) -> tuple[float, float, float]:
    """Persona-paired bootstrap of the cell-contrast correlation of arm a minus arm b (same persona ids)."""
    rng = np.random.default_rng(seed)
    bi = {r["persona_id"]: r for r in b}
    pairs = [(r, bi[r["persona_id"]]) for r in a if r["persona_id"] in bi]
    point = (
        cell_contrast([x for x, _ in pairs], real, items, order)[0]
        - cell_contrast([y for _, y in pairs], real, items, order)[0]
    )
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(pairs), len(pairs))
        sa = [pairs[k][0] for k in idx]
        sb = [pairs[k][1] for k in idx]
        vals.append(
            cell_contrast(sa, real, items, order)[0] - cell_contrast(sb, real, items, order)[0]
        )
    lo, hi = np.percentile([v for v in vals if np.isfinite(v)], [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main() -> int:
    items = list(H.QUESTIONS)
    tau, _ = H.load_frozen_tau()
    rows: list[dict] = []
    for wave in (2024, 2022, 2020):
        refs = json.loads((H.wave_dir(wave) / "ssr_refs.json").read_text(encoding="utf-8"))
        refs = {
            i: {str(k): (v if isinstance(v, list) else [str(v)]) for k, v in o.items()}
            for i, o in refs.items()
        }
        order = {i: list(o) for i, o in refs.items()}
        real = H.real_item_frame(wave)
        bl = pd.read_csv(settings.outputs_dir / "reports" / "era_holdout_baselines.csv")
        bl = bl[bl.wave == wave]
        rows.append(
            {
                "wave": wave,
                "record": "baseline",
                "arm": "marginal_lookup",
                "value": float(bl.marginal_lookup_tvd.mean()),
            }
        )
        rows.append(
            {
                "wave": wave,
                "record": "baseline",
                "arm": "real_sample",
                "value": float(bl.real_sample_tvd.mean()),
            }
        )
        for reader in ("qwen", "exaone", "gemma"):
            for method in ("ssr", "logprob"):
                if method == "logprob" and (wave != 2024 or reader == "gemma"):
                    continue
                arms: dict[str, list[dict]] = {}
                raw_arms: dict[str, list[dict]] = {}
                for arm in ("release", "synthanchors", "twin", "prior"):
                    if method == "ssr":
                        recs = load_ssr(reader, arm, wave, refs, tau) if arm != "prior" else None
                    else:
                        recs = load_lp(reader, arm)
                        if arm == "prior" and recs and "release" in arms:
                            recs = L.prior_as_lookup(recs[0], arms["release"])
                        elif arm == "prior":
                            recs = None
                    if recs is None:
                        continue
                    raw = P.mean_cell_tvd(recs, real, items, order)
                    rec_ = recalibrate(recs, real, items, order)
                    arms[arm] = rec_
                    raw_arms[arm] = recs
                    corr, mae, mae0 = cell_contrast(recs, real, items, order)
                    row = {
                        "wave": wave,
                        "record": "level",
                        "reader": reader,
                        "method": method,
                        "arm": arm,
                        "raw": raw,
                        "value": P.mean_cell_tvd(rec_, real, items, order),
                        "n": len(recs),
                        "cell_corr": corr,
                        "cell_mae": mae,
                        "cell_mae_lookup": mae0,
                    }
                    if arm != "prior":
                        null_mean, null_p5, pval = shuffle_null(recs, real, items, order)
                        lo, hi = corr_ci(recs, real, items, order)
                        row.update(
                            {
                                "shuffle_mean": null_mean,
                                "shuffle_p5": null_p5,
                                "shuffle_p": pval,
                                "cell_corr_lo": lo,
                                "cell_corr_hi": hi,
                            }
                        )
                    rows.append(row)
                    log.info(
                        "wave %d %s %s %s raw %.3f recal %.3f corr %.2f",
                        wave,
                        reader,
                        method,
                        arm,
                        raw,
                        row["value"],
                        corr,
                    )
                if {"release", "synthanchors"} <= set(raw_arms):
                    pt, lo, hi = corr_diff_ci(
                        raw_arms["release"], raw_arms["synthanchors"], real, items, order
                    )
                    rows.append(
                        {
                            "wave": wave,
                            "record": "corr_diff",
                            "reader": reader,
                            "method": method,
                            "arm": "release-synthanchors",
                            "value": pt,
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )
                if {"release", "synthanchors", "twin"} <= set(arms):
                    for row in P.bootstrap_diffs(
                        {
                            "release": arms["synthanchors"],
                            "oracle": arms["release"],
                            "twin": arms["twin"],
                        },
                        real,
                        items,
                        order,
                    ):
                        name = {
                            "oracle-release": "release-synthanchors",
                            "release-twin": "synthanchors-twin",
                            "oracle-twin": "release-twin",
                        }[row["contrast"]]
                        rows.append(
                            {
                                "wave": wave,
                                "record": "contrast",
                                "reader": reader,
                                "method": method,
                                "arm": name,
                                "value": row["diff"],
                                "ci_lo": row["ci_lo"],
                                "ci_hi": row["ci_hi"],
                                "n": row["n_shared"],
                            }
                        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT, index=False)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
