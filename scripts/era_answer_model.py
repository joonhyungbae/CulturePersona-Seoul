"""Era distilled answer model — anchors → distribution over 9 holdout items (ADR-010 / A1).

Training uses only the release SSR soft distribution (no microdata). Applied to real
respondents' anchors and used as the `distilled` predictor in `era_holdout_ppi`.

The algorithm: soft-target expansion + per-item multinomial logistic regression.

Run:
  uv run python scripts/era_answer_model.py --waves 2024 2022 2020
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.schema import ERA_CORE_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_answer_model")

OUT = settings.data_root / "eval" / "era_holdout"
C_DEFAULT = 0.5


def _load_holdout_mod():
    path = Path(__file__).resolve().parent / "era_holdout_run.py"
    spec = importlib.util.spec_from_file_location("era_holdout_run", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _artifact_dir(ehr, wave: int) -> Path:
    """Write under w{wave}/; for 2024 also keep flat OUT writable via resolve fallback."""
    d = ehr.wave_dir(wave)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _refs_path(ehr, wave: int) -> Path:
    p = ehr.wave_dir(wave) / "ssr_refs.json"
    if p.exists():
        return p
    if wave == 2024 and (OUT / "ssr_refs.json").exists():
        return OUT / "ssr_refs.json"
    return p


def fit_item(X, P: np.ndarray, opts: list[str], C: float) -> LogisticRegression | None:
    """Expand a soft target (n, k) into (n·k) rows and use the probabilities as weights."""
    ok = ~np.isnan(P[:, 0])
    if int(ok.sum()) < 50:
        return None
    Xo, Po = X[ok], P[ok]
    rows, ys, ws = [], [], []
    for j, o in enumerate(opts):
        w = Po[:, j]
        keep = w > 1e-4
        if keep.sum() == 0:
            continue
        rows.append(Xo[keep])
        ys.extend([o] * int(keep.sum()))
        ws.extend(w[keep].tolist())
    if len(set(ys)) < 2:
        return None
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit(sparse.vstack(rows), np.array(ys), sample_weight=np.array(ws))
    return clf


def _anchor_cols(frame: pd.DataFrame, quant_keys: set[str]) -> list[str]:
    return [c for c in ERA_CORE_ANCHORS if c in frame.columns and c in quant_keys]


def _soft_matrix(
    ans: list[dict],
    items: list[str],
    options: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    n = len(ans)
    for var in items:
        opts = options[var]
        P = np.full((n, len(opts)), np.nan)
        for i, r in enumerate(ans):
            p = (r.get("probs") or {}).get(var)
            if not p:
                continue
            P[i] = np.array([float(p.get(o, 0.0)) for o in opts], dtype=float)
        out[var] = P
    return out


def run_wave(wave: int, ehr, tau: float, C: float) -> Path:
    items = list(ehr.QUESTIONS)
    refs_path = _refs_path(ehr, wave)
    if not refs_path.exists():
        raise FileNotFoundError(refs_path)
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    options = {var: list(refs[var].keys()) for var in items if var in refs}

    ans_path = ehr.resolve_answers_path(wave, "answers_release_narrative_ssr.jsonl")
    if not ans_path.exists():
        raise FileNotFoundError(ans_path)
    ans, _ = ehr.load_cos_probs(ans_path, refs, tau)
    if len(ans) < 50:
        raise SystemExit(f"wave={wave}: release answers n={len(ans)} < 50")

    # Prefer the frozen eval bank — still reproducible if a full release generation
    # overwrites the narrative path (ADR-012).
    recs = {r["persona_id"]: r for r in ehr.load_eval_bank(wave)}
    quant_rows = []
    keep_ans = []
    for r in ans:
        pid = r["persona_id"]
        rec = recs.get(pid)
        if not rec:
            log.warning("wave=%d missing narrative for %s — skip", wave, pid)
            continue
        quant_rows.append(rec["quant"])
        keep_ans.append(r)
    if len(keep_ans) < 50:
        raise SystemExit(f"wave={wave}: joined personas {len(keep_ans)} < 50")

    meta = pd.DataFrame(quant_rows)
    frame = ehr.real_item_frame(wave)
    cols = _anchor_cols(frame, set(meta.columns) | set(frame.columns))
    if len(cols) < 3:
        raise SystemExit(f"wave={wave}: too few anchor cols {cols}")
    log.info("wave=%d anchors=%d train_n=%d real_n=%d", wave, len(cols), len(keep_ans), len(frame))

    soft = _soft_matrix(keep_ans, items, options)
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(pd.concat([meta[cols], frame[cols]], axis=0).fillna("NA").astype(str))
    Xp = enc.transform(meta[cols].fillna("NA").astype(str))
    Xr = enc.transform(frame[cols].fillna("NA").astype(str))

    models: dict[str, LogisticRegression] = {}
    preds: dict[str, np.ndarray] = {}
    for var in items:
        if var not in options:
            continue
        clf = fit_item(Xp, soft[var], options[var], C)
        if clf is None:
            log.warning("wave=%d %s: fit failed", wave, var)
            continue
        models[var] = clf
        pr = (
            pd.DataFrame(clf.predict_proba(Xr), columns=clf.classes_.astype(str))
            .reindex(columns=options[var])
            .fillna(0.0)
            .to_numpy()
        )
        preds[var] = pr / np.clip(pr.sum(1, keepdims=True), 1e-12, None)

    art = _artifact_dir(ehr, wave)
    pkl_path = art / "answer_model_ssr.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "encoder": enc,
                "anchor_cols": cols,
                "items": {v: options[v] for v in models},
                "models": models,
                "wave": wave,
                "tau": tau,
                "trained_on": f"{len(keep_ans)} release SSR personas",
                "note": "Trained only on released synthetic SSR answers; no microdata rows.",
            },
            f,
        )

    npz_path = art / "pred_real_distilled_ssr.npz"
    payload = {"row_uid": frame["row_uid"].astype(str).to_numpy()}
    for var, P in preds.items():
        payload[f"p__{var}"] = P
    payload["options_json"] = np.asarray(
        [json.dumps({v: options[v] for v in models}, ensure_ascii=False)]
    )
    np.savez_compressed(npz_path, **payload)

    # 2024 flat mirror for legacy readers
    if wave == 2024:
        for src in (pkl_path, npz_path):
            dst = OUT / src.name
            dst.write_bytes(src.read_bytes())

    log.info("wave=%d wrote %s · %s (%d items)", wave, pkl_path, npz_path, len(models))
    return npz_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, nargs="+", default=[2024, 2022, 2020])
    ap.add_argument("--C", type=float, default=C_DEFAULT)
    args = ap.parse_args()
    ehr = _load_holdout_mod()
    tau, meta = ehr.load_frozen_tau()
    log.info("tau=%.3f sha=%s", tau, meta.get("sha256"))
    for wave in args.waves:
        if wave not in ehr.ALLOWED_WAVES:
            log.warning("skip wave %s", wave)
            continue
        run_wave(wave, ehr, tau, args.C)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
