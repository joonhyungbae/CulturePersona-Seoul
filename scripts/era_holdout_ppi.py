"""Era holdout computational PPI — learned · twin_ssr · distilled ρ² (ADR-010, no humans).

Inputs: era_holdout_run and era_answer_model outputs.

Run:
  uv run python scripts/era_holdout_ppi.py
  uv run python scripts/era_holdout_ppi.py --waves 2024 2022 2020
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_ppi")

REPORTS = settings.outputs_dir / "reports"
OUT = settings.data_root / "eval" / "era_holdout"
SEED = 20260910
N_BOOT = 1000
N_PERM = 1000
ANCHOR_COLS = ["sex", "age_band", "attend_any"]


def add_bh(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini–Hochberg over every (wave, item, predictor) row: `p_bh`, `above_null_bh`."""
    p = df["p_perm"].to_numpy(dtype=float)
    m = int(np.isfinite(p).sum())
    order = np.argsort(p)
    ranked = np.empty(len(p))
    ranked[order] = np.arange(1, len(p) + 1)
    q = p * m / ranked
    # Monotone correction (cumulative minimum from the back)
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_adj = np.empty(len(p))
    q_adj[order] = np.minimum(q_sorted, 1.0)
    out = df.copy()
    out["p_bh"] = q_adj
    out["above_null_bh"] = out["p_bh"] < alpha
    return out


def _load_holdout_mod():
    path = Path(__file__).resolve().parent / "era_holdout_run.py"
    spec = importlib.util.spec_from_file_location("era_holdout_run", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def wvar(x: np.ndarray, w: np.ndarray) -> float:
    m = np.average(x, weights=w)
    return float(np.average((x - m) ** 2, weights=w))


def wcov(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return float(
        np.average((x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w)
    )


def _rho2_core(Y: np.ndarray, F: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """(Var(Y)-weighted mean ρ², plain ratio). λ is the per-option optimum Cov/Var(F)."""
    num = den = 0.0
    plain_num = 0.0
    for o in range(Y.shape[1]):
        vy = wvar(Y[:, o], w)
        if vy <= 0:
            continue
        vf = wvar(F[:, o], w)
        c = wcov(Y[:, o], F[:, o], w)
        rho2 = (c * c / (vy * vf)) if vf > 0 else 0.0
        num += rho2 * vy
        den += vy
        plain_num += wvar(Y[:, o] - F[:, o], w)
    if den <= 0:
        return 0.0, 1.0
    return num / den, plain_num / den


def ppi_rho2(
    Y: np.ndarray,
    F: np.ndarray,
    w: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> dict[str, float]:
    """Y, F: (n, k) one-hot / probs. ρ² + bootstrap CI + chance level.

    Two fixes, 2026-09-11.

    1. The previous version's `rect_difficulty` column was removed. With λ=Cov/Var(F),
       Var(Y−λF) = Var(Y)(1−ρ²) holds identically, so that column was **exactly the same
       number** as `ratio_pp` (max observed difference 4e-16). A column with no new
       information.
    2. ρ² is a squared correlation, so it rises even under the null. 1/(n−2) is the value
       **when there is a single predictor** (0.0034 when n=300), and it does not fit a
       predictor fit on sex×age×attend cells, as with `learned`.
       From 2026-09-20 a **permutation null** is reported alongside it: shuffle Y's rows to
       break the pairing, repeat the same calculation, and the 95th percentile of that
       distribution is `rho2_perm95`. The decision (`above_null`) uses this. Because λ is
       fit on the evaluation sample, `rho2` itself is also biased upward — read it via
       the CI.

    Note: ρ² is a squared statistic that meets a boundary at 0, so when the true value is
    near 0 the bootstrap distribution piles up above the point estimate and **the percentile
    CI may fail to cover the point estimate**. The CI is therefore used only as a
    **one-sided screen for exceeding chance level**, not as "error bars on the point
    estimate" (`above_null` = CI lower bound > `rho2_null`).
    """
    n = len(Y)
    saved, plain = _rho2_core(Y, F, w)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b, _p = _rho2_core(Y[idx], F[idx], w[idx])
        boots.append(b)
    lo, hi = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if boots
        else (np.nan, np.nan)
    )
    null = 1.0 / max(n - 2, 1)
    perms = []
    for _ in range(n_perm):
        b, _p = _rho2_core(Y[rng.permutation(n)], F, w)
        perms.append(b)
    perm95 = float(np.percentile(perms, 95)) if perms else np.nan
    # Permutation p-value for multiplicity over 81 rows. `add_bh`
    # attaches BH after the whole file is pooled.
    p_perm = (
        float((1 + sum(1 for b in perms if b >= saved)) / (1 + len(perms))) if perms else np.nan
    )
    return {
        "rho2": saved,
        "rho2_lo": lo,
        "rho2_hi": hi,
        "rho2_null": null,
        "rho2_perm95": perm95,
        "p_perm": p_perm,
        "above_null": bool(lo > perm95),
        "ratio_pp": 1.0 - saved,
        "plain_ratio": plain,
    }


def onehot_y(y: pd.Series, opts: list[str]) -> np.ndarray:
    return np.stack([(y.astype(str) == o).to_numpy(dtype=float) for o in opts], axis=1)


def learned_oof(
    frame: pd.DataFrame,
    items: list[str],
    options: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Return {item: (N, k)} OOF probability matrices aligned to frame index."""
    A = frame[ANCHOR_COLS].fillna("NA").astype(str)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X = enc.fit_transform(A)
    w = frame["w"].astype(float)
    kf = KFold(5, shuffle=True, random_state=SEED)
    out: dict[str, np.ndarray] = {}
    for var in items:
        opts = options[var]
        y = frame[var]
        ok = y.notna().to_numpy()
        idx = np.flatnonzero(ok)
        L = np.full((len(frame), len(opts)), np.nan)
        if len(idx) < 20 or y.nunique(dropna=True) < 2:
            out[var] = L
            continue
        for tr, te in kf.split(idx):
            tr_i, te_i = idx[tr], idx[te]
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr_i], y.iloc[tr_i].astype(str), sample_weight=w.iloc[tr_i])
            pr = (
                pd.DataFrame(clf.predict_proba(X[te_i]), columns=clf.classes_.astype(str))
                .reindex(columns=opts)
                .fillna(0.0)
                .to_numpy()
            )
            L[te_i] = pr / np.clip(pr.sum(1, keepdims=True), 1e-12, None)
        out[var] = L
    return out


def twin_ssr_preds(
    wave: int,
    frame: pd.DataFrame,
    items: list[str],
    options: dict[str, list[str]],
    tau: float,
    ehr,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Map twin answers → (N,k) aligned to frame; return mask of twin rows."""
    ans_path = ehr.resolve_answers_path(wave, "answers_twin_anchors_ssr.jsonl")
    if not ans_path.exists():
        raise FileNotFoundError(ans_path)
    refs_path = ehr.wave_dir(wave) / "ssr_refs.json"
    if not refs_path.exists() and wave == 2024:
        refs_path = OUT / "ssr_refs.json"
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    # ensure option order matches refs when present
    for var in items:
        if var in refs:
            options[var] = list(refs[var].keys())

    ans, _ = ehr.load_cos_probs(ans_path, refs, tau)

    uid_to_i = {str(u): i for i, u in enumerate(frame["row_uid"].astype(str))}
    pred = {var: np.full((len(frame), len(options[var])), np.nan) for var in items}
    twin_mask = np.zeros(len(frame), dtype=bool)
    n_hit = 0
    for r in ans:
        pid = str(r["persona_id"])
        uid = pid.removeprefix("twin-") if pid.startswith("twin-") else pid
        i = uid_to_i.get(uid)
        if i is None:
            continue
        twin_mask[i] = True
        n_hit += 1
        probs = r.get("probs") or {}
        for var in items:
            p = probs.get(var)
            if not p:
                continue
            opts = options[var]
            pred[var][i] = np.array([float(p.get(o, 0.0)) for o in opts], dtype=float)
    log.info("wave=%d twin_ssr matched %d / %d answers", wave, n_hit, len(ans))
    return pred, twin_mask


def distilled_preds(
    wave: int,
    frame: pd.DataFrame,
    items: list[str],
    options: dict[str, list[str]],
    ehr,
) -> dict[str, np.ndarray]:
    """Load A1 pred_real_distilled_ssr.npz aligned to frame row_uid."""
    candidates = [
        ehr.wave_dir(wave) / "pred_real_distilled_ssr.npz",
        OUT / "pred_real_distilled_ssr.npz" if wave == 2024 else None,
    ]
    path = next((p for p in candidates if p is not None and p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"pred_real_distilled_ssr.npz missing for wave={wave}")
    z = np.load(path, allow_pickle=True)
    uid_pred = z["row_uid"].astype(str)
    uid_to_i = {u: i for i, u in enumerate(uid_pred)}
    opt_json = {}
    if "options_json" in z.files:
        opt_json = json.loads(str(z["options_json"][0]))
    pred: dict[str, np.ndarray] = {
        var: np.full((len(frame), len(options[var])), np.nan) for var in items
    }
    n_hit = 0
    for j, uid in enumerate(frame["row_uid"].astype(str)):
        i = uid_to_i.get(uid)
        if i is None:
            continue
        n_hit += 1
        for var in items:
            key = f"p__{var}"
            if key not in z.files:
                continue
            P = z[key]
            src_opts = list(opt_json.get(var, options[var]))
            row = {o: float(P[i, k]) for k, o in enumerate(src_opts) if k < P.shape[1]}
            pred[var][j] = np.array([row.get(o, 0.0) for o in options[var]], dtype=float)
            s = pred[var][j].sum()
            if s > 0:
                pred[var][j] /= s
    log.info("wave=%d distilled matched %d / %d rows from %s", wave, n_hit, len(frame), path.name)
    return pred


def eval_wave(wave: int, ehr, tau: float) -> list[dict]:
    frame = ehr.real_item_frame(wave)
    items = list(ehr.QUESTIONS)
    # option sets from observed labels (+ Likert for domain)
    options: dict[str, list[str]] = {}
    for var in items:
        vals = sorted(frame[var].dropna().astype(str).unique().tolist())
        options[var] = vals

    # prefer refs order for SSR alignment
    refs_path = ehr.wave_dir(wave) / "ssr_refs.json"
    if not refs_path.exists() and wave == 2024:
        refs_path = OUT / "ssr_refs.json"
    if refs_path.exists():
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        for var in items:
            if var in refs:
                options[var] = list(refs[var].keys())

    learned = learned_oof(frame, items, options)
    try:
        twin, twin_mask = twin_ssr_preds(wave, frame, items, options, tau, ehr)
    except FileNotFoundError as exc:
        log.warning("wave=%d skip twin_ssr: %s", wave, exc)
        twin, twin_mask = {}, np.zeros(len(frame), dtype=bool)

    try:
        distilled = distilled_preds(wave, frame, items, options, ehr)
    except FileNotFoundError as exc:
        log.warning("wave=%d skip distilled: %s", wave, exc)
        distilled = {}

    rows = []
    w_all = frame["w"].astype(float).to_numpy()
    for var in items:
        y = frame[var]
        ok = y.notna().to_numpy() & twin_mask
        if twin_mask.sum() == 0:
            # fall back: evaluate learned on all non-null
            ok = y.notna().to_numpy()
        if ok.sum() < 30:
            continue
        opts = options[var]
        Y = onehot_y(y[ok].astype(str), opts)
        ww = w_all[ok]
        # learned
        L = learned[var][ok]
        good = ~np.isnan(L).any(axis=1)
        if good.sum() >= 30:
            m = ppi_rho2(Y[good], L[good], ww[good])
            rows.append(
                {"wave": wave, "item": var, "predictor": "learned", "n": int(good.sum()), **m}
            )
        # twin_ssr
        if var in twin and twin_mask.any():
            T = twin[var][ok]
            good_t = ~np.isnan(T).any(axis=1)
            if good_t.sum() >= 30:
                m = ppi_rho2(Y[good_t], T[good_t], ww[good_t])
                rows.append(
                    {
                        "wave": wave,
                        "item": var,
                        "predictor": "twin_ssr",
                        "n": int(good_t.sum()),
                        **m,
                    }
                )
        # distilled (A1)
        if var in distilled:
            D = distilled[var][ok]
            good_d = ~np.isnan(D).any(axis=1)
            if good_d.sum() >= 30:
                m = ppi_rho2(Y[good_d], D[good_d], ww[good_d])
                rows.append(
                    {
                        "wave": wave,
                        "item": var,
                        "predictor": "distilled",
                        "n": int(good_d.sum()),
                        **m,
                    }
                )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, nargs="+", default=[2024, 2022, 2020])
    args = ap.parse_args()
    ehr = _load_holdout_mod()
    tau, tau_meta = ehr.load_frozen_tau()
    log.info("tau=%.3f sha=%s", tau, tau_meta.get("sha256"))

    all_rows: list[dict] = []
    for wave in args.waves:
        if wave not in ehr.ALLOWED_WAVES and wave not in (2020, 2022, 2024):
            log.warning("skip wave %s", wave)
            continue
        try:
            all_rows.extend(eval_wave(wave, ehr, tau))
        except Exception:
            log.exception("wave=%d failed", wave)

    if not all_rows:
        raise SystemExit("no PPI rows — twin/answers missing?")

    df = pd.DataFrame(all_rows)
    summary = (
        df.groupby("predictor")
        .agg(
            n_rows=("rho2", "size"),
            mean_rho2=("rho2", "mean"),
            median_rho2=("rho2", "median"),
            n_above_null=("above_null", "sum"),
        )
        .reset_index()
        .round(4)
    )
    csv_path = REPORTS / "era_holdout_ppi.csv"
    df = add_bh(df)
    df.to_csv(csv_path, index=False)

    lines = [
        "# Era holdout PPI (전산) — ADR-010",
        "",
        f"- τ={tau} (sha `{tau_meta.get('sha256')}`)",
        "- 예측자: `learned` (sex×age×attend_any OOF) · `twin_ssr` (동결 τ) · `distilled` (release SSR → 다항 로지스틱, A1)",
        "- 평가 표본: twin 앵커 행과 교집합 (가능 시)",
        "- 인간 제외. H2는 수치만 보고 — 기각/채택 단정 금지.",
        (
            f"- CI: 비모수 부트스트랩 {N_BOOT}회 퍼센타일. 판정 기준은 순열 귀무 {N_PERM}회의"
            f" 95 백분위 (`rho2_perm95`); `rho2_null`=1/(n−2) 는 예측자 하나짜리 참고값 —"
        ),
        "  ρ² 는 상관제곱이라 예측력이 없어도 이만큼 뜬다. `above_null` 은 CI 하한이 그 위인지.",
        "- λ 를 평가 표본에서 적합하므로 `rho2` 는 위로 편향된다. 점추정이 아니라 CI 로 읽어라.",
        (
            "- ρ² 는 0 에서 경계를 만나므로, 참값이 0 근처면 CI 가 점추정을 포함하지 않을 수 있다."
            " — CI 는 오차 막대가 아니라 `above_null` 선별용이다."
        ),
        "- 이전 판의 `rect_difficulty` 열은 삭제했다 — λ 최적값에서 `1−ρ²` 와 항등이라 정보가 없었다.",
        "",
        "## ρ² (절감률)",
        "",
        df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False),
        "",
        "## 우연 수준 대비 요약",
        "",
        (
            summary.to_markdown(index=False)
            if hasattr(summary, "to_markdown")
            else summary.to_string(index=False)
        ),
        "",
        f"- 우연 수준을 CI 하한으로 넘긴 행: {int(df['above_null'].sum())}/{len(df)}",
        f"- 보정 없는 PPI(λ=1)가 평균보다 나쁜 행(`plain_ratio`>1): {int((df['plain_ratio'] > 1).sum())}/{len(df)}",
        "",
        f"CSV: `{csv_path.name}`",
        "",
    ]
    md = REPORTS / "era_holdout_ppi.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s (%d rows)", md, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
