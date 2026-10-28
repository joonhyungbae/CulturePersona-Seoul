"""Per-era evaluation suite (ADR-007 C4 + progress Step1 revision).

- Measured comparisons use **survey weights**.
- Width reports age-band MAE (neutral to population/sample mix) together with the weighted
  overall mean.
- T ablation: genre φ MAD + genre-pair joint TVD (structure metrics).

Run: uv run python scripts/era_eval_suite.py
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.error_study import genre_phi
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import (
    AGE_BANDS,
    ERA_CORE_ANCHORS,
    ERA_WAVES,
    GENRE_ANCHORS,
    ROOT_ANCHORS,
    UNWEIGHTED_WAVES,
    WEIGHTED_WAVES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_eval")


def _tvd_marginal(real: pd.DataFrame, syn: pd.DataFrame, col: str) -> float:
    if col not in real.columns or col not in syn.columns:
        return float("nan")
    pr = real.dropna(subset=[col]).groupby(col)["w"].sum()
    pr = pr / pr.sum()
    ps = syn.dropna(subset=[col]).groupby(col).size().astype(float)
    ps = ps / ps.sum()
    idx = pr.index.union(ps.index)
    return float(0.5 * (pr.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).abs().sum())


def _genre_pair_joint_tvd(real: pd.DataFrame, syn: pd.DataFrame) -> float:
    """Mean genre-pair joint TVD (structure)."""
    syn = syn.copy()
    syn["w"] = 1.0
    tvds = []
    for a, b in itertools.combinations(GENRE_ANCHORS, 2):
        if a not in real.columns or b not in syn.columns:
            continue

        def joint(df: pd.DataFrame, _a: str = a, _b: str = b) -> pd.Series:
            g = df.groupby([_a, _b])["w"].sum()
            return g / g.sum()

        jr, js = joint(real), joint(syn)
        idx = jr.index.union(js.index)
        tvds.append(
            0.5 * (jr.reindex(idx, fill_value=0) - js.reindex(idx, fill_value=0)).abs().sum()
        )
    return float(np.mean(tvds)) if tvds else float("nan")


def _ensure_width(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "width" not in out.columns:
        cols = [g for g in GENRE_ANCHORS if g in out.columns]
        out["width"] = (out[cols] == "예").sum(axis=1)
    return out


def eval_wave(wave: int, ablation: str) -> dict:
    real = _ensure_width(build_official_anchor_frame(wave))
    path = settings.synthetic_dir / f"era_{wave}_quant" / f"personas_{ablation}.parquet"
    if not path.exists() and ablation != "margins":
        path = settings.synthetic_dir / f"era_{wave}_quant" / "personas_margins.parquet"
    if not path.exists():
        return {"wave": wave, "ablation": ablation, "error": "missing_personas"}
    syn = _ensure_width(pd.read_parquet(path))
    syn["w"] = 1.0
    for g in GENRE_ANCHORS:
        if g not in syn.columns:
            syn[g] = "아니오"

    anchors = [
        a
        for a in ERA_CORE_ANCHORS
        if a in real.columns and a in syn.columns and a not in ROOT_ANCHORS
    ]
    # root anchors: still report but note R0 vs survey
    tvds = {a: _tvd_marginal(real, syn, a) for a in anchors}
    root_tvds = {a: _tvd_marginal(real, syn, a) for a in ROOT_ANCHORS if a in syn.columns}
    mean_tvd = float(np.nanmean(list(tvds.values()))) if tvds else float("nan")

    w_real = float(np.average(real["width"], weights=real["w"]))
    w_syn = float(syn["width"].mean())
    width_mae_weighted = abs(w_real - w_syn)

    # age-conditional width MAE (equal weight per age band present in syn)
    band_maes = []
    for band in AGE_BANDS:
        rw = real[real["age_band"] == band]
        sw = syn[syn["age_band"] == band]
        if len(rw) == 0 or len(sw) == 0:
            continue
        band_maes.append(
            abs(float(np.average(rw["width"], weights=rw["w"])) - float(sw["width"].mean()))
        )
    width_mae_by_age = float(np.mean(band_maes)) if band_maes else float("nan")

    phi_r = genre_phi(real)
    phi_s = genre_phi(syn)
    phi_mad = float(np.abs(phi_r.values - phi_s.values)[~np.eye(len(phi_r), dtype=bool)].mean())
    pair_tvd = _genre_pair_joint_tvd(real, syn)

    return {
        "wave": wave,
        "ablation": ablation,
        "n_syn": len(syn),
        "mean_marginal_tvd_nonroot": mean_tvd,
        "root_tvd_age_band": root_tvds.get("age_band"),
        "width_mean_real_weighted": w_real,
        "width_mean_syn": w_syn,
        "width_mae_weighted": width_mae_weighted,
        "width_mae_by_age": width_mae_by_age,
        "depth_mean_real_weighted": float(np.average(real["depth"], weights=real["w"])),
        "genre_phi_mad": phi_mad,
        "genre_pair_joint_tvd": pair_tvd,
        "top_tvd_anchors": sorted(tvds.items(), key=lambda kv: -kv[1])[:5],
    }


def main() -> int:
    out_dir = settings.outputs_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for wave in ERA_WAVES:
        for ab in ("margins", "structure_T"):
            if wave in UNWEIGHTED_WAVES and ab == "structure_T":
                continue
            r = eval_wave(wave, ab)
            rows.append(r)
            log.info("eval %s", {k: v for k, v in r.items() if k != "top_tvd_anchors"})

    flat = [{k: v for k, v in r.items() if k != "top_tvd_anchors"} for r in rows]
    pd.DataFrame(flat).to_csv(out_dir / "era_eval_suite.csv", index=False)
    (out_dir / "era_eval_suite.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # T win summary on weighted transfer targets (2022, 2024); 2020 = prior source
    lines = [
        "# Era eval suite (revised)",
        "",
        "실측은 **가중** 기준. 폭은 `width_mae_weighted`와 `width_mae_by_age`를 본다.",
        "R0 뿌리라 `age_band` 주변 TVD가 클 수 있다 (ADR-002) — 비뿌리 앵커·연령조건부 폭이 품질 지표.",
        "T 승패: `genre_pair_joint_tvd` · `genre_phi_mad` (작을수록 좋음). 2020은 prior 출처(ADR-008).",
        "",
    ]
    for wave in WEIGHTED_WAVES:
        m = next((r for r in flat if r["wave"] == wave and r["ablation"] == "margins"), None)
        t = next((r for r in flat if r["wave"] == wave and r["ablation"] == "structure_T"), None)
        if not m or not t:
            continue
        win_phi = t["genre_phi_mad"] < m["genre_phi_mad"]
        win_pair = t["genre_pair_joint_tvd"] < m["genre_pair_joint_tvd"]
        lines.append(
            f"- **{wave}**: width_mae_by_age margins={m['width_mae_by_age']:.3f} T={t['width_mae_by_age']:.3f}; "
            f"φ MAD m={m['genre_phi_mad']:.4f} T={t['genre_phi_mad']:.4f} ({'T win' if win_phi else 'margins'}); "
            f"pair TVD m={m['genre_pair_joint_tvd']:.4f} T={t['genre_pair_joint_tvd']:.4f} ({'T win' if win_pair else 'margins'})"
        )
    lines += ["", "상세: `era_eval_suite.csv` · `era_eval_suite.json`"]
    (out_dir / "era_eval_suite.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for wave in ERA_WAVES:
        d = settings.data_root / "eval" / f"era_{wave}"
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# eval era_{wave}\n\nholdout · SSR · judge 산출을 여기에 둔다.\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
