"""Score era holdouts — synthetic vs observed (ADR-009 Step 2).

The four overlapping canonical variables are absent from generated anchors/quant
by design:
  - weighted observed margins plus a noise floor (resampling the observed data
    at the synthetic n) are recorded here
  - item-level synthetic TVD is filled by the SSR answer layer
    (`era_holdout_ssr_smoke`)
Width and depth are scored directly as syn vs real from quant (ADR-008).

Run:
  uv run python scripts/era_holdout_score.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.official import (
    build_official_anchor_frame,
    depth_proxy_from_band,
)
from sfac_persona.schema import WEIGHTED_WAVES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_score")

HOLDOUT_CANON = [
    "leisure_activity_1st",
    "leisure_activity_wanted_1st",
    "leisure_time_weekday",
    "leisure_time_weekend",
]

DOMAIN_PRIMARY = [
    ("attitude", "other", "arts_importance"),
    ("satisfaction", "satisfaction", "attendance_overall"),
    ("satisfaction", "satisfaction", "cultural_life_overall"),
]

SEED = 20260910
N_BOOT = 20


def _tvd(p: pd.Series, q: pd.Series) -> float:
    idx = sorted(set(map(str, p.index)) | set(map(str, q.index)))
    p_map = {str(k): float(v) for k, v in p.items()}
    q_map = {str(k): float(v) for k, v in q.items()}
    pv = np.array([p_map.get(k, 0.0) for k in idx], dtype=float)
    qv = np.array([q_map.get(k, 0.0) for k in idx], dtype=float)
    if pv.sum() > 0:
        pv /= pv.sum()
    if qv.sum() > 0:
        qv /= qv.sum()
    return 0.5 * float(np.abs(pv - qv).sum())


def _weighted_dist(values: pd.Series, weights: pd.Series) -> pd.Series:
    df = pd.DataFrame({"v": values.astype(str), "w": weights.astype(float)})
    df = df[df["w"] > 0]
    tab = df.groupby("v", dropna=False)["w"].sum()
    return tab / tab.sum() if tab.sum() > 0 else tab


def _noise_floor(real_dist: pd.Series, n: int, rng: np.random.Generator) -> float:
    """Resample n from the observed margin; mean TVD vs the full margin (sampling-noise floor)."""
    cats = real_dist.index.to_numpy()
    probs = real_dist.to_numpy(dtype=float)
    probs = probs / probs.sum()
    tvs = []
    for _ in range(N_BOOT):
        draws = rng.choice(cats, size=n, p=probs)
        syn = pd.Series(draws).value_counts(normalize=True)
        tvs.append(_tvd(real_dist, syn))
    return float(np.mean(tvs))


def _canon_labeled(wave: int, var: str) -> tuple[pd.Series, pd.Series]:
    """row-aligned values + weights for general frame."""
    keys = pd.read_parquet(settings.official_person_dir / "official_person_keys.parquet")
    k = keys[(keys["wave"] == wave) & (keys["frame_group"] == "general")]
    wmap = k.set_index("row_uid")["weight"]
    canon = pd.read_parquet(settings.official_person_dir / "official_person_canonical.parquet")
    sub = canon[(canon["wave"] == wave) & (canon["canonical_var"] == var)]
    sub = sub[sub["row_uid"].isin(wmap.index)].copy()
    # label via source_var if possible
    from sfac_persona.grounding.official import load_official_value_labels

    labels = load_official_value_labels(wave)
    src = str(sub["source_var"].iloc[0]) if len(sub) and "source_var" in sub.columns else ""
    labmap = labels.get(src, {})

    def _lab(x: object) -> str:
        if pd.isna(x):
            return "NA"
        key = str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else str(x)
        if key.endswith(".0"):
            key = str(int(float(key)))
        return labmap.get(key, labmap.get(str(x), key))

    vals = sub["value"].map(_lab)
    ww = sub["row_uid"].map(wmap).astype(float)
    return vals, ww


def _domain_values(
    wave: int, family: str, measure: str, item: str
) -> tuple[pd.Series, pd.Series] | None:
    keys = pd.read_parquet(settings.official_person_dir / "official_person_keys.parquet")
    k = keys[(keys["wave"] == wave) & (keys["frame_group"] == "general")]
    wmap = k.set_index("row_uid")["weight"]
    dom = pd.read_parquet(settings.official_person_dir / "official_person_domain.parquet")
    sub = dom[
        (dom["wave"] == wave)
        & (dom["family"] == family)
        & (dom["measure"] == measure)
        & (dom["canonical_item"] == item)
        & (dom["row_uid"].isin(wmap.index))
    ]
    if sub.empty:
        return None
    # one row per person if duplicates
    sub = sub.drop_duplicates("row_uid", keep="first")
    vals = sub["value"].map(
        lambda x: (
            str(int(float(x))) if pd.notna(x) and str(x).replace(".", "", 1).isdigit() else str(x)
        )
    )
    ww = sub["row_uid"].map(wmap).astype(float)
    return vals, ww


def _ensure_width(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "width" not in out.columns:
        gcols = [
            c
            for c in out.columns
            if c.startswith("attend_") and c not in ("attend_any", "attend_count_band")
        ]
        out["width"] = (out[gcols] == "예").sum(axis=1)
    return out


def width_depth_rows() -> list[dict]:
    rows = []
    for wave in WEIGHTED_WAVES:
        real = _ensure_width(build_official_anchor_frame(wave))
        for ablation in ("margins", "structure_T"):
            path = settings.synthetic_dir / f"era_{wave}_quant" / f"personas_{ablation}.parquet"
            if not path.exists():
                continue
            syn = _ensure_width(pd.read_parquet(path))
            w_real = float(np.average(real["width"], weights=real["w"]))
            # MAE by age band. The unweighted band mean (one vote per band) is canonical,
            # but band-size differences can move the metric, so a population-weighted
            # mean is reported alongside it (2026-09-11).
            band_maes: list[float] = []
            band_sizes: list[float] = []
            for age, sw in syn.groupby("age_band"):
                rw = real[real["age_band"] == age]
                if len(rw) == 0 or len(sw) == 0:
                    continue
                band_maes.append(
                    abs(float(np.average(rw["width"], weights=rw["w"])) - float(sw["width"].mean()))
                )
                band_sizes.append(float(len(sw)))
            # Depth: synthetic rows only have band midpoints. Fold observed counts onto
            # the **same bands and the same midpoints** before comparing (fix 2026-09-11).
            # Subtracting raw counts treats discretization bias as model error.
            d_real_raw = float(np.average(real["depth"], weights=real["w"]))
            rp = depth_proxy_from_band(real["attend_count_band"])
            rp_ok = rp.notna()
            d_real = float(np.average(rp[rp_ok], weights=real["w"][rp_ok]))
            discretization_bias = d_real_raw - d_real
            d_syn = float(depth_proxy_from_band(syn["attend_count_band"]).mean())
            rows.append(
                {
                    "layer": "width_depth",
                    "wave": wave,
                    "ablation": ablation,
                    "item": "width",
                    "metric": "mae_by_age",
                    "syn_vs_real": float(np.mean(band_maes)) if band_maes else float("nan"),
                    "noise_floor": float("nan"),
                    "ratio_vs_floor": float("nan"),
                    "n_syn": len(syn),
                    "note": f"width_mean_real={w_real:.3f} syn={syn['width'].mean():.3f}",
                }
            )
            rows.append(
                {
                    "layer": "width_depth",
                    "wave": wave,
                    "ablation": ablation,
                    "item": "width",
                    "metric": "mae_by_age_popw",
                    "syn_vs_real": (
                        float(np.average(band_maes, weights=band_sizes))
                        if band_maes
                        else float("nan")
                    ),
                    "noise_floor": float("nan"),
                    "ratio_vs_floor": float("nan"),
                    "n_syn": len(syn),
                    "note": "age-band MAE weighted by synthetic band size",
                }
            )
            rows.append(
                {
                    "layer": "width_depth",
                    "wave": wave,
                    "ablation": ablation,
                    "item": "depth",
                    "metric": "abs_gap",
                    "syn_vs_real": abs(d_real - d_syn),
                    "noise_floor": float("nan"),
                    "ratio_vs_floor": float("nan"),
                    "n_syn": len(syn),
                    "note": (
                        f"like-for-like: real_binned={d_real:.3f} syn_proxy={d_syn:.3f} "
                        f"(raw real={d_real_raw:.3f}; discretization bias={discretization_bias:+.3f})"
                    ),
                }
            )
    return rows


def holdout_canon_rows(rng: np.random.Generator) -> list[dict]:
    rows = []
    for wave in WEIGHTED_WAVES:
        # n_syn from structure_T (representative)
        path = settings.synthetic_dir / f"era_{wave}_quant" / "personas_structure_T.parquet"
        n_syn = len(pd.read_parquet(path)) if path.exists() else 15000
        for var in HOLDOUT_CANON:
            vals, ww = _canon_labeled(wave, var)
            real_d = _weighted_dist(vals, ww)
            floor = _noise_floor(real_d, n_syn, rng)
            rows.append(
                {
                    "layer": "canonical_holdout",
                    "wave": wave,
                    "ablation": "n/a",
                    "item": var,
                    "metric": "tvd_syn_vs_real",
                    "syn_vs_real": float("nan"),
                    "noise_floor": floor,
                    "ratio_vs_floor": float("nan"),
                    "n_syn": n_syn,
                    "note": "not_in_quant; syn TVD via SSR answer layer",
                }
            )
            # also store entropy for context
            p = real_d.to_numpy(dtype=float)
            p = p[p > 0]
            ent = float(-(p * np.log(p)).sum())
            rows.append(
                {
                    "layer": "canonical_holdout",
                    "wave": wave,
                    "ablation": "n/a",
                    "item": var,
                    "metric": "real_entropy",
                    "syn_vs_real": ent,
                    "noise_floor": floor,
                    "ratio_vs_floor": float("nan"),
                    "n_syn": n_syn,
                    "note": f"n_cats={len(real_d)}",
                }
            )
    return rows


def domain_rows(rng: np.random.Generator) -> list[dict]:
    rows = []
    skipped = []
    for wave in WEIGHTED_WAVES:
        path = settings.synthetic_dir / f"era_{wave}_quant" / "personas_structure_T.parquet"
        n_syn = len(pd.read_parquet(path)) if path.exists() else 15000
        for family, measure, item in DOMAIN_PRIMARY:
            got = _domain_values(wave, family, measure, item)
            if got is None:
                skipped.append(f"{wave}:{item}")
                continue
            vals, ww = got
            real_d = _weighted_dist(vals, ww)
            floor = _noise_floor(real_d, n_syn, rng)
            rows.append(
                {
                    "layer": "domain_primary",
                    "wave": wave,
                    "ablation": "n/a",
                    "item": item,
                    "metric": "tvd_syn_vs_real",
                    "syn_vs_real": float("nan"),
                    "noise_floor": floor,
                    "ratio_vs_floor": float("nan"),
                    "n_syn": n_syn,
                    "note": f"{family}/{measure}; not_in_quant",
                }
            )
    if skipped:
        log.info("domain skipped: %s", skipped)
    return rows


def main() -> int:
    rng = np.random.default_rng(SEED)
    out_dir = settings.outputs_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = width_depth_rows() + holdout_canon_rows(rng) + domain_rows(rng)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "era_holdout_scores.csv"
    df.to_csv(csv_path, index=False)

    # summary markdown
    wd = df[df["layer"] == "width_depth"]
    floors = df[(df["layer"] == "canonical_holdout") & (df["metric"] == "tvd_syn_vs_real")]
    lines = [
        "# Era holdout scores (ADR-009 Step 2)",
        "",
        "게이트 기준: **점수가 존재**한다. 이분 통과/실패 컷 없음 (Atlas-first).",
        "",
        "## 폭·깊이 (quant syn vs 가중 실측)",
        "",
    ]
    if hasattr(wd, "to_markdown"):
        lines.append(
            wd[["wave", "ablation", "item", "metric", "syn_vs_real", "note"]].to_markdown(
                index=False
            )
        )
    else:
        lines.append(wd.to_string(index=False))
    lines += [
        "",
        "## 교집합 canonical — 잡음 바닥 (합성 n 재추출)",
        "",
        "문항은 생성 quant에 **없음** (holdout). 문항급 `syn_vs_real` TVD는 SSR 스모크에서 채움 → [`era_holdout_ssr_smoke.md`](era_holdout_ssr_smoke.md).",
        "",
        "2024 SSR smoke (n=20) 대 잡음바닥(~0.01): activity TVD ~0.43–0.52, time ~0.11–0.15 (배관 검증용; `era_holdout_ssr_smoke.csv`).",
        "",
    ]
    if not floors.empty:
        summ = floors.groupby("wave")["noise_floor"].mean().reset_index()
        summ.columns = ["wave", "mean_noise_floor_tvd"]
        lines.append(
            summ.to_markdown(index=False)
            if hasattr(summ, "to_markdown")
            else summ.to_string(index=False)
        )
        lines.append("")
        lines.append(floors[["wave", "item", "noise_floor", "n_syn"]].to_markdown(index=False))
    lines += [
        "",
        "## Domain primary",
        "",
        "arts_importance · attendance_overall · cultural_life_overall — 동일하게 not_in_quant; 잡음 바닥만.",
        "",
        f"- 상세 CSV: `{csv_path.name}`",
        "- 레지스트리: `data/eval/era_holdout_registry.json`",
        "",
    ]
    md_path = out_dir / "era_holdout_scores.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s (%d rows) %s", csv_path, len(df), md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
