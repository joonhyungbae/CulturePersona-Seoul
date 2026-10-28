"""Era holdout intersection and width/depth metrics (ADR-007 / ADR-008).

Intersection: non-anchor items shared by the three weighted eras (2020, 2022, 2024).
R5-only: Tier3 / domain candidates present only in 2024 (not placed in generation anchors).

Run:
  uv run python scripts/era_holdout_metrics.py
"""

from __future__ import annotations

import json
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
log = logging.getLogger("era_holdout")

# canonical_var already used as generation anchors — excluded from the holdout intersection
_ANCHOR_CANON = {
    "attendance_companion",
    "education",
    "household_income",
    "household_size",
    "leisure_companion",
    "leisure_time_sufficiency",
    "occupation",
    "years_in_seoul",
    "marital_status",  # 2020 only — an anchor candidate, but not wired in v1
}

# Intersection domain (family, measure, item) — rechecked against official at startup
_INTERSECTION_DOMAIN_SEED = [
    ("attitude", "frequency", "effect_leisure_well_used"),
    ("attitude", "frequency", "effect_mood_improved"),
    ("attitude", "frequency", "effect_overall_happiness"),
    ("attitude", "frequency", "effect_stress_relieved"),
    ("attitude", "other", "arts_importance"),
    ("intention_past", "intention", "movie"),
    ("intention_past", "intention", "exhibition"),
    ("intention_past", "intention", "museum"),
    ("satisfaction", "satisfaction", "attendance_overall"),
    ("satisfaction", "satisfaction", "cultural_life_overall"),
]

_R5_ONLY_EXAMPLES = [
    "ai_artwork_participation_intent",
    "arts_contribute_community_happiness",
    "attitude_youth_education_support",
    "digital_arts_event",
]


def _official_dir() -> Path:
    return settings.official_person_dir


def _keys(wave: int) -> pd.DataFrame:
    keys = pd.read_parquet(_official_dir() / "official_person_keys.parquet")
    return keys[(keys["wave"] == wave) & (keys["frame_group"] == "general")]


def intersection_canonical_holdouts() -> list[str]:
    canon = pd.read_parquet(_official_dir() / "official_person_canonical.parquet")
    sets: list[set[str]] = []
    for w in WEIGHTED_WAVES:
        uids = set(_keys(w)["row_uid"])
        vars_ = set(
            canon[(canon["wave"] == w) & (canon["row_uid"].isin(uids))]["canonical_var"].unique()
        )
        sets.append(vars_ - _ANCHOR_CANON)
    inter = sets[0].intersection(*sets[1:])
    return sorted(inter)


def intersection_domain_triples() -> list[tuple[str, str, str]]:
    dom = pd.read_parquet(_official_dir() / "official_person_domain.parquet")
    sets: list[set[tuple[str, str, str]]] = []
    for w in WEIGHTED_WAVES:
        uids = set(_keys(w)["row_uid"])
        sub = dom[(dom["wave"] == w) & (dom["row_uid"].isin(uids))]
        sets.append(set(zip(sub["family"], sub["measure"], sub["canonical_item"], strict=True)))
    inter = sets[0].intersection(*sets[1:])
    # Seed first, then the rest sorted
    seed = [t for t in _INTERSECTION_DOMAIN_SEED if t in inter]
    rest = sorted(inter - set(seed))
    return seed + rest


def r5_only_domain_items(limit: int = 40) -> list[str]:
    dom = pd.read_parquet(_official_dir() / "official_person_domain.parquet")
    by_wave: dict[int, set[str]] = {}
    for w in WEIGHTED_WAVES:
        uids = set(_keys(w)["row_uid"])
        sub = dom[(dom["wave"] == w) & (dom["row_uid"].isin(uids))]
        by_wave[w] = set(sub["canonical_item"].unique())
    only = sorted(by_wave[2024] - by_wave[2022] - by_wave[2020])
    preferred = [x for x in _R5_ONLY_EXAMPLES if x in only]
    return preferred + [x for x in only if x not in preferred][:limit]


def _weighted_tvd(p: pd.Series, q: pd.Series) -> float:
    idx = sorted(set(p.index) | set(q.index))
    pv = np.array([float(p.get(i, 0.0)) for i in idx], dtype=float)
    qv = np.array([float(q.get(i, 0.0)) for i in idx], dtype=float)
    if pv.sum() > 0:
        pv /= pv.sum()
    if qv.sum() > 0:
        qv /= qv.sum()
    return 0.5 * float(np.abs(pv - qv).sum())


def width_depth_rows() -> list[dict]:
    rows = []
    for wave in WEIGHTED_WAVES:
        real = build_official_anchor_frame(wave)
        if "width" not in real.columns:
            gcols = [
                c
                for c in real.columns
                if c.startswith("attend_") and c not in ("attend_any", "attend_count_band")
            ]
            real["width"] = (real[gcols] == "예").sum(axis=1)
        for ablation in ("margins", "structure_T"):
            path = settings.synthetic_dir / f"era_{wave}_quant" / f"personas_{ablation}.parquet"
            if not path.exists():
                continue
            syn = pd.read_parquet(path)
            if "width" not in syn.columns:
                gcols = [
                    c
                    for c in syn.columns
                    if c.startswith("attend_") and c not in ("attend_any", "attend_count_band")
                ]
                syn["width"] = (syn[gcols] == "예").sum(axis=1)
            w_real = float(np.average(real["width"], weights=real["w"]))
            d_real = float(np.average(real["depth"], weights=real["w"]))
            # age-conditional MAE for width
            band_maes = []
            for age, sw in syn.groupby("age_band"):
                rw = real[real["age_band"] == age]
                if len(rw) == 0 or len(sw) == 0:
                    continue
                band_maes.append(
                    abs(float(np.average(rw["width"], weights=rw["w"])) - float(sw["width"].mean()))
                )
            # depth: if the synthetic still holds the raw count, use it as-is; otherwise the
            # band's representative value.
            # Fold the observed series into **the same representation as the synthetic**
            # before comparing (fix 2026-09-11).
            if "depth" in syn.columns:
                d_syn = float(syn["depth"].mean())
                d_real_cmp = d_real
            else:
                d_syn = float(depth_proxy_from_band(syn["attend_count_band"]).mean())
                rp = depth_proxy_from_band(real["attend_count_band"])
                rp_ok = rp.notna()
                d_real_cmp = float(np.average(rp[rp_ok], weights=real["w"][rp_ok]))
            rows.append(
                {
                    "wave": wave,
                    "ablation": ablation,
                    "width_mean_real_weighted": w_real,
                    "width_mean_syn": float(syn["width"].mean()),
                    "width_mae_by_age": float(np.mean(band_maes)) if band_maes else float("nan"),
                    "depth_mean_real_weighted": d_real,
                    "depth_mean_real_like_for_like": d_real_cmp,
                    "depth_mean_syn_proxy": d_syn,
                    "depth_abs_gap": abs(d_real_cmp - d_syn),
                    "depth_discretization_bias": d_real - d_real_cmp,
                }
            )
    return rows


def canonical_holdout_distributions(vars_: list[str]) -> pd.DataFrame:
    canon = pd.read_parquet(_official_dir() / "official_person_canonical.parquet")
    rows = []
    for wave in WEIGHTED_WAVES:
        k = _keys(wave)
        wmap = k.set_index("row_uid")["weight"]
        for var in vars_:
            sub = canon[(canon["wave"] == wave) & (canon["canonical_var"] == var)]
            sub = sub[sub["row_uid"].isin(wmap.index)]
            if sub.empty:
                continue
            ww = sub["row_uid"].map(wmap).astype(float)
            # normalize missing weights
            ww = ww.fillna(0.0)
            if ww.sum() <= 0:
                continue
            tab = sub.assign(_w=ww).groupby("value", dropna=False)["_w"].sum()
            tab = tab / tab.sum()
            for val, p in tab.items():
                rows.append(
                    {
                        "wave": wave,
                        "canonical_var": var,
                        "value": val,
                        "p_weighted": float(p),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = settings.outputs_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = settings.data_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    canon_hold = intersection_canonical_holdouts()
    domain_hold = intersection_domain_triples()
    r5_only = r5_only_domain_items()

    registry = {
        "weighted_waves": list(WEIGHTED_WAVES),
        "intersection_canonical_holdouts": canon_hold,
        "intersection_domain_n": len(domain_hold),
        "intersection_domain_primary": [
            {"family": a, "measure": b, "item": c} for a, b, c in domain_hold[:15]
        ],
        "r5_only_domain_items_sample": r5_only[:25],
        "metrics_policy": {
            "binary_ok": True,
            "require_width_depth_spend": True,
            "forbid_gap_narrative": True,
            "mode_scale_note": "이분 격차 서사 금지; 폭·깊이·지출은 분포 재현 지표로만",
        },
    }
    reg_path = eval_dir / "era_holdout_registry.json"
    reg_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    dist = canonical_holdout_distributions(canon_hold)
    dist_path = out_dir / "era_holdout_intersection_dist.csv"
    dist.to_csv(dist_path, index=False)

    # cross-era TVD on holdout canonical (same var, wave pairs)
    tvd_rows = []
    for var in canon_hold:
        piv = dist[dist["canonical_var"] == var].pivot_table(
            index="value", columns="wave", values="p_weighted", fill_value=0.0
        )
        for a, b in ((2020, 2022), (2022, 2024), (2020, 2024)):
            if a in piv.columns and b in piv.columns:
                tvd_rows.append(
                    {
                        "canonical_var": var,
                        "wave_a": a,
                        "wave_b": b,
                        "tvd": _weighted_tvd(piv[a], piv[b]),
                    }
                )
    tvd_df = pd.DataFrame(tvd_rows)
    tvd_path = out_dir / "era_holdout_intersection_cross_era_tvd.csv"
    tvd_df.to_csv(tvd_path, index=False)

    wd = pd.DataFrame(width_depth_rows())
    wd_path = out_dir / "era_holdout_width_depth.csv"
    wd.to_csv(wd_path, index=False)

    lines = [
        "# Era holdout — 교집합 · 폭/깊이",
        "",
        "## 설계",
        "",
        "- **교집합 문항**: 가중 3시대에 공통이고 생성 앵커가 아닌 canonical / domain.",
        "- **R5 전용**: 2024만 있는 domain item — 시대 횡비교 금지, 2024 서사·holdout 전용.",
        "- **지표**: 이분뿐 아니라 **폭(width)·깊이(depth)·지출** 분포. 격차(«줄었다») 서사 금지.",
        "",
        "- registry: `data/eval/era_holdout_registry.json`",
        f"- canonical holdouts: `{canon_hold}`",
        f"- domain intersection n={len(domain_hold)} (primary {len(registry['intersection_domain_primary'])})",
        f"- R5-only sample n={len(r5_only)}",
        "",
        "## 폭·깊이 (합성 vs 가중 실측)",
        "",
        wd.to_markdown(index=False) if hasattr(wd, "to_markdown") else wd.to_string(index=False),
        "",
        "## 교집합 canonical — 시대 간 TVD (실측 가중, 참고)",
        "",
        "생성 유효성이 아니라 **문항이 시대마다 같은 의미를 갖는지** 진단용.",
        "",
    ]
    if not tvd_df.empty:
        summary = tvd_df.groupby(["wave_a", "wave_b"])["tvd"].mean().reset_index()
        lines.append(
            summary.to_markdown(index=False)
            if hasattr(summary, "to_markdown")
            else summary.to_string(index=False)
        )
    md_path = out_dir / "era_holdout_metrics.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s %s %s %s", reg_path, dist_path, wd_path, md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
