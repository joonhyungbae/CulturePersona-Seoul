"""Era appendix audit — variance preservation · segment holdout (ADR-012).

Variance preservation and segment holdout on the official era frame (appendix numbers).

Run:
  uv run python scripts/era_appendix_audit.py --wave 2024
  uv run python scripts/era_appendix_audit.py --wave 2024 --skip-segment
Outputs: outputs/reports/era_variance.csv · era_segment_holdout.csv · era_appendix_audit.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.chain import (
    ChainSampler,
    add_derived_columns,
    build_spec,
    era_core_chain,
)
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import (
    AGE_BANDS,
    ERA_CORE_ANCHORS,
    GENRE_ANCHORS,
    ROOT_ANCHORS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("era_appendix_audit")
for _n in ("sfac_persona.grounding.chain", "sfac_persona.grounding.official"):
    logging.getLogger(_n).setLevel(logging.WARNING)

SEED = 20260911
REPORTS = settings.outputs_dir / "reports"


def _non_root(frame: pd.DataFrame) -> list[str]:
    return [a for a in ERA_CORE_ANCHORS if a in frame.columns and a not in ROOT_ANCHORS]


def _genres(frame: pd.DataFrame) -> list[str]:
    return [g for g in GENRE_ANCHORS if g in frame.columns]


# ---------------------------------------------------------------- variance


def _entropy_bits(codes: np.ndarray, w: np.ndarray) -> float:
    s = pd.Series(w).groupby(codes).sum()
    p = s / s.sum()
    return float(-(p * np.log2(p)).sum())


def _mean_hamming(x: np.ndarray, w: np.ndarray) -> float:
    p = np.average(x, axis=0, weights=w)
    return float((2 * p * (1 - p)).sum())


def _variance_metrics(df: pd.DataFrame, w: np.ndarray, label: str, genres: list[str]) -> list[dict]:
    x = (df[genres] == "예").to_numpy(dtype=int)
    codes = x @ (1 << np.arange(x.shape[1]))
    n_genres = x.sum(axis=1)
    rows = []
    for a in AGE_BANDS:
        m = (df["age_band"] == a).to_numpy()
        if m.sum() == 0:
            continue
        mu = np.average(n_genres[m], weights=w[m])
        rows.append(
            {
                "source": label,
                "age_band": a,
                "n": int(m.sum()),
                "profile_entropy": _entropy_bits(codes[m], w[m]),
                "mean_pairwise_ham": _mean_hamming(x[m], w[m]),
                "n_genres_mean": float(mu),
                "n_genres_sd": float(np.sqrt(np.average((n_genres[m] - mu) ** 2, weights=w[m]))),
                "distinct_profiles": len(np.unique(codes[m])),
            }
        )
    ages = df["age_band"].to_numpy()
    grand = np.average(n_genres, weights=w)
    within = between = 0.0
    for a in AGE_BANDS:
        m = ages == a
        if m.sum() == 0:
            continue
        wa = w[m].sum()
        mu = np.average(n_genres[m], weights=w[m])
        within += np.average((n_genres[m] - mu) ** 2, weights=w[m]) * wa
        between += (mu - grand) ** 2 * wa
    rows.append(
        {
            "source": label,
            "age_band": "ALL",
            "n": len(df),
            # The single-persona baseline has within=0: each age band is only its modal
            # profile, and between is also 0 if every band's mode is "attended nothing."
            "within_share": (within / (within + between))
            if (within + between) > 0
            else float("nan"),
        }
    )
    return rows


def _single_persona_baseline(obs: pd.DataFrame, w: np.ndarray, genres: list[str]) -> pd.DataFrame:
    """The 'representative persona' practice: replace everyone with the age band's modal profile."""
    x = (obs[genres] == "예").to_numpy(dtype=int)
    codes = x @ (1 << np.arange(x.shape[1]))
    rows = []
    for a in AGE_BANDS:
        m = (obs["age_band"] == a).to_numpy()
        if m.sum() == 0:
            continue
        mode = pd.Series(w[m]).groupby(codes[m]).sum().idxmax()
        bits = [(int(mode) >> i) & 1 for i in range(len(genres))]
        rows.extend(
            [
                {
                    "age_band": a,
                    **{g: ("예" if b else "아니오") for g, b in zip(genres, bits, strict=True)},
                }
            ]
            * int(m.sum())
        )
    return pd.DataFrame(rows)


def run_variance(wave: int, frame: pd.DataFrame) -> pd.DataFrame:
    genres = _genres(frame)
    w_obs = frame["w"].to_numpy(dtype=float)
    syn_path = settings.synthetic_dir / f"era_{wave}_quant" / "personas_structure_T.parquet"
    if not syn_path.exists():
        syn_path = settings.synthetic_dir / f"era_{wave}_quant" / "personas_margins.parquet"
    syn = pd.read_parquet(syn_path)
    base = _single_persona_baseline(frame, w_obs, genres)
    rows = (
        _variance_metrics(frame, w_obs, "microdata", genres)
        + _variance_metrics(syn, np.ones(len(syn)), "chain", genres)
        + _variance_metrics(base, w_obs.copy(), "single_persona", genres)
    )
    df = pd.DataFrame(rows)
    df.insert(0, "wave", wave)
    log.info("variance rows=%d (syn=%s)", len(df), syn_path.name)
    return df


# ---------------------------------------------------------- segment holdout


def _dist(s: pd.Series, w: pd.Series | None = None) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    if w is None:
        return s.value_counts(normalize=True)
    d = w.loc[s.index].groupby(s).sum()
    return d / d.sum()


def _tvd(p: pd.Series, q: pd.Series) -> float:
    if p.empty or q.empty:
        return float("nan")
    idx = p.index.union(q.index)
    return float(0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())


def _sample_cell(
    spec: dict, fixed: dict[str, str], root_mix: pd.Series, n: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampler = ChainSampler(spec)
    rows = []
    for combo, share in root_mix.items():
        k = round(n * share)
        if k <= 0:
            continue
        free = dict(
            zip(root_mix.index.names, combo if isinstance(combo, tuple) else (combo,), strict=False)
        )
        rows.append(
            sampler.sample(k, seed=int(rng.integers(1 << 31)), fixed_root={**free, **fixed})
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_segment(
    wave: int, frame: pd.DataFrame, by: str, n_units: int, n_sample: int, min_n: int
) -> pd.DataFrame:
    w = frame["w"]
    group_cols = ["living_area", "age_band"] if by == "cell" else [by]
    counts = frame.groupby(group_cols, observed=True).size().sort_values(ascending=False)
    eligible = counts[counts >= min_n]
    if eligible.empty:
        log.warning("by=%s: no unit with n>=%d", by, min_n)
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
    pick = np.linspace(0, len(eligible) - 1, min(n_units, len(eligible))).round().astype(int)
    keys = [eligible.index[i] for i in dict.fromkeys(pick)]
    cells = [k if isinstance(k, tuple) else (k,) for k in keys]
    log.info("by=%s: %d eligible → %d tested", by, len(eligible), len(cells))

    # The released chain structure (era_core_chain), refitted on what remains after the segment is held out.
    spec_full = build_spec(frame, root_source="survey", nodes=era_core_chain())
    # Compare only nodes the chain actually produces.
    produced = {n.get("name") for n in spec_full["nodes"] if n.get("name")}
    anchors = [a for a in _non_root(frame) if a in produced]
    log.info("by=%s: comparing %d anchors produced by the chain", by, len(anchors))
    naive = {a: _dist(frame[a], w) for a in anchors}
    rows = []
    for i, cell in enumerate(cells, 1):
        mask = np.logical_and.reduce([frame[c] == v for c, v in zip(group_cols, cell, strict=True)])
        real = frame[mask]
        free_cols = [c for c in ROOT_ANCHORS if c not in group_cols]
        root_mix = real.groupby(free_cols, observed=True).size()
        root_mix = (root_mix / root_mix.sum()).sort_values(ascending=False)
        fixed = dict(zip(group_cols, cell, strict=True))
        log.info("[%d/%d] %s (n=%d) — refit", i, len(cells), " × ".join(map(str, cell)), len(real))
        spec_out = build_spec(frame[~mask], root_source="survey", nodes=era_core_chain())
        syn_out = _sample_cell(spec_out, fixed, root_mix, n_sample, SEED + i)
        syn_orc = _sample_cell(spec_full, fixed, root_mix, n_sample, SEED + i)

        pw = (w.loc[real.index] / w.loc[real.index].sum()).to_numpy()
        floor = []
        for _ in range(30):
            boot = real.iloc[rng.choice(len(real), size=len(real), replace=True, p=pw)]
            floor.append(np.nanmean([_tvd(_dist(boot[a]), _dist(real[a], w)) for a in anchors]))
        for a in anchors:
            rows.append(
                {
                    "wave": wave,
                    "by": by,
                    "cell": " × ".join(map(str, cell)),
                    "n_real": len(real),
                    "anchor": a,
                    "tvd_holdout": _tvd(_dist(syn_out[a]), _dist(real[a], w))
                    if len(syn_out)
                    else np.nan,
                    "tvd_oracle": _tvd(_dist(syn_orc[a]), _dist(real[a], w))
                    if len(syn_orc)
                    else np.nan,
                    "tvd_naive": _tvd(naive[a], _dist(real[a], w)),
                }
            )
        rows[-1]["floor_cell"] = float(np.mean(floor))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2024)
    ap.add_argument("--skip-segment", action="store_true")
    ap.add_argument("--cells", type=int, default=15)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--min-n", type=int, default=40)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = add_derived_columns(build_official_anchor_frame(args.wave))
    log.info("wave=%d n=%d anchors=%d", args.wave, len(frame), len(_non_root(frame)))

    var = run_variance(args.wave, frame)
    var.to_csv(REPORTS / "era_variance.csv", index=False)

    seg = pd.DataFrame()
    if not args.skip_segment:
        parts = [
            run_segment(args.wave, frame, by, args.cells, args.sample, args.min_n)
            for by in ("age_band", "living_area", "cell")
        ]
        seg = pd.concat([p for p in parts if len(p)], ignore_index=True)
        seg.to_csv(REPORTS / "era_segment_holdout.csv", index=False)

    piv = (
        var[var.age_band != "ALL"]
        .pivot(
            index="age_band",
            columns="source",
            values=["profile_entropy", "mean_pairwise_ham", "n_genres_sd"],
        )
        .reindex([a for a in AGE_BANDS if a in set(var.age_band)])
    )
    within = var[var.age_band == "ALL"].set_index("source")["within_share"]
    lines = [
        "# Era 부록 감사 — 분산 보존 · 세그먼트 홀드아웃 (ADR-012)",
        "",
        f"wave={args.wave} · 프레임 n={len(frame):,} · 앵커 {len(_non_root(frame))}개",
        "",
        "## 분산 보존 (연령대 안의 관람 프로파일 산포)",
        "",
        piv.round(3).to_markdown(),
        "",
        "## 장르 수 분산 중 연령대 내 비율",
        "",
        within.round(4).to_markdown(),
        "",
        "단일 페르소나 기준선은 연령대 안 분산이 0 이다. 체인이 마이크로데이터의 within_share 를 재현하는가가 지표다.",
        "",
    ]
    if len(seg):
        by_unit = seg.groupby(["by", "cell", "n_real"], as_index=False)[
            ["tvd_holdout", "tvd_oracle", "tvd_naive"]
        ].mean()
        floors = seg.dropna(subset=["floor_cell"]).groupby(["by", "cell"])["floor_cell"].first()
        by_unit["floor"] = by_unit.set_index(["by", "cell"]).index.map(floors)
        summ = seg.groupby("by", as_index=False)[["tvd_holdout", "tvd_oracle", "tvd_naive"]].mean()
        summ["floor"] = summ["by"].map(by_unit.groupby("by")["floor"].mean())
        lines += [
            "## 세그먼트 홀드아웃 (단위를 통째로 빼고 체인 재구축)",
            "",
            summ.round(3).to_markdown(index=False),
            "",
            "외삽이 oracle 에 가까울수록 그 단위를 표집하지 않아도 체인이 복원한다는 뜻이다.",
            "",
            by_unit.round(3).to_markdown(index=False),
            "",
        ]
    lines.append("CSV: `era_variance.csv` · `era_segment_holdout.csv`")
    (REPORTS / "era_appendix_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote era_appendix_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
