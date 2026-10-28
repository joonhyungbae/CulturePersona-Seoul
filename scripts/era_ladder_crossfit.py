"""Out-of-sample (split-half cross-fit) remeasurement of the five error-ladder rungs.

The ladder in manuscript Table 1
(indep → cond_age → cond_age_sex → cond_age_route → cond_age_attend)
estimates conditionals on all respondents and scores them on the same
respondents' joint: an in-sample value. Here the frame is split with the same
seed and the same halves as `era_chain_crossfit.split_half`:

  - conditionals P(X|C)·P(Y|C) and margins are estimated only on fit half A
  - each pair's reconstructed joint is scored as TVD against score half B's
    weighted observed joint P_B(X,Y).

Two versions of P(C) are reported.

  oos    Σ_c P_B(c) P_A(X|c) P_A(Y|c)     — default. Conditionals from A; mixture weights from the score sample.
  oos_a  Σ_c P_A(c) P_A(X|c) P_A(Y|c)     — variant. The whole table is built from A, as in the chain.

`indep` has no conditioner, so the two columns match: P_A(X)P_A(Y). The floor
is the mean observed-joint TVD of A vs B on the same 187 pairs. Person rows
stay in memory; outputs are aggregate tables only.

Run: uv run python scripts/era_ladder_crossfit.py [--waves 2020 2022 2024] [--splits 3]
Outputs: outputs/reports/era_ladder_crossfit.csv · era_ladder_crossfit.md
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era_chain_crossfit import SEED

from sfac_persona.config import settings
from sfac_persona.grounding.error_study import (
    ATTEND_BLOCK,
    RECON,
    _align,
    _cond_indep,
    _cond_route,
    _joint,
    ladder_mask,
    routed_values,
)
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import ERA_CORE_ANCHORS, ROOT_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _name in ("sfac_persona.grounding.official", "sfac_persona.grounding.chain"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("era_ladder_crossfit")

WAVES = (2020, 2022, 2024)
N_LADDER = 187

# Rung → condition variables. route's mixture weight is also age band
# (π and the stratum distribution are conditionals, so they stay on A).
_BY: dict[str, list[str]] = {
    "cond_age": ["age_band"],
    "cond_age_sex": ["age_band", "sex"],
    "cond_age_route": ["age_band"],
    "cond_age_attend": ["age_band", "attend_any"],
}

COLUMNS = [
    "wave",
    "rung",
    "in_sample",
    "oos_mean",
    "oos_min",
    "oos_max",
    "floor",
    "chain_oos",
    "chain_in",
    "oos_a_mean",
    "oos_a_min",
    "oos_a_max",
]


@dataclass(frozen=True)
class SplitResult:
    """Mean over 187 pairs for one half. tvd_b uses P_B(C); tvd_a uses P_A(C)."""

    n_fit: int
    n_score: int
    n_pairs: int
    tvd_b: dict[str, float]
    tvd_a: dict[str, float]
    floor: float


def core_anchors(frame: pd.DataFrame) -> list[str]:
    return [a for a in ERA_CORE_ANCHORS if a in frame.columns and a not in ROOT_ANCHORS]


def split_halves(frame: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same seed and mask as `era_chain_crossfit.split_half`. Weights only split the rows."""
    rng = np.random.default_rng(SEED + k)
    mask = rng.random(len(frame)) < 0.5
    fit = frame[mask].reset_index(drop=True)
    score = frame[~mask].reset_index(drop=True)
    return fit, score


def ladder_pairs(anchors: list[str]) -> list[tuple[str, str]]:
    rows = [
        {
            "anchor_a": a,
            "anchor_b": b,
            "structural": a in ATTEND_BLOCK and b in ATTEND_BLOCK,
            "conditioning": "attend_any" in (a, b),
        }
        for a, b in itertools.combinations(anchors, 2)
    ]
    kept = pd.DataFrame(rows)
    kept = kept[ladder_mask(kept)]
    return list(zip(kept.anchor_a, kept.anchor_b, strict=True))


def _tvd(p: pd.DataFrame, q: pd.DataFrame) -> float:
    left, right = _align(p, q)
    return float(0.5 * np.abs(left - right).sum())


def _indep(obs: pd.DataFrame) -> pd.DataFrame:
    """Independence joint from the observed joint's margins. Matches indep in `pair_errors`."""
    return pd.DataFrame(
        np.outer(obs.sum(axis=1).to_numpy(), obs.sum(axis=0).to_numpy()),
        index=obs.index,
        columns=obs.columns,
    )


def _reweight_pc(fit: pd.DataFrame, score: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Keep within-group weight shares from fit; match only group totals to score's P(by).

    Condition cells that do not overlap are dropped. `_cond_indep` renormalizes on the
    remaining weight, so the mixture is P_score(c) over c present in both halves.
    """
    w_fit = fit.groupby(by, observed=True, dropna=True)["w"].sum()
    w_score = score.groupby(by, observed=True, dropna=True)["w"].sum()
    if float(w_fit.sum()) <= 0.0 or float(w_score.sum()) <= 0.0:
        raise ValueError("reweight: empty weight")
    p_fit = w_fit / w_fit.sum()
    p_score = w_score / w_score.sum()
    tab = pd.concat({"p_fit": p_fit, "p_score": p_score}, axis=1)
    tab["factor"] = (tab["p_score"] / tab["p_fit"]).replace([np.inf, -np.inf], np.nan)
    tab = tab.reset_index()
    out = fit.merge(tab[[*by, "factor"]], on=by, how="left")
    null_key = fit[by].isna().any(axis=1).to_numpy()
    factor = out["factor"].to_numpy(dtype=float)
    factor = np.where(null_key, 1.0, factor)
    factor = np.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0)
    out["w"] = fit["w"].to_numpy(dtype=float) * factor
    kept = out.loc[out["w"] > 0.0, list(fit.columns)]
    if kept.empty:
        raise ValueError("reweight: no overlapping conditioner cells")
    return kept.reset_index(drop=True)


def _recon(df: pd.DataFrame, a: str, b: str, rung: str, routed: dict[str, str]) -> pd.DataFrame:
    if rung == "indep":
        return _indep(_joint(df, a, b))
    if rung == "cond_age_route":
        return _cond_route(df, a, b, routed)
    return _cond_indep(df, a, b, _BY[rung])


def score_split(fit: pd.DataFrame, score: pd.DataFrame, anchors: list[str]) -> SplitResult:
    """Estimate the five rungs on A and score them on B's observed joint. If A=B, this is the in-sample TVD."""
    if "w" not in fit.columns or "w" not in score.columns:
        raise ValueError("frame needs weight column 'w'")
    pairs = ladder_pairs(anchors)
    if not pairs:
        raise ValueError("no ladder pairs")
    routed = routed_values(fit, anchors)
    sum_b = dict.fromkeys(RECON, 0.0)
    sum_a = dict.fromkeys(RECON, 0.0)
    floor_sum = 0.0
    for a, b in pairs:
        obs_b = _joint(score, a, b)
        obs_a = _joint(fit, a, b)
        floor_sum += _tvd(obs_b, obs_a)
        fit_sub = fit.dropna(subset=[a, b])
        score_sub = score.dropna(subset=[a, b])
        for rung in RECON:
            q_a = _recon(fit, a, b, rung, routed)
            if rung == "indep":
                q_b = q_a
            else:
                q_b = _recon(_reweight_pc(fit_sub, score_sub, _BY[rung]), a, b, rung, routed)
            sum_a[rung] += _tvd(obs_b, q_a)
            sum_b[rung] += _tvd(obs_b, q_b)
    n = len(pairs)
    return SplitResult(
        n_fit=len(fit),
        n_score=len(score),
        n_pairs=n,
        tvd_b={k: v / n for k, v in sum_b.items()},
        tvd_a={k: v / n for k, v in sum_a.items()},
        floor=floor_sum / n,
    )


def in_sample_from_atlas(path: Path, wave: int) -> dict[str, float]:
    pairs = pd.read_csv(path)
    sub = pairs[pairs["wave"] == wave]
    if sub.empty:
        raise ValueError(f"no atlas pairs for wave {wave}")
    kept = sub[ladder_mask(sub)]
    if len(kept) != N_LADDER:
        raise ValueError(f"wave {wave}: expected {N_LADDER} ladder pairs, got {len(kept)}")
    return {rung: float(kept[f"tvd_{rung}"].mean()) for rung in RECON}


def chain_split_means(path: Path) -> dict[int, tuple[float, float]]:
    """wave → (out of sample, in sample). Split-half rows only. Waves missing from the file are omitted."""
    if not path.exists():
        log.warning("chain crossfit csv missing: %s", path)
        return {}
    df = pd.read_csv(path)
    sh = df[df["setting"] == "split-half"]
    out: dict[int, tuple[float, float]] = {}
    for wave, g in sh.groupby("wave"):
        out[int(wave)] = (
            float(g["tvd_out_of_sample"].mean()),
            float(g["tvd_in_sample"].mean()),
        )
    return out


def summarize_wave(
    parts: list[SplitResult],
    in_sample: dict[str, float],
    chain_oos: float,
    chain_in: float,
    wave: int,
) -> list[dict[str, float | int | str]]:
    floor = float(np.mean([p.floor for p in parts]))
    rows: list[dict[str, float | int | str]] = []
    for rung in RECON:
        b = [p.tvd_b[rung] for p in parts]
        a = [p.tvd_a[rung] for p in parts]
        rows.append(
            {
                "wave": wave,
                "rung": rung,
                "in_sample": in_sample[rung],
                "oos_mean": float(np.mean(b)),
                "oos_min": float(np.min(b)),
                "oos_max": float(np.max(b)),
                "floor": floor,
                "chain_oos": chain_oos,
                "chain_in": chain_in,
                "oos_a_mean": float(np.mean(a)),
                "oos_a_min": float(np.min(a)),
                "oos_a_max": float(np.max(a)),
            }
        )
    return rows


def _inversions(g: pd.DataFrame, col: str) -> list[str]:
    seq = [(r, float(g.loc[r, col])) for r in RECON]
    bad: list[str] = []
    for (r1, v1), (r2, v2) in itertools.pairwise(seq):
        if v1 + 1e-6 < v2:
            bad.append(f"{r2} {v2:.4f} > {r1} {v1:.4f}")
    return bad


def _fmt(v: float) -> str:
    return "—" if not np.isfinite(v) else f"{v:.4f}"


def interpret(table: pd.DataFrame) -> list[str]:
    """Three lines: order, age rung vs chain, overfitting share (in − oos)."""
    waves = sorted(int(w) for w in table["wave"].unique())
    wlabel = "·".join(str(w) for w in waves)

    def bad_of(col: str) -> dict[int, list[str]]:
        found: dict[int, list[str]] = {}
        for wave in waves:
            g = table[table["wave"] == wave].set_index("rung")
            inv = _inversions(g, col)
            if inv:
                found[wave] = inv
        return found

    bad_b, bad_a = bad_of("oos_mean"), bad_of("oos_a_mean")
    if not bad_b and not bad_a:
        line1 = (
            f"표본 밖에서 다섯 단의 순서는 B의 P(C)와 A의 P(C) 두 변형 모두 {wlabel} 에서 유지된다."
        )
    else:
        bits: list[str] = []
        for label, bad in (("B의 P(C)", bad_b), ("A의 P(C)", bad_a)):
            if not bad:
                bits.append(f"{label}는 {wlabel} 에서 순서가 유지된다")
            else:
                detail = "; ".join(f"{w}: {', '.join(v)}" for w, v in bad.items())
                bits.append(f"{label}에서 순서가 뒤집힌 곳 — {detail}")
        line1 = "표본 밖 순서. " + ". ".join(bits) + "."

    focus = table[table["wave"] == waves[-1]].set_index("rung")
    wave_f = waves[-1]
    age = focus.loc["cond_age"]
    line2 = (
        f"{wave_f} 연령 단 표본 밖은 {_fmt(float(age.oos_mean))} (B의 P(C)), "
        f"{_fmt(float(age.oos_a_mean))} (A의 P(C))이고, "
        f"체인 표본 밖은 {_fmt(float(age.chain_oos))} "
        f"(체인 표본 내 {_fmt(float(age.chain_in))}), "
        f"같은 187 쌍의 A–B 바닥은 {_fmt(float(age.floor))} 이다."
    )
    gaps_b = ", ".join(
        f"{r} {float(focus.loc[r, 'in_sample'] - focus.loc[r, 'oos_mean']):+.4f}" for r in RECON
    )
    gaps_a = ", ".join(
        f"{r} {float(focus.loc[r, 'in_sample'] - focus.loc[r, 'oos_a_mean']):+.4f}" for r in RECON
    )
    line3 = (
        f"과적합 몫(in − oos, {wave_f})은 B의 P(C)에서 {gaps_b} 이고, "
        f"A의 P(C)에서 {gaps_a} 이다. 음수는 표본 밖 TVD 가 표본 내보다 크다는 뜻이다."
    )
    return [line1, line2, line3]


def write_md(table: pd.DataFrame, path: Path, splits: int) -> None:
    lines = [
        "# 오차 사다리 반분 교차적합",
        "",
        (
            f"> 자동 생성: `scripts/era_ladder_crossfit.py` "
            f"(분할 {splits}, 시드 {SEED} = `era_chain_crossfit.split_half`)."
        ),
        "",
        (
            "조건부 P(X|C)·P(Y|C) 와 주변분포는 적합 절반 A 에서만 추정하고, "
            "채점은 점수 절반 B 의 가중 관측 결합에 대한 TVD 다."
        ),
        (
            "기본 열 `oos_*` 는 A 의 조건부에 B 의 P(C) 를 곱해 더한 재구성이다. "
            "`oos_a_*` 는 같은 조건부에 A 의 P(C) 를 쓴 변형이다 "
            "(체인 샘플러가 A 의 표 전체를 쓰는 것과 같은 쪽)."
        ),
        "`indep` 는 조건 변수가 없어 두 열이 같다: P_A(X)P_A(Y) 대 P_B(X,Y).",
        (
            "바닥은 같은 187 쌍에서 A 와 B 의 관측 결합 TVD 를 분할에 대해 평균한 값이다. "
            "표본 내는 `decade_error_atlas_pairs.csv` 의 `ladder_mask` 187 쌍 평균이고, "
            "체인 열은 `era_chain_crossfit.csv` 의 split-half 평균이다 (파일에 있는 파동만)."
        ),
        "",
        (
            "| wave | rung | in_sample | oos_mean | oos_min | oos_max | oos_a_mean "
            "| floor | chain_oos | chain_in |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {int(r['wave'])} | {r['rung']} | {r['in_sample']:.4f} | {r['oos_mean']:.4f} | "
            f"{r['oos_min']:.4f} | {r['oos_max']:.4f} | {r['oos_a_mean']:.4f} | "
            f"{r['floor']:.4f} | {_fmt(float(r['chain_oos']))} | {_fmt(float(r['chain_in']))} |"
        )
    lines.append("")
    lines.extend(interpret(table))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(waves: list[int], splits: int) -> pd.DataFrame:
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atlas = reports / "decade_error_atlas_pairs.csv"
    chain = chain_split_means(reports / "era_chain_crossfit.csv")
    rows: list[dict[str, float | int | str]] = []
    for wave in waves:
        log.info("wave %s — loading frame", wave)
        frame = build_official_anchor_frame(wave)
        anchors = core_anchors(frame)
        base = in_sample_from_atlas(atlas, wave)
        if wave in chain:
            c_oos, c_in = chain[wave]
        else:
            log.warning("wave %s: no split-half rows in era_chain_crossfit.csv", wave)
            c_oos, c_in = float("nan"), float("nan")
        parts: list[SplitResult] = []
        for k in range(splits):
            fit, score = split_halves(frame, k)
            log.info("wave %s split %d — n_fit %d n_score %d", wave, k, len(fit), len(score))
            part = score_split(fit, score, anchors)
            if part.n_pairs != N_LADDER:
                raise RuntimeError(
                    f"wave {wave} split {k}: {part.n_pairs} pairs, expected {N_LADDER}"
                )
            parts.append(part)
            log.info(
                "wave %s split %d floor %.4f | oos_b %s | oos_a %s",
                wave,
                k,
                part.floor,
                " ".join(f"{r}={part.tvd_b[r]:.4f}" for r in RECON),
                " ".join(f"{r}={part.tvd_a[r]:.4f}" for r in RECON),
            )
        rows.extend(summarize_wave(parts, base, c_oos, c_in, wave))
    table = pd.DataFrame(rows)[COLUMNS]
    table.to_csv(reports / "era_ladder_crossfit.csv", index=False)
    write_md(table, reports / "era_ladder_crossfit.md", splits)
    text = "\n".join(interpret(table))
    log.info("\n%s", text)
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, nargs="+", default=list(WAVES))
    ap.add_argument("--splits", type=int, default=3)
    args = ap.parse_args()
    run(list(args.waves), args.splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
