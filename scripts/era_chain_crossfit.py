"""Out-of-sample checks for the chain sampler — split-half cross-fit, prior-wave fit, and a
placebo conditioner.

The manuscript's chain error (`tvdChain`) builds the table on 2024 respondents and scores
those same respondents. Three checks ask whether that in-sample value contains overfitting.

  split-half   Split respondents at random into halves; build the chain on A and score it
               on B's observed joints (out of sample). Report that beside the same chain
               scored on A (in sample) and the distance between the A and B observed joints
               (sampling floor).
  prior-wave   Score a chain built on 2022 respondents on the 2024 observed joints —
               "build this year from last wave's tables". Baseline: the distance of the
               2022 observed joints themselves against 2024 (last wave's microdata used
               as-is).
  placebo      Whether the attendance rung of the error ladder is an artifact of in-sample
               fit: compare the reduction when an independent random binary with the same
               marginal as the attendance rate is conditioned on together with age band
               (same cell count) with the reduction from attendance.
  design-OOS   Downstream 2 (design pretest) out of sample: a synthetic population from A,
               A's age table, and A's marginals predict B's cell-level attendance rates.

Run: uv run python scripts/era_chain_crossfit.py [--wave 2024] [--splits 3] [--sample 30000]
Outputs: outputs/reports/era_chain_crossfit.csv · era_chain_crossfit_design.csv ·
era_chain_crossfit.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era_design_pretest import CELL_KEYS, MIN_CELL, PUBLISHED_KEYS, cell_rates, required_n
from era_sampler_audit import build_variant, pair_tvd

from sfac_persona.config import settings
from sfac_persona.grounding import chain as C
from sfac_persona.grounding.error_study import _align, _cond_indep, _joint, ladder_mask, pair_errors
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import ERA_CORE_ANCHORS, ROOT_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for _n in ("sfac_persona.grounding.official", "sfac_persona.grounding.chain"):
    logging.getLogger(_n).setLevel(logging.WARNING)
log = logging.getLogger("chain_crossfit")

SEED = 20260923
N_PLACEBO = 5


def _anchors(frame: pd.DataFrame) -> list[str]:
    return [a for a in ERA_CORE_ANCHORS if a in frame.columns and a not in ROOT_ANCHORS]


def _obs_tvd(a: pd.DataFrame, b: pd.DataFrame, anchors: list[str]) -> float:
    """Pair-joint distance on non-structure pairs of two observed frames (sampling floor)."""
    from itertools import combinations

    from sfac_persona.grounding.error_study import ATTEND_BLOCK

    ds = []
    for x, y in combinations(anchors, 2):
        if x in ATTEND_BLOCK and y in ATTEND_BLOCK:
            continue
        o, s = _align(_joint(a, x, y), _joint(b, x, y))
        ds.append(float(0.5 * np.abs(o - s).sum()))
    return float(np.mean(ds))


def split_half(frame: pd.DataFrame, n_sample: int, splits: int) -> tuple[list[dict], list[dict]]:
    anchors = _anchors(frame)
    rows, design = [], []
    for k in range(splits):
        rng = np.random.default_rng(SEED + k)
        mask = rng.random(len(frame)) < 0.5
        fa, fb = frame[mask].reset_index(drop=True), frame[~mask].reset_index(drop=True)
        spec = build_variant(fa, "released")
        syn = C.ChainSampler(spec).sample(n_sample, seed=SEED + k)
        ins, _, n_ns, _ = pair_tvd(fa, syn, anchors)
        oos, _, _, _ = pair_tvd(fb, syn, anchors)
        floor = _obs_tvd(fa, fb, anchors)
        rows.append(
            {
                "setting": "split-half",
                "split": k,
                "n_fit": len(fa),
                "n_score": len(fb),
                "tvd_in_sample": ins,
                "tvd_out_of_sample": oos,
                "tvd_sampling_floor": floor,
                "n_pairs": n_ns,
            }
        )
        design += _design_oos(fa, fb, syn, k)
        log.info("split %d: in %.4f · out %.4f · floor %.4f", k, ins, oos, floor)
    return rows, design


def _design_oos(fa: pd.DataFrame, fb: pd.DataFrame, syn: pd.DataFrame, k: int) -> list[dict]:
    """Design pretest out of sample: A's tables and synthetic sample predict B's cells."""
    outcome = "attend_any"
    rows = []
    marginal = float((fa[outcome].astype(str) == "예").astype(float).mul(fa.w).sum() / fa.w.sum())
    by_age = cell_rates(fa, ["age_band"], outcome, "w")
    for keys in CELL_KEYS:
        kk = list(keys)
        truth = cell_rates(fb, kk, outcome, "w")
        counts = fb.groupby(kk).size()
        truth = truth[counts >= MIN_CELL]
        syn_p = cell_rates(syn, kk, outcome, None)
        for cell, p_true in truth.items():
            cell_t = cell if isinstance(cell, tuple) else (cell,)
            age_val = cell_t[kk.index("age_band")] if "age_band" in kk else None
            p_age = float(by_age.get(age_val, marginal)) if age_val is not None else marginal
            p_syn = float(syn_p.get(cell, np.nan))
            for label, p_hat in (
                ("marginal", marginal),
                ("age-conditional", p_age),
                ("synthetic", p_syn),
            ):
                if not np.isfinite(p_hat):
                    continue
                rows.append(
                    {
                        "split": k,
                        "cell_key": "+".join(kk),
                        "published_axis": tuple(kk) in PUBLISHED_KEYS,
                        "method": label,
                        "abs_err_pp": 100 * abs(p_hat - float(p_true)),
                        "n_ratio": required_n(p_hat) / required_n(float(p_true)),
                    }
                )
    return rows


def prior_wave(target: int, source: int, n_sample: int) -> dict:
    ft = build_official_anchor_frame(target)
    fs = build_official_anchor_frame(source)
    anchors = [a for a in _anchors(ft) if a in fs.columns]
    spec = build_variant(fs, "released")
    syn = C.ChainSampler(spec).sample(n_sample, seed=SEED)
    oos, _, n_ns, _ = pair_tvd(ft, syn, anchors)
    micro = _obs_tvd(fs, ft, anchors)
    spec_t = build_variant(ft, "released")
    syn_t = C.ChainSampler(spec_t).sample(n_sample, seed=SEED)
    ins, _, _, _ = pair_tvd(ft, syn_t, anchors)
    return {
        "setting": f"prior-wave {source}->{target}",
        "split": -1,
        "n_fit": len(fs),
        "n_score": len(ft),
        "tvd_in_sample": ins,
        "tvd_out_of_sample": oos,
        "tvd_sampling_floor": micro,  # here: distance of "last-wave microdata as-is"
        "n_pairs": n_ns,
    }


def placebo(frame: pd.DataFrame) -> dict:
    """Error reduction under age band × placebo (an independent binary with the same
    marginal as the attendance rate) versus age band × attendance.
    """
    anchors = _anchors(frame)
    p = pair_errors(frame, anchors=anchors)
    p = p[ladder_mask(p)]
    pairs = list(zip(p.anchor_a, p.anchor_b, strict=True))
    age = float(p.tvd_cond_age.mean())
    att = float(p.tvd_cond_age_attend.mean())
    share = float(np.average((frame.attend_any == "예").astype(float), weights=frame.w))
    cuts = []
    for s in range(N_PLACEBO):
        rng = np.random.default_rng(SEED + 100 + s)
        f = frame.assign(placebo=np.where(rng.random(len(frame)) < share, "예", "아니오"))
        ds = []
        for a, b in pairs:
            o, r = _align(_joint(f, a, b), _cond_indep(f, a, b, ["age_band", "placebo"]))
            ds.append(float(0.5 * np.abs(o - r).sum()))
        cuts.append(1.0 - float(np.mean(ds)) / age)
    return {
        "tvd_cond_age": age,
        "tvd_cond_age_attend": att,
        "attend_cut_pct": 100 * (1 - att / age),
        "placebo_cut_pct_mean": 100 * float(np.mean(cuts)),
        "placebo_cut_pct_max": 100 * float(np.max(cuts)),
        "n_placebo": N_PLACEBO,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2024)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--sample", type=int, default=30000)
    args = ap.parse_args()
    out = settings.outputs_dir / "reports"
    frame = build_official_anchor_frame(args.wave)
    rows, design = split_half(frame, args.sample, args.splits)
    rows.append(prior_wave(args.wave, 2022, args.sample))
    log.info("prior-wave: %s", rows[-1])
    pl = placebo(frame)
    log.info("placebo: %s", pl)
    res = pd.DataFrame(rows)
    res.insert(0, "wave", args.wave)
    res.to_csv(out / "era_chain_crossfit.csv", index=False)
    d = pd.DataFrame(design)
    d.to_csv(out / "era_chain_crossfit_design.csv", index=False)
    pd.DataFrame([{"wave": args.wave, **pl}]).to_csv(
        out / "era_chain_crossfit_placebo.csv", index=False
    )
    sh = res[res.setting == "split-half"]
    dsum = d.groupby(["published_axis", "method"]).abs_err_pp.mean().unstack()
    lines = [
        "# 체인 표본 밖 검증",
        "",
        f"> 자동 생성: `scripts/era_chain_crossfit.py` (wave {args.wave}, 반분 {args.splits} 회, 표본 {args.sample:,}).",
        "",
        "| 설정 | 표본 내 | 표본 밖 | 바닥/기준 | 쌍 |",
        "|---|---|---|---|---|",
        f"| 반분 평균 | {sh.tvd_in_sample.mean():.4f} | {sh.tvd_out_of_sample.mean():.4f} | {sh.tvd_sampling_floor.mean():.4f} (A–B 관측) | {int(sh.n_pairs.iloc[0])} |",
    ]
    pw = res[res.setting.str.startswith("prior")].iloc[0]
    lines.append(
        f"| {pw.setting} | {pw.tvd_in_sample:.4f} (2024 적합) | {pw.tvd_out_of_sample:.4f} | {pw.tvd_sampling_floor:.4f} (2022 관측 그대로) | {int(pw.n_pairs)} |"
    )
    lines += [
        "",
        f"위약: 연령대×관람 여부 절감 {pl['attend_cut_pct']:.1f}% 대 연령대×위약 이진 변수 절감 평균 {pl['placebo_cut_pct_mean']:.1f}% (최대 {pl['placebo_cut_pct_max']:.1f}%, {N_PLACEBO} 회).",
        "",
        "설계 사전검토 (표본 밖, 칸별 |p̂−p| %p 평균):",
        "",
        dsum.round(2).to_markdown(),
    ]
    (out / "era_chain_crossfit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
