"""Era sampler audit — ablation, invariant-enforcement bias, feasibility (ADR-012).

Runs the sampler ablation, feasibility and invariant-enforcement checks on the
**official era frame**; the paper's sampler and appendix numbers come from here.

Ablations (all on the same scale: mean TVD of non-structural anchor pairs)
  released       public chain (same structure as structure_T)
  no_attend      drop attendance from the parents of the attendance block
  age_only       condition every non-root anchor on age band alone
  independent    every anchor from its margin only
  no_genre_tree  the 10 genres without a Chow–Liu tree
  no_count_node  drop the genre-count node and let the tree set the count

Invariant enforcement (margin bias)
  raw_old / whole_old / local_old / local_new  — the four methods compared

Run: uv run python scripts/era_sampler_audit.py --wave 2024 --sample 30000
Outputs: outputs/reports/era_sampler_audit.{md,csv} · era_sampler_{ablation,rejection,feasibility}.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sfac_persona.config import settings
from sfac_persona.grounding import chain as C
from sfac_persona.grounding.error_study import ATTEND_BLOCK, _align, _joint
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import ERA_CORE_ANCHORS, GENRE_ANCHORS, ROOT_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("era_sampler_audit")
for _n in ("sfac_persona.grounding.chain", "sfac_persona.grounding.official"):
    logging.getLogger(_n).setLevel(logging.WARNING)

SEED = 20260911
REPORTS = settings.outputs_dir / "reports"
SAME_CONSTRUCT = {
    "viewing_companion": "leisure_companion",
    "child_cohabiting": "marital",
    "subjective_class": "household_income_band",
}
COUNT_MAX = {"0회": 0, "1~2회": 2, "3~5회": 5, "6~10회": 10, "11회 이상": 10**6}
# Axes on which sampling zeros are counted. Axes absent from era anchors drop out
# automatically; the axes actually used are written as coverage_keys in
# era_sampler_audit.json, which the manuscript cites as-is.
COVERAGE_KEYS = ["age_band", "education", "occupation", "household_income_band", "marital"]

# The public bank is built with era_core_chain() (structure_priors.py). The ablation
# baseline must match — the version that used default_chain() omitted
# household_size_band, so pairs were 184 and could not be compared with the atlas (204).
_ORIGINAL_CHAIN = C.era_core_chain
ABLATIONS = [
    "released",
    "no_attend",
    "age_only",
    "independent",
    "no_genre_tree",
    "no_count_node",
    "no_same_construct",
]


def _non_root(frame: pd.DataFrame) -> list[str]:
    return [a for a in ERA_CORE_ANCHORS if a in frame.columns and a not in ROOT_ANCHORS]


def variant_nodes(kind: str) -> list[C.Node]:
    nodes = deepcopy(_ORIGINAL_CHAIN())
    if kind == "released":
        return nodes
    if kind == "tables_only":
        # Genre count derived from the genres (no table for it is printed); then age-band parents.
        nodes = variant_nodes("no_count_node")
    if kind == "no_count_node":
        out = [nd for nd in nodes if nd.name != "n_genres_band"]
        i = next(k for k, nd in enumerate(out) if nd.name == "__genre_tree__")
        out.insert(i + 1, C.Node("n_genres_band", derived="n_genres_band"))
        return out
    out = []
    for nd in nodes:
        if nd.name in ("root", "__genre_tree__") or nd.derived:
            out.append(nd)
            continue
        if kind == "age_only":
            nd.parents = ["age_band"]
        elif kind == "tables_only":
            # What a builder holding only the printed report has: every anchor conditional on age band;
            # the attendance block conditional on attendance status and age band, which the report's
            # age-band conditionals plus the questionnaire's skip logic determine (the ladder's skip rung).
            nd.parents = (
                ["attend_any", "age_band"]
                if nd.name in ("attend_count_band", "paid_spend_band", "viewing_companion")
                else ["age_band"]
            )
        elif kind == "independent":
            nd.parents = []
        elif kind == "no_attend":
            nd.parents = [p for p in nd.parents if p != "attend_any"] or ["age_band"]
        elif kind == "no_same_construct" and nd.name in SAME_CONSTRUCT:
            nd.parents = [p for p in nd.parents if p != SAME_CONSTRUCT[nd.name]] or ["age_band"]
        out.append(nd)
    return out


def build_variant(
    frame: pd.DataFrame, kind: str, root_source: str = "survey", r0_path: Path | None = None
) -> dict:
    C.default_chain = lambda: variant_nodes(kind)  # type: ignore[assignment]
    try:
        spec = C.build_spec(frame, root_source=root_source, r0_path=r0_path)
    finally:
        C.default_chain = _ORIGINAL_CHAIN  # type: ignore[assignment]
    if kind in ("no_genre_tree", "independent", "age_only", "tables_only"):
        keep = {"independent": [], "age_only": ["age_band"]}.get(kind, ["attend_any", "age_band"])
        for i, node in enumerate(spec["nodes"]):
            if node.get("name") in GENRE_ANCHORS:
                spec["nodes"][i] = C.conditional_table(
                    frame, node["name"], list(keep), settings.cell_floor
                )
        spec["genre_tree"] = []
    return spec


def pair_tvd(
    obs: pd.DataFrame, syn: pd.DataFrame, anchors: list[str]
) -> tuple[float, float, int, int]:
    syn = syn.assign(w=1.0)
    ns, st = [], []
    for a, b in itertools.combinations(anchors, 2):
        if a not in syn.columns or b not in syn.columns:
            continue
        o, s = _align(_joint(obs, a, b), _joint(syn, a, b))
        d = float(0.5 * np.abs(o - s).sum())
        if a in ATTEND_BLOCK and b in ATTEND_BLOCK:
            st.append(d)
        elif "attend_any" in (a, b):
            continue  # same pair set as the ladder (ladder_mask): drop pairs that include a conditioner
        else:
            ns.append(d)
    return (
        float(np.mean(ns)) if ns else float("nan"),
        float(np.mean(st)) if st else float("nan"),
        len(ns),
        len(st),
    )


def _phi(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = (df[cols] == "예").to_numpy(dtype=float)
    ww = df["w"].to_numpy(dtype=float) if "w" in df else np.ones(len(df))
    mean = np.average(x, axis=0, weights=ww)
    xc = x - mean
    cov = (xc * ww[:, None]).T @ xc / ww.sum()
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def genre_phi_mae(obs: pd.DataFrame, syn: pd.DataFrame) -> float:
    cols = [g for g in GENRE_ANCHORS if g in obs.columns and g in syn.columns]
    if len(cols) < 2:
        return float("nan")
    a, b = _phi(obs, cols), _phi(syn.assign(w=1.0), cols)
    iu = np.triu_indices(len(cols), 1)
    return float(np.abs(a[iu] - b[iu]).mean())


def run_ablations(frame: pd.DataFrame, n: int) -> tuple[pd.DataFrame, dict]:
    anchors = _non_root(frame)
    rows, extra = [], {}
    for v in ABLATIONS:
        log.info("ablation %s — spec", v)
        spec = build_variant(frame, v)
        sampler = C.ChainSampler(spec)
        syn = sampler.sample(n, seed=SEED)
        ns, st, n_ns, n_st = pair_tvd(frame, syn, anchors)
        rows.append(
            {
                "variant": v,
                "tvd_nonstructural": ns,
                "tvd_structural": st,
                "genre_phi_mae": genre_phi_mae(frame, syn),
                "n_pairs_nonstructural": n_ns,
                "n_pairs_structural": n_st,
                "rejections": sampler.rejections,
                "genre_fallbacks": sampler.genre_fallbacks,
            }
        )
        log.info("  ns=%.4f st=%.4f phi=%.4f", ns, st, rows[-1]["genre_phi_mae"])
        if v == "released":
            bo = sampler.backoff_report()
            if not bo.empty:
                worst = bo.loc[bo["full_parents_share"].idxmin()]
                extra["backoff_min"] = float(worst["full_parents_share"])
                extra["backoff_min_node"] = str(worst["node"])
                extra["backoff_marginal_max"] = float(bo["marginal_share"].max())
            root_g = next(
                (e["genre"] for e in spec.get("genre_tree", []) if not e.get("parent")), ""
            )
            extra["genre_root"] = root_g.removeprefix("attend_")
            extra["released_syn"] = syn
    df = pd.DataFrame(rows)
    base = df[df.variant == "released"].iloc[0]
    df["delta_nonstructural"] = df.tvd_nonstructural - base.tvd_nonstructural
    return df, extra


def _tvd_series(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return float(0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())


def run_rejection(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """How the four invariant-enforcement methods push the margins."""
    w = frame["w"]
    genres = [g for g in GENRE_ANCHORS if g in frame.columns]
    ng_real = (frame[genres] == "예").sum(axis=1).clip(upper=5)
    ref_ng = w.groupby(ng_real).sum() / w.sum()

    def ref(anchor: str) -> pd.Series:
        s = w.groupby(frame[anchor]).sum()
        return s / s.sum()

    nodes_old = [nd for nd in deepcopy(_ORIGINAL_CHAIN()) if nd.name != "n_genres_band"]
    i = next(k for k, nd in enumerate(nodes_old) if nd.name == "__genre_tree__")
    nodes_old.insert(i + 1, C.Node("n_genres_band", derived="n_genres_band"))
    spec_old = C.build_spec(frame, root_source="survey", nodes=nodes_old)
    spec_new = C.build_spec(frame, root_source="survey")

    def sample_raw(spec: dict, whole_rejection: bool) -> pd.DataFrame:
        s = C.ChainSampler(spec)
        s.enforce_local = False
        rng = np.random.default_rng(SEED)
        rows = []
        while len(rows) < n:
            st = s._sample_once(rng)
            if not whole_rejection or s._consistent(st):
                rows.append(st)
        return pd.DataFrame(rows)

    variants = {
        "raw_old": sample_raw(spec_old, False),
        "whole_old": sample_raw(spec_old, True),
        "local_old": C.ChainSampler(spec_old).sample(n, seed=SEED),
        "local_new": C.ChainSampler(spec_new).sample(n, seed=SEED),
    }
    rows = []
    for name, syn in variants.items():
        any_genre = (syn[genres] == "예").any(axis=1)
        att = syn["attend_any"] == "예"
        ng_syn = (syn[genres] == "예").sum(axis=1).clip(upper=5)
        viol_r1 = float((att != any_genre).mean())
        viol_r6 = float(
            (
                att & (syn["attend_count_band"].map(COUNT_MAX).fillna(10**6) < ng_syn.clip(lower=1))
            ).mean()
        )
        rows.append(
            {
                "variant": name,
                "violation_r1": viol_r1,
                "violation_r6": viol_r6,
                "nonattender_share": float((syn["attend_any"] == "아니오").mean()),
                "tvd_attend_any": _tvd_series(
                    ref("attend_any"), syn["attend_any"].value_counts(normalize=True)
                ),
                "tvd_n_genres": _tvd_series(ref_ng, ng_syn.value_counts(normalize=True)),
                "tvd_attend_count_band": _tvd_series(
                    ref("attend_count_band"), syn["attend_count_band"].value_counts(normalize=True)
                ),
            }
        )
        log.info("rejection %s: r1=%.4f nonatt=%.4f", name, viol_r1, rows[-1]["nonattender_share"])
    return pd.DataFrame(rows)


def run_feasibility(frame: pd.DataFrame, syn: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Hard invariant violations (structural zeros) and sampling zeros (combinations absent from the observed data)."""
    genres = [g for g in GENRE_ANCHORS if g in syn.columns]
    any_genre = (syn[genres] == "예").any(axis=1)
    att = syn["attend_any"] == "예"
    non = syn["attend_any"] == "아니오"
    ng = (syn[genres] == "예").sum(axis=1)
    rules = {
        "R1 attend_any ⇔ 장르": float((att != any_genre).mean()),
        "R2 무관람 ⇒ 0회": float((non & (syn["attend_count_band"] != "0회")).mean()),
        "R3 관람 ⇒ 0회 아님": float((att & (syn["attend_count_band"] == "0회")).mean()),
        "R6 횟수 상한 ≥ 장르 수": float(
            (
                att & (syn["attend_count_band"].map(COUNT_MAX).fillna(10**6) < ng.clip(lower=1))
            ).mean()
        ),
    }
    if "marital" in syn.columns and "child_cohabiting" in syn.columns:
        rules["R4 미혼 ⇒ 동거자녀 없음"] = float(
            ((syn["marital"] == "미혼") & (syn["child_cohabiting"] != "없음")).mean()
        )
    viol = pd.DataFrame([{"rule": k, "violation_share": v} for k, v in rules.items()])
    # combination coverage
    keys = [k for k in COVERAGE_KEYS if k in syn.columns and k in frame.columns]
    real_combos = set(map(tuple, frame[keys].astype(str).to_numpy()))
    syn_combos_all = list(map(tuple, syn[keys].astype(str).to_numpy()))
    syn_combos = set(syn_combos_all)
    new = syn_combos - real_combos
    zero_share = float(np.mean([c in new for c in syn_combos_all]))
    summary = {
        "feasible_share": 100.0 * (1.0 - max(rules.values())),
        "sampling_zero_share": 100.0 * zero_share,
        "n_new_combos": len(new),
        "n_real_combos": len(real_combos),
        "coverage_keys": keys,
    }
    log.info("feasibility: %s", {k: v for k, v in summary.items() if k != "coverage_keys"})
    return viol, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2024)
    ap.add_argument("--sample", type=int, default=30000)
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = C.add_derived_columns(build_official_anchor_frame(args.wave))
    log.info("wave=%d frame n=%d anchors=%d", args.wave, len(frame), len(_non_root(frame)))

    abl, extra = run_ablations(frame, args.sample)
    released_syn = extra.pop("released_syn")
    rej = run_rejection(frame, args.sample)
    viol, feas = run_feasibility(frame, released_syn)

    for df, name in ((abl, "ablation"), (rej, "rejection"), (viol, "feasibility")):
        df.insert(0, "wave", args.wave)
        df.to_csv(REPORTS / f"era_sampler_{name}.csv", index=False)

    meta = {"wave": args.wave, "n_sample": args.sample, "seed": SEED, **extra, **feas}
    (REPORTS / "era_sampler_audit.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def md(df: pd.DataFrame) -> str:
        return (
            df.round(4).to_markdown(index=False)
            if hasattr(df, "to_markdown")
            else df.to_string(index=False)
        )

    lines = [
        "# Era 샘플러 감사 — 절제 · 불변식 · 실행가능성 (ADR-012)",
        "",
        f"wave={args.wave} · n={args.sample:,} · seed={SEED} · 프레임 n={len(frame):,}",
        "",
        "official era 프레임에서 낸 샘플러 절제·실행 가능성·불변식 보정 결과.",
        "",
        "## 절제 (비구조 앵커 쌍 평균 TVD)",
        "",
        md(abl.drop(columns=["wave"])),
        "",
        "`delta_nonstructural` 이 클수록 그 설계 결정이 값을 한 것이다. 0 에 가까우면 근거로 내세우면 안 된다.",
        "",
        (
            f"백오프: 최저 완전부모 비율 {extra.get('backoff_min', float('nan')):.3f}"
            f" (`{extra.get('backoff_min_node', 'n/a')}`) · 주변 fallback 최대"
            f" {extra.get('backoff_marginal_max', float('nan')):.3f}"
            f" · 장르 트리 뿌리 `{extra.get('genre_root', 'n/a')}`"
        ),
        "",
        "## 불변식 강제 방식별 주변분포 편향",
        "",
        md(rej.drop(columns=["wave"])),
        "",
        "## 실행가능성 · 조합 커버리지",
        "",
        md(viol.drop(columns=["wave"])),
        "",
        f"- feasible share: {feas['feasible_share']:.1f}%",
        (
            f"- sampling zero share: {feas['sampling_zero_share']:.1f}%"
            f" ({feas['n_new_combos']} new vs {feas['n_real_combos']} observed;"
            f" keys={feas['coverage_keys']})"
        ),
        "",
        "CSV: `era_sampler_ablation.csv` · `era_sampler_rejection.csv` · `era_sampler_feasibility.csv` · `era_sampler_audit.json`",
        "",
    ]
    (REPORTS / "era_sampler_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote era_sampler_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
