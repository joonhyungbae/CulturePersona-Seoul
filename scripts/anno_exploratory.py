"""Exploratory human-audit analysis (outside the preregistration, ADR-039). Registered decisions
are fixed by anno_analysis.py; this file only aids interpretation.

Uses independent scores only (each rater's marks before consensus). Item text and person-level
rows are not written out. Every estimate gets a 95% interval, and tests that emit a p-value are
one exploratory family under Holm. Because this is exploratory, use the results only as grounds
for interpretation, not as discoveries.

  A. Agreement — how sensitive the indices are to score pile-up (ceiling)
     Four scales including groundedness: exact agreement, ±1 agreement, quadratically weighted
     agreement, Gwet AC2 (quadratic), Brennan–Prediger (quadratic), Krippendorff α (ordinal,
     the registered index). Binary contradiction: agreement, Cohen κ, Gwet AC1, positive/negative
     agreement. Intervals are stratum-fixed bootstraps (clean, flagged).
  B. Detection — 36 planted contradictions
     Detection rates for each rater, the judge, either human, and human∪judge (exact
     Clopper–Pearson 95%), detection that also names the right field, exact McNemar of rater
     vs judge (paired), and detection counts by trap type.
  C. Separation by score — AUC of groundedness (lower means trap) separating traps from clean,
     and a paired bootstrap of the judge − rater difference.
  D. Task load — trend of per-item time against the order the rater actually worked (saved_at)
     (Kendall τ, permutation test), medians of the first and last quarter of that order, and
     trends of trap detection and clean-item contradiction marks against order (order AUC,
     permutation test). Caution: time and detection are not causal — finding a contradiction
     takes longer because the evidence sentence must be written. Time→detection is therefore
     not tested.
  E. On clean items, did the judge also mark contradiction where a human did (aggregates only).

Run: uv run python scripts/anno_exploratory.py [--boot 5000]
Outputs: outputs/reports/anno_exploratory.csv (key,value for manuscript macros) ·
anno_exploratory_tests.csv · anno_exploratory.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest, kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from human_audit_sample_size import (
    kripp_alpha_ordinal,
)  # same implementation as the registered analysis

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_exploratory")

RATERS = ("R1", "R2")
SCALES = ("grounded", "coherent", "plausible", "fluent")
K = 5
SECS_CAP = 1800  # items left idle (over 30 minutes) are capped at 30 minutes
SEED = 20261007
N_PERM = 20_000


# --------------------------------------------------------------------------- read


def latest(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                a = json.loads(line)
                out[a["item_id"]] = a
    return out


# --------------------------------------------------------------------------- agreement


def _w(k: int = K) -> np.ndarray:
    i = np.arange(k)
    return 1 - ((i[:, None] - i[None, :]) / (k - 1)) ** 2


def weighted_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(_w()[a, b].mean())


def gwet_ac2(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet (2014) AC2, quadratic weights, two raters, no missing values. a, b are 0..K-1."""
    w = _w()
    pa = w[a, b].mean()
    pi = (np.bincount(a, minlength=K) + np.bincount(b, minlength=K)) / (2 * len(a))
    pe = w.sum() / (K * (K - 1)) * float((pi * (1 - pi)).sum())
    return float((pa - pe) / (1 - pe))


def brennan_prediger(a: np.ndarray, b: np.ndarray) -> float:
    """Weighted agreement with chance agreement under a uniform distribution (quadratic)."""
    w = _w()
    pe = w.sum() / K**2
    return float((w[a, b].mean() - pe) / (1 - pe))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe)


def pos_neg_agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    both, neither = int((a & b).sum()), int((~a & ~b).sum())
    pa = 2 * both / (a.sum() + b.sum()) if (a.sum() + b.sum()) else float("nan")
    na = 2 * neither / ((~a).sum() + (~b).sum())
    return float(pa), float(na)


def strat_boot(strata: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Index lists drawn with replacement inside each stratum."""
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    return [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(n_boot)]


def ci(xs: list[float]) -> tuple[float, float]:
    v = np.asarray([x for x in xs if np.isfinite(x)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# --------------------------------------------------------------------------- detection


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar for paired binary outcomes (two-sided). (x only, y only, p)."""
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, float(p)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(trap score < clean score) + ½P(tie). Oriented so a lower score means trap."""
    diff = pos[:, None] - neg[None, :]
    return float(((diff < 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# --------------------------------------------------------------------------- run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=settings.outputs_dir / "anno")
    ap.add_argument("--reports", type=Path, default=settings.outputs_dir / "reports")
    ap.add_argument(
        "--items",
        type=Path,
        default=ROOT / "annotation" / "data_private" / "items_full.json",
    )
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    key = {
        r["item_id"]: r
        for r in csv.DictReader((args.store / "answer_key.csv").open(encoding="utf-8"))
    }
    subset_path = args.store / "subset_v2.json"
    sub = (
        set(json.loads(subset_path.read_text(encoding="utf-8"))["ids"])
        if subset_path.exists()
        else {i for i, r in key.items() if r["stratum"] != "practice"}
    )
    ann = {r: latest(args.store / "annotations" / f"{r}.jsonl") for r in RATERS}
    judge = latest(args.store / "llm_judge.jsonl")
    for r in RATERS:
        if sub - set(ann[r]):
            raise SystemExit(
                f"{r}: {len(sub - set(ann[r]))} unfinished items — run after scoring is complete"
            )
    if sub - set(judge):
        raise SystemExit(f"items with no judge result: {len(sub - set(judge))}")

    def yes(r: str, i: str) -> bool:
        return ann[r][i]["stage2"]["contradiction"] == "yes"

    def jyes(i: str) -> bool:
        return judge[i]["contradiction"] == "yes"

    macros: dict[str, float | int] = {}
    tests: list[dict] = []  # exploratory family — Holm applies to rows that have a p

    def row(
        group: str,
        name: str,
        est: float,
        lo: float | None = None,
        hi: float | None = None,
        n: int | None = None,
        p: float | None = None,
        note: str = "",
    ) -> None:
        tests.append(
            {
                "group": group,
                "name": name,
                "estimate": est,
                "lo": lo,
                "hi": hi,
                "n": n,
                "p": p,
                "note": note,
            }
        )

    # ---- A. agreement (clean + flagged)
    cf = sorted(i for i in sub if key[i]["stratum"] in ("clean", "flagged"))
    strata = np.array([key[i]["stratum"] for i in cf])
    boots = strat_boot(strata, args.boot, rng)
    for sc in SCALES:
        a = np.array([int(ann["R1"][i]["stage2"][sc]) - 1 for i in cf])
        b = np.array([int(ann["R2"][i]["stage2"][sc]) - 1 for i in cf])
        top = (float((a == K - 1).mean()), float((b == K - 1).mean()))
        metrics = {
            "정확 일치율": lambda x, y: float((x == y).mean()),
            "±1 일치율": lambda x, y: float((abs(x - y) <= 1).mean()),
            "이차 가중 일치율": weighted_agreement,
            "Gwet AC2 (이차)": gwet_ac2,
            "Brennan–Prediger (이차)": brennan_prediger,
            "Krippendorff α (서열, 등록 지표)": kripp_alpha_ordinal,
        }
        for mname, f in metrics.items():
            est = f(a, b)
            lo, hi = ci([f(a[ix], b[ix]) for ix in boots])
            row(
                "A 일치도",
                f"{sc} · {mname}",
                est,
                lo,
                hi,
                len(cf),
                note=f"만점 비율 R1 {top[0]:.2f} · R2 {top[1]:.2f}",
            )
        if sc == "grounded":
            macros.update(
                {
                    "g_top_R1": round(top[0] * 100),
                    "g_top_R2": round(top[1] * 100),
                    "g_ac2": round(gwet_ac2(a, b), 2),
                    "g_wagree": round(weighted_agreement(a, b), 2),
                    "g_exact": round(float((a == b).mean()) * 100),
                    "g_alpha": round(kripp_alpha_ordinal(a, b), 2),
                }
            )
            lo, hi = ci([gwet_ac2(a[ix], b[ix]) for ix in boots])
            macros.update({"g_ac2_lo": round(lo, 2), "g_ac2_hi": round(hi, 2)})
    ya = np.array([yes("R1", i) for i in cf])
    yb = np.array([yes("R2", i) for i in cf])
    for mname, f in {
        "일치율": lambda x, y: float((x == y).mean()),
        "Cohen κ": cohen_kappa,
        "Gwet AC1": gwet_ac1,
        "양성 일치": lambda x, y: pos_neg_agreement(x, y)[0],
        "음성 일치": lambda x, y: pos_neg_agreement(x, y)[1],
    }.items():
        est = f(ya, yb)
        lo, hi = ci([f(ya[ix], yb[ix]) for ix in boots])
        row(
            "A 일치도",
            f"모순 여부 · {mname}",
            est,
            lo,
            hi,
            len(cf),
            note=f"모순 표시 R1 {int(ya.sum())} · R2 {int(yb.sum())} · 둘 다 {int((ya & yb).sum())}",
        )
    macros.update(
        {
            "c_ac1": round(gwet_ac1(ya, yb), 2),
            "c_kappa": round(cohen_kappa(ya, yb), 2),
            "c_agree": round(float((ya == yb).mean()) * 100),
            "c_both": int((ya & yb).sum()),
        }
    )

    # ---- B. detection (planted contradictions)
    traps = sorted(i for i in sub if key[i]["stratum"] == "trap")
    n_t = len(traps)
    det = {r: np.array([yes(r, i) for i in traps]) for r in RATERS}
    det["judge"] = np.array([jyes(i) for i in traps])
    det["human_any"] = det["R1"] | det["R2"]
    det["union"] = det["human_any"] | det["judge"]
    field_ok = {
        r: np.array(
            [
                yes(r, i) and key[i]["trap_field"] in ann[r][i]["stage2"]["contra_fields"]
                for i in traps
            ]
        )
        for r in RATERS
    }
    field_ok["judge"] = np.array(
        [jyes(i) and key[i]["trap_field"] in judge[i].get("contra_fields", []) for i in traps]
    )
    names = {
        "R1": "평가자 1",
        "R2": "평가자 2",
        "judge": "판정기",
        "human_any": "평가자 한 명이라도",
        "union": "평가자∪판정기",
    }
    for k_, lab in names.items():
        k = int(det[k_].sum())
        lo, hi = cp(k, n_t)
        row("B 탐지", f"{lab} 탐지율", k / n_t, lo, hi, n_t, note=f"{k}/{n_t}")
        if k_ in field_ok:
            kf = int(field_ok[k_].sum())
            lo, hi = cp(kf, n_t)
            row(
                "B 탐지", f"{lab} 탐지율 (맞는 문단까지)", kf / n_t, lo, hi, n_t, note=f"{kf}/{n_t}"
            )
    for x, y in (("R1", "judge"), ("R2", "judge"), ("human_any", "judge")):
        b_, c_, p = mcnemar_exact(det[x], det[y])
        row(
            "B 탐지",
            f"{names[x]} 대 {names[y]} (정확 McNemar)",
            (det[x].sum() - det[y].sum()) / n_t,
            n=n_t,
            p=p,
            note=f"{names[x]}만 {b_} · {names[y]}만 {c_}",
        )
    macros.update(
        {
            "n_traps": n_t,
            "trap_human_any": int(det["human_any"].sum()),
            "trap_judge": int(det["judge"].sum()),
            "trap_union": int(det["union"].sum()),
            "trap_human_only": int((det["human_any"] & ~det["judge"]).sum()),
            "trap_judge_only": int((det["judge"] & ~det["human_any"]).sum()),
            "trap_R1_field": int(field_ok["R1"].sum()),
            "trap_R2_field": int(field_ok["R2"].sum()),
            "trap_judge_field": int(field_ok["judge"].sum()),
        }
    )
    _, _, p_hj = mcnemar_exact(det["human_any"], det["judge"])
    macros["mcnemar_human_judge_p"] = round(p_hj, 2)
    lo, hi = cp(int(det["union"].sum()), n_t)
    macros.update({"union_lo": round(lo * 100), "union_hi": round(hi * 100)})
    by_type: dict[str, list[int]] = defaultdict(list)
    for k, i in enumerate(traps):
        by_type[key[i]["trap_type"]].append(k)
    for t, ix in sorted(by_type.items()):
        h, j = int(det["human_any"][ix].sum()), int(det["judge"][ix].sum())
        row(
            "B 탐지",
            f"종류 {t}",
            h / len(ix),
            n=len(ix),
            note=f"평가자 {h} · 판정기 {j} / {len(ix)}",
        )
        macros.update({f"type_{t}_n": len(ix), f"type_{t}_human": h, f"type_{t}_judge": j})

    # ---- C. separation by score (groundedness, trap vs clean)
    clean = sorted(i for i in sub if key[i]["stratum"] == "clean")
    score = {
        r: (
            np.array([int(ann[r][i]["stage2"]["grounded"]) for i in traps]),
            np.array([int(ann[r][i]["stage2"]["grounded"]) for i in clean]),
        )
        for r in RATERS
    }
    score["judge"] = (
        np.array([int(judge[i]["grounded"]) for i in traps]),
        np.array([int(judge[i]["grounded"]) for i in clean]),
    )
    bt = [
        (rng.integers(0, n_t, n_t), rng.integers(0, len(clean), len(clean)))
        for _ in range(args.boot)
    ]
    aucs = {}
    for k_, (pt, pc) in score.items():
        aucs[k_] = [auc(pt[i], pc[j]) for i, j in bt]
        lo, hi = ci(aucs[k_])
        row("C 점수로 가려내기", f"{names[k_]} 근거성 AUC", auc(pt, pc), lo, hi, n_t + len(clean))
        macros[f"auc_{k_}"] = round(auc(pt, pc), 2)
    for r in RATERS:
        d = np.asarray(aucs["judge"]) - np.asarray(aucs[r])
        p = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
        lo, hi = ci(d.tolist())
        row(
            "C 점수로 가려내기",
            f"판정기 − {names[r]} AUC (짝 부트스트랩)",
            auc(*score["judge"]) - auc(*score[r]),
            lo,
            hi,
            n_t + len(clean),
            p=max(p, 1 / args.boot),
        )

    # ---- D. task load (the order the rater actually worked)
    for r in RATERS:
        done = sorted(sub, key=lambda i: ann[r][i]["saved_at"])
        pos = np.arange(len(done))
        s1 = np.array([min(int(ann[r][i]["stage1"].get("secs") or 0), SECS_CAP) for i in done])
        s2 = np.array([min(int(ann[r][i]["stage2"].get("secs") or 0), SECS_CAP) for i in done])
        tot = s1 + s2
        q = len(done) // 4
        macros[f"{r}_total_secs_first_q"] = int(np.median(tot[:q]))
        macros[f"{r}_total_secs_last_q"] = int(np.median(tot[-q:]))
        macros[f"{r}_stage2_secs_first_q"] = int(np.median(s2[:q]))
        macros[f"{r}_stage2_secs_last_q"] = int(np.median(s2[-q:]))
        tau = kendalltau(pos, tot).statistic
        perm = np.array([kendalltau(pos, rng.permutation(tot)).statistic for _ in range(2000)])
        p = float(max((np.abs(perm) >= abs(tau)).mean(), 1 / 2000))
        row(
            "D 과제 부하",
            f"{names[r]} 순서 대 문항당 시간 (Kendall τ, 순열)",
            float(tau),
            n=len(done),
            p=p,
            note=f"앞 1/4 {int(np.median(tot[:q]))}초 → 뒤 1/4 {int(np.median(tot[-q:]))}초",
        )
        macros[f"{r}_tau"] = round(float(tau), 2)
        rank = {i: k for k, i in enumerate(done)}
        for lab, ids in (("함정 탐지", traps), ("정상 문항 모순 표시", clean)):
            pp = np.array([rank[i] for i in ids], dtype=float) / len(done)
            fl = np.array([yes(r, i) for i in ids])
            if fl.all() or not fl.any():
                continue
            a_ = auc(pp[fl], pp[~fl])  # P(a marked item is earlier in order); 0.5 = no trend
            perms = (rng.permutation(fl) for _ in range(N_PERM // 4))
            perm = np.array([auc(pp[z], pp[~z]) for z in perms])
            p = float(max((np.abs(perm - 0.5) >= abs(a_ - 0.5)).mean(), 4 / N_PERM))
            row(
                "D 과제 부하",
                f"{names[r]} 순서 대 {lab} (앞쪽일 확률, 순열)",
                a_,
                n=len(ids),
                p=p,
                note=f"표시 {int(fl.sum())}/{len(ids)} · 0.5 초과면 앞쪽에 몰림",
            )
    items = {i["item_id"]: i for i in json.loads(args.items.read_text(encoding="utf-8"))}
    macros["narr_chars_median"] = int(
        statistics.median(sum(len(v) for v in items[i]["narrative"].values()) for i in sub)
    )

    # ---- E. clean-item flags and the judge
    flags = [i for i in clean if yes("R1", i) or yes("R2", i)]
    macros.update(
        {
            "clean_human_flags": len(flags),
            "clean_human_flags_judge_yes": sum(jyes(i) for i in flags),
            "clean_judge_flags": sum(jyes(i) for i in clean),
            "n_clean": len(clean),
        }
    )

    # ---- Holm (exploratory family)
    pidx = [k for k, t in enumerate(tests) if t["p"] is not None]
    for k, ph in zip(pidx, holm([tests[k]["p"] for k in pidx]), strict=True):
        tests[k]["p_holm"] = ph

    args.reports.mkdir(parents=True, exist_ok=True)
    with (args.reports / "anno_exploratory.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in macros.items():
            w.writerow([k, v])
    cols = ["group", "name", "estimate", "lo", "hi", "n", "p", "p_holm", "note"]
    with (args.reports / "anno_exploratory_tests.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in tests:
            w.writerow({c: t.get(c) for c in cols})

    def fmt(v: object, d: int = 3) -> str:
        return (
            ""
            if v is None or (isinstance(v, float) and not np.isfinite(v))
            else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
        )

    lines = [
        "# 인간 검수 탐색 분석 (사전등록 밖)",
        "",
        "독립 채점만 사용. 구간은 95% (A·C 부트스트랩, B 정확 Clopper–Pearson). p 는 탐색 족 전체 Holm 보정을 함께 적는다.",
        "등록한 판정을 바꾸지 않는다. D 의 시간은 탐지와 인과 관계가 아니다 (모순을 찾으면 근거를 적느라 시간이 는다).",
        "",
    ]
    for g in dict.fromkeys(t["group"] for t in tests):
        lines += [
            f"## {g}",
            "",
            "| 항목 | 추정 | 95% 구간 | n | p | Holm p | 비고 |",
            "|---|---|---|---|---|---|---|",
        ]
        for t in (t for t in tests if t["group"] == g):
            span = f"[{fmt(t['lo'])}, {fmt(t['hi'])}]" if t["lo"] is not None else ""
            lines.append(
                f"| {t['name']} | {fmt(t['estimate'])} | {span} | {fmt(t['n'])} | {fmt(t['p'])} | {fmt(t.get('p_holm'))} | {t['note']} |"
            )
        lines.append("")
    (args.reports / "anno_exploratory.md").write_text("\n".join(lines), encoding="utf-8")
    log.info(
        "wrote anno_exploratory.csv · anno_exploratory_tests.csv · anno_exploratory.md "
        "(tests %d, Holm family %d)",
        len(tests),
        len(pidx),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
