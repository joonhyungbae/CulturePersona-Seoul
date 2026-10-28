"""인간 검수 분석 — 사전등록 v2 그대로 (docs/annotation_prereg.md §6, ADR-023 보강).

**이 파일은 첫 채점 전에 커밋된다.** 판정 기준·통계·제외 규칙은 사전등록 문서와 한 줄씩 대응한다.
결과를 본 뒤 여기를 고치지 않는다. 고쳐야 하면 원래 분석도 함께 내고 사전등록의 변경 기록에 남긴다.

v2 (2026-09-23, 첫 채점 전 — 에이전트 검토 반영):
  - 모든 판정은 **단측 95%** 하나로 (P1 의 CP 상한 = 정확 단측 p, P2 의 부트스트랩 5 백분위).
  - P1 은 합의 기준 + 독립 OR·AND 를 나란히, 함정 민감도로 보정한 상한을 민감도 분석으로.
  - 이진 모순 판정의 일치도(명목 α·κ·양성/음성 일치)를 보고한다 (검정하지 않음 — clean 은 「없음」이
    대부분이라 κ 가 원리적으로 낮게 나온다: 유병률 역설).
  - 함정 v2(국소 편집 48 개, 난이도 T1–T3)로 S1(H₀ 민감도 ≤ 0.5)·S2(평가자별 McNemar).
    통째 교체 6 개(trap_gross)는 조작 확인 — 검정하지 않는다.
  - S4 는 clean 만, 평가자 추측 분포를 보존한 순열 귀무.
  - S5 는 LLM 의 이진 모순 판정: 함정 민감도 비열등(한계 0.10) · clean 특이도 > 0.90.
  - Holm 가족은 8 개로 고정 (S1a/b·S2a/b·S4a/b·S5a/b; 없으면 p=1).

입력 (기본 outputs/anno/, --store 로 바꿀 수 있다)
  annotations/R1.jsonl · R2.jsonl   독립 채점 (문항별 마지막 저장이 유효)
  adjudication.jsonl                합의 판정
  answer_key.csv                    층·함정·검증기·정답 (anno_build_sample.py)
  llm_judge.jsonl                   LLM 채점 (anno_llm_judge.py, 있으면)
  llm_bank.jsonl                    뱅크 전수 LLM 판정 (있으면 PPI 추정)
출력 (--out, 기본 outputs/reports/): anno_results.csv · anno_results.md
실행: uv run python scripts/anno_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from human_audit_sample_size import kripp_alpha_ordinal

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_analysis")

SEED = 20260923
N_BOOT = 20_000  # 판정에 쓰는 부트스트랩·순열 (p≈0.006 근처 몬테카를로 오차를 줄인다)
N_BOOT_DESC = 2_000  # 기술 통계 구간
ALPHA = 0.05  # 단측
SCALES = ("grounded", "coherent", "plausible", "fluent")

# 사전등록 §6 의 임계값. 여기 숫자를 바꾸면 사전등록 위반이다.
P1_BOUND = 0.10
P2_BOUND = 0.667
S1_NULL = 0.50
S5_MARGIN = 0.10
S5_SPEC = 0.90
HOLM_FAMILY = ("S1a", "S1b", "S2a", "S2b", "S4a", "S4b", "S5a", "S5b")
ANNO_WEB = ROOT / "sfac_persona_anno" / "public" / "data"


# --------------------------------------------------------------------------- 불러오기


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def latest(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        out[r["item_id"]] = r
    return out


def load(
    store: Path, key_path: Path
) -> tuple[pd.DataFrame, dict[str, dict], pd.DataFrame | None, pd.DataFrame | None]:
    key = pd.read_csv(key_path, dtype=str).fillna("")
    key = key[key.stratum != "practice"].copy()  # 사전등록 §4: 연습은 전부 제외

    wide = key.set_index("item_id")
    for r in ("R1", "R2"):
        cols = [f"{r}_contra", *(f"{r}_{sc}" for sc in SCALES), f"{r}_age", f"{r}_attend"]
        for c in cols:
            wide[c] = np.nan if not c.endswith(("_age", "_attend")) else None
        ann = latest(read_jsonl(store / "annotations" / f"{r}.jsonl"))
        for iid in wide.index:
            a = ann.get(iid)
            if a is None:
                continue
            s1, s2 = a["stage1"], a["stage2"]
            wide.loc[iid, f"{r}_contra"] = 1 if s2["contradiction"] == "yes" else 0
            for sc in SCALES:
                wide.loc[iid, f"{r}_{sc}"] = int(s2[sc])
            wide.loc[iid, f"{r}_age"] = s1["age_guess"]
            wide.loc[iid, f"{r}_attend"] = s1["attend_guess"]
    adj = latest(read_jsonl(store / "adjudication.jsonl"))
    llm_rows = read_jsonl(store / "llm_judge.jsonl")
    llm = (
        pd.DataFrame(llm_rows).drop_duplicates("item_id", keep="last").set_index("item_id")
        if llm_rows
        else None
    )
    bank_rows = read_jsonl(store / "llm_bank.jsonl")
    bank = pd.DataFrame(bank_rows) if bank_rows else None
    return wide.reset_index(), adj, llm, bank


def consensus(df: pd.DataFrame, adj: dict[str, dict]) -> pd.DataFrame:
    """사전등록 §2: 판정한 문항은 판정값, 아니면 두 평가자의 값(모순)·평균(앵커 부합)."""
    df = df.copy()
    df["cons_contra"] = np.where(df.R1_contra == df.R2_contra, df.R1_contra, np.nan)
    df["cons_grounded"] = (df.R1_grounded + df.R2_grounded) / 2
    for i, row in df.iterrows():
        a = adj.get(row.item_id)
        if a:
            df.loc[i, "cons_contra"] = 1 if a["consensus"]["contradiction"] == "yes" else 0
            df.loc[i, "cons_grounded"] = float(a["consensus"]["grounded"])
    # 판정이 안 된 모순 불일치는 **모순으로 센다** (사전등록 §4). 빼면 모순률이 아래로 치우친다.
    unresolved = df.cons_contra.isna() & df.R1_contra.notna() & df.R2_contra.notna()
    if unresolved.any():
        log.warning("판정되지 않은 모순 불일치 %d 문항 — 모순으로 센다", int(unresolved.sum()))
        df.loc[unresolved, "cons_contra"] = 1
    df["unresolved"] = unresolved
    return df


# --------------------------------------------------------------------------- 통계


def cp_upper(k: int, n: int) -> float:
    """단측 95% Clopper–Pearson 상한. hi < p0 ⇔ 정확 단측 p(H0: p ≥ p0) < 0.05."""
    return 1.0 if k >= n else float(beta.ppf(1 - ALPHA, k + 1, n - k))


def cp_lower(k: int, n: int) -> float:
    """단측 95% Clopper–Pearson 하한."""
    return 0.0 if k <= 0 else float(beta.ppf(ALPHA, k, n - k + 1))


def p_upper(k: int, n: int, p0: float) -> float:
    """H0: p ≥ p0 대 H1: p < p0 — 정확 단측."""
    return float(binom.cdf(k, n, p0))


def p_lower(k: int, n: int, p0: float) -> float:
    """H0: p ≤ p0 대 H1: p > p0 — 정확 단측."""
    return float(binom.sf(k - 1, n, p0))


def mcnemar_greater(x: np.ndarray, y: np.ndarray) -> tuple[float, int, int]:
    """정확 McNemar 단측 (x 가 y 보다 더 잡는다). 반환: p, x 만, y 만."""
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    return (float(binom.sf(b - 1, b + c, 0.5)) if b + c else 1.0), b, c


def strat_boot_alpha(
    a: np.ndarray, b: np.ndarray, strata: np.ndarray, rng: np.random.Generator, n_boot: int
) -> np.ndarray:
    """층을 고정한 부트스트랩 (flagged 비율이 표본마다 흔들리지 않게)."""
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        out[i] = kripp_alpha_ordinal(a[idx] - 1, b[idx] - 1)
    return out


def binary_agreement(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """두 평가자 이진 판정의 명목 α · Cohen κ · 양성/음성 일치 (Cicchetti & Feinstein)."""
    a = int(((x == 1) & (y == 1)).sum())
    d = int(((x == 0) & (y == 0)).sum())
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    n = a + b + c + d
    po = (a + d) / n
    p1, p2 = (a + b) / n, (a + c) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    q = (2 * a + b + c) / (2 * n)  # 합친 양성 비율
    de = 2 * q * (1 - q) * (2 * n) / (2 * n - 1)
    alpha = 1 - ((b + c) / n) / de if de > 0 else float("nan")
    pa = 2 * a / (2 * a + b + c) if (2 * a + b + c) else float("nan")
    na = 2 * d / (2 * d + b + c) if (2 * d + b + c) else float("nan")
    return {"alpha": alpha, "kappa": kappa, "PA": pa, "NA": na, "n": n, "both_yes": a}


def perm_accuracy(guess: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> float:
    """H0: 추측과 정답이 무관 — 평가자 자신의 추측 분포와 정답 분포를 둘 다 보존한 순열 귀무."""
    obs = float(np.mean(guess == truth))
    hits = sum(float(np.mean(rng.permutation(guess) == truth)) >= obs for _ in range(N_BOOT))
    return (1 + hits) / (1 + N_BOOT)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


# --------------------------------------------------------------------------- 분석


def analyse(
    df: pd.DataFrame,
    adj: dict[str, dict],
    llm: pd.DataFrame | None,
    bank: pd.DataFrame | None,
    orders: dict[str, list[str]],
) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    df = consensus(df, adj)
    clean = df[df.stratum == "clean"]
    cf = df[df.stratum.isin(["clean", "flagged"])].dropna(subset=["R1_grounded", "R2_grounded"])
    trap = df[df.stratum == "trap"]
    gross = df[df.stratum == "trap_gross"]

    # ---- 함정 민감도 (P1 보정·S1·S5 가 쓴다): 합의 기준
    tc = trap.dropna(subset=["cons_contra"])
    se_k, se_n = int(tc.cons_contra.sum()), len(tc)
    se_lo = cp_lower(se_k, se_n) if se_n else float("nan")

    # ---- P1 평가자가 찾은 모순률 (clean, 합의)
    c = clean.dropna(subset=["cons_contra"])
    k, n = int(c.cons_contra.sum()), len(c)
    hi = cp_upper(k, n)
    rows.append(
        {
            "id": "P1",
            "kind": "primary",
            "claim": "평가자가 찾은 모순률 < 10% (합의)",
            "estimate": k / n,
            "hi": hi,
            "n": n,
            "p": p_upper(k, n, P1_BOUND),
            "holds": hi < P1_BOUND,
            "note": f"단측 95% CP 상한. 판정 안 된 불일치 {int(c.unresolved.sum())} 문항은 모순으로 셈",
        }
    )
    both = clean.dropna(subset=["R1_contra", "R2_contra"])
    for tag, v in (
        ("OR", ((both.R1_contra + both.R2_contra) > 0)),
        ("AND", ((both.R1_contra + both.R2_contra) == 2)),
    ):
        kk = int(v.sum())
        rows.append(
            {
                "id": f"P1-{tag}",
                "kind": "sensitivity",
                "claim": f"독립 판정 {tag} 모순률",
                "estimate": kk / len(both),
                "hi": cp_upper(kk, len(both)),
                "n": len(both),
            }
        )
    for r in ("R1", "R2"):
        x = clean[f"{r}_contra"].dropna()
        rows.append(
            {
                "id": f"P1-{r}",
                "kind": "sensitivity",
                "claim": f"{r} 독립 모순률",
                "estimate": x.mean(),
                "hi": cp_upper(int(x.sum()), len(x)),
                "n": len(x),
            }
        )
    if se_n and se_lo > 0:
        rows.append(
            {
                "id": "P1-corrected",
                "kind": "sensitivity",
                "claim": "함정 민감도로 보정한 모순률 상한 (hi / Se 하한)",
                "estimate": min(1.0, (k / n) / (se_k / se_n)) if se_k else float("nan"),
                "hi": min(1.0, hi / se_lo),
                "n": n,
                "note": f"Se(합의, 국소 편집 함정) {se_k}/{se_n}, 단측 하한 {se_lo:.3f}. 국소 편집 함정이 자연 오류보다 쉬우면 이 상한도 낙관적이다",
            }
        )

    # ---- P2 평가자 일치도 (독립 앵커 부합, clean+flagged, 층 고정 부트스트랩)
    a1, a2 = cf.R1_grounded.to_numpy(int), cf.R2_grounded.to_numpy(int)
    point = kripp_alpha_ordinal(a1 - 1, a2 - 1)
    bs = strat_boot_alpha(a1, a2, cf.stratum.to_numpy(), rng, N_BOOT)
    lo = float(np.nanpercentile(bs, 100 * ALPHA))
    rows.append(
        {
            "id": "P2",
            "kind": "primary",
            "claim": "평가자 일치도 α(앵커 부합, 서열) > 0.667",
            "estimate": point,
            "lo": lo,
            "n": len(cf),
            "p": float(np.mean(bs <= P2_BOUND)),
            "holds": lo > P2_BOUND,
            "note": "단측 95% 하한 (층 고정 부트스트랩 20,000)",
        }
    )
    cl = cf[cf.stratum == "clean"]
    b_cl = strat_boot_alpha(
        cl.R1_grounded.to_numpy(int),
        cl.R2_grounded.to_numpy(int),
        cl.stratum.to_numpy(),
        rng,
        N_BOOT_DESC,
    )
    rows.append(
        {
            "id": "P2-clean",
            "kind": "descriptive",
            "claim": "α(앵커 부합) clean 만",
            "estimate": kripp_alpha_ordinal(
                cl.R1_grounded.to_numpy(int) - 1, cl.R2_grounded.to_numpy(int) - 1
            ),
            "lo": float(np.nanpercentile(b_cl, 100 * ALPHA)),
            "n": len(cl),
        }
    )
    for sc in SCALES[1:]:
        x = cf.dropna(subset=[f"R1_{sc}", f"R2_{sc}"])
        b = strat_boot_alpha(
            x[f"R1_{sc}"].to_numpy(int),
            x[f"R2_{sc}"].to_numpy(int),
            x.stratum.to_numpy(),
            rng,
            N_BOOT_DESC,
        )
        rows.append(
            {
                "id": f"P2-{sc}",
                "kind": "descriptive",
                "claim": f"평가자 일치도 α({sc})",
                "estimate": kripp_alpha_ordinal(
                    x[f"R1_{sc}"].to_numpy(int) - 1, x[f"R2_{sc}"].to_numpy(int) - 1
                ),
                "lo": float(np.nanpercentile(b, 100 * ALPHA)),
                "n": len(x),
            }
        )

    # ---- 이진 모순 판정의 일치도 (보고, 검정하지 않음)
    bb = cf.dropna(subset=["R1_contra", "R2_contra"])
    ag = binary_agreement(bb.R1_contra.to_numpy(int), bb.R2_contra.to_numpy(int))
    rows.append(
        {
            "id": "D-agree-contra",
            "kind": "descriptive",
            "claim": "이진 모순 판정 일치도 (clean+flagged)",
            "estimate": ag["alpha"],
            "n": ag["n"],
            "note": f"명목 α {ag['alpha']:.3f} · κ {ag['kappa']:.3f} · 양성 일치 {ag['PA']:.3f} · 음성 일치 {ag['NA']:.3f} · 둘 다 모순 {ag['both_yes']}",
        }
    )

    secondary: dict[str, float] = {}

    # ---- S1 함정 민감도 (독립, 국소 편집 함정)
    for r in ("R1", "R2"):
        x = trap[f"{r}_contra"].dropna()
        k, n = int(x.sum()), len(x)
        pid = f"S1{'a' if r == 'R1' else 'b'}"
        secondary[pid] = p_lower(k, n, S1_NULL) if n else 1.0
        rows.append(
            {
                "id": pid,
                "kind": "secondary",
                "claim": f"{r} 함정 민감도 > 0.5 (국소 편집 48)",
                "estimate": k / n if n else float("nan"),
                "lo": cp_lower(k, n) if n else float("nan"),
                "n": n,
                "p": secondary[pid],
            }
        )
        for t, g in trap.groupby("tier"):
            y = g[f"{r}_contra"].dropna()
            rows.append(
                {
                    "id": f"S1-{r}-{t}",
                    "kind": "descriptive",
                    "claim": f"{r} 함정 {t} 탐지율",
                    "estimate": y.mean(),
                    "lo": cp_lower(int(y.sum()), len(y)),
                    "n": len(y),
                }
            )
        y = gross[f"{r}_contra"].dropna()
        rows.append(
            {
                "id": f"M-gross-{r}",
                "kind": "manipulation",
                "claim": f"{r} 통째 교체 함정 탐지 (조작 확인)",
                "estimate": y.mean() if len(y) else float("nan"),
                "n": len(y),
            }
        )

    # ---- S2 평가자별 vs 자동 검증기 (국소 편집 함정, 정확 McNemar 단측)
    valid_all = (trap.validator_checks != "").astype(int)
    for r in ("R1", "R2"):
        t2 = trap.dropna(subset=[f"{r}_contra"])
        h = t2[f"{r}_contra"].to_numpy(int)
        v = valid_all.loc[t2.index].to_numpy(int)
        p, b_, c_ = mcnemar_greater(h, v)
        pid = f"S2{'a' if r == 'R1' else 'b'}"
        secondary[pid] = p
        rows.append(
            {
                "id": pid,
                "kind": "secondary",
                "claim": f"{r} 이 검증기보다 함정을 더 잡는다",
                "estimate": h.mean() - v.mean() if len(h) else float("nan"),
                "n": len(t2),
                "p": p,
                "note": f"{r} 만 {b_} · 검증기만 {c_} · {r} {h.mean():.2f} · 검증기 {v.mean():.2f}"
                if len(h)
                else "",
            }
        )

    # ---- S4 연령대 추측 (clean 만, 순열 귀무) · 관람 추측 (기술)
    for r in ("R1", "R2"):
        x = clean.dropna(subset=[f"{r}_age"])
        x = x[x[f"{r}_age"].astype(str) != ""]
        n = len(x)
        g, t = x[f"{r}_age"].to_numpy(str), x.true_age_band.to_numpy(str)
        pid = f"S4{'a' if r == 'R1' else 'b'}"
        secondary[pid] = perm_accuracy(g, t, rng) if n else 1.0
        mode_rate = x.true_age_band.value_counts(normalize=True).max() if n else float("nan")
        rows.append(
            {
                "id": pid,
                "kind": "secondary",
                "claim": f"{r} 연령대 추측이 우연(순열 귀무)보다 낫다 (clean)",
                "estimate": float(np.mean(g == t)) if n else float("nan"),
                "n": n,
                "p": secondary[pid],
                "note": f"최빈 연령대 비율 {mode_rate:.3f}",
            }
        )
        nt = x[x.true_age_band != "10대"]
        rows.append(
            {
                "id": f"S4-{r}-no10s",
                "kind": "sensitivity",
                "claim": f"{r} 연령대 추측 (10대 제외)",
                "estimate": float(np.mean(nt[f"{r}_age"] == nt.true_age_band))
                if len(nt)
                else float("nan"),
                "n": len(nt),
            }
        )
        k = int((x[f"{r}_attend"] == x.true_attend).sum())
        rows.append(
            {
                "id": f"G-attend-{r}",
                "kind": "descriptive",
                "claim": f"{r} 관람 여부 추측 (clean)",
                "estimate": k / n if n else float("nan"),
                "n": n,
            }
        )

    # ---- S5 LLM 이진 모순 판정 (있을 때만)
    if llm is not None and "contradiction" in llm.columns:
        lj = (llm.contradiction == "yes").astype(int)
        tt = trap.dropna(subset=["R1_contra", "R2_contra"])
        tt = tt[tt.item_id.isin(lj.index)]
        L = lj.loc[tt.item_id].to_numpy(int)
        H = ((tt.R1_contra.to_numpy(int) + tt.R2_contra.to_numpy(int)) / 2).astype(float)
        m = len(tt)
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, m, m)
            diffs[i] = L[idx].mean() - H[idx].mean()
        secondary["S5a"] = float(np.mean(diffs <= -S5_MARGIN))
        rows.append(
            {
                "id": "S5a",
                "kind": "secondary",
                "claim": "LLM 함정 민감도가 사람(평균)에 비열등 (한계 0.10)",
                "estimate": L.mean() - H.mean(),
                "lo": float(np.percentile(diffs, 100 * ALPHA)),
                "n": m,
                "p": secondary["S5a"],
                "note": f"LLM {L.mean():.3f} · 사람 평균 {H.mean():.3f}",
            }
        )
        neg = clean[(clean.cons_contra == 0) & clean.item_id.isin(lj.index)]
        kk = int((lj.loc[neg.item_id] == 0).sum())
        secondary["S5b"] = p_lower(kk, len(neg), S5_SPEC) if len(neg) else 1.0
        rows.append(
            {
                "id": "S5b",
                "kind": "secondary",
                "claim": "LLM 특이도 > 0.90 (clean, 합의 모순 없음)",
                "estimate": kk / len(neg) if len(neg) else float("nan"),
                "lo": cp_lower(kk, len(neg)) if len(neg) else float("nan"),
                "n": len(neg),
                "p": secondary["S5b"],
            }
        )
        n_fail = int(len(df) - len(lj.index.intersection(df.item_id)))
        rows.append(
            {
                "id": "D-llm-missing",
                "kind": "descriptive",
                "claim": "LLM 판정을 못 받은 문항",
                "estimate": n_fail,
            }
        )
        # PPI: 뱅크 전수 LLM 판정 + clean 에서 사람(합의) − LLM 보정
        if bank is not None and "contradiction" in bank.columns:
            lb = (bank.contradiction == "yes").astype(float).to_numpy()
            cc = clean.dropna(subset=["cons_contra"])
            cc = cc[cc.item_id.isin(lj.index)]
            rect = cc.cons_contra.to_numpy(float) - lj.loc[cc.item_id].to_numpy(float)
            theta = lb.mean() + rect.mean()
            se = float(np.sqrt(lb.var(ddof=1) / len(lb) + rect.var(ddof=1) / len(rect)))
            rows.append(
                {
                    "id": "D-ppi",
                    "kind": "descriptive",
                    "claim": "뱅크 전수 모순률 (PPI: LLM 전수 + 사람 보정)",
                    "estimate": theta,
                    "hi": theta + 1.645 * se,
                    "n": len(lb),
                    "note": f"보정 표본 {len(rect)} · LLM 원 추정 {lb.mean():.4f}",
                }
            )
    else:
        rows.append(
            {
                "id": "S5",
                "kind": "secondary",
                "claim": "LLM 판정",
                "note": "LLM 채점 없음 — p=1 로 가족에 넣음",
            }
        )

    # ---- Holm (가족 8 로 고정 — 빠진 검정은 p=1)
    fam = {k: secondary.get(k, 1.0) for k in HOLM_FAMILY}
    adj_p = holm(fam)
    for r in rows:
        if r["id"] in adj_p:
            r["p_holm"] = adj_p[r["id"]]
            r["holds"] = adj_p[r["id"]] < ALPHA

    # ---- 기술 통계
    fl = df[df.stratum == "flagged"]
    for chk in sorted({c for v in fl.validator_checks for c in str(v).split(";") if c}):
        g = fl[fl.validator_checks.str.contains(chk)]
        if chk == "han_characters":
            v = g.dropna(subset=["R1_fluent", "R2_fluent"])
            est = float((((v.R1_fluent + v.R2_fluent) / 2) <= 2).mean()) if len(v) else float("nan")
            claim = "검증기 정밀도 han_characters (자연스러움 ≤2 비율)"
            n_ = len(v)
        else:
            v = g.dropna(subset=["cons_contra"])
            est = float(v.cons_contra.mean()) if len(v) else float("nan")
            claim = f"검증기 정밀도 {chk} (합의 모순 비율)"
            n_ = len(v)
        rows.append(
            {
                "id": f"D-precision-{chk}",
                "kind": "descriptive",
                "claim": claim,
                "estimate": est,
                "n": n_,
            }
        )
    k = int((trap.validator_checks != "").sum())
    rows.append(
        {
            "id": "D-recall",
            "kind": "descriptive",
            "claim": "검증기 재현율 (국소 편집 함정, 같은 검증기로 다시 채점)",
            "estimate": k / len(trap) if len(trap) else float("nan"),
            "n": len(trap),
        }
    )
    for r in ("R1", "R2"):
        x = clean[f"{r}_contra"].dropna()
        rows.append(
            {
                "id": f"neg-{r}",
                "kind": "descriptive",
                "claim": f"{r} 음성 판정률 (clean 에서 모순 없음 — clean 에도 모순이 있을 수 있어 특이도가 아니다)",
                "estimate": 1 - x.mean(),
                "n": len(x),
            }
        )
    # 위치(피로): 본 문항 순번 사분위별 모순률 (순서가 무작위라 편향이 아니라 잡음 — 보고만)
    for r in ("R1", "R2"):
        order = [i for i in orders.get(r, []) if i in set(clean.item_id)]
        if not order:
            continue
        pos = {iid: i for i, iid in enumerate(order)}
        x = clean.dropna(subset=[f"{r}_contra"]).copy()
        x["q"] = pd.qcut(x.item_id.map(pos), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        rates = x.groupby("q", observed=True)[f"{r}_contra"].mean().round(3).to_dict()
        rows.append(
            {
                "id": f"D-position-{r}",
                "kind": "descriptive",
                "claim": f"{r} 순번 사분위별 모순률 (clean)",
                "note": str(rates),
            }
        )
    for sc in SCALES:
        v = ((clean[f"R1_{sc}"] + clean[f"R2_{sc}"]) / 2).dropna().to_numpy()
        if not len(v):
            continue
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT_DESC)]
        rows.append(
            {
                "id": f"mean-{sc}",
                "kind": "descriptive",
                "claim": f"{sc} 평균 (두 평가자 평균, clean)",
                "estimate": v.mean(),
                "lo": float(np.percentile(bs, 100 * ALPHA)),
                "hi": float(np.percentile(bs, 100 * (1 - ALPHA))),
                "n": len(v),
            }
        )
    changed = (
        pd.Series([a.get("changed_by") for a in adj.values()]).value_counts().to_dict()
        if adj
        else {}
    )
    rows.append(
        {
            "id": "adjudication",
            "kind": "descriptive",
            "claim": "판정한 불일치 문항",
            "estimate": len(adj),
            "note": f"누가 바꿨나 {changed} (none = 합의 불가, 엄격한 쪽으로 기록)",
        }
    )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=settings.outputs_dir / "anno")
    ap.add_argument("--key", type=Path, default=settings.outputs_dir / "anno" / "answer_key.csv")
    ap.add_argument("--out", type=Path, default=settings.outputs_dir / "reports")
    args = ap.parse_args()

    df, adj, llm, bank = load(args.store, args.key)
    missing = {r: int(df[f"{r}_contra"].isna().sum()) for r in ("R1", "R2")}
    log.info("문항 %d (연습 제외) · 끝내지 못한 문항 %s", len(df), missing)
    orders = {}
    for r in ("R1", "R2"):
        p = ANNO_WEB / f"order_{r}.json"
        if p.exists():
            orders[r] = json.loads(p.read_text(encoding="utf-8"))
    rows = analyse(df, adj, llm, bank, orders)

    res = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out / "anno_results.csv", index=False)
    show = res.copy()
    for c in ("estimate", "lo", "hi", "p", "p_holm"):
        if c in show:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    lines = [
        "# 인간 검수 결과 (사전등록 v2 분석)",
        "",
        f"사전등록: docs/annotation_prereg.md §6 · 끝내지 못한 문항 {missing}",
        "",
        show.fillna("").to_markdown(index=False),
        "",
        "주 주장(P1, P2)은 둘 다 성립해야 「검수 통과」다 (각각 단측 α=0.05). 부 주장 8 개는 Holm 보정 p 로 판정한다.",
    ]
    (args.out / "anno_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
