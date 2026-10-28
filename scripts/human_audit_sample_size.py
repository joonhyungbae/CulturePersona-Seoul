"""Set the human-audit sample size from the claims it must support (ADR-023).

"Statistically sufficient" is computable only once **what is being claimed** is
fixed. The audit has to support four claims in the manuscript. The sample each
one requires is computed separately, and the **largest** is taken.

  C1  contradiction rate     Estimate, to ±d, the share of personas whose narrative contradicts the anchors.
                             (Exact Clopper–Pearson interval. Smallest n whose expected half-width is ≤ d)
  C2  rate upper bound       When the true rate is 5%, power ≥ 0.8 to show "rate < 10%" as a 95% upper bound.
  C3  rater agreement        Bootstrap 95% half-width ≤ 0.10 for two raters' Krippendorff α (ordinal),
                             and power ≥ 0.8 that the lower bound exceeds 0.667 (provisional reliability bar) when true α = 0.80.
  C4  trap detection         The rate at which a rater catches personas with a contradiction planted on purpose (sensitivity).
                             Power ≥ 0.8 to show a 95% lower bound ≥ 0.70 when true sensitivity is 0.90 → number of trap items.

C1–C3 set the number of **ordinary personas**; C4 sets the number of **trap personas**. Their sum is what one rater reads.
Every calculation is a seeded simulation or the exact binomial — not an approximation formula.

Run: uv run python scripts/human_audit_sample_size.py
Outputs: outputs/reports/human_audit_sample_size.md · .csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("audit_n")

SEED = 20260923
ALPHA = 0.05
POWER = 0.80
N_SIM = 2000
N_BOOT = 400
MINUTES_PER_PERSONA = (
    2.5  # read ~940 characters across 12 fields + score 4 items (recalibrate on the pilot)
)


# --------------------------------------------------------------------------- binomial


def cp_interval(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def expected_half_width(p: float, n: int) -> float:
    ks = np.arange(n + 1)
    pmf = binom.pmf(ks, n, p)
    widths = np.array([(cp_interval(int(k), n)[1] - cp_interval(int(k), n)[0]) / 2 for k in ks])
    return float((pmf * widths).sum())


def min_n_for_half_width(p: float, d: float, n_max: int = 3000) -> int:
    lo, hi = 10, n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if expected_half_width(p, mid) <= d:
            hi = mid
        else:
            lo = mid + 1
    return lo


def power_upper_below(p_true: float, bound: float, n: int) -> float:
    """Probability that the two-sided 95% CP upper bound is below bound when the true rate is p_true."""
    ks = np.arange(n + 1)
    ok = np.array([cp_interval(int(k), n)[1] < bound for k in ks])
    return float(binom.pmf(ks, n, p_true)[ok].sum())


def power_lower_above(p_true: float, bound: float, n: int) -> float:
    """Probability that the two-sided 95% CP lower bound exceeds bound when the true rate is p_true."""
    ks = np.arange(n + 1)
    ok = np.array([cp_interval(int(k), n)[0] > bound for k in ks])
    return float(binom.pmf(ks, n, p_true)[ok].sum())


def min_n_for_power(fn, n_max: int = 3000) -> int:
    # Binomial power is sawtoothed in n, so take the first n whose next 10 values also clear the bar.
    for n in range(5, n_max):
        if fn(n) >= POWER and all(fn(m) >= POWER for m in range(n, min(n + 10, n_max))):
            return n
    return n_max


# --------------------------------------------------------------------------- Krippendorff α


def kripp_alpha_ordinal(r1: np.ndarray, r2: np.ndarray, k: int = 5) -> float:
    """Two raters, no missing values, ordinal distance. The coincidence-matrix definition as stated."""
    o = np.zeros((k, k))
    for a, b in zip(r1, r2, strict=True):
        o[a, b] += 1
        o[b, a] += 1
    nc = o.sum(axis=1)
    n = nc.sum()
    # Ordinal distance: δ(c,k) = (Σ_{g=c}^{k} n_g − (n_c + n_k)/2)^2
    delta = np.zeros((k, k))
    for c in range(k):
        for kk in range(k):
            lo, hi = min(c, kk), max(c, kk)
            delta[c, kk] = (nc[lo : hi + 1].sum() - (nc[c] + nc[kk]) / 2) ** 2
    do = (o * delta).sum() / n
    de = (np.outer(nc, nc) * delta).sum() / (n * (n - 1))
    return float(1 - do / de) if de > 0 else float("nan")


CUTS = np.array(
    [-1.5, -0.6, 0.3, 1.2]
)  # a realistic distribution piled on the top of the scale (4–5)
_NOISE_CACHE: dict[float, float] = {}


def _draw(
    n: int, noise_sd: float, rng: np.random.Generator, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    t = rng.normal(size=n)
    r1 = np.digitize(t + rng.normal(scale=noise_sd, size=n), CUTS).clip(0, k - 1)
    r2 = np.digitize(t + rng.normal(scale=noise_sd, size=n), CUTS).clip(0, k - 1)
    return r1, r2


def calibrate_noise(target_alpha: float) -> float:
    """Cutting to 5 points makes α lower than the latent reliability. Binary-search the noise that makes the realized α equal the target."""
    if target_alpha in _NOISE_CACHE:
        return _NOISE_CACHE[target_alpha]
    rng = np.random.default_rng(SEED + 1)
    lo, hi = 0.01, 3.0
    for _ in range(30):
        mid = (lo + hi) / 2
        a = kripp_alpha_ordinal(*_draw(40_000, mid, rng))
        if a > target_alpha:
            lo = mid
        else:
            hi = mid
    _NOISE_CACHE[target_alpha] = (lo + hi) / 2
    log.info("noise for alpha=%.2f -> sd=%.4f", target_alpha, _NOISE_CACHE[target_alpha])
    return _NOISE_CACHE[target_alpha]


def simulate_raters(
    n: int, target_alpha: float, rng: np.random.Generator, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Latent quality plus per-rater noise → a 5-point scale. Noise is calibrated so realized α equals the target."""
    return _draw(n, calibrate_noise(target_alpha), rng, k)


def alpha_ci(n: int, true_alpha: float, rng: np.random.Generator) -> tuple[float, float]:
    """(mean half-width, share with lower bound > 0.667). Simulation × bootstrap."""
    widths, passes = [], []
    for _ in range(N_SIM // 10):
        r1, r2 = simulate_raters(n, true_alpha, rng)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            boots.append(kripp_alpha_ordinal(r1[idx], r2[idx]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        widths.append((hi - lo) / 2)
        passes.append(lo > 0.667)
    return float(np.mean(widths)), float(np.mean(passes))


# --------------------------------------------------------------------------- run


def main() -> int:
    rng = np.random.default_rng(SEED)
    rows = []

    # C1 — contradiction rate to ±d. The true rate is unknown, so scan the whole plausible range.
    for p in (0.02, 0.05, 0.10, 0.15):
        for d in (0.03, 0.04):
            n = min_n_for_half_width(p, d)
            rows.append({"claim": "C1 모순률 추정", "assumption": f"p={p:.2f}, ±{d:.2f}", "n": n})
            log.info("C1 p=%.2f d=%.2f -> n=%d", p, d, n)

    # C2 — power to show "contradiction rate < 10%" as a 95% upper bound.
    n_c2 = min_n_for_power(lambda n: power_upper_below(0.05, 0.10, n))
    rows.append({"claim": "C2 모순률 상한 <10%", "assumption": "참 p=0.05, 검정력 0.8", "n": n_c2})
    log.info("C2 -> n=%d", n_c2)

    # C3 — agreement. Half-width ≤ 0.10 and power for "lower bound > 0.667", together.
    c3 = []
    for n in (60, 90, 120, 150, 180, 240, 300):
        hw, pw = alpha_ci(n, 0.80, rng)
        c3.append({"n": n, "half_width": hw, "power_lb_gt_0667": pw})
        log.info("C3 n=%d half-width=%.3f power=%.2f", n, hw, pw)
    c3_df = pd.DataFrame(c3)
    ok = c3_df[(c3_df.half_width <= 0.10) & (c3_df.power_lb_gt_0667 >= POWER)]
    n_c3 = int(ok.n.min()) if len(ok) else int(c3_df.n.max())
    rows.append(
        {
            "claim": "C3 평가자 일치도 α",
            "assumption": "참 α=0.80, 반폭≤0.10 · 하한>0.667 검정력 0.8",
            "n": n_c3,
        }
    )

    # C4 — number of trap items (sensitivity lower bound ≥ 0.70, true 0.90).
    n_c4 = min_n_for_power(lambda n: power_lower_above(0.90, 0.70, n))
    rows.append(
        {"claim": "C4 함정 탐지 민감도", "assumption": "참 0.90, 하한>0.70 검정력 0.8", "n": n_c4}
    )
    log.info("C4 -> n=%d", n_c4)

    res = pd.DataFrame(rows)
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    res.to_csv(reports / "human_audit_sample_size.csv", index=False)
    c3_df.to_csv(reports / "human_audit_alpha_sim.csv", index=False)

    c1_binding = int(
        res[res.claim.str.startswith("C1") & res.assumption.str.contains("p=0.10, ±0.04")].n.iloc[0]
    )
    n_clean = max(c1_binding, n_c2, n_c3)
    n_trap = n_c4
    total = n_clean + n_trap
    hours = total * MINUTES_PER_PERSONA / 60
    lines = [
        "# 인간 검수 표본 크기 (ADR-023)",
        "",
        "각 주장이 요구하는 표본을 따로 구하고 가장 큰 것을 택한다. 정확 이항 · 시드 고정 모의실험.",
        "",
        res.to_markdown(index=False),
        "",
        "## C3 모의실험 (2 평가자, 5 점 서열, 참 α = 0.80, 부트스트랩 400)",
        c3_df.round(3).to_markdown(index=False),
        "",
        "## 결론",
        f"- 정상 페르소나: **{n_clean}** (C1 은 p≤0.10·±0.04 를 구속 조건으로, C2·C3 과 비교해 최댓값)",
        f"- 함정 페르소나: **{n_trap}** (C4)",
        f"- 평가자 1 인당 **{total}** 건, 건당 {MINUTES_PER_PERSONA} 분이면 약 **{hours:.1f} 시간**",
        "",
        "건당 분은 파일럿 10 건으로 다시 잰다. 느리면 C1 의 ±d 를 넓히는 쪽으로 조정하고, 그 선택을 원고에 적는다.",
    ]
    (reports / "human_audit_sample_size.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
