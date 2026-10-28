"""Statistical power of the human-validation design (ADR-023).

Under the constraint of 2 validators and about 6.5 hours each, compute **what
can actually be distinguished**. Item counts in the plan have to come from this
calculation, not from a guess that "about this many should do".

Four outputs.

  S1 precision     Standard error of the mean item score (item variance + rater noise / number of raters)
  S1 negative control  Power to catch a deliberately broken narrative on groundedness
  S1 agreement     Bootstrap interval width of Krippendorff α for 2 raters · n items (Monte Carlo)
  S2 forced choice Minimum effect at which accuracy can be said to differ from chance (50%)
  S3 AUC           Standard error of the human upper bound — distinguishable from logistic 0.77 and upper bound 0.83?

Every assumption is named. If an assumption is wrong the conclusion is wrong, so after scoring, rerun on the observed values and report how far the plan was.

Run: uv run python scripts/human_validation_power.py
Output: a markdown report of the power calculation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("human_power")

SEED = 20260921
Z95 = 1.959964
Z80 = 0.8416212  # one-sided z for 80% power

# --- assumptions (5-point rubric) ---------------------------------------------
# How far people disagree (rater noise) and how far personas truly differ (item variance).
# Conservative values taken from typical 5-point rubric reports.
SIGMA_ITEM = 0.60  # SD of the true value across items (personas)
SIGMA_RATER = 0.50  # SD of disagreement when two people score the same item
N_RATERS = 2

# Minutes per item — set without a pilot measurement. Update from the calibration round.
MIN_PER_ITEM = {"S1": 2.5, "S2": 0.33, "S3": 0.75}
BUDGET_MIN = 390  # 6.5 hours per person


def mean_se(n_items: int, k: int = N_RATERS) -> float:
    """Standard error of the item mean. Var(item mean) = σ_item² + σ_rater²/k."""
    return float(np.sqrt(SIGMA_ITEM**2 + SIGMA_RATER**2 / k) / np.sqrt(n_items))


def control_power(n_control: int, n_normal: int, true_gap: float) -> float:
    """Power to detect a negative control scoring below ordinary items (two-sided t, α=.05)."""
    v = SIGMA_ITEM**2 + SIGMA_RATER**2 / N_RATERS
    se = np.sqrt(v / n_control + v / n_normal)
    ncp = true_gap / se
    return float(stats.norm.cdf(ncp - Z95) + stats.norm.cdf(-ncp - Z95))


def krippendorff_ordinal(matrix: np.ndarray, n_levels: int = 5) -> float:
    """Ordinal Krippendorff α. matrix: (coders, items), assuming no missing values."""
    coders, items = matrix.shape
    values = np.arange(1, n_levels + 1)
    counts = np.zeros(n_levels)
    for lv in range(n_levels):
        counts[lv] = (matrix == values[lv]).sum()

    def delta(a: int, b: int) -> float:
        lo, hi = (min(a, b), max(a, b))
        seg = counts[lo : hi + 1].copy()
        s = seg.sum() - (counts[lo] + counts[hi]) / 2.0
        return float(s**2)

    do = 0.0
    n_pairs = 0
    for j in range(items):
        col = matrix[:, j]
        for x in range(coders):
            for y in range(x + 1, coders):
                do += delta(int(col[x]) - 1, int(col[y]) - 1)
                n_pairs += 1
    do /= max(n_pairs, 1)

    de = 0.0
    tot_pairs = 0.0
    for a in range(n_levels):
        for b in range(n_levels):
            if counts[a] == 0 or counts[b] == 0:
                continue
            w = counts[a] * (counts[b] - (1 if a == b else 0))
            de += w * delta(a, b)
            tot_pairs += w
    de /= max(tot_pairs, 1)
    return float(1 - do / de) if de > 0 else float("nan")


def simulate_alpha_ci(n_items: int, sigma_rater: float, n_sim: int = 300) -> tuple[float, float]:
    """Add rater noise to true scores to make a 2-rater scoring, and look at the bootstrap interval width of α."""
    rng = np.random.default_rng(SEED)
    alphas = []
    for _ in range(n_sim):
        truth = rng.normal(4.0, SIGMA_ITEM, n_items)
        obs = np.clip(
            np.round(truth[None, :] + rng.normal(0, sigma_rater, (N_RATERS, n_items))), 1, 5
        ).astype(int)
        alphas.append(krippendorff_ordinal(obs))
    a = np.array([x for x in alphas if np.isfinite(x)])
    return float(np.mean(a)), float(np.percentile(a, 97.5) - np.percentile(a, 2.5))


def binary_mde(n_items: int, p0: float = 0.5) -> float:
    """Smallest difference at which accuracy can be said to differ from chance (two-sided α=.05, power 80%)."""
    se = np.sqrt(p0 * (1 - p0) / n_items)
    return float((Z95 + Z80) * se)


def binary_power(n_items: int, p_true: float, p0: float = 0.5) -> float:
    se = np.sqrt(p0 * (1 - p0) / n_items)
    return float(stats.norm.cdf(abs(p_true - p0) / se - Z95))


def auc_se(n_items: int, positive_rate: float, auc: float) -> float:
    """Hanley--McNeil approximate standard error."""
    n1 = max(round(n_items * positive_rate), 2)
    n2 = max(n_items - n1, 2)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (n1 * n2)
    return float(np.sqrt(var))


def auc_n_for_se(target_se: float, positive_rate: float, auc: float) -> int:
    n = 40
    while n < 5000 and auc_se(n, positive_rate, auc) > target_se:
        n += 10
    return n


def main() -> int:
    log.info(
        "assumptions: sigma_item=%s sigma_rater=%s raters=%d", SIGMA_ITEM, SIGMA_RATER, N_RATERS
    )

    lines: list[str] = [
        "# 인간 검증 설계의 통계적 유효성 (ADR-023)",
        "",
        "`scripts/human_validation_power.py` 가 생성한다. 문항 수는 이 계산에서 나온다.",
        "",
        f"**가정**: 항목 간 표준편차 σ_item={SIGMA_ITEM}, 평가자 불일치 σ_rater={SIGMA_RATER},",
        f"평가자 {N_RATERS} 명, 1 인 예산 {BUDGET_MIN} 분.",
        "문항당 소요는 S1 {S1}분 · S2 {S2}분 · S3 {S3}분으로 잡았다 (보정 회차에서 실측 후 갱신).".format(
            **MIN_PER_ITEM
        ),
        "",
        "## S1 — 루브릭 평균의 정밀도",
        "",
        "| 항목 수 | 평균의 표준오차 | 95% 반폭 | 채점 시간(1인) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n in (50, 80, 100, 150, 200):
        se = mean_se(n)
        lines.append(f"| {n} | {se:.3f} | ±{Z95 * se:.2f} | {n * MIN_PER_ITEM['S1']:.0f}분 |")
    lines += [
        "",
        "5 점 척도에서 ±0.15 보다 좁으면 「4.3 과 4.6 은 다르다」를 말할 수 있다.",
        "",
        "## S1 — 음성 통제 검정력",
        "",
        "일부러 앵커와 모순되게 만든 서사가 groundedness 에서 얼마나 확실히 걸리는가.",
        "",
        "| 통제 수 | 정상 수 | 참 격차 1.0 | 참 격차 1.5 | 참 격차 2.5 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for nc in (5, 10, 15, 20):
        nn = 100 - nc
        row = [f"{control_power(nc, nn, gap):.2f}" for gap in (1.0, 1.5, 2.5)]
        lines.append(f"| {nc} | {nn} | {row[0]} | {row[1]} | {row[2]} |")
    lines += [
        "",
        "통제는 **설계상 크게 어긋나게** 만들 것이므로 참 격차 2.5 가 현실적이다.",
        "그 경우 10 개면 충분하다. 격차가 1.0 밖에 안 난다면 통제가 너무 미묘했다는 뜻이고,",
        "그때는 통제 설계를 고쳐야지 문항 수를 늘릴 일이 아니다.",
        "",
        "## S1 — 일치도(α) 구간 폭 (몬테카를로 300 회)",
        "",
        "| 항목 수 | σ_rater | 평균 α | 95% 구간 폭 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for n in (50, 100, 150):
        for sr in (0.35, 0.50, 0.80):
            mean_a, width = simulate_alpha_ci(n, sr)
            lines.append(f"| {n} | {sr} | {mean_a:.2f} | ±{width / 2:.2f} |")
    lines += [
        "",
        "2 명 · 100 항목이면 α 구간이 대략 ±0.1 이다. 「α ≥ 0.5 면 쓴다」는 기준을",
        "점추정으로 판정하면 위험하고, **구간 하한**으로 판정해야 한다.",
        "",
        "## S2 — 강제선택으로 판별 가능한 최소 효과",
        "",
        "| 문항 수 | 판별 가능한 최소 차이 | 60% 를 잡을 검정력 | 70% 를 잡을 검정력 | 시간(1인) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in (60, 100, 150, 200, 300):
        lines.append(
            f"| {n} | {100 * binary_mde(n):.1f}%p | {binary_power(n, 0.60):.2f} | "
            f"{binary_power(n, 0.70):.2f} | {n * MIN_PER_ITEM['S2']:.0f}분 |"
        )
    lines += [
        "",
        "프로브는 생활권을 0.21 로 잡았고 기준선이 0.16 이다 — 사람이 50% 근처일 것이라는",
        "예측이다. **귀무(우연)를 기각하지 못하는 것이 이 연구의 성공**이므로, 판별 가능한",
        "최소 효과가 작을수록 「우연과 구별되지 않는다」는 말이 세진다.",
        "",
        "## S3 — 인간 AUC 의 표준오차",
        "",
        "| 문항 수 | 양성비 0.4 · AUC 0.80 | 로지스틱 0.77 과 구분? | 상한 0.83 과 구분? | 시간(1인) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in (80, 120, 200, 300, 400):
        se = auc_se(n, 0.4, 0.80)
        d_low = abs(0.80 - 0.77) / se
        d_high = abs(0.83 - 0.80) / se
        lines.append(
            f"| {n} | ±{Z95 * se:.3f} | {'예' if d_low > Z95 else '아니오'} "
            f"| {'예' if d_high > Z95 else '아니오'} | {n * MIN_PER_ITEM['S3']:.0f}분 |"
        )
    need = auc_n_for_se(0.015, 0.4, 0.80)
    lines += [
        "",
        f"0.03 차이를 구간으로 가르려면 표준오차 0.015 가 필요하고, 그러려면 문항이 약 {need} 개 든다",
        f"({need * MIN_PER_ITEM['S3']:.0f}분). 예산 안에서는 불가능하다.",
        "**그래서 S3 의 목표를 바꾼다**: 「사람이 0.77/0.83 보다 높은가 낮은가」를 가리는 대신,",
        "「사람이 우연(0.5)보다 얼마나 높은가」와 「대략 어느 구간인가」를 보고한다.",
        "기계와의 미세 비교는 문항을 늘릴 수 있을 때로 미룬다.",
        "",
        "## 예산 배분",
        "",
        "| 배분 | S1 | S2 | S3 | 합계(분) | 얻는 것 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, (a, b, c) in {
        "초안": (100, 130, 80),
        "권장": (80, 200, 160),
        "S1 중심": (140, 100, 60),
    }.items():
        total = a * MIN_PER_ITEM["S1"] + b * MIN_PER_ITEM["S2"] + c * MIN_PER_ITEM["S3"]
        gain = (
            f"평균 ±{Z95 * mean_se(a):.2f} · S2 최소효과 {100 * binary_mde(b):.0f}%p · "
            f"S3 ±{Z95 * auc_se(c, 0.4, 0.8):.03f}"
        )
        lines.append(f"| {label} | {a} | {b} | {c} | {total:.0f} | {gain} |")
    lines += [
        "",
        f"예산은 1 인 {BUDGET_MIN} 분이다. **권장 배분**은 S1 을 80 명으로 줄이고(정밀도 손실이",
        "0.01 수준이다) 그 시간을 S2·S3 로 옮긴 것이다. S2 는 문항당 20 초라 늘리는 값이 싸고,",
        "S3 는 지금 배분에서 구간이 너무 넓다.",
        "",
        "## 채점 뒤에 할 일",
        "",
        "여기 쓰인 σ 는 **가정**이다. 보정 회차 10 명에서 두 사람의 차이를 실측해 σ_rater 를",
        "갱신하고 이 문서를 다시 생성한다. 실측이 가정보다 크면 문항을 늘리거나 주장을 줄인다.",
    ]
    out = Path(__file__).resolve().parents[1] / "docs" / "human_validation_power.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", out)
    print("\n".join(lines[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
