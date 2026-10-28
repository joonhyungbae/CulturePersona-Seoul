"""Downstream 2 — survey-design pretest (ADR-019).

Question: if someone designing a new survey sets subgroup sample sizes from the synthetic
population **before drawing a real sample**, how far off are they?

The procedure is what a statistical office actually does. How many people are needed to
estimate a subgroup's attendance rate (for example, people in their 50s · 서북생활권) at a
target precision — the required sample is proportional to p(1−p), so a wrong prior
estimate p̂ makes the sample size wrong too.

  baseline A  published marginal (one overall attendance rate for every cell)
  baseline B  published age-band conditional
  ours        cell-level rates in the synthetic population
  truth       cell-level weighted rates in the same wave's microdata

Report: cell-level |p̂ − p| and the **multiplicative error in the required sample** that
follows (planned n / required n). A ratio below 1 is **undersampling** — drawing the
planned n misses the target precision.

No cultural-participation interpretation. What is measured is the error of the design
calculation.

Run: uv run python scripts/era_design_pretest.py
Outputs: outputs/reports/era_design_pretest.csv · era_design_pretest.md
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
from sfac_persona.grounding.official import build_official_anchor_frame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("design_pretest")

HALF_WIDTH = 0.05  # Target: 95% CI half-width ±5 pp.
Z = 1.959964
MIN_CELL = 30
# Cells fall into two classes.
#   published    Axes the report actually prints (age band · living area · age×sex).
#                Here the "age-conditional" baseline is correct by definition, so the
#                comparison is meaningless — this is the control.
#   unpublished  Crossings that are not printed (income×education, and so on). If the
#                synthetic population is worth anything, it has to show up here.
PUBLISHED_KEYS = (("age_band",), ("living_area",), ("age_band", "sex"), ("age_band", "living_area"))
UNPUBLISHED_KEYS = (
    ("household_income_band",),
    ("education",),
    ("occupation",),
    ("education", "household_income_band"),
    ("age_band", "household_income_band"),
    ("age_band", "education"),
    ("leisure_sufficiency", "age_band"),
    ("household_size_band", "age_band"),
)
CELL_KEYS = PUBLISHED_KEYS + UNPUBLISHED_KEYS


def required_n(p: float, half_width: float = HALF_WIDTH) -> float:
    """Required sample under simple random sampling (no finite-population correction)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float((Z**2) * p * (1 - p) / (half_width**2))


def synthetic_bank(wave: int) -> pd.DataFrame:
    p = settings.data_root / "synthetic" / f"era_{wave}_quant" / "personas_structure_T.parquet"
    return pd.read_parquet(p)


def cell_rates(df: pd.DataFrame, keys: list[str], outcome: str, weight: str | None) -> pd.Series:
    y = (df[outcome].astype(str) == "예").astype(float)
    if weight is None:
        return y.groupby([df[k] for k in keys]).mean()
    w = df[weight].astype(float)
    num = (y * w).groupby([df[k] for k in keys]).sum()
    den = w.groupby([df[k] for k in keys]).sum()
    return num / den


def run(wave: int, outcome: str = "attend_any") -> pd.DataFrame:
    real = build_official_anchor_frame(wave)
    syn = synthetic_bank(wave)
    rows = []
    for keys in CELL_KEYS:
        k = list(keys)
        truth = cell_rates(real, k, outcome, "w")
        counts = real.groupby(k).size()
        truth = truth[counts >= MIN_CELL]
        syn_p = cell_rates(syn, k, outcome, None)
        marginal = float(
            (real[outcome].astype(str) == "예").astype(float).mul(real.w).sum() / real.w.sum()
        )
        by_age = cell_rates(real, ["age_band"], outcome, "w")
        for cell, p_true in truth.items():
            cell_t = cell if isinstance(cell, tuple) else (cell,)
            p_syn = float(syn_p.get(cell, np.nan))
            age_val = cell_t[k.index("age_band")] if "age_band" in k else None
            p_age = float(by_age.get(age_val, marginal)) if age_val is not None else marginal
            n_true = required_n(float(p_true))
            for label, p_hat in (
                ("marginal", marginal),
                ("age-conditional", p_age),
                ("synthetic", p_syn),
            ):
                if not np.isfinite(p_hat):
                    continue
                rows.append(
                    {
                        "wave": wave,
                        "cell_key": "+".join(k),
                        "published_axis": tuple(k) in PUBLISHED_KEYS,
                        "cell": " / ".join(str(c) for c in cell_t),
                        "method": label,
                        "p_true": float(p_true),
                        "p_hat": float(p_hat),
                        "abs_err_pp": 100 * abs(float(p_hat) - float(p_true)),
                        "n_planned": required_n(float(p_hat)),
                        "n_required": n_true,
                        "n_ratio": required_n(float(p_hat)) / n_true,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, action="append", default=None)
    args = ap.parse_args()
    waves = args.wave or [2024, 2022, 2020]

    res = pd.concat([run(w) for w in waves], ignore_index=True)
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    res.to_csv(reports / "era_design_pretest.csv", index=False)

    agg = res.groupby(["wave", "cell_key", "method"]).agg(
        cells=("cell", "size"),
        mean_abs_err_pp=("abs_err_pp", "mean"),
        p90_abs_err_pp=("abs_err_pp", lambda x: float(np.percentile(x, 90))),
        median_n_ratio=("n_ratio", "median"),
        under_planned_pct=("n_ratio", lambda x: 100.0 * float((x < 0.9).mean())),
    )
    overall = res.groupby(["published_axis", "method"]).agg(
        mean_abs_err_pp=("abs_err_pp", "mean"),
        median_n_ratio=("n_ratio", "median"),
        under_planned_pct=("n_ratio", lambda x: 100.0 * float((x < 0.9).mean())),
    )
    lines = [
        "# Downstream 2 — 조사 설계 사전검토 (ADR-019)",
        "",
        f"소집단 관람률을 ±{HALF_WIDTH:.0%} 로 추정하는 데 필요한 표본을, 실제 표본 없이 정한다.",
        "",
        "## 공표 축 / 비공표 교차",
        overall.round(3).to_markdown(),
        "",
        "## 칸 정의별",
        agg.round(3).to_markdown(),
        "",
        "`n_ratio` 는 계획한 표본 / 실제 필요한 표본이다. 1 보다 작으면 과소표집이고,",
        "`under_planned_pct` 는 10% 이상 모자라는 칸의 비율이다.",
        "참값은 같은 파동 마이크로데이터의 가중 비율, 칸은 비가중 30 명 이상만 본다.",
    ]
    (reports / "era_design_pretest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", reports / "era_design_pretest.csv")
    print(overall.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
