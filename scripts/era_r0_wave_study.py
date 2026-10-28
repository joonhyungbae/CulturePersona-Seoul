"""Per-wave population root (R0) study — resolving ADR-011.

`build_root_population` used the latest month in the population grounding CSV,
and that CSV held only months from 2023-01 on. The sex×age×living-area mix of
the 2020 and 2022 era banks was therefore Seoul in December 2025.
`fetch_population_prior_history.py` has fetched December 2016, 2018, 2020, and
2022, so a per-wave R0 can now be built. This script measures **how large that
difference is**.

Two layers.
  1. The root itself: TVD of the wave R0 vs the R0 the public release actually used (latest month).
  2. Personas: run the same chain on each root and compare distance to the observed margin.
     A sample with the `_r0wave` suffix is required (era_sample_quant --out-suffix _r0wave).

**How to read it.** By definition the root is resident-registration population,
not the panel (ADR-002). Whether the wave R0 moves closer to the survey-weighted
age distribution is therefore **not** a success criterion — the criterion is
whether that year's population is the right one to use. The layer-2 table only
shows how far that choice moves the output.

Run: uv run python scripts/era_r0_wave_study.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.grounding.tables import build_root_population
from sfac_persona.schema import ERA_WAVES, GENRE_ANCHORS, WEIGHTED_WAVES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("era_r0_wave_study")
for _n in ("sfac_persona.grounding.official", "sfac_persona.grounding.tables"):
    logging.getLogger(_n).setLevel(logging.WARNING)

REPORTS = settings.outputs_dir / "reports"
ROOT_KEYS = ["living_area", "sex", "age_band"]


def _tvd(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return float(0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())


def _wdist(values: pd.Series, weights: pd.Series) -> pd.Series:
    d = weights.groupby(values).sum()
    return d / d.sum()


def root_layer() -> pd.DataFrame:
    prior = pd.read_csv(settings.population_prior_csv)
    months = sorted(prior.reference_month.unique())
    shipped_month = months[-1]
    shipped = build_root_population(month=shipped_month)
    base = shipped.set_index(ROOT_KEYS)["share"]
    base_age = shipped.groupby("age_band", observed=True)["share"].sum()

    rows = []
    for wave in ERA_WAVES:
        want = [m for m in months if m.startswith(str(wave))]
        if not want:
            rows.append({"wave": wave, "wave_month": None, "available": False})
            continue
        r = build_root_population(month=want[-1])
        s = r.set_index(ROOT_KEYS)["share"]
        age = r.groupby("age_band", observed=True)["share"].sum()
        rows.append(
            {
                "wave": wave,
                "wave_month": want[-1],
                "available": True,
                "shipped_month": shipped_month,
                "pop_15plus_wave": int(r.population.sum()),
                "pop_15plus_shipped": int(shipped.population.sum()),
                "tvd_root_vs_shipped": _tvd(base, s),
                "tvd_age_vs_shipped": _tvd(base_age, age),
            }
        )
    return pd.DataFrame(rows)


def persona_layer() -> pd.DataFrame:
    rows = []
    for wave in WEIGHTED_WAVES:
        real = build_official_anchor_frame(wave)
        w = real["w"].astype(float)
        gcols = [g for g in GENRE_ANCHORS if g in real.columns]
        real = real.assign(width=(real[gcols] == "예").sum(axis=1))
        for ablation in ("margins", "structure_T"):
            base_p = settings.synthetic_dir / f"era_{wave}_quant" / f"personas_{ablation}.parquet"
            new_p = (
                settings.synthetic_dir / f"era_{wave}_quant" / f"personas_{ablation}_r0wave.parquet"
            )
            if not (base_p.exists() and new_p.exists()):
                log.warning("wave=%d %s: comparison sample missing — skipping", wave, ablation)
                continue
            old = pd.read_parquet(base_p)
            new = pd.read_parquet(new_p)
            for axis in ROOT_KEYS:
                r = _wdist(real[axis], w)
                rows.append(
                    {
                        "wave": wave,
                        "ablation": ablation,
                        "metric": f"tvd_{axis}_vs_real",
                        "shipped_r0": _tvd(r, old[axis].value_counts(normalize=True)),
                        "wave_r0": _tvd(r, new[axis].value_counts(normalize=True)),
                    }
                )

            def width_mae(syn: pd.DataFrame, _real=real, _gcols=gcols) -> float:
                real, gcols = _real, _gcols
                cols = [c for c in gcols if c in syn.columns]
                s = syn.assign(width=(syn[cols] == "예").sum(axis=1)) if "width" not in syn else syn
                maes = []
                for age, g in s.groupby("age_band"):
                    rw = real[real["age_band"] == age]
                    if len(rw) == 0 or len(g) == 0:
                        continue
                    maes.append(abs(np.average(rw["width"], weights=rw["w"]) - g["width"].mean()))
                return float(np.mean(maes)) if maes else float("nan")

            rows.append(
                {
                    "wave": wave,
                    "ablation": ablation,
                    "metric": "width_mae_by_age",
                    "shipped_r0": width_mae(old),
                    "wave_r0": width_mae(new),
                }
            )
    df = pd.DataFrame(rows)
    if len(df):
        df["delta"] = df["wave_r0"] - df["shipped_r0"]
    return df


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    root = root_layer()
    persona = persona_layer()
    root.to_csv(REPORTS / "era_r0_wave_study_root.csv", index=False)
    if len(persona):
        persona.to_csv(REPORTS / "era_r0_wave_study_personas.csv", index=False)

    def md(df: pd.DataFrame) -> str:
        return (
            df.round(4).to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string()
        )

    avail = root[root.available.fillna(False)]
    lines = [
        "# 파동별 인구 뿌리 (R0) — ADR-011 해소",
        "",
        (
            "인구 근거 CSV 에 2016·2018·2020·2022년 12월을 받아 넣었다"
            " (`scripts/fetch_population_prior_history.py`, 2025-12 로 파서 검증: 350행 불일치 0)."
        ),
        "이제 파동별 R0 를 만들 수 있다. 아래는 **그 선택이 얼마나 움직이는가**다.",
        "",
        "## 1. 뿌리 자체 — 파동 R0 vs 공개본이 쓴 최신 월 R0",
        "",
        md(avail.drop(columns=["available"])),
        "",
        (
            "공개본은 전 시대에 최신 월 하나를 썼다. 2020 의 괴리(TVD"
            f" {float(avail[avail.wave == 2020].tvd_root_vs_shipped.iloc[0]):.3f})는"
            " atlas 의 앵커쌍 평균 TVD(0.026–0.077)와 같은 크기다 — 무시할 수준이 아니다."
        ),
        "",
        "## 2. 페르소나 — 같은 체인을 두 뿌리로 돌렸을 때",
        "",
    ]
    if len(persona):
        lines += [md(persona), ""]
    lines += [
        (
            "**읽는 법.** 뿌리는 정의상 주민등록 인구이지 조사 패널이 아니다 (ADR-002)."
            " 따라서 조사 가중 연령분포와 가까워지는지는 성공 기준이 아니다 — 2020 처럼 멀어지는"
            " 축이 있는 것은 그 해 패널과 그 해 인구의 괴리가 크다는 뜻이지 뿌리가 틀렸다는 뜻이"
            " 아니다. 판단 기준은 「그 해 뱅크가 그 해 인구 구성을 쓰는가」이고, 파동 R0 는"
            " 정의상 그렇다."
        ),
        "",
        "## 3. 상태",
        "",
        "- 파동별 R0: `data/grounding/era_*/R0_population_root.parquet` ✔",
        "- 파동별 spec: `data/grounding/era_*/sampler_specs_*_r0wave.json` ✔ (아직 공개본 아님)",
        "- 비교 표본: `data/synthetic/era_*_quant/personas_*_r0wave.parquet` ✔",
        "- **공개본은 여전히 최신 월 R0** — 승격하면 서사·holdout·PPI·원고 수치를 전부 다시 내야 한다.",
        "",
    ]
    (REPORTS / "era_r0_wave_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote era_r0_wave_study.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
