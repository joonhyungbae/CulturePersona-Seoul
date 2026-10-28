"""Era holdout cross-wave summary — weighted 2020, 2022, and 2024 (ADR-010 computational).

Merge: era_holdout_{wave}_scores.csv · _baselines.csv
Output: outputs/reports/era_holdout_crosswave.md · .csv

Run: uv run python scripts/era_holdout_crosswave.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_crosswave")

REPORTS = settings.outputs_dir / "reports"
WAVES = (2020, 2022, 2024)


def main() -> int:
    score_parts: list[pd.DataFrame] = []
    base_parts: list[pd.DataFrame] = []
    for wave in WAVES:
        sp = REPORTS / f"era_holdout_{wave}_scores.csv"
        bp = REPORTS / f"era_holdout_{wave}_baselines.csv"
        if not sp.exists():
            log.warning("missing %s", sp)
            continue
        s = pd.read_csv(sp)
        if "wave" not in s.columns:
            s.insert(0, "wave", wave)
        score_parts.append(s)
        if bp.exists():
            b = pd.read_csv(bp)
            b.insert(0, "wave", wave)
            base_parts.append(b)

    if not score_parts:
        raise SystemExit("no era_holdout_*_scores.csv found")

    scores = pd.concat(score_parts, ignore_index=True)
    bases = pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame()

    # pivot release vs twin per item
    wide_rows = []
    for (wave, item), g in scores.groupby(["wave", "item"]):
        row = {"wave": wave, "item": item}
        for _, r in g.iterrows():
            row[f"tvd_{r['tag']}"] = r["cell_mean_tvd"]
            row["tau"] = r["tau"]
            row["n"] = r["n"]
        if "tvd_release_narrative" in row and "tvd_twin_anchors" in row:
            row["delta_twin_minus_release"] = row["tvd_twin_anchors"] - row["tvd_release_narrative"]
        if not bases.empty:
            b = bases[(bases["wave"] == wave) & (bases["item"] == item)]
            if len(b):
                row["noise_floor_tvd"] = float(b.iloc[0]["noise_floor_tvd"])
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows).sort_values(["item", "wave"]).reset_index(drop=True)

    out_csv = REPORTS / "era_holdout_crosswave.csv"
    wide.to_csv(out_csv, index=False)
    scores.to_csv(REPORTS / "era_holdout_crosswave_long.csv", index=False)

    lines = [
        "# Era holdout 횡단 — ADR-010 전산",
        "",
        "- 파동: 가중 2020 · 2022 · 2024",
        "- τ: 2024 동결 0.05 재사용",
        "- H1 확정 금지 — Atlas-first 측정 표",
        "",
        "## 셀 가중 TVD (release vs twin)",
        "",
    ]
    lines.append(
        wide.to_markdown(index=False)
        if hasattr(wide, "to_markdown")
        else wide.to_string(index=False)
    )
    if not bases.empty:
        lines += ["", "## 기준선 noise floor (참고)", ""]
        floor = bases[["wave", "item", "noise_floor_tvd", "n_real", "n_cats"]]
        lines.append(
            floor.to_markdown(index=False)
            if hasattr(floor, "to_markdown")
            else floor.to_string(index=False)
        )
    lines += ["", f"CSV: `{out_csv.name}`", ""]
    md = REPORTS / "era_holdout_crosswave.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s (%d rows)", md, len(wide))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
