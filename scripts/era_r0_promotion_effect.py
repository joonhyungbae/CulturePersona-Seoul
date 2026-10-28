"""How much promoting a per-wave R0 moved the holdout (ADR-013).

Pre-promotion numbers are not copied by hand — they are **read from git**. The
default base commit is the one just before promotion (`5b272e9^`). Each item is
compared with the same file in the current worktree.

How to read it: the twin arm answers from **real respondent anchors**, so it
does not depend on the population root. The twin's shift is therefore the noise
floor of this comparison (model regeneration noise), and the release shift
should be read against that floor.

Run: uv run python scripts/era_r0_promotion_effect.py [--base 5b272e9^]
"""

from __future__ import annotations

import argparse
import io
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("era_r0_promotion_effect")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = settings.outputs_dir / "reports"
WAVES = (2020, 2022, 2024)


def _from_git(rev: str, rel_path: str) -> pd.DataFrame | None:
    out = subprocess.run(
        ["git", "show", f"{rev}:{rel_path}"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0 or not out.stdout.strip():
        log.warning("git show %s:%s failed", rev, rel_path)
        return None
    return pd.read_csv(io.StringIO(out.stdout))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="5b272e9^", help="commit just before promotion")
    args = ap.parse_args()

    rows = []
    for wave in WAVES:
        rel = f"outputs/reports/era_holdout_{wave}_scores.csv"
        before = _from_git(args.base, rel)
        after_path = REPORTS / f"era_holdout_{wave}_scores.csv"
        if before is None or not after_path.exists():
            continue
        after = pd.read_csv(after_path)
        key = ["tag", "item"]
        m = before.merge(after, on=key, suffixes=("_before", "_after"))
        m["delta"] = m["cell_mean_tvd_after"] - m["cell_mean_tvd_before"]
        m.insert(0, "wave", wave)
        rows.append(m[[*["wave"], *key, "cell_mean_tvd_before", "cell_mean_tvd_after", "delta"]])

    if not rows:
        raise SystemExit("nothing to compare — check the base commit")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(REPORTS / "era_r0_promotion_effect.csv", index=False)

    summ = (
        df.groupby(["wave", "tag"])
        .agg(
            mean_before=("cell_mean_tvd_before", "mean"),
            mean_after=("cell_mean_tvd_after", "mean"),
            max_abs_item_delta=("delta", lambda s: float(s.abs().max())),
        )
        .reset_index()
    )
    summ["mean_delta"] = summ.mean_after - summ.mean_before
    summ.to_csv(REPORTS / "era_r0_promotion_effect_summary.csv", index=False)

    rel_max = float(summ[summ.tag == "release_narrative"].mean_delta.abs().max())
    twin_max = float(summ[summ.tag == "twin_anchors"].mean_delta.abs().max())

    def md(d: pd.DataFrame) -> str:
        return d.round(4).to_markdown(index=False) if hasattr(d, "to_markdown") else d.to_string()

    lines = [
        "# 파동별 R0 승격의 효과 — holdout (ADR-013)",
        "",
        f"기준(승격 직전) 커밋: `{args.base}`. 승격 전 수치는 git 에서 읽는다.",
        "",
        "## 파동 × 팔",
        "",
        md(summ),
        "",
        (
            f"- release 팔 평균 이동 최대 **{rel_max:.3f}**"
            f" · twin 팔 평균 이동 최대 **{twin_max:.3f}**"
        ),
        (
            "- twin 은 실제 응답자 앵커에서 답하므로 인구 뿌리와 **무관**하다."
            " 그 이동폭이 이 비교의 잡음 바닥이다."
        ),
        "",
        "## 문항별",
        "",
        md(df),
        "",
        "CSV: `era_r0_promotion_effect.csv` · `era_r0_promotion_effect_summary.csv`",
        "",
    ]
    (REPORTS / "era_r0_promotion_effect.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("release max |Δmean|=%.4f · twin max |Δmean|=%.4f", rel_max, twin_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
