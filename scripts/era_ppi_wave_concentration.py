"""Diagnose wave concentration of PPI significant rows — after the ADR-017 decision rule.

Once `above_null` was tightened to the 95th percentile of the permutation null,
passing rows fell from 71/81 to 24/81, and 13 of those 24 sit in wave 2020.
2020 is the worst holdout wave, so that needs an explanation.

Three hypotheses are separated using existing outputs only — no new computation
and no access to microdata.
  1. Sample size · λ bias   (does ρ² inflate when n is small)
  2. Threshold              (does perm95 differ by wave)
  3. Non-attender share     (same axis as the atlas reduction rate)

Run: uv run python scripts/era_ppi_wave_concentration.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics as st
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PPI_CSV = REPO_ROOT / "outputs" / "reports" / "era_holdout_ppi.csv"
ATLAS_MD = REPO_ROOT / "outputs" / "reports" / "decade_error_atlas.md"
OUT_DIR = REPO_ROOT / "outputs" / "reports"
WAVES = ("2020", "2022", "2024")
# Item whose routing differs across waves: 2020 asks everyone; 2022 and 2024 ask attenders only.
ROUTED_ITEM = "attendance_overall"
# The canonical report is the row-level output of sfac-persona-a6. This file is a
# cross-check that does not open microdata.
STEM = "era_ppi_wave_concentration_crosscheck"


def read_atlas_shares(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Read non-attender share and reduction rate from weighted rows of decade_error_atlas.md."""
    header: list[str] = []
    nonatt: dict[str, float] = {}
    cut: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "wave":
            header = cells
            continue
        if not header or len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        if row.get("mode") != "weighted" or row["wave"] not in WAVES:
            continue
        nonatt[row["wave"]] = 1.0 - float(row["attend_any_share"])
        cut[row["wave"]] = float(row["attend_cut"])
    missing = set(WAVES) - nonatt.keys()
    if missing:
        raise ValueError(f"waves with no weighted row in the atlas: {sorted(missing)}")
    return nonatt, cut


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ppi-csv", type=Path, default=PPI_CSV)
    ap.add_argument("--atlas-md", type=Path, default=ATLAS_MD)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = list(csv.DictReader(args.ppi_csv.open(encoding="utf-8")))
    logger.info("read %d PPI rows (%s)", len(rows), args.ppi_csv.name)
    nonatt, atlas_cut = read_atlas_shares(args.atlas_md)

    def agg(group: list[dict[str, str]]) -> dict[str, float]:
        return {
            "rows": len(group),
            "pass": sum(r["above_null"] == "True" for r in group),
            "rho2": st.mean(float(r["rho2"]) for r in group),
            "perm95": st.mean(float(r["rho2_perm95"]) for r in group),
            "ci": st.mean(float(r["rho2_hi"]) - float(r["rho2_lo"]) for r in group),
        }

    wave_stat = {w: agg([r for r in rows if r["wave"] == w]) for w in WAVES}
    # ROUTED_ITEM confounds wave comparisons because routing differs by wave (report section «채널»).
    excl_stat = {
        w: agg([r for r in rows if r["wave"] == w and r["item"] != ROUTED_ITEM]) for w in WAVES
    }
    by_n = {k: agg(v) for k, v in sorted(_group(rows, lambda r: (r["wave"], int(r["n"]))).items())}
    items = sorted({r["item"] for r in rows})
    item_stat = {
        (it, w): agg([r for r in rows if r["item"] == it and r["wave"] == w])
        for it in items
        for w in WAVES
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_wave_csv(args.out_dir, wave_stat, nonatt, atlas_cut)
    _write_item_csv(args.out_dir, items, item_stat)
    _write_report(args.out_dir, wave_stat, excl_stat, by_n, items, item_stat, nonatt, atlas_cut)


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def _write_wave_csv(out_dir, wave_stat, nonatt, atlas_cut) -> None:
    path = out_dir / f"{STEM}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "wave",
                "rows",
                "n_above_null",
                "mean_rho2",
                "mean_perm95",
                "mean_ci_width",
                "nonattender_share",
                "atlas_attend_cut",
            ]
        )
        for wave in WAVES:
            s = wave_stat[wave]
            w.writerow(
                [
                    wave,
                    s["rows"],
                    s["pass"],
                    f"{s['rho2']:.4f}",
                    f"{s['perm95']:.4f}",
                    f"{s['ci']:.4f}",
                    f"{nonatt[wave]:.3f}",
                    f"{atlas_cut[wave]:.4f}",
                ]
            )
    logger.info("%s", path.name)


def _write_item_csv(out_dir, items, item_stat) -> None:
    path = out_dir / f"{STEM}_by_item.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item"] + [f"{x}_{k}" for x in WAVES for k in ("pass", "mean_rho2")])
        for it in items:
            row: list[object] = [it]
            for wave in WAVES:
                s = item_stat[(it, wave)]
                row += [s["pass"], f"{s['rho2']:.4f}"]
            w.writerow(row)
    logger.info("%s", path.name)


def _write_report(out_dir, wave_stat, excl_stat, by_n, items, item_stat, nonatt, atlas_cut) -> None:
    nl = "\n"
    wave_rows = nl.join(
        f"| {w} | {100 * nonatt[w]:.1f}% | {wave_stat[w]['pass']}/{wave_stat[w]['rows']} "
        f"| {wave_stat[w]['rho2']:.4f} | {wave_stat[w]['perm95']:.4f} "
        f"| {wave_stat[w]['ci']:.4f} | {100 * atlas_cut[w]:.1f}% |"
        for w in WAVES
    )
    n_rows = nl.join(
        f"| {k[0]} | {k[1]} | {s['rows']} | {s['rho2']:.4f} | {s['pass']} |"
        for k, s in by_n.items()
    )
    item_rows = nl.join(
        f"| `{it}` | "
        + " | ".join(
            f"{item_stat[(it, w)]['pass']}/3 · {item_stat[(it, w)]['rho2']:.4f}" for w in WAVES
        )
        + " |"
        for it in items
    )

    excl_rows = nl.join(
        f"| {w} | {excl_stat[w]['pass']}/{excl_stat[w]['rows']} "
        f"| {excl_stat[w]['rho2']:.4f} | {100 * nonatt[w]:.1f}% |"
        for w in WAVES
    )
    best = max(item_stat.items(), key=lambda kv: kv[1]["rho2"])
    path = out_dir / f"{STEM}.md"
    path.write_text(
        f"""# PPI 유의 행의 파동 편중 — 진단 (2026-09-20)

ADR-017 로 `above_null` 판정이 순열 귀무 1000회의 95 백분위 기준으로 강화되면서
통과 행이 71/81 → {sum(s["pass"] for s in wave_stat.values())}/81 로 줄었다. 그 통과 행이 파동별로 고르지 않다.
2020 파동에 {wave_stat["2020"]["pass"]}행이 몰리는데, 2020 은 홀드아웃 mean cell TVD 가 0.403 으로
세 파동 중 가장 나쁜 파동이다. 설명 없이 두면 "1.6% 조차 어디서 온 건지 모른다" 가 된다.

세 가설을 구분했다. **새 연산도 원자료 접근도 없다** — 기존 산출물
(`era_holdout_ppi.csv`, `decade_error_atlas.md`) 만으로 갈린다.

## 이 보고서의 위치 — 대조본이다

정본은 `era_ppi_wave_concentration.{{md,csv}}` (ADR-018, 행 단위 3,429행) 다. 이쪽은
**원자료를 열지 않는 독립 재현**이라 값이 갈리면 신호가 되는 용도다. 입력이
집계 산출물 두 개뿐이므로 재현이 싸다.

정본이 이 보고서를 넘어선 곳을 먼저 적는다. 이 세 가지는 정본을 따라라.

1. **역설이 애초에 없다.** 홀드아웃 TVD 는 응답층의 분포 적합을, ρ² 는 앵커–응답
   연관을 잰다. 한 파동이 전자에서 나쁘고 후자에서 좋을 수 있다. "가장 나쁜 파동이
   유의 행을 가장 많이 낸다" 는 설명이 필요한 모순이 아니다.
2. **응답 분산 가설은 반대 방향으로 기각된다.** 2020 이 가장 *덜* 퍼져 있다
   (정규화 엔트로피 0.769 / 0.782 / 0.842). 통과와 붙는 것은 분산이 아니라
   |Cov(Y,F)| (ρ_s=+0.48) 다. 즉 통과는 「ρ² 가 뜰 여지」가 아니라 실제 연관에서 온다.
   아래 «채널» 절의 "관람이 붕괴해 설명할 분산이 커진다" 류의 서술은 이 값으로 대체된다.
3. **{ROUTED_ITEM} 제외 후에도 남는 수(10/6/5)는 안정된 통계량이 아니다.** 재표집
   seed 만 바꾸면 24/81 → 25/81 로 움직이고, 2024 는 27행 중 10행이 문턱
   |margin|<0.002 안에 있다. **「24 of 81」을 정확한 수처럼 인용하면 안 된다.**

가설 번호는 이 보고서와 정본이 다르다: 여기의 가설 1(n·λ) = 정본 H3,
가설 2(문턱) = 정본 H3 의 일부, 가설 3(비관람자 비율) = 정본 H2. 정본의 H1
(응답 분산) 은 이 보고서가 다루지 않는다.

## 결과

| wave | 비관람자 비율(가중) | 통과 | 평균 ρ² | 평균 perm95 | 평균 CI 폭 | atlas attend_cut |
| --- | --- | --- | --- | --- | --- | --- |
{wave_rows}

**가설 1 (표본 크기 · λ 편향) — 기각.** λ 를 평가 표본에서 적합하므로 n 이 작을수록
ρ² 가 위로 뜰 것을 걱정했다. 방향이 반대다. 2020 은 27행 전부 n=300 이고,
n 이 작은 행(2022·2024 각 3행)은 **전부 탈락**한다.

| wave | n | rows | 평균 ρ² | 통과 |
| --- | --- | --- | --- | --- |
{n_rows}

**가설 2 (문턱이 낮아서) — 기각.** perm95 평균은 2022 가 가장 높고 2024 가 가장 낮다.
2020 은 중간이다. 쉬운 바를 넘은 것이 아니고 ρ² 자체가 높다.

**가설 3 (비관람자 비율) — 방향은 맞지만 처음 본 것보다 약하다.** 통과 수와 평균 ρ²
가 비관람자 비율과 같은 순서로 정렬하고, atlas 가 보고한 관계(절감률과 비관람자
비율의 상관 서울 +0.94 · KCAS +0.91) 와 같은 축이다. 그런데 이 정렬의 몫이 라우팅이
다른 문항 하나(`{ROUTED_ITEM}`, 아래 «채널») 에서 온다. 그 문항을 빼면 이렇게 된다.

| wave | 통과 | 평균 ρ² | 비관람자 비율(가중) |
| --- | --- | --- | --- |
{excl_rows}

통과 수의 순서는 남지만 2020 과 2022 의 평균 ρ² 차이는 사실상 사라진다
({excl_stat["2020"]["rho2"]:.4f} vs {excl_stat["2022"]["rho2"]:.4f}). 세 파동으로는
2020 과 2022 를 가릴 힘이 없다고 보는 것이 정확하다.

## 채널

| item | 2020 (통과·ρ²) | 2022 | 2024 |
| --- | --- | --- | --- |
{item_rows}

`{best[0][0]}` 이 {best[0][1]} 에서 ρ²={best[1]["rho2"]:.4f} — 전체 81행에서 가장 높은 값인데,
같은 문항이 2024 에서는 {item_stat[(best[0][0], "2024")]["pass"]}/3, {item_stat[(best[0][0], "2024")]["rho2"]:.4f} 이다. **이 격차는 실질이 아니라 설문
라우팅이다** (`sfac-persona-a6` 확인). 2020 은 이 문항을 응답자 전원에게 물었고
2022·2024 는 관람자에게만 물어 평가 표본이 n=241·243 으로 줄어든다. 그 부분집합에서는
`attend_any` 가 상수이므로 예측자 세 종(`learned` · `twin_ssr` · `distilled`) 이 모두
최강 앵커를 잃는다 (2024 `learned` ρ²=0.0007 vs 2020 0.0537). n 이 작은 행이 곧
이 문항이라는 점도 여기서 설명된다 — 가설 1 의 «작은 n 은 전부 탈락» 은 표본 크기
효과가 아니라 이 라우팅의 결과다.

남는 신호는 `effect_stress_relieved` · `effect_overall_happiness` · `leisure_time_weekday`
쪽이다. 이 문항들은 파동 간 라우팅이 같으므로 비관람자 비율 설명이 여기서는 성립한다.

## 함의

1. **편중의 일부는 인공물이고 일부는 아니다.** `{ROUTED_ITEM}` 이 기여한 몫
   (2020 통과 3행, ρ² 최고값) 은 라우팅 차이라 파동 비교에서 빠져야 한다. 그것을
   빼고 남는 부분은 atlas 의 "절감은 비관람자 비율을 따라간다" 와 같은 축이므로,
   두 관찰을 하나의 진술로 묶는 근거는 남는다 — 다만 파동 3개로는 약하다.
2. **1.6% 의 성격이 바뀐다.** 그 평균은 관람이 붕괴한 파동 하나와 정상 파동 둘을
   섞은 값이고, 라우팅이 다른 문항 하나가 그 안에 섞여 있다. 정상 파동만 보면
   2024 평균 ρ² {wave_stat["2024"]["rho2"]:.4f}, 27행 중 {wave_stat["2024"]["pass"]}행이다.
3. **보고 형식 제안 (미서명).** atlas 가 파동별로 비관람자 비율을 나란히 내고 있으므로
   PPI 도 같은 형식으로 간다 — 단일 평균 대신 **파동별 + 비관람자 비율 병기**.
   원고 보고 형식 변경이므로 ADR 이 필요하다.

## 재현

    uv run python scripts/era_ppi_wave_concentration.py

입력은 `era_holdout_ppi.csv` (ADR-017 판정 기준) 와 `decade_error_atlas.md` 의
weighted 행뿐이다. 비관람자 비율은 그 행의 `1 - attend_any_share` 다. 가중치가 관람
분포에 레이킹돼 있으므로 비가중으로 보면 2020 0.3125 · 2022 0.2157 · 2024 0.2165 다.
2020 은 그대로 가장 높지만 **2022 와 2024 는 0.0008 차이로 사실상 동률이고 순서가
뒤집힌다** — 비가중 축에서는 그 두 파동을 이 값으로 가릴 수 없다. `sfac-persona-a6`
가 평가 표본에서 잰 비가중 값(0.312 / 0.216 / 0.216) 도 같은 결론이다.
""",
        encoding="utf-8",
    )
    logger.info("%s", path.name)


if __name__ == "__main__":
    main()
