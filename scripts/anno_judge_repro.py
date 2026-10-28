"""Judge reproducibility — rescore a sample of bank decisions with the same settings and prompt.

Measures agreement of the unseeded gemma4 decisions (audit item 4). Scoring calls
`anno_llm_judge.ask` and `render` unchanged. Settings are not changed.

Run:
  uv run python scripts/anno_judge_repro.py --n 1000 --seed 20260926
  uv run python scripts/anno_judge_repro.py --n 3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from anno_llm_judge import MODEL, OUT_BANK, SCALES, ask, render

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_judge_repro")

DEFAULT_N = 1000
DEFAULT_SEED = 20260926
N_RETRIES = 3
RUN_TAG = "repro"
OUT_REPRO = settings.outputs_dir / "anno" / "llm_bank_repro.jsonl"
REPORT_MD = settings.outputs_dir / "reports" / "anno_judge_repro.md"
REPORT_CSV = settings.outputs_dir / "reports" / "anno_judge_repro.csv"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        out[str(row["item_id"])] = row
    return out


def judged_ids_by_wave(judged: set[str]) -> dict[int, list[str]]:
    """Group clean quant_ids that have a first-pass decision, by wave. Same filter as `--bank`."""
    from anno_build_sample import WAVES, is_clean, load_bank

    grouped: dict[int, list[str]] = {int(w): [] for w in WAVES}
    for wave in WAVES:
        for row in load_bank(wave):
            if not is_clean(row):
                continue
            qid = str(row["quant_id"])
            if qid in judged:
                grouped[int(wave)].append(qid)
    for wave, ids in grouped.items():
        grouped[wave] = sorted(set(ids))
    return grouped


def stratified_sample(ids_by_wave: dict[int, list[str]], n: int, seed: int) -> list[str]:
    """The same count per wave. If n is not divisible by the number of waves, take n // n_waves
    per wave.
    """
    waves = list(ids_by_wave)
    if not waves:
        raise SystemExit("no waves")
    per = n // len(waves)
    if per < 1:
        raise SystemExit(f"n={n} divided across {len(waves)} waves is 0")
    if n % len(waves):
        log.warning(
            "n=%d is not divisible by %d waves. %d per wave, %d in total",
            n,
            len(waves),
            per,
            per * len(waves),
        )
    rng = random.Random(seed)
    picked: list[str] = []
    for wave in waves:
        pool = ids_by_wave[wave]
        if len(pool) < per:
            raise SystemExit(f"wave {wave}: first-pass decisions {len(pool)} < sample {per}")
        picked.extend(rng.sample(pool, per))
    return picked


def build_items(wanted: set[str]) -> list[dict[str, Any]]:
    """Items in the same shape as the `anno_llm_judge.py --bank` branch. Keep only wanted."""
    from anno_build_sample import FIELDS, WAVES, anchor_card, genre_lists, is_clean, load_bank

    items: list[dict[str, Any]] = []
    for wave in WAVES:
        for row in load_bank(wave):
            if not is_clean(row):
                continue
            qid = str(row["quant_id"])
            if qid not in wanted:
                continue
            yes, no = genre_lists(row["quant"])
            items.append(
                {
                    "item_id": row["quant_id"],
                    "wave": wave,
                    "anchors": anchor_card(row["quant"]),
                    "genres_attended": yes,
                    "genres_not": no,
                    "fields": [{"key": f.key, "label": f.label_ko} for f in FIELDS],
                    "narrative": {f.key: row["narrative"].get(f.key, "") for f in FIELDS},
                }
            )
    return items


def failure_row(item_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "item_id": item_id,
        "contradiction": None,
        "contra_fields": None,
        "model": MODEL,
        "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": RUN_TAG,
    }
    for scale in SCALES:
        row[scale] = None
    return row


def judge_item(item: dict[str, Any]) -> dict[str, Any]:
    """Up to 3 calls with the same prompt. If all three fail, contradiction is null."""
    parsed: dict[str, Any] | None = None
    for _ in range(N_RETRIES):
        parsed = ask(render(item))
        if parsed:
            break
    if parsed is None:
        log.warning("unparsed after retries: %s", item["item_id"])
        return failure_row(str(item["item_id"]))
    parsed.update(
        item_id=item["item_id"],
        model=MODEL,
        judged_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        run=RUN_TAG,
    )
    return parsed


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rate(numer: int, denom: int) -> float:
    if denom == 0:
        return float("nan")
    return numer / denom


def cohen_kappa(first: list[str], second: list[str]) -> float:
    """Binary Cohen κ. nan when expected agreement is 1."""
    n = len(first)
    if n == 0 or n != len(second):
        return float("nan")
    agree = sum(a == b for a, b in zip(first, second, strict=True))
    po = agree / n
    pe = 0.0
    for label in ("yes", "no"):
        pe += (sum(a == label for a in first) / n) * (sum(b == label for b in second) / n)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def _fields(row: dict[str, Any]) -> tuple[str, ...]:
    raw = row.get("contra_fields") or []
    return tuple(sorted(str(x) for x in raw))


def _scored(row: dict[str, Any] | None) -> bool:
    return row is not None and row.get("contradiction") in ("yes", "no")


def binary_stats(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, float | int]:
    first = [a["contradiction"] for a, _ in pairs]
    second = [b["contradiction"] for _, b in pairs]
    n = len(pairs)
    n_agree = sum(a == b for a, b in zip(first, second, strict=True))
    return {
        "n": n,
        "n_agree": n_agree,
        "agreement": _rate(n_agree, n),
        "kappa": cohen_kappa(first, second),
        "n_yes_first": sum(v == "yes" for v in first),
        "n_yes_repro": sum(v == "yes" for v in second),
        "yes_rate_first": _rate(sum(v == "yes" for v in first), n),
        "yes_rate_repro": _rate(sum(v == "yes" for v in second), n),
    }


def aggregate(
    first: dict[str, dict[str, Any]],
    repro: dict[str, dict[str, Any]],
    wave_of: dict[str, int],
    sample_ids: list[str],
) -> dict[str, Any]:
    """Pair first-pass decisions with rescores. Parse failures (contradiction null) are dropped
    from the agreement rate.
    """
    n_fail = 0
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pair_wave: list[int] = []
    for item_id in sample_ids:
        left = first.get(item_id)
        right = repro.get(item_id)
        if not _scored(right):
            n_fail += 1
            continue
        if not _scored(left):
            continue
        assert left is not None and right is not None
        pairs.append((left, right))
        pair_wave.append(wave_of[item_id])

    yes_pairs = [(a, b) for a, b in pairs if a["contradiction"] == "yes"]
    n_fields_exact = sum(_fields(a) == _fields(b) for a, b in yes_pairs)

    scales: list[dict[str, float | int | str]] = []
    for name in SCALES:
        exact = within = 0
        for left, right in pairs:
            gap = abs(int(left[name]) - int(right[name]))
            exact += int(gap == 0)
            within += int(gap <= 1)
        n = len(pairs)
        scales.append(
            {
                "name": name,
                "n": n,
                "n_exact": exact,
                "n_within1": within,
                "exact": _rate(exact, n),
                "within1": _rate(within, n),
            }
        )

    by_wave: dict[int, dict[str, float | int]] = {}
    for wave in sorted({wave_of[i] for i in sample_ids}):
        subset = [pair for pair, w in zip(pairs, pair_wave, strict=True) if w == wave]
        by_wave[wave] = binary_stats(subset)

    return {
        "n_sample": len(sample_ids),
        "n_fail": n_fail,
        "binary": binary_stats(pairs),
        "by_wave": by_wave,
        "fields_n": len(yes_pairs),
        "fields_exact": n_fields_exact,
        "fields_rate": _rate(n_fields_exact, len(yes_pairs)),
        "scales": scales,
    }


def _fmt(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def report_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        metric: str,
        wave: str,
        n: int,
        numer: int | None,
        value: float,
    ) -> None:
        rows.append(
            {
                "metric": metric,
                "wave": wave,
                "n": str(n),
                "numerator": "" if numer is None else str(numer),
                "value": _fmt(value),
            }
        )

    binary = summary["binary"]
    add("n_fail", "all", int(summary["n_sample"]), int(summary["n_fail"]), float(summary["n_fail"]))
    add(
        "contradiction_agreement",
        "all",
        int(binary["n"]),
        int(binary["n_agree"]),
        binary["agreement"],
    )
    add("cohen_kappa", "all", int(binary["n"]), None, binary["kappa"])
    add(
        "yes_rate_first",
        "all",
        int(binary["n"]),
        int(binary["n_yes_first"]),
        binary["yes_rate_first"],
    )
    add(
        "yes_rate_repro",
        "all",
        int(binary["n"]),
        int(binary["n_yes_repro"]),
        binary["yes_rate_repro"],
    )
    add(
        "contra_fields_exact",
        "all",
        int(summary["fields_n"]),
        int(summary["fields_exact"]),
        summary["fields_rate"],
    )
    for scale in summary["scales"]:
        add(f"{scale['name']}_exact", "all", int(scale["n"]), int(scale["n_exact"]), scale["exact"])
        add(
            f"{scale['name']}_within1",
            "all",
            int(scale["n"]),
            int(scale["n_within1"]),
            scale["within1"],
        )
    for wave, stats in summary["by_wave"].items():
        add(
            "contradiction_agreement",
            str(wave),
            int(stats["n"]),
            int(stats["n_agree"]),
            stats["agreement"],
        )
        add("cohen_kappa", str(wave), int(stats["n"]), None, stats["kappa"])
        add(
            "yes_rate_first",
            str(wave),
            int(stats["n"]),
            int(stats["n_yes_first"]),
            stats["yes_rate_first"],
        )
        add(
            "yes_rate_repro",
            str(wave),
            int(stats["n"]),
            int(stats["n_yes_repro"]),
            stats["yes_rate_repro"],
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "wave", "n", "numerator", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    if math.isnan(value):
        return "—"
    return f"{value:.1%}"


def _num(value: float) -> str:
    if math.isnan(value):
        return "—"
    return f"{value:.3f}"


def write_md(path: Path, summary: dict[str, Any], seed: int) -> None:
    binary = summary["binary"]
    lines = [
        "# 판정기 재현율",
        "",
        f"1차 판정 `{OUT_BANK.name}` 에서 파동별 같은 수로 뽑은 표본을",
        f"`{MODEL}` · temperature 0 · think false · num_predict 200 · 시드 {seed} 로 다시 채점했다.",
        "프롬프트와 호출은 `anno_llm_judge.render`·`ask` 이다.",
        "일치율은 재채점이 파싱된 쌍만 쓴다. 개인 서사는 싣지 않는다.",
        "",
        f"- 표본 {summary['n_sample']}명, 파싱 실패 {summary['n_fail']}명,",
        f"  비교 {binary['n']}쌍",
        f"- 모순 yes/no 일치 {_pct(float(binary['agreement']))} ({binary['n_agree']}/{binary['n']}),",
        f"  Cohen κ {_num(float(binary['kappa']))}",
        f"- 1차 yes {_pct(float(binary['yes_rate_first']))},",
        f"  재채점 yes {_pct(float(binary['yes_rate_repro']))}",
        f"- 1차 모순 yes {summary['fields_n']}명 중 `contra_fields` 집합 일치",
        f"  {_pct(float(summary['fields_rate']))} ({summary['fields_exact']}/{summary['fields_n']})",
        "",
        "## 척도",
        "",
        "| 척도 | 정확 일치 | ±1 이내 |",
        "| --- | --- | --- |",
    ]
    for scale in summary["scales"]:
        lines.append(
            f"| {scale['name']} | {_pct(float(scale['exact']))} "
            f"({scale['n_exact']}/{scale['n']}) | {_pct(float(scale['within1']))} "
            f"({scale['n_within1']}/{scale['n']}) |"
        )
    lines.extend(
        [
            "",
            "## 파동별 모순 판정",
            "",
            "| 파동 | n | 일치 | κ | 1차 yes | 재채점 yes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for wave, stats in summary["by_wave"].items():
        lines.append(
            f"| {wave} | {stats['n']} | {_pct(float(stats['agreement']))} | "
            f"{_num(float(stats['kappa']))} | {_pct(float(stats['yes_rate_first']))} | "
            f"{_pct(float(stats['yes_rate_repro']))} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def check_items(items: list[dict[str, Any]], wanted: set[str]) -> None:
    """Log only the item skeleton. Narrative text is not written to the log."""
    from anno_build_sample import FIELDS

    missing = wanted - {str(it["item_id"]) for it in items}
    if missing:
        raise SystemExit(f"failed to build items for {len(missing)} ids")
    if len(items) != len(wanted):
        raise SystemExit(f"items {len(items)} != sample {len(wanted)}")
    field_keys = [f.key for f in FIELDS]
    for item in items:
        if [f["key"] for f in item["fields"]] != field_keys:
            raise SystemExit(f"fields mismatch: {item['item_id']}")
        if set(item["narrative"]) != set(field_keys):
            raise SystemExit(f"narrative key mismatch: {item['item_id']}")
        if not item["anchors"]:
            raise SystemExit(f"no anchors: {item['item_id']}")
    prompt_len = len(render(items[0]))
    by_wave: dict[int, int] = {}
    for item in items:
        by_wave[int(item["wave"])] = by_wave.get(int(item["wave"]), 0) + 1
    log.info(
        "items %d (waves %s), fields %d, anchors %d, prompt %d chars",
        len(items),
        by_wave,
        len(field_keys),
        len(items[0]["anchors"]),
        prompt_len,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="bank judge reproducibility")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="sample size (equal per wave)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run", action="store_true", help="check the sample and items only; do not score"
    )
    args = parser.parse_args()

    bank_rows = index_by_id(read_jsonl(OUT_BANK))
    if not bank_rows:
        raise SystemExit(f"no first-pass decisions: {OUT_BANK}")
    ids_by_wave = judged_ids_by_wave(set(bank_rows))
    for wave, ids in ids_by_wave.items():
        log.info("first-pass decisions %d: %d", wave, len(ids))
    sample = stratified_sample(ids_by_wave, args.n, args.seed)
    wave_of = {qid: wave for wave, ids in ids_by_wave.items() for qid in ids}
    log.info("sample %d (seed %d)", len(sample), args.seed)

    items = build_items(set(sample))
    check_items(items, set(sample))
    if args.dry_run:
        log.info("dry-run: skipping scoring and reports")
        return 0

    done = index_by_id(read_jsonl(OUT_REPRO))
    todo = [it for it in items if str(it["item_id"]) not in done]
    log.info("model=%s todo=%d (already %d)", MODEL, len(todo), len(done))
    for n, item in enumerate(todo, 1):
        append_row(OUT_REPRO, judge_item(item))
        if n % 25 == 0:
            log.info("%d/%d", n, len(todo))
    log.info("wrote %s", OUT_REPRO)

    repro_rows = index_by_id(read_jsonl(OUT_REPRO))
    summary = aggregate(bank_rows, repro_rows, wave_of, sample)
    rows = report_rows(summary)
    write_csv(REPORT_CSV, rows)
    write_md(REPORT_MD, summary, args.seed)
    log.info(
        "compared %s pairs agreement %s κ %s failures %s",
        summary["binary"]["n"],
        _fmt(summary["binary"]["agreement"]),
        _fmt(summary["binary"]["kappa"]),
        summary["n_fail"],
    )
    log.info("wrote %s", REPORT_MD)
    log.info("wrote %s", REPORT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
