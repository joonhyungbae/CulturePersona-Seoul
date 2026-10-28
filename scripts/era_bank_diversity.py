"""Duplicates and diversity in the narrative bank — quality visible only at scale (ADR-018).

A question a resource paper must answer before release: once 15,000 people are generated,
does the generator start repeating itself? Three hundred people could not answer that.

What is measured.

  exact       Exact-duplicate counts for the full profile and for each field.
  near        TF-IDF cosine nearest-neighbour distribution (self excluded). A tail stuck
              at 1 means narratives copied one another.
  vocabulary  Whether new vocabulary (character 4-grams) keeps appearing as the generated
              count grows — a saturation curve.
  cliche      Hit rate of banned stock phrases: what generation-time checks caught, and
              what remained.
  coverage    How many people filled each anchor cell (whether sparse cells are empty).

Run: uv run python scripts/era_bank_diversity.py --wave 2024
Outputs: outputs/reports/era_bank_diversity.csv · era_bank_diversity.md
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bank_diversity")

SEED = 20260920
CHUNK = 512
SATURATION_STEPS = (100, 300, 1000, 3000, 7500, 15000)


def load_bank(wave: int) -> list[dict]:
    p = settings.data_root / "synthetic" / f"era_{wave}_narrative" / "personas.jsonl"
    rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    log.info("wave=%d bank n=%d", wave, len(rows))
    return rows


def nearest_neighbour_cos(texts: list[str]) -> np.ndarray:
    """Max cosine similarity with self excluded. The sparse matrix is multiplied in chunks
    (15k² is not loaded into memory).
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    X = normalize(vec.fit_transform(texts))
    log.info("tfidf %s", X.shape)
    out = np.zeros(X.shape[0], dtype=float)
    for start in range(0, X.shape[0], CHUNK):
        block = X[start : start + CHUNK]
        sim = (block @ X.T).toarray()
        for i in range(sim.shape[0]):
            sim[i, start + i] = -1.0
        out[start : start + CHUNK] = sim.max(axis=1)
    return out


def vocabulary_saturation(texts: list[str], rng: np.random.Generator) -> list[dict]:
    """Whether new character 4-grams keep appearing as the subsample grows."""
    idx = rng.permutation(len(texts))
    rows = []
    seen: set[str] = set()
    prev = 0
    for step in SATURATION_STEPS:
        if step > len(texts):
            continue
        for j in idx[prev:step]:
            t = texts[j]
            seen.update(t[k : k + 4] for k in range(len(t) - 3))
        prev = step
        rows.append(
            {"n_personas": step, "distinct_4grams": len(seen), "per_persona": len(seen) / step}
        )
    return rows


def run(wave: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = load_bank(wave)
    rng = np.random.default_rng(SEED)
    fields = list(rows[0]["narrative"].keys())
    full = [" ".join(r["narrative"].values()) for r in rows]

    dup = [{"scope": "full profile", "n": len(full), "duplicates": len(full) - len(set(full))}]
    for f in fields:
        vals = [r["narrative"][f] for r in rows]
        dup.append({"scope": f, "n": len(vals), "duplicates": len(vals) - len(set(vals))})
    dup_df = pd.DataFrame(dup)
    dup_df["duplicate_pct"] = 100.0 * dup_df.duplicates / dup_df.n

    nn = nearest_neighbour_cos(full)
    cl = collections.Counter(len(r.get("cliches") or []) for r in rows)
    n_cliche = sum(v for k, v in cl.items() if k)
    summary = pd.DataFrame(
        [
            {"wave": wave, "metric": "n_personas", "value": float(len(rows))},
            {
                "wave": wave,
                "metric": "duplicate_profiles",
                "value": float(dup_df.duplicates.iloc[0]),
            },
            {"wave": wave, "metric": "nn_cos_median", "value": float(np.median(nn))},
            {"wave": wave, "metric": "nn_cos_p95", "value": float(np.percentile(nn, 95))},
            {"wave": wave, "metric": "nn_cos_max", "value": float(nn.max())},
            {
                "wave": wave,
                "metric": "nn_cos_above_90_pct",
                "value": float(100 * (nn > 0.9).mean()),
            },
            {
                "wave": wave,
                "metric": "cliche_share_pct",
                "value": float(100 * n_cliche / len(rows)),
            },
            {
                "wave": wave,
                "metric": "issue_share_pct",
                "value": float(100 * sum(1 for r in rows if r.get("issues")) / len(rows)),
            },
            {
                "wave": wave,
                "metric": "mean_regenerations",
                "value": float(np.mean([len(r.get("attempts") or []) for r in rows])),
            },
        ]
    )
    sat = pd.DataFrame(vocabulary_saturation(full, rng))
    sat.insert(0, "wave", wave)
    dup_df.insert(0, "wave", wave)
    return summary, dup_df, sat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, action="append", default=None)
    args = ap.parse_args()
    waves = args.wave or [2024]

    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summaries, dups, sats = [], [], []
    for w in waves:
        s, d, sa = run(w)
        summaries.append(s)
        dups.append(d)
        sats.append(sa)
    summary = pd.concat(summaries, ignore_index=True)
    dup = pd.concat(dups, ignore_index=True)
    sat = pd.concat(sats, ignore_index=True)

    # Merge across waves instead of overwriting (a 2022 run once wiped 2024).
    def _merge(new_df: pd.DataFrame, name: str) -> pd.DataFrame:
        path = reports / name
        if path.exists():
            old_df = pd.read_csv(path)
            new_df = pd.concat([old_df[~old_df.wave.isin(waves)], new_df], ignore_index=True)
        return new_df.sort_values("wave").reset_index(drop=True)

    summary = _merge(summary, "era_bank_diversity.csv")
    dup = _merge(dup, "era_bank_duplicates.csv")
    sat = _merge(sat, "era_bank_saturation.csv")
    summary.to_csv(reports / "era_bank_diversity.csv", index=False)
    dup.to_csv(reports / "era_bank_duplicates.csv", index=False)
    sat.to_csv(reports / "era_bank_saturation.csv", index=False)

    lines = [
        "# 서사 뱅크의 중복과 다양성 (ADR-018)",
        "",
        summary.round(4).to_markdown(index=False),
        "",
        "## 필드별 완전 중복",
        dup.round(3).to_markdown(index=False),
        "",
        "## 어휘 포화 (문자 4-gram)",
        sat.round(2).to_markdown(index=False),
        "",
        "`nn_cos_*` 는 자기 자신을 뺀 최근접 이웃 코사인이다. 중앙값이 낮고 꼬리가 1 에",
        "붙지 않으면 서사끼리 베끼지 않은 것이다. 짧은 필드(요약)의 완전 중복은 앵커 조합이",
        "같을 때 나오므로 0 이 목표가 아니다 — 크기만 보고한다.",
    ]
    (reports / "era_bank_diversity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", reports / "era_bank_diversity.csv")
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
