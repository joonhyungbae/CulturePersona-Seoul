"""Narrative-layer probes — can anchors be recovered from the narrative (ADR-018).

This fills the empty `tab:probes` in the manuscript. A 300-person eval bank
could not be split into train and test; that became possible once the 2024
full bank (15,000) was finished.

Two conditions.

  raw     Full narrative. Because the narrative **states anchors in sentences**,
          most scores are near 1.0. That is a property of the resource, not a
          finding — it is the baseline.
  masked  After deleting **every surface form of an anchor value** and genre
          names, replacing them with [MASK]. Recovery is then from the remaining
          prose. Recovery after deletion means the narrative carried the anchor
          in the prose; failure means it only copied the value.

The masking vocabulary is **built from the data**: every anchor-value string
that actually appears in the bank, plus schema genre labels, plus sex synonyms.
It is not a hand-picked list, so it runs unchanged on other waves.

The table is also evidence for attribute inference in the ethics section.
Income bands are never stated, yet they leak from spending and barrier prose.

Run: uv run python scripts/era_probes.py --wave 2024
Outputs: outputs/reports/era_probes.csv · era_probes.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.schema import ANCHORS, ERA_CORE_ANCHORS, GENRE_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_probes")

SEED = 20260920
TEST_SIZE = 0.25
# Leaving a mask token makes "how often it was masked" a feature by itself
# (genre-mention count → attendance). Delete to whitespace instead of a token.
MASK = " "

# Probe targets — era core anchors minus the 10 genre flags. Genres are pooled as attendance.
TARGETS = [a for a in ERA_CORE_ANCHORS if a not in GENRE_ANCHORS]

# Sex is also written these ways, besides the values ("남성"/"여성"). Genre labels come from the schema.
SEX_SYNONYMS = ["남자", "여자", "남성", "여성"]


def genre_terms() -> list[str]:
    """Surface forms actually used in narratives, taken from genre labels ('미술관·전시회 관람' → 미술관, 전시회)."""
    out: list[str] = []
    for spec in ANCHORS:
        if spec.name not in GENRE_ANCHORS:
            continue
        lab = spec.label_ko.replace(" 관람", "")
        lab = re.sub(r"\(.*?\)", "", lab)
        for part in re.split(r"[·,]", lab):
            part = part.strip()
            if part:
                out.append(part)
    out += ["뮤지컬", "영화", "공연", "전시"]
    return out


def mask_terms(rows: list[dict]) -> list[str]:
    """Every anchor-value string in the bank, plus genre surface forms, plus sex synonyms."""
    vals: set[str] = set()
    for r in rows:
        for v in r["quant"].values():
            if isinstance(v, str) and len(v) >= 2 and v not in ("예", "아니오"):
                vals.add(v)
    terms = sorted(vals | set(genre_terms()) | set(SEX_SYNONYMS), key=len, reverse=True)
    return terms


def apply_mask(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(MASK, text)


def load_bank(wave: int) -> list[dict]:
    p = settings.data_root / "synthetic" / f"era_{wave}_narrative" / "personas.jsonl"
    rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    log.info("wave=%d bank n=%d", wave, len(rows))
    return rows


def fit_probe(X, y: np.ndarray, seed: int = SEED) -> dict[str, float]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    maj = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    strat = DummyClassifier(strategy="stratified", random_state=seed).fit(Xtr, ytr)
    return {
        "accuracy": float((pred == yte).mean()),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "majority": float(maj.score(Xte, yte)),
        "stratified": float(strat.score(Xte, yte)),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
    }


def run(wave: int, min_count: int = 30) -> pd.DataFrame:
    rows = load_bank(wave)
    texts_raw = [" ".join(r["narrative"].values()) for r in rows]
    terms = mask_terms(rows)
    pattern = re.compile("|".join(re.escape(t) for t in terms))
    texts_masked = [apply_mask(t, pattern) for t in texts_raw]
    log.info(
        "mask terms=%d, chars kept %.1f%%",
        len(terms),
        100 * np.mean([len(m) / len(t) for m, t in zip(texts_masked, texts_raw, strict=True)]),
    )

    out = []
    for cond, texts in (("raw", texts_raw), ("masked", texts_masked)):
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=5,
            max_features=200_000,
            sublinear_tf=True,
        )
        X = vec.fit_transform(texts)
        log.info("condition=%s tfidf=%s", cond, X.shape)
        for target in TARGETS:
            y = np.array([str(r["quant"].get(target)) for r in rows])
            counts = pd.Series(y).value_counts()
            keep = counts[counts >= min_count].index
            mask = np.isin(y, keep)
            if mask.sum() < min_count * 2 or len(keep) < 2:
                log.warning("skip %s (classes=%d)", target, len(keep))
                continue
            m = fit_probe(X[mask], y[mask])
            m.update(wave=wave, condition=cond, target=target, n_classes=len(keep))
            out.append(m)
            log.info(
                "%-24s %-6s acc=%.3f f1=%.3f maj=%.3f",
                target,
                cond,
                m["accuracy"],
                m["macro_f1"],
                m["majority"],
            )
    cols = [
        "wave",
        "condition",
        "target",
        "n_classes",
        "accuracy",
        "macro_f1",
        "majority",
        "stratified",
        "n_train",
        "n_test",
    ]
    return pd.DataFrame(out)[cols]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, action="append", default=None)
    args = ap.parse_args()
    waves = args.wave or [2024]

    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    res = pd.concat([run(w) for w in waves], ignore_index=True)

    path = reports / "era_probes.csv"
    if path.exists():
        old = pd.read_csv(path)
        res = pd.concat([old[~old.wave.isin(waves)], res], ignore_index=True)
    res.sort_values(["wave", "target", "condition"]).to_csv(path, index=False)

    wide = res.pivot_table(
        index=["wave", "target", "n_classes", "majority"], columns="condition", values="accuracy"
    ).reset_index()
    lines = [
        "# 서사층 프로브 (ADR-018)",
        "",
        "서사에서 앵커를 되찾는 문자 n-gram TF-IDF 분류기. `raw` 는 서사 전문,",
        "`masked` 는 앵커 값 표면형과 장르 이름을 지운 뒤. 마스킹 어휘는 데이터에서 만든다.",
        "",
        wide.round(3).to_markdown(index=False),
        "",
        "`raw` 가 높은 것은 발견이 아니라 자원의 성질이다 (서사가 앵커를 문장으로 말한다).",
        "읽어야 할 것은 **masked 가 다수결 위인지** — 그만큼이 산문에 실려 있는 정보다.",
        "명시되지 않는 앵커(소득·가구원 수)의 masked 값은 윤리 절의 속성추론 근거이기도 하다.",
    ]
    (reports / "era_probes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    print(wide.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
