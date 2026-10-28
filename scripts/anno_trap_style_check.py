"""Trap v2 edit-trace check — does the inserted field show up stylistically
(preregistration v2 §6.1, condition AUC < 0.60).

Compare within the same field: the new field inserted into the trap (1) versus the trap owner's
original field (0). Character 2–4 gram TF-IDF + logistic regression, 5-fold CV AUC split by
owner. Length is reported separately (a new field that is conspicuously longer or shorter is a
cue). A high AUC here means a rater could spot the trap as "only this paragraph was written
differently."

Run: uv run python scripts/anno_trap_style_check.py
→ outputs/reports/anno_trap_style.md (aggregates only)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_trap_style")

AUC_LIMIT = 0.60


def main() -> int:
    rows = [
        json.loads(line)
        for line in (settings.outputs_dir / "anno" / "traps_v2.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    texts = [r["new_field"] for r in rows] + [r["original_field"] for r in rows]
    y = np.array([1] * len(rows) + [0] * len(rows))
    groups = np.array(list(range(len(rows))) * 2)  # both fields of one owner share a fold
    scores = np.zeros(len(texts))
    for tr, te in GroupKFold(n_splits=5).split(texts, y, groups):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        xtr = vec.fit_transform([texts[i] for i in tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(xtr, y[tr])
        scores[te] = clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
    auc = roc_auc_score(y, scores)
    new_len = np.array([len(r["new_field"]) for r in rows])
    old_len = np.array([len(r["original_field"]) for r in rows])
    ratio = new_len / np.maximum(old_len, 1)
    auc_len = roc_auc_score(y, np.concatenate([new_len, old_len]))
    by_tier = {
        t: int(sum(1 for r in rows if r["tier"] == t)) for t in sorted({r["tier"] for r in rows})
    }
    detected = sum(1 for r in rows if r["validator_checks"])
    lines = [
        "# 함정 v2 편집 흔적 검사",
        "",
        f"- 함정 {len(rows)} 개 (층 {by_tier}) · 생성 시점 검증기로 다시 채점해 적발 {detected}",
        (
            f"- 문자 n-gram 분류기 AUC (끼운 칸 대 원래 칸, 주인 단위 5 겹): **{auc:.3f}** "
            f"(조건 < {AUC_LIMIT}: {'통과' if auc < AUC_LIMIT else '실패'})"
        ),
        (
            f"- 길이만으로 가르는 AUC: {auc_len:.3f} · 새 칸/원래 칸 길이 비 중앙값 "
            f"{np.median(ratio):.2f} (사분위 {np.percentile(ratio, 25):.2f}–{np.percentile(ratio, 75):.2f})"
        ),
    ]
    # Content-matched comparison: inserted field vs the same field of clean personas that
    # **actually hold** the replaced anchor values.
    # The comparison above (original field) also picks up content differences, because the
    # inserted field must state different facts (e.g. retirement) — use this one for style only.
    bank = []
    for w in (2020, 2022, 2024):
        p_ = settings.data_root / "synthetic" / f"era_{w}_narrative" / "personas.jsonl"
        bank += [json.loads(x) for x in p_.open(encoding="utf-8") if x.strip()]
    bank = [b for b in bank if b.get("ok") and not b.get("issues")]
    rng = np.random.default_rng(0)
    texts_m, y_m, g_m = [], [], []
    for gi, r in enumerate(rows):
        match = [
            b
            for b in bank
            if all(b["quant"].get(k) == v for k, v in r["to"].items())
            and b["quant_id"] != r["quant_id"]
        ]
        pick = (
            [match[i] for i in rng.choice(len(match), size=min(3, len(match)), replace=False)]
            if match
            else []
        )
        texts_m.append(r["new_field"])
        y_m.append(1)
        g_m.append(gi)
        for b in pick:
            texts_m.append(b["narrative"][r["field"]])
            y_m.append(0)
            g_m.append(gi)
    y_m, g_m = np.array(y_m), np.array(g_m)
    sc_m = np.zeros(len(texts_m))
    for tr, te in GroupKFold(n_splits=5).split(texts_m, y_m, g_m):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            vec.fit_transform([texts_m[i] for i in tr]), y_m[tr]
        )
        sc_m[te] = clf.predict_proba(vec.transform([texts_m[i] for i in te]))[:, 1]
    auc_m = roc_auc_score(y_m, sc_m)
    lines.append(
        f"- 내용을 맞춘 비교 (끼운 칸 대 바꾼 앵커 값을 실제로 가진 페르소나의 같은 칸, 함정당 최대 3): "
        f"AUC **{auc_m:.3f}** (조건 < {AUC_LIMIT}: {'통과' if auc_m < AUC_LIMIT else '실패'})"
    )
    lines.append(
        "- 해석: 원래 칸과의 비교는 끼운 칸이 다른 사실을 말해야 하므로 내용 차이를 함께 잡는다. "
        "문체 흔적은 내용을 맞춘 비교로, 평가자가 알아챌 수 있는지는 아래 문항 단위 비교로 본다."
    )
    # Rater's view: can trap items be separated from clean items by narrative alone, with no
    # anchors (masked narrative = stage 1, source text = stage 2).
    import pandas as pd

    key = pd.read_csv(settings.outputs_dir / "anno" / "answer_key.csv", dtype=str)
    items = json.loads(
        (ROOT / "annotation" / "data_private" / "items_full.json").read_text("utf-8")
    )
    by_id = {i["item_id"]: i for i in items}
    sub = key[key.stratum.isin(["clean", "trap"])]
    y2 = (sub.stratum == "trap").to_numpy(int)
    for view in ("masked", "narrative"):
        docs = [" ".join(by_id[i][view].values()) for i in sub.item_id]
        sc = np.zeros(len(docs))
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(docs, y2):
            vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True
            )
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                vec.fit_transform([docs[i] for i in tr]), y2[tr]
            )
            sc[te] = clf.predict_proba(vec.transform([docs[i] for i in te]))[:, 1]
        lines.append(
            f"- 서사만으로 함정 대 clean 가르기 ({'가린 서사, 1 단계' if view == 'masked' else '원문, 앵커 없이'}): "
            f"AUC {roc_auc_score(y2, sc):.3f} (함정 {int(y2.sum())} · clean {int((1 - y2).sum())})"
        )
    out = settings.outputs_dir / "reports" / "anno_trap_style.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
