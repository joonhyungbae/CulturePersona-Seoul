"""Narrative augmentation B — an anchor–narrative consistency checker trained on
counterfactual pairs (2026-09-24, ideation B).

Information outside the anchors cannot be added, but "a narrative for these anchors" and
"a narrative for other anchors" can be paired without limit. A checker trained on those
pairs to answer "does this field contradict these anchors" is compared with the regex
validator (3 of 48 traps) and the LLM judge (gemma4:12b, 27/48, specificity 99.6%) on
whether it catches the human-audit traps better. The test set is the audit items
(clean 243 · trap 48) and does not overlap the personas used for training.

Training pairs (2024 bank, ok · no issues · audit items excluded):
  positive (match)      (p's anchor card, p's field f)
  negative (contradict) (p's anchor card, q's field f) — q differs from p on at least one
                         required anchor of f (half are hard: exactly one differs)
Features: BGE-m3-ko embeddings [a, f, a⊙f, |a−f|] → logistic regression (persona-level
8:2 split).
Test: score each of the item's 12 fields; item score = maximum. AUC (trap vs clean),
sensitivity at clean specificity 0.99, and the trap-field localization rate.

Run: python scripts/era_consistency_checker.py   (sentence-transformers · CUDA environment)
Outputs: outputs/reports/era_consistency_checker.csv · .md (aggregates only)
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfac_persona.config import settings
from sfac_persona.narrative import FIELDS
from sfac_persona.schema import ANCHOR_BY_NAME, ERA_CORE_ANCHORS, GENRE_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("consistency_checker")

SEED = 20260924
MODEL = "dragonkue/BGE-m3-ko"
N_PER_FIELD = 1500
WAVE = 2024


def anchor_card(q: dict) -> str:
    parts = []
    for a in ERA_CORE_ANCHORS:
        if a in GENRE_ANCHORS or a not in q:
            continue
        spec = ANCHOR_BY_NAME.get(a)
        parts.append(f"{spec.label_ko if spec else a}: {q[a]}")

    def gname(g: str) -> str:
        return ANCHOR_BY_NAME[g].label_ko.replace(" 관람", "")

    yes = [gname(g) for g in GENRE_ANCHORS if str(q.get(g)) == "예"]
    no = [gname(g) for g in GENRE_ANCHORS if str(q.get(g)) == "아니오"]
    parts.append("지난 1년 관람 장르: " + (", ".join(yes) if yes else "없음"))
    parts.append("관람하지 않은 장르: " + (", ".join(no) if no else "없음"))
    return "; ".join(parts)


def item_card(item: dict) -> str:
    parts = [f"{a['label']}: {a['value']}" for a in item["anchors"]]
    parts.append("지난 1년 관람 장르: " + (", ".join(item["genres_attended"]) or "없음"))
    parts.append("관람하지 않은 장르: " + (", ".join(item["genres_not"]) or "없음"))
    return "; ".join(parts)


def build_pairs(bank: list[dict], rng: random.Random) -> pd.DataFrame:
    by_field_anchors = {f.key: [a for a in f.anchors if a in bank[0]["quant"]] for f in FIELDS}
    rows = []
    for f in FIELDS:
        req = by_field_anchors[f.key]
        cands = [p for p in bank if p["narrative"].get(f.key)]
        rng.shuffle(cands)
        pos = cands[:N_PER_FIELD]
        for p in pos:
            rows.append(
                {
                    "field": f.key,
                    "pid": p["quant_id"],
                    "card": anchor_card(p["quant"]),
                    "text": p["narrative"][f.key],
                    "y": 0,
                    "hard": 0,
                }
            )
        # Negative: the same field from a donor that differs on a required anchor.
        made = 0
        tries = 0
        while made < N_PER_FIELD and tries < 50 * N_PER_FIELD:
            tries += 1
            p, q = cands[rng.randrange(len(cands))], cands[rng.randrange(len(cands))]
            if p["quant_id"] == q["quant_id"]:
                continue
            diff = [a for a in req if str(p["quant"].get(a)) != str(q["quant"].get(a))]
            if not diff:
                continue
            want_hard = made % 2 == 0
            if want_hard and len(diff) != 1:
                continue
            rows.append(
                {
                    "field": f.key,
                    "pid": p["quant_id"],
                    "card": anchor_card(p["quant"]),
                    "text": q["narrative"][f.key],
                    "y": 1,
                    "hard": int(len(diff) == 1),
                }
            )
            made += 1
    return pd.DataFrame(rows)


def embed(model, texts: list[str]) -> np.ndarray:
    uniq = list(dict.fromkeys(texts))
    e = model.encode(uniq, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    idx = {t: i for i, t in enumerate(uniq)}
    return e[[idx[t] for t in texts]]


def features(a: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.hstack([a, f, a * f, np.abs(a - f)])


def main() -> int:
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = random.Random(SEED)
    key = pd.read_csv(settings.outputs_dir / "anno" / "answer_key.csv", dtype=str)
    audit_qids = set(key.quant_id.dropna()) | set(key.donor_quant_id.dropna())
    bank = [
        json.loads(l)
        for l in (
            settings.data_root / "synthetic" / f"era_{WAVE}_narrative" / "personas.jsonl"
        ).open(encoding="utf-8")
        if l.strip()
    ]
    bank = [
        b
        for b in bank
        if b.get("ok") and not b.get("issues") and str(b.get("quant_id")) not in audit_qids
    ]
    log.info("bank personas for training: %d", len(bank))
    pairs = build_pairs(bank, rng)
    log.info("pairs: %d (neg %d, hard %d)", len(pairs), int(pairs.y.sum()), int(pairs.hard.sum()))

    model = SentenceTransformer(MODEL)
    ea = embed(model, pairs.card.tolist())
    ef = embed(model, pairs.text.tolist())
    X = features(ea, ef)
    y = pairs.y.to_numpy()
    pids = pairs.pid.to_numpy()
    uniq = sorted(set(pids))
    rng.shuffle(uniq)
    val_ids = set(uniq[: len(uniq) // 5])
    va = np.array([p in val_ids for p in pids])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[~va], y[~va])
    val_auc = roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1])
    per_field = {}
    for f in FIELDS:
        m = va & (pairs.field.to_numpy() == f.key)
        if m.sum() > 20 and len(set(y[m])) == 2:
            per_field[f.key] = float(roc_auc_score(y[m], clf.predict_proba(X[m])[:, 1]))
    log.info("synthetic held-out AUC %.3f", val_auc)

    # Test: audit items.
    items = {
        i["item_id"]: i
        for i in json.loads(
            (ROOT / "annotation" / "data_private" / "items_full.json").read_text(encoding="utf-8")
        )
    }
    test = key[key.stratum.isin(["clean", "trap", "trap_gross", "flagged"])]
    cards, texts, meta = [], [], []
    for r in test.itertuples():
        it = items.get(r.item_id)
        if not it:
            continue
        card = item_card(it)
        for f in FIELDS:
            t = it["narrative"].get(f.key)
            if not t:
                continue
            cards.append(card)
            texts.append(t)
            meta.append((r.item_id, r.stratum, f.key, r.trap_field))
    Xt = features(embed(model, cards), embed(model, texts))
    sc = clf.predict_proba(Xt)[:, 1]
    mdf = pd.DataFrame(meta, columns=["item_id", "stratum", "field", "trap_field"]).assign(score=sc)
    item = (
        mdf.groupby(["item_id", "stratum"])
        .agg(
            score=("score", "max"),
            top_field=("score", lambda s: mdf.loc[s.idxmax(), "field"]),
            trap_field=("trap_field", "first"),
        )
        .reset_index()
    )
    clean = item[item.stratum == "clean"]
    trap = item[item.stratum == "trap"]
    gross = item[item.stratum == "trap_gross"]
    flagged = item[item.stratum == "flagged"]
    auc = roc_auc_score([0] * len(clean) + [1] * len(trap), np.r_[clean.score, trap.score])
    thr99 = float(np.quantile(clean.score, 0.99))
    thr95 = float(np.quantile(clean.score, 0.95))
    sens99 = float((trap.score > thr99).mean())
    sens95 = float((trap.score > thr95).mean())
    loc = float((trap.top_field == trap.trap_field).mean())
    gross_sens = float((gross.score > thr99).mean()) if len(gross) else float("nan")
    flagged_rate = float((flagged.score > thr99).mean()) if len(flagged) else float("nan")
    # By trap tier.
    tiers = key.set_index("item_id").tier
    trap = trap.assign(tier=trap.item_id.map(tiers))
    tier_sens = trap.groupby("tier").apply(lambda g: float((g.score > thr99).mean())).to_dict()
    rows = {
        "synthetic_val_auc": val_auc,
        "audit_auc_trap_vs_clean": auc,
        "sens_at_spec99": sens99,
        "sens_at_spec95": sens95,
        "localization": loc,
        "gross_sens_at_spec99": gross_sens,
        "flagged_rate_at_spec99": flagged_rate,
        "n_clean": len(clean),
        "n_trap": len(trap),
        "n_pairs": len(pairs),
        "baseline_validator_sens": 3 / 48,
        "baseline_judge_sens": 27 / 48,
        "baseline_judge_spec": 1 - 0.004,
        **{f"tier_sens_{k}": v for k, v in tier_sens.items()},
        **{f"field_auc_{k}": v for k, v in per_field.items()},
    }
    out = settings.outputs_dir / "reports"
    pd.DataFrame([rows]).to_csv(out / "era_consistency_checker.csv", index=False)
    lines = [
        "# 서사 증강 B — 반사실 쌍으로 학습한 앵커–서사 일관성 검사기",
        "",
        f"> 자동 생성: `scripts/era_consistency_checker.py` (학습 쌍 {len(pairs):,} · 페르소나 {len(bank):,} · {MODEL} + 로지스틱).",
        "",
        f"- 합성 held-out AUC (페르소나 분할): **{val_auc:.3f}**",
        f"- 검수 문항 trap(48) vs clean(243) AUC: **{auc:.3f}**",
        f"- clean 특이도 0.99 에서 함정 민감도: **{sens99:.2f}** (0.95 에서 {sens95:.2f}) — 기준선: 정규식 검증기 {3 / 48:.2f}, LLM 판정기 {27 / 48:.2f} (특이도 0.996)",
        f"- 함정 칸 특정률(최고 점수 칸 = 심은 칸): {loc:.2f} (LLM 판정기가 칸을 맞힌 비율 {27 / 48:.2f})",
        f"- 층별 민감도(특이도 0.99): {', '.join(f'{k} {v:.2f}' for k, v in sorted(tier_sens.items()))}",
        f"- 조작 확인(통째 교체) 6 건 민감도: {gross_sens:.2f} · flagged 46 건 중 경보율: {flagged_rate:.2f}",
        "",
        "칸별 합성 AUC: " + ", ".join(f"{k} {v:.2f}" for k, v in per_field.items()),
    ]
    (out / "era_consistency_checker.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
