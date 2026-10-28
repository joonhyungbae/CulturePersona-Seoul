"""Coding real open-ended answers with a classifier trained on synthetic narrative text only.

A task the anchors cannot do: the survey's open item (what Seoul needs to become a cultural city) must be coded
into themes by hand or by a model. The 2024 bank carries 15,000 synthetic answers to that item (the field
written from a theme drawn from T7, era_theme_v2.py), each labelled with its theme. This script trains a theme
classifier on those synthetic texts alone and applies it to the real answers, whose labels are EXAONE 4.0 32B's
coding under the same codebook (the reference used throughout Appendix open-text; not human labels).

Design fixed before running (2026-10-09):
  features    sentence embeddings (dragonkue/BGE-m3-ko, the SSR embedding model) + multinomial logistic
              regression (C=1); a character 2-4-gram TF-IDF + logistic regression as a second feature set
  test set    real answers of split B (never used to build the codebook or the theme table), all eight labels;
              split-A answers serve only as the train-on-real reference
  conditions  majority label; zero-shot nearest codebook description; train on synthetic (15,000) -> B;
              train on real A -> B (reference); train on synthetic + real A -> B (augmentation)
  primary     accuracy on B of the synthetic-trained embedding classifier minus the majority label, 95% interval
              from a bootstrap over B answers (2,000 draws)
  secondary   macro-F1; the gap to the train-on-real reference; the seven substantive themes only (the sampler
              never draws "other", so synthetic training cannot predict it)
Every result is reported, whatever its sign.

Stages: data (needs the microdata) -> embed (CPU, separate embedding environment) -> eval
Run: uv run python scripts/era_theme_coding.py data && uv run python scripts/era_theme_coding.py embed
     && uv run python scripts/era_theme_coding.py eval
Output: outputs/reports/era_theme_coding.csv (aggregates only; texts and labels stay in the local cache)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("theme_coding")

CACHE = settings.outputs_dir / "logs" / "theme_cache"
DATA = CACHE / "coding_data.parquet"
EMB = CACHE / "coding_emb.npy"
CODEBOOK = settings.grounding_dir / "era_2024" / "T7_theme_codebook.csv"
WISH = settings.synthetic_dir / "era_2024_narrative" / "wish_for_seoul.jsonl"
EMBED_MODEL = "dragonkue/BGE-m3-ko"
OTHER = 7
N_BOOT = 2000
SEED = 20261009


def stage_data() -> None:
    import era_theme_v2 as T

    real = T.real_frame()
    codes = T.code_all("real", {})
    real["label"] = real.row_uid.astype(str).map(codes).fillna(-1).astype(int)
    real = real[real.label >= 0]
    rows = [
        {
            "source": "real",
            "id": str(r.row_uid),
            "text": r.text,
            "label": int(r.label),
            "split": r.split,
        }
        for r in real.itertuples()
    ]
    for line in WISH.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("text") and r.get("wish_theme") is not None:
            rows.append(
                {
                    "source": "syn",
                    "id": r["persona_id"],
                    "text": r["text"],
                    "label": int(r["wish_theme"]),
                    "split": "train",
                }
            )
    cb = pd.read_csv(CODEBOOK)
    for r in cb.itertuples():
        rows.append(
            {
                "source": "codebook",
                "id": str(r.theme),
                "text": f"{r.name}: {r.description}",
                "label": int(r.theme),
                "split": "desc",
            }
        )
    d = pd.DataFrame(rows)
    d.to_parquet(DATA, index=False)
    log.info("data: %s", d.groupby(["source", "split"]).size().to_dict())


def stage_embed() -> None:
    py = ROOT / ".venv-embed" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    texts = CACHE / "coding_texts.json"
    texts.write_text(
        json.dumps(pd.read_parquet(DATA).text.tolist(), ensure_ascii=False), encoding="utf-8"
    )
    code = f"""
import json, numpy as np
from sentence_transformers import SentenceTransformer
t = json.load(open({str(texts)!r}, encoding="utf-8"))
m = SentenceTransformer({EMBED_MODEL!r}, device="cpu")
e = m.encode(t, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
np.save({str(EMB)!r}, e.astype(np.float32))
print("embedded", e.shape)
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    subprocess.check_call([str(py), "-c", code], cwd=str(ROOT), env=env)


def _fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(xtr, ytr)
    return clf.predict(xte)


def _tfidf(train_texts: list[str], test_texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    return v.fit_transform(train_texts), v.transform(test_texts)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score

    return {"accuracy": float((y == p).mean()), "macro_f1": float(f1_score(y, p, average="macro"))}


def stage_eval() -> None:
    d = pd.read_parquet(DATA)
    e = np.load(EMB)
    assert len(d) == len(e)
    syn = d.source == "syn"
    a = (d.source == "real") & (d.split == "A")
    b = (d.source == "real") & (d.split == "B")
    cb = d.source == "codebook"
    yb = d.label[b].to_numpy()
    rows: list[dict] = []
    preds: dict[str, np.ndarray] = {}

    majority = int(pd.Series(d.label[syn]).mode().iloc[0])
    preds["majority"] = np.full(len(yb), majority)
    # Zero-shot: nearest codebook description in embedding space.
    sims = e[b.to_numpy()] @ e[cb.to_numpy()].T
    preds["zero_shot_codebook"] = d.label[cb].to_numpy()[sims.argmax(axis=1)]
    for feat in ("embedding", "tfidf"):
        for cond, tr in (("synthetic", syn), ("real_A", a), ("synthetic+real_A", syn | a)):
            if feat == "embedding":
                xtr, xte = e[tr.to_numpy()], e[b.to_numpy()]
            else:
                xtr, xte = _tfidf(d.text[tr].tolist(), d.text[b].tolist())
            preds[f"{cond}|{feat}"] = _fit_predict(xtr, d.label[tr].to_numpy(), xte)
            log.info("%s %s done", cond, feat)

    seven = yb != OTHER
    for name, p in preds.items():
        m = _metrics(yb, p)
        m7 = _metrics(yb[seven], p[seven])
        rows.append(
            {
                "record": "level",
                "condition": name,
                "subset": "all8",
                "n": len(yb),
                **m,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
            }
        )
        rows.append(
            {
                "record": "level",
                "condition": name,
                "subset": "seven",
                "n": int(seven.sum()),
                **m7,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
            }
        )

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(yb))
    for cond, ref in (
        ("synthetic|embedding", "majority"),
        ("synthetic|embedding", "zero_shot_codebook"),
        ("synthetic|embedding", "real_A|embedding"),
        ("synthetic|tfidf", "majority"),
    ):
        hit_c = (preds[cond] == yb).astype(float)
        hit_r = (preds[ref] == yb).astype(float)
        diff = float(hit_c.mean() - hit_r.mean())
        boot = []
        for _ in range(N_BOOT):
            s = rng.choice(idx, len(idx))
            boot.append(float(hit_c[s].mean() - hit_r[s].mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(
            {
                "record": "contrast",
                "condition": f"{cond} - {ref}",
                "subset": "all8",
                "n": len(yb),
                "accuracy": diff,
                "macro_f1": np.nan,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(settings.outputs_dir / "reports" / "era_theme_coding.csv", index=False)
    print(out.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["data", "embed", "eval"])
    a = ap.parse_args()
    {"data": stage_data, "embed": stage_embed, "eval": stage_eval}[a.stage]()


if __name__ == "__main__":
    main()
