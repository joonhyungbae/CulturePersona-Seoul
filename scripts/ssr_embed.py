"""SSR embedding step — cosine between free-response sentences and per-option reference sentences.

**Run in a separate venv**: `.venv-embed/bin/python scripts/ssr_embed.py`
Putting sentence-transformers/torch in the main venv would replace packages the 3-day 30k generation uses.
The GPU is also left alone (CUDA_VISIBLE_DEVICES is cleared) — the generator is loaded on the 24GB.

Input  data/eval/holdout/ssr_refs.json          per-option reference sentences (see `src/sfac_persona/ssr_refs.py`)
       data/eval/holdout/answers_*_ssr.jsonl    per-persona free responses
Output data/eval/holdout/cos_<tag>_<condition>.jsonl   {persona_id, cos: {item: [per-option cosine]}}

The softmax temperature is not chosen here. Cosines are left as-is; the temperature is applied at scoring time.

  --selftest   Print only cosines among reference sentences. First check that options separate in embedding space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", ""
)  # before import — the GPU belongs to the generator
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("ssr_embed")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "dragonkue/BGE-m3-ko"


def load_holdout_dir() -> Path:
    """Find paths without importing config (read .env only) — this venv has no pydantic."""
    root = ROOT / "data"
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SFP_DATA_ROOT="):
                root = Path(line.split("=", 1)[1].strip().strip('"').strip("'")).expanduser()
    return root / "eval" / "holdout"


def embed(model: SentenceTransformer, texts: list[str], batch: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def option_embeddings(
    model: SentenceTransformer, opts: dict[str, list[str]], batch: int
) -> np.ndarray:
    """If an option has several reference sentences, average the embeddings and renormalize."""
    flat, span = [], []
    for sents in opts.values():
        span.append((len(flat), len(flat) + len(sents)))
        flat.extend(sents)
    E = embed(model, flat, batch)
    M = np.stack([E[i:j].mean(axis=0) for i, j in span])
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def selftest(model: SentenceTransformer, refs: dict, batch: int) -> None:
    """Do per-option reference sentences separate from each other. The least-separated item is where SSR is weakest."""
    rows = []
    for var, opts in refs.items():
        names = list(opts)
        C = option_embeddings(model, opts, batch) @ option_embeddings(model, opts, batch).T
        iu = np.triu_indices(len(names), 1)
        worst = int(np.argmax(C[iu]))
        rows.append(
            (
                var,
                len(names),
                float(C[iu].mean()),
                float(C[iu].max()),
                f"{names[iu[0][worst]]} ~ {names[iu[1][worst]]}",
            )
        )
    rows.sort(key=lambda r: -r[3])
    print(f"{'item':<10}{'opts':>4}{'mean_cos':>9}{'max_cos':>9}  closest option pair")
    for var, k, mean, mx, pair in rows:
        print(f"{var:<10}{k:>4}{mean:>9.3f}{mx:>9.3f}  {pair[:70]}")
    mx_all = max(r[3] for r in rows)
    # The bar is gate G4: max cosine < 0.90.
    ok = mx_all < 0.90
    print(
        f"\nMax cosine among reference-sentence pairs {mx_all:.3f}. "
        + (
            "Every option pair separates."
            if ok
            else "A pair is at or above 0.90 — SSR cannot separate those options. Revise the reference sentences."
        )
    )
    print(f"GATE G4-refs {'PASS' if ok else 'FAIL'} max_cos={mx_all:.3f} (bar <0.90)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pattern", default="answers_*_ssr.jsonl")
    ap.add_argument(
        "--dir",
        default=None,
        help="holdout directory (default data/eval/holdout). era smoke uses data/eval/era_holdout",
    )
    a = ap.parse_args()

    out = Path(a.dir) if a.dir else load_holdout_dir()
    if not out.is_absolute():
        out = ROOT / out
    refs_path = out / "ssr_refs.json"
    if not refs_path.exists():
        raise SystemExit(f"{refs_path} is missing.")
    refs: dict[str, dict[str, list[str] | str]] = json.loads(refs_path.read_text(encoding="utf-8"))
    # normalize: value -> list[str]
    norm_refs: dict[str, dict[str, list[str]]] = {}
    for var, opts in refs.items():
        norm_refs[var] = {k: (v if isinstance(v, list) else [v]) for k, v in opts.items()}
    refs = norm_refs

    log.info("loading model %s (CPU)", a.model)
    model = SentenceTransformer(a.model, device="cpu")
    model.max_seq_length = min(model.max_seq_length, 256)

    if a.selftest:
        selftest(model, refs, a.batch)
        return

    # Reference-sentence embeddings (per item, (k, d))
    ref_emb = {var: option_embeddings(model, opts, a.batch) for var, opts in refs.items()}
    ref_opts = {var: list(opts) for var, opts in refs.items()}
    log.info("embedded %d reference sentences", sum(len(v) for v in ref_opts.values()))

    files = sorted(out.glob(a.pattern))
    if not files:
        raise SystemExit(f"no answer files in {out}/{a.pattern}.")
    for path in files:
        # answers_foo_ssr.jsonl → cos_foo.jsonl ; smoke_answers.jsonl → cos_smoke_answers.jsonl
        if path.name.startswith("answers_") and path.name.endswith("_ssr.jsonl"):
            dst = out / ("cos_" + path.name[len("answers_") : -len("_ssr.jsonl")] + ".jsonl")
        else:
            dst = out / f"cos_{path.stem}.jsonl"
        if dst.resolve() == path.resolve():
            raise SystemExit(f"cos output is the same file as the input: {path}")
        done = (
            {
                json.loads(l)["persona_id"]
                for l in dst.read_text(encoding="utf-8").splitlines()
                if l.strip()
            }
            if dst.exists()
            else set()
        )
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        todo = [r for r in recs if r["persona_id"] not in done]
        log.info("%s: %d people, %d remaining → %s", path.name, len(recs), len(todo), dst.name)
        if not todo:
            continue

        # The same sentence repeats (the model uses stock phrases). Embed unique sentences only.
        uniq: dict[str, str] = {}
        for r in todo:
            for s in (r.get("texts") or {}).values():
                if s:
                    uniq.setdefault(hashlib.sha1(s.encode()).hexdigest(), s)
        keys, sents = list(uniq), list(uniq.values())
        log.info(
            "  %d unique sentences (of %d)",
            len(sents),
            sum(1 for r in todo for s in (r.get("texts") or {}).values() if s),
        )
        E = np.zeros((len(sents), model.get_sentence_embedding_dimension()), dtype=np.float32)
        step = max(a.batch * 16, 256)
        for i in range(0, len(sents), step):
            E[i : i + step] = embed(model, sents[i : i + step], a.batch)
            log.info("  embedding %d/%d", min(i + step, len(sents)), len(sents))
        pos = {k: i for i, k in enumerate(keys)}

        with open(dst, "a", encoding="utf-8") as f:
            for r in todo:
                cos = {}
                for var, s in (r.get("texts") or {}).items():
                    if not s or var not in ref_emb:
                        continue
                    cos[var] = np.round(
                        ref_emb[var] @ E[pos[hashlib.sha1(s.encode()).hexdigest()]], 5
                    ).tolist()
                f.write(
                    json.dumps({"persona_id": r["persona_id"], "cos": cos}, ensure_ascii=False)
                    + "\n"
                )
        log.info("  → %s", dst.name)


if __name__ == "__main__":
    main()
