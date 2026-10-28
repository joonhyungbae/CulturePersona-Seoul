"""Open-item reproduction at the narrative layer — H3 「서울이 문화도시가 되려면」 vs persona
`view_on_seoul`.

Anchors reproduce tables of closed items but cannot answer an open item. An open
item is needed to see what the narrative layer adds. Free-text H3 from the 2024
survey (9,263 responses in 2024, what one hopes for so that Seoul becomes a
cultural city) asks the same question as the last slot of the persona narrative
(one's view of Seoul as a cultural city). Personas have never seen this item —
the prompt gives only anchors and slot names.

Measures.
  theme        Split real-response embeddings into k themes with k-means (fit on real only)
               and assign persona text to the nearest theme. Per cell (age 4 × attendance 2),
               TVD of the theme distribution, real (weighted) vs persona.
  baselines    marginal  = the overall real theme distribution on every cell (no cell information) ·
               shuffled  = shuffle the persona's cell label (does the narrative follow the cell) ·
               split-half = between two halves of real respondents (sampling floor).
  nearest-cell Share of persona embeddings closest to their own cell's real centroid (chance 1/8).

Person-level text is not written to disk (embedding cache is a scratchpad; only aggregate
tables go to outputs/reports).

Run: python scripts/era_open_text.py [--wave 2024] [--k 10]
(an environment with sentence-transformers + CUDA; .venv-embed conflicts with anaconda NumPy)
Outputs: outputs/reports/era_open_text.csv · era_open_text_themes.csv · era_open_text.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame, load_general_master

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("open_text")

SEED = 20260923
MODEL = "dragonkue/BGE-m3-ko"
ITEM = "H3"
FIELD = "view_on_seoul"
MIN_CHARS = 10
AGE_GROUP = {"10대": 0, "20대": 0, "30대": 1, "40대": 1, "50대": 2, "60대": 2, "70대 이상": 3}
N_REP = 20


def _embed(texts: list[str], cache: Path) -> np.ndarray:
    if cache.exists():
        e = np.load(cache)
        if len(e) == len(texts):
            return e
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    e = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    np.save(cache, e)
    return e


def _cells(age: pd.Series, att: pd.Series) -> pd.Series:
    return age.map(AGE_GROUP).astype(int).astype(str) + "|" + att.astype(str)


def _dist(labels: np.ndarray, k: int, w: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(labels)) if w is None else w
    out = np.bincount(labels, weights=w, minlength=k)
    return out / max(out.sum(), 1e-12)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def load_real(wave: int) -> pd.DataFrame:
    master = load_general_master(wave)
    frame = build_official_anchor_frame(wave)
    frame = frame.assign(row_uid=master["row_uid"].to_numpy())
    text = pd.read_parquet(settings.official_person_dir / "official_person_text.parquet")
    text = text[(text.wave == wave) & (text.variable == ITEM)]
    text = text.assign(text=text.text.astype(str).str.strip())
    text = text[text.text.str.len() >= MIN_CHARS]
    d = text.merge(frame[["row_uid", "w", "age_band", "attend_any", "sex"]], on="row_uid")
    return d.reset_index(drop=True)


def load_personas(wave: int, path: Path | None = None) -> pd.DataFrame:
    p = path or settings.data_root / "synthetic" / f"era_{wave}_narrative" / "personas.jsonl"
    rows = []
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("ok") or r.get("issues"):
            continue
        rows.append(
            {
                "text": str(r["narrative"].get(FIELD, "")).strip(),
                "age_band": r["quant"]["age_band"],
                "attend_any": r["quant"]["attend_any"],
            }
        )
    d = pd.DataFrame(rows)
    return d[d.text.str.len() >= MIN_CHARS].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2024)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--narrative-path",
        type=Path,
        default=None,
        help="narrative bank from another generator (era_second_generator)",
    )
    ap.add_argument("--tag", default="", help="output filename suffix")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    from sklearn.cluster import KMeans

    scratch = Path(settings.outputs_dir / "logs" / "open_text_cache")
    scratch.mkdir(parents=True, exist_ok=True)
    real = load_real(args.wave)
    syn = load_personas(args.wave, args.narrative_path)
    log.info("real %d answers · personas %d", len(real), len(syn))
    e_real = _embed(real.text.tolist(), scratch / f"real_{args.wave}.npy")
    e_syn = _embed(syn.text.tolist(), scratch / f"syn_{args.wave}{suffix}.npy")

    km = KMeans(n_clusters=args.k, n_init=10, random_state=SEED).fit(e_real)
    real["theme"] = km.labels_
    syn["theme"] = km.predict(e_syn)
    real["cell"] = _cells(real.age_band, real.attend_any)
    syn["cell"] = _cells(syn.age_band, syn.attend_any)
    cells = sorted(real.cell.unique())
    w_cell = real.groupby("cell").w.sum()
    w_cell = w_cell / w_cell.sum()
    k = args.k
    marginal = _dist(real.theme.to_numpy(), k, real.w.to_numpy())

    def cell_tvd(syn_labels: pd.Series, syn_cells: pd.Series) -> float:
        tot = 0.0
        for c in cells:
            p = _dist(real[real.cell == c].theme.to_numpy(), k, real[real.cell == c].w.to_numpy())
            m = syn_cells == c
            q = _dist(syn_labels[m].to_numpy(), k) if m.any() else marginal
            tot += float(w_cell[c]) * _tvd(p, q)
        return tot

    rng = np.random.default_rng(SEED)
    persona_tvd = cell_tvd(syn.theme, syn.cell)
    marginal_tvd = sum(
        float(w_cell[c])
        * _tvd(
            _dist(real[real.cell == c].theme.to_numpy(), k, real[real.cell == c].w.to_numpy()),
            marginal,
        )
        for c in cells
    )
    shuffled = [
        cell_tvd(syn.theme, pd.Series(rng.permutation(syn.cell.to_numpy()))) for _ in range(N_REP)
    ]
    halves = []
    for _ in range(N_REP):
        m = rng.random(len(real)) < 0.5
        a, b = real[m], real[~m]
        tot = 0.0
        for c in cells:
            pa = _dist(a[a.cell == c].theme.to_numpy(), k, a[a.cell == c].w.to_numpy())
            pb = _dist(b[b.cell == c].theme.to_numpy(), k, b[b.cell == c].w.to_numpy())
            tot += float(w_cell[c]) * _tvd(pa, pb)
        halves.append(tot)
    # Cell discrimination: is the persona embedding closest to its own cell's real centroid?
    cent = {c: e_real[(real.cell == c).to_numpy()].mean(axis=0) for c in cells}
    cmat = np.stack([cent[c] / np.linalg.norm(cent[c]) for c in cells])
    nearest = np.array(cells)[(e_syn @ cmat.T).argmax(axis=1)]
    hit = float((nearest == syn.cell.to_numpy()).mean())
    # Mean similarity to real responses in the same cell versus other cells
    own = np.array([float(e_syn[i] @ cmat[cells.index(c)]) for i, c in enumerate(syn.cell)])
    other = np.array(
        [
            float(np.mean([e_syn[i] @ cmat[j] for j, cc in enumerate(cells) if cc != c]))
            for i, c in enumerate(syn.cell)
        ]
    )
    rows = [
        {"wave": args.wave, "metric": "cell_theme_tvd", "method": "personas", "value": persona_tvd},
        {
            "wave": args.wave,
            "metric": "cell_theme_tvd",
            "method": "marginal",
            "value": marginal_tvd,
        },
        {
            "wave": args.wave,
            "metric": "cell_theme_tvd",
            "method": "shuffled",
            "value": float(np.mean(shuffled)),
            "lo": float(np.percentile(shuffled, 2.5)),
            "hi": float(np.percentile(shuffled, 97.5)),
        },
        {
            "wave": args.wave,
            "metric": "cell_theme_tvd",
            "method": "split_half",
            "value": float(np.mean(halves)),
            "lo": float(np.percentile(halves, 2.5)),
            "hi": float(np.percentile(halves, 97.5)),
        },
        {"wave": args.wave, "metric": "nearest_cell_hit", "method": "personas", "value": hit},
        {
            "wave": args.wave,
            "metric": "nearest_cell_hit",
            "method": "chance",
            "value": 1.0 / len(cells),
        },
        {
            "wave": args.wave,
            "metric": "cos_own_minus_other",
            "method": "personas",
            "value": float((own - other).mean()),
        },
        {"wave": args.wave, "metric": "n_real", "method": "-", "value": len(real)},
        {"wave": args.wave, "metric": "n_personas", "method": "-", "value": len(syn)},
        {"wave": args.wave, "metric": "k", "method": "-", "value": k},
    ]
    out = settings.outputs_dir / "reports"
    pd.DataFrame(rows).to_csv(out / f"era_open_text{suffix}.csv", index=False)
    # Theme summary: per-cell distributions (aggregates) and representative terms per theme (no person text)
    theme_rows = []
    for t in range(k):
        rt = real[real.theme == t]
        theme_rows.append(
            {
                "theme": t,
                "real_share": float(rt.w.sum() / real.w.sum()),
                "persona_share": float((syn.theme == t).mean()),
                "real_mean_chars": float(rt.text.str.len().mean()),
            }
        )
    pd.DataFrame(theme_rows).to_csv(out / f"era_open_text_themes{suffix}.csv", index=False)
    lines = [
        "# 서사 층의 열린 문항 재현 (H3 대 `view_on_seoul`)",
        "",
        f"> 자동 생성: `scripts/era_open_text.py` (wave {args.wave}, 실제 {len(real):,} · 페르소나 {len(syn):,} · 주제 k={k} · {MODEL}).",
        "",
        "| 방법 | 셀별 주제 분포 TVD (실제 대비, 가중 평균) |",
        "|---|---|",
        f"| 페르소나 `view_on_seoul` | {persona_tvd:.3f} |",
        f"| 주변분포 (셀 정보 없음) | {marginal_tvd:.3f} |",
        f"| 페르소나, 셀 표지 섞음 | {np.mean(shuffled):.3f} [{np.percentile(shuffled, 2.5):.3f}, {np.percentile(shuffled, 97.5):.3f}] |",
        f"| 실제 반분 (표집 바닥) | {np.mean(halves):.3f} [{np.percentile(halves, 2.5):.3f}, {np.percentile(halves, 97.5):.3f}] |",
        "",
        (
            f"셀 분별: 페르소나가 자기 셀의 실제 중심에 가장 가까운 비율 {hit:.3f} (우연 {1 / len(cells):.3f}); "
            f"자기 셀 − 다른 셀 평균 코사인 {float((own - other).mean()):+.4f}."
        ),
        "",
        (
            "해석 규칙: 페르소나 TVD 가 주변분포·섞음보다 낮으면 서사가 셀을 따라간다는 뜻이고, 반분 바닥에 가까울수록 실제 분포에 가깝다. "
            "높으면 서사가 셀을 따라가지 않거나 주제 자체가 다르다는 뜻이다 — 어느 쪽이든 그대로 보고한다."
        ),
    ]
    (out / f"era_open_text{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
