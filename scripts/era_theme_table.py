"""E1 step 1 — code free text (H3 「서울이 문화도시가 되려면」) into themes and build cell × theme aggregate table T7 (ADR-032).

Narratives built from anchors cannot say more than the anchors (ADR-027, 029, 030, 031). The only channel for new information is an **aggregate table**.
Real respondents' free text is coded into themes, a theme-distribution table is built per cell (age 4 × attendance 2), and that table is given to the narrative as a 25th anchor.
Person records are used only to build the table; what is published is theme names, descriptions, and per-cell shares (unweighted cell ≥ 30).

Procedure (respondents are split into halves A/B — the table uses only A, evaluation uses only B):
  1. Embed 2024 H3 answers (≥10 characters, linked to the frame) (BGE-m3-ko).
  2. k-means on A's embeddings (k themes). For each theme, show gemma4:12b the 20 answers closest to the centroid and have it write a theme name and a one-line description
     (the model only names; answer text is not kept anywhere).
  3. Assign every answer (A and B) to the theme of the nearest centroid. Write a human-check CSV (data_private, gitignore).
  4. T7: A's cell × theme weighted shares (shrink toward the margin with Dirichlet α; themes with unweighted n < 5 inside a cell become '기타'). Also emit the margin.
  5. Store B's cell × theme distribution (truth), the A–B split-half floor, and the oracle that applies A's table to B unchanged → used by step-2 evaluation.

Run: python scripts/era_theme_table.py [--k 12]
Outputs: data/grounding/era_2024/T7_theme_by_cell.parquet (public candidate) · outputs/reports/era_theme_table.md ·
         outputs/logs/theme_cache/ (embeddings, centroids, assignments, person-level — gitignore) · data_private check CSV
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era_open_text import AGE_GROUP, MIN_CHARS, _embed, load_real

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("theme_table")

SEED = 20260924
NAMER = "gemma4:12b"
OLLAMA = f"{settings.llm_base_url}/api/chat"
ALPHA = 10.0  # Dirichlet shrinkage strength (in respondent counts)
MIN_THEME_CELL = 5


def cells(df: pd.DataFrame) -> pd.Series:
    return df.age_band.map(AGE_GROUP).astype(int).astype(str) + "|" + df.attend_any.astype(str)


def name_theme(examples: list[str]) -> dict:
    prompt = (
        "다음은 「서울이 문화도시가 되기 위해 바라는 점」에 대한 설문 자유응답 20 개다. 공통 주제를 한국어 명사구 6 자 이내로 이름 짓고, "
        '한 문장(30 자 이내)으로 설명하라. JSON 으로만 답하라: {"name": ..., "description": ...}\n\n'
        + "\n".join(f"- {e}" for e in examples)
    )
    body = json.dumps(
        {
            "model": NAMER,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
            "think": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read().decode("utf-8"))
    try:
        return json.loads(out["message"]["content"])
    except Exception:  # noqa: BLE001
        return {"name": "?", "description": out["message"]["content"][:60]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2024)
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()
    from sklearn.cluster import KMeans

    cache = settings.outputs_dir / "logs" / "theme_cache"
    cache.mkdir(parents=True, exist_ok=True)
    real = load_real(args.wave)
    real = real[real.text.str.len() >= MIN_CHARS].reset_index(drop=True)
    real["cell"] = cells(real)
    rng = np.random.default_rng(SEED)
    # Split in half within each cell (stratified)
    half = np.zeros(len(real), dtype=bool)
    for idx in real.groupby("cell").indices.values():
        idx = np.array(idx)
        rng.shuffle(idx)
        half[idx[: len(idx) // 2]] = True
    real["split"] = np.where(half, "A", "B")
    e = _embed(real.text.tolist(), cache / f"real_{args.wave}_all.npy")
    a_mask = (real.split == "A").to_numpy()
    km = KMeans(n_clusters=args.k, n_init=10, random_state=SEED).fit(e[a_mask])
    real["theme"] = km.predict(e)
    np.save(cache / "centroids.npy", km.cluster_centers_)

    # Name the themes (20 answers nearest A's centroid)
    codebook = []
    ea = e[a_mask]
    ra = real[a_mask].reset_index(drop=True)
    for t in range(args.k):
        m = (ra.theme == t).to_numpy()
        if m.sum() == 0:
            codebook.append({"theme": t, "name": f"주제{t}", "description": "", "n_A": 0})
            continue
        d = ea[m] @ km.cluster_centers_[t]
        top = ra[m].iloc[np.argsort(-d)[:20]].text.tolist()
        nm = name_theme(top)
        codebook.append(
            {
                "theme": t,
                "name": str(nm.get("name", "?"))[:20],
                "description": str(nm.get("description", ""))[:80],
                "n_A": int(m.sum()),
            }
        )
        log.info(
            "theme %d (n=%d): %s — %s",
            t,
            int(m.sum()),
            codebook[-1]["name"],
            codebook[-1]["description"],
        )
    cb = pd.DataFrame(codebook)

    # T7: A's cell × theme weighted shares, with shrinkage
    A = real[a_mask]
    marg = A.groupby("theme").w.sum()
    marg = (marg / marg.sum()).reindex(range(args.k)).fillna(0.0)
    rows = []
    for c, g in A.groupby("cell"):
        n_unw = len(g)
        wsum = g.w.sum()
        counts = g.groupby("theme").size().reindex(range(args.k)).fillna(0).astype(int)
        wshare = (g.groupby("theme").w.sum().reindex(range(args.k)).fillna(0.0)) / wsum
        shrunk = (wshare * wsum + ALPHA * marg) / (wsum + ALPHA)
        for t in range(args.k):
            rows.append(
                {
                    "cell": c,
                    "theme": t,
                    "share": float(shrunk[t]),
                    "raw_share": float(wshare[t]),
                    "n_unweighted": int(counts[t]),
                    "n_cell": n_unw,
                    "suppressed": bool(counts[t] < MIN_THEME_CELL),
                }
            )
    t7 = pd.DataFrame(rows)
    # Public table: mass of suppressed themes is pooled into '기타' (theme=-1) — cells ≥ 30 only
    pub_rows = []
    for c, g in t7.groupby("cell"):
        if int(g.n_cell.iloc[0]) < settings.cell_floor:
            continue
        other = float(g[g.suppressed].share.sum())
        for r in g[~g.suppressed].itertuples():
            pub_rows.append(
                {
                    "cell": c,
                    "theme": r.theme,
                    "share": r.share,
                    "n_unweighted": r.n_unweighted,
                    "n_cell": r.n_cell,
                }
            )
        pub_rows.append(
            {
                "cell": c,
                "theme": -1,
                "share": other,
                "n_unweighted": int(g[g.suppressed].n_unweighted.sum()),
                "n_cell": int(g.n_cell.iloc[0]),
            }
        )
    t7_pub = pd.DataFrame(pub_rows)
    gdir = settings.era_grounding_dir(args.wave)
    t7_pub.to_parquet(gdir / "T7_theme_by_cell_release.parquet", index=False)
    cb.to_csv(gdir / "T7_theme_codebook.csv", index=False)
    pd.DataFrame({"theme": range(args.k), "share": marg.to_numpy()}).to_parquet(
        gdir / "T7_theme_marginal_release.parquet", index=False
    )

    # Save for evaluation (person-level stays in the cache only)
    real[["row_uid", "split", "cell", "theme", "w"]].to_parquet(
        cache / "assignments.parquet", index=False
    )
    # For human check (data_private, gitignore): 15 per theme
    chk = real.groupby("theme", group_keys=False).apply(
        lambda g: g.sample(min(15, len(g)), random_state=SEED)
    )
    chk = chk.merge(cb[["theme", "name", "description"]], on="theme")[
        ["theme", "name", "description", "text"]
    ]
    priv = ROOT / "annotation" / "data_private"
    chk.to_csv(priv / "theme_check_2024.csv", index=False)

    # Truth (B), floor, and oracle
    B = real[~a_mask]

    def dist(g: pd.DataFrame) -> np.ndarray:
        d = g.groupby("theme").w.sum().reindex(range(args.k)).fillna(0.0)
        return (d / d.sum()).to_numpy()

    tv = []
    wts = []
    for c, gb in B.groupby("cell"):
        ga = A[A.cell == c]
        if len(ga) < 5 or len(gb) < 5:
            continue
        pb = dist(gb)
        oracle = t7[t7.cell == c].sort_values("theme").share.to_numpy()
        tv.append(
            {
                "cell": c,
                "n_B": len(gb),
                "floor_AB": 0.5 * np.abs(dist(ga) - pb).sum(),
                "oracle_T7": 0.5 * np.abs(oracle - pb).sum(),
                "marginal": 0.5 * np.abs(marg.to_numpy() - pb).sum(),
            }
        )
        wts.append(len(gb))
    ev = pd.DataFrame(tv)
    w = np.array(wts, dtype=float)
    summary = {
        k: float(np.average(ev[k], weights=w)) for k in ("floor_AB", "oracle_T7", "marginal")
    }
    ev.to_csv(cache / "eval_cells.csv", index=False)
    lines = [
        f"# T7 주제 집계표 ({args.wave}, k={args.k})",
        "",
        f"응답 {len(real):,} 건 (A {int(a_mask.sum()):,} · B {int((~a_mask).sum()):,}), 셀 8 개, 수축 α={ALPHA}, 셀 안 {MIN_THEME_CELL} 미만 주제는 '기타'.",
        "",
        "| theme | 이름 | 설명 | n_A |",
        "|---|---|---|---|",
        *[f"| {r.theme} | {r.name} | {r.description} | {r.n_A} |" for r in cb.itertuples()],
        "",
        (
            "B 에 대한 셀별 주제 TVD (가중 평균): "
            f"A 표를 그대로(오라클 T7) **{summary['oracle_T7']:.3f}** · A 주변분포 {summary['marginal']:.3f} · A–B 반분 바닥 {summary['floor_AB']:.3f}"
        ),
        "",
        "해석: 오라클 T7 이 주변분포보다 낮으면 셀별 주제 정보가 실재한다 (2 단계에서 서사가 이 정보를 실어 나르는지 본다).",
    ]
    (settings.outputs_dir / "reports" / "era_theme_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    log.info("\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
