"""E1 v2 (ADR-033) — fold free-text theme table T7 into the dataset as the 25th anchor `wish_theme`.

Drop ADR-032's k-means codebook (12 overlapping themes) and same-model scoring (circular):
  codebook  7 themes plus 기타, merged by the author from the k-means names (CODEBOOK)
  coder     EXAONE 4.0 32B — different from gemma4:12b, which named them, and from Qwen, which generated them
  split     ADR-032's stratified A/B split unchanged (outputs/logs/theme_cache/assignments.parquet)

Stages (in order; each stage can resume):
  code-real   EXAONE codes every real H3 answer → cache/v2_codes_real.jsonl
  table       T7 = attendance (attend_any) × theme. Value cells (theme rows) also keep only unweighted n ≥ 30; below that, fold into 기타.
              Eval table = half A (cache, for the pilot and the gate) · public table = all respondents (data/grounding/era_2024/T7_*, for bank sampling) —
              refitting on the full sample after the split check is the same practice as T1–T6. reports/era_theme_v2_table.md
  pilot       Frozen bank, 300 people: regenerate C1 (margin theme) and C2 (cell-table theme) with Qwen → EXAONE codes C0, C1, C2 → reports/era_theme_v2_pilot.md
              Gate: exit 0 if C2 < C0 and realization ≥ 0.7, else 2
  bank        2024 bank, 15k: sample wish_theme (public T7; 기타 is resampled from the margin) + generate wish_for_seoul → data/synthetic/era_2024_narrative/wish_for_seoul.jsonl
  bank-eval   EXAONE codes a cell-stratified 1,000 → TVD vs the public table (renderer fidelity) and realization, and vs B as a reference → reports/era_theme_v2_bank.md

Run: uv run python scripts/era_theme_v2.py <stage>   (GPU: EXAONE and Qwen take turns — the judge wrapper stops the full count)
Person text is used only as coding input; only aggregates are kept.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from era_consistency_checker import anchor_card
from era_open_text import AGE_GROUP, MIN_CHARS, load_real

from sfac_persona.config import settings
from sfac_persona.narrative import WISH_FIELD

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("theme_v2")

SEED = 20260924
WAVE = 2024
OLLAMA = f"{settings.llm_base_url}/api/chat"
CODER = "hf.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF:IQ4_XS"
GEN_MODEL = "qwen3.8:27b-mtp-q4_K_M"
CONC = 2
ALPHA = 10.0  # Dirichlet shrinkage (in respondent counts)
MIN_THEME_CELL = (
    settings.cell_floor
)  # value cells (theme rows) use the same floor as the other tables (tests/test_release_surface.py)
OTHER = 7
N_BANK_EVAL = 1000

# Author-merged codebook (built from the 12 k-means theme names; fixed before seeing results)
CODEBOOK: list[tuple[int, str, str]] = [
    (0, "문화 공간·시설 확충", "동네 가까이에 공연장·전시장·도서관 같은 문화 공간을 늘리는 것"),
    (1, "예산·정책 지원 확대", "문화 예산과 예술인·단체 지원을 늘리고 체감되는 정책을 펴는 것"),
    (2, "홍보·정보 제공 강화", "어떤 행사가 언제 어디서 열리는지 알리는 홍보와 정보 제공"),
    (3, "비용 부담 완화", "무료·저가 공연과 전시, 할인 등으로 경제적 부담을 줄이는 것"),
    (4, "프로그램·콘텐츠 다양화", "장르·행사·축제·체험 프로그램의 종류와 수준을 넓히는 것"),
    (5, "참여 기회·포용 확대", "세대·계층·지역 구분 없이 누구나 참여할 수 있는 기회와 접근성"),
    (6, "현재 만족·특별한 의견 없음", "지금도 충분하다, 잘 하고 있다, 특별히 바라는 점이 없다"),
    (OTHER, "기타", "위 어디에도 맞지 않는 내용"),
]
K = len(CODEBOOK)  # 8 codes including 기타 (the table is published as 8 codes)
THEMES = [
    t for t in range(K) if t != OTHER
]  # TVD is on the 7 themes excluding 기타 (the sampler does not draw 기타)
CACHE = settings.outputs_dir / "logs" / "theme_cache"
GDIR = settings.era_grounding_dir(WAVE)
REPORTS = settings.outputs_dir / "reports"
BANK_PATH = settings.synthetic_dir / f"era_{WAVE}_narrative" / "personas.jsonl"
WISH_PATH = settings.synthetic_dir / f"era_{WAVE}_narrative" / "wish_for_seoul.jsonl"
PILOT_DIR = settings.synthetic_dir / f"era_{WAVE}_narrative_theme_v2"
EVAL_BANK = ROOT / "data" / "eval" / "era_holdout" / f"w{WAVE}" / "eval_bank.jsonl"


def codebook_text() -> str:
    return "\n".join(f"{t}: {n} — {d}" for t, n, d in CODEBOOK)


def _post(payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def code_one(text: str) -> int:
    prompt = (
        "다음 글은 「서울이 문화도시가 되기 위해 바라는 점」에 대한 한 사람의 의견이다. "
        "아래 코드 가운데 가장 잘 맞는 하나를 고르라. 두 가지가 섞였으면 더 비중이 큰 쪽을 고르고, "
        f"어디에도 맞지 않으면 {OTHER} 을 고르라.\n\n{codebook_text()}\n\n글: {text}\n\n"
        'JSON 으로만 답하라: {"code": <번호>}'
    )
    out = _post(
        {
            "model": CODER,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 40, "num_ctx": 4096},
            "think": False,
        },
        timeout=180,
    )
    content = out["message"]["content"]
    try:
        c = int(json.loads(content)["code"])
    except Exception:  # noqa: BLE001
        m = re.search(r"\d+", content)
        c = int(m.group()) if m else -1
    return c if 0 <= c < K else -1


def code_all(name: str, texts: dict[str, str]) -> dict[str, int]:
    path = CACHE / f"v2_codes_{name}.jsonl"
    done: dict[str, int] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r["code"]
    todo = {k: v for k, v in texts.items() if k not in done or done[k] < 0}
    log.info("code %s: %d done, %d todo", name, len(done), len(todo))
    t0 = time.time()
    with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=CONC) as pool:
        futs = {pool.submit(code_one, v): k for k, v in todo.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            k = futs[fut]
            try:
                c = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: %s", k, exc)
                continue
            f.write(json.dumps({"id": k, "code": c}) + "\n")
            f.flush()
            done[k] = c
            if i % 200 == 0:
                log.info(
                    "  %s %d/%d (%.0f/min)", name, i, len(todo), i / max(time.time() - t0, 1) * 60
                )
    return done


FIRST_PERSON = re.compile(r"\d|나는|나에게|내가|나의|저는|제가|저의|우리는")


def gen_one(card: str, theme: int) -> str:
    name = CODEBOOK[theme][1]
    desc = CODEBOOK[theme][2]
    prompt = (
        "다음은 서울에 사는 한 사람의 설문 응답 요약이다.\n"
        f"{card}\n\n"
        f"이 사람의 「{WISH_FIELD.label_ko}」 칸을 3인칭('이 사람은 …')으로 {WISH_FIELD.max_sentences} 문장 이내로 쓰라. "
        f"{WISH_FIELD.guide}\n"
        f"이 사람이 바라는 점의 주제는 「{name}」({desc})이다. "
        "그 주제를 이 사람의 상황과 목소리로 구체적으로 말하되, 1인칭·숫자·실명·특정 장소명은 쓰지 말라. 본문만 출력하라."
    )
    text = ""
    for _ in range(
        3
    ):  # if first person or a digit leaks in, retry under the same conditions (at most 2 retries)
        out = _post(
            {
                "model": GEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.6, "num_predict": 300, "num_ctx": 4096},
                "think": False,
            },
            timeout=300,
        )
        text = out["message"]["content"].strip()
        if text and not FIRST_PERSON.search(text):
            break
    return text


def gen_all(path: Path, units: list[tuple[str, str, str, int]]) -> dict[str, dict]:
    """units: (persona_id, cell, card, theme). Resumable; result {pid: rec}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["persona_id"]] = r
    todo = [u for u in units if u[0] not in done]
    log.info("gen %s: %d done, %d todo", path.name, len(done), len(todo))
    t0 = time.time()
    with path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=CONC) as pool:
        futs = {pool.submit(gen_one, u[2], u[3]): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            u = futs[fut]
            try:
                text = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("%s failed: %s", u[0], exc)
                continue
            rec = {"persona_id": u[0], "cell": u[1], "wish_theme": u[3], "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[u[0]] = rec
            if i % 100 == 0:
                log.info(
                    "  %s %d/%d (%.1f/min)",
                    path.name,
                    i,
                    len(todo),
                    i / max(time.time() - t0, 1) * 60,
                )
    return done


# ---------- real answers ----------


def real_frame() -> pd.DataFrame:
    real = load_real(WAVE)
    real = real[real.text.str.len() >= MIN_CHARS].reset_index(drop=True)
    real["cell"] = (
        real.age_band.map(AGE_GROUP).astype(int).astype(str) + "|" + real.attend_any.astype(str)
    )
    split = pd.read_parquet(CACHE / "assignments.parquet")[["row_uid", "split"]]
    real = real.merge(split, on="row_uid", how="left")
    if real.split.isna().any():  # new rows are assigned at random, matching the stratified split
        rng = np.random.default_rng(SEED)
        m = real.split.isna()
        real.loc[m, "split"] = rng.choice(["A", "B"], size=int(m.sum()))
    return real


def wdist(g: pd.DataFrame, col: str = "code") -> np.ndarray:
    """Weighted theme distribution — normalized on the 7 themes excluding 기타."""
    g = g[(g[col] >= 0) & (g[col] != OTHER)]
    d = g.groupby(col).w.sum().reindex(THEMES).fillna(0.0)
    tot = d.sum()
    return (d / tot).to_numpy() if tot > 0 else np.full(len(THEMES), 1.0 / len(THEMES))


def wdist8(g: pd.DataFrame, col: str = "code") -> np.ndarray:
    """For building the table: distribution over 8 codes including 기타."""
    g = g[g[col] >= 0]
    d = g.groupby(col).w.sum().reindex(range(K)).fillna(0.0)
    tot = d.sum()
    return (d / tot).to_numpy() if tot > 0 else np.full(K, 1.0 / K)


def stage_code_real() -> int:
    real = real_frame()
    code_all("real", {str(r.row_uid): r.text for r in real.itertuples()})
    return 0


def _real_coded() -> pd.DataFrame:
    real = real_frame()
    codes = code_all("real", {})  # read the cache only
    real["code"] = real.row_uid.astype(str).map(codes).fillna(-1).astype(int)
    return real


def att_of(cell: str) -> str:
    """Eval cell 'age4|att' → T7 cell (att)."""
    return cell.split("|", 1)[1]


def build_t7(src: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Attendance × theme table (α shrinkage). Themes with a value cell under 30 fold into 기타; if 기타 is also under 30, drop the cell."""
    src = src[src.code >= 0]
    marg = pd.Series(wdist8(src), index=range(K))
    rows = []
    for c, g in src.groupby(src.attend_any.astype(str)):
        n_unw = len(g)
        if n_unw < settings.cell_floor:
            continue
        counts = g.groupby("code").size().reindex(range(K)).fillna(0).astype(int)
        shrunk = (wdist8(g) * g.w.sum() + ALPHA * marg.to_numpy()) / (g.w.sum() + ALPHA)
        fold = [t for t in range(K) if t != OTHER and counts[t] < MIN_THEME_CELL]
        other_n = int(counts[OTHER] + sum(counts[t] for t in fold))
        if other_n < MIN_THEME_CELL:
            continue
        for t in range(K):
            if t in fold:
                continue
            share = float(shrunk[t]) + (float(sum(shrunk[u] for u in fold)) if t == OTHER else 0.0)
            n = other_n if t == OTHER else int(counts[t])
            rows.append({"cell": c, "theme": t, "share": share, "n_unweighted": n, "n_cell": n_unw})
    return pd.DataFrame(rows), marg


def stage_table() -> int:
    real = _real_coded()
    A = real[real.split == "A"]
    B = real[real.split == "B"]
    t7_eval, marg_A = build_t7(A)
    t7_eval.to_parquet(CACHE / "v2_T7_eval_A.parquet", index=False)
    pd.DataFrame({"theme": range(K), "share": marg_A.to_numpy()}).to_parquet(
        CACHE / "v2_T7_marginal_A.parquet", index=False
    )
    t7_rel, marg_all = build_t7(real)
    t7_rel.to_parquet(GDIR / "T7_theme_by_cell_release.parquet", index=False)
    pd.DataFrame({"theme": range(K), "share": marg_all.to_numpy()}).to_parquet(
        GDIR / "T7_theme_marginal_release.parquet", index=False
    )
    pd.DataFrame(CODEBOOK, columns=["theme", "name", "description"]).assign(
        n_all=[int((real.code == t).sum()) for t, _, _ in CODEBOOK]
    ).to_csv(GDIR / "T7_theme_codebook.csv", index=False)
    # Baselines vs B truth (eval cell age4|att): A's margin / A–B split-half floor / oracle that applies A's eval table to the eval cell
    ev = []
    smp = Sampler(np.random.default_rng(0), source="eval")
    marg7 = wdist(A)
    for c, gb in B.groupby("cell"):
        ga = A[A.cell == c]
        pb = wdist(gb)
        oracle = smp.expected(
            c
        )  # sampler's expected distribution (기타 mass goes to the margin) — the limit of draw_only
        ev.append(
            {
                "cell": c,
                "n_B": int((gb.code >= 0).sum()),
                "marginal": 0.5 * np.abs(marg7 - pb).sum(),
                "floor_AB": 0.5 * np.abs(wdist(ga) - pb).sum(),
                "oracle_T7": 0.5 * np.abs(oracle - pb).sum(),
            }
        )
    ev = pd.DataFrame(ev)
    ev.to_csv(CACHE / "v2_eval_cells.csv", index=False)
    w = ev.n_B.to_numpy(dtype=float)
    base = {k: float(np.average(ev[k], weights=w)) for k in ("marginal", "floor_AB", "oracle_T7")}
    invalid = int((real.code < 0).sum())
    fold_rel = 1 - t7_rel[t7_rel.theme != OTHER].n_unweighted.sum() / max(
        t7_rel.n_cell.drop_duplicates().sum(), 1
    )
    lines = [
        f"# T7 v2 주제 집계표 ({WAVE}, 코더 EXAONE 4.0 32B, 코드북 {K} 코드, 셀 = 관람 여부)",
        "",
        f"응답 {len(real):,} 건 (A {len(A):,} · B {len(B):,}; 코딩 실패 {invalid}), 수축 α={ALPHA}, 값 셀 {MIN_THEME_CELL} 미만 주제는 기타로.",
        f"공개표(전체 응답자): 셀 {t7_rel.cell.nunique()} 개 · 행 {len(t7_rel)} · 기타로 접힌 몫 {fold_rel:.2f} — 평가표(A 절반)는 파일럿·문지기에만 쓴다.",
        "",
        "| 코드 | 이름 | 설명 | n_all | 전체 가중 비율 |",
        "|---|---|---|---|---|",
        *[
            f"| {t} | {n} | {d} | {int((real.code == t).sum())} | {marg_all[t]:.3f} |"
            for t, n, d in CODEBOOK
        ],
        "",
        "| 공개표 셀 | 주제 | share | n |",
        "|---|---|---|---|",
        *[
            f"| {r.cell} | {CODEBOOK[int(r.theme)][1]} | {r.share:.3f} | {int(r.n_unweighted)} |"
            for r in t7_rel.itertuples()
        ],
        "",
        f"B(평가 셀 age4×att, 기타 제외 7 주제) 대비 주제 TVD (가중 평균): A 평가표의 샘플러 기대 분포(오라클) **{base['oracle_T7']:.3f}** · A 주변분포 {base['marginal']:.3f} · A–B 반분 바닥 {base['floor_AB']:.3f}",
    ]
    (REPORTS / "era_theme_v2_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0


# ---------- sampling ----------


class Sampler:
    """Draw a theme from T7 (attendance × theme). source='eval' = half-A eval table (pilot), 'release' = public table (bank)."""

    def __init__(self, rng: np.random.Generator, source: str = "release"):
        self.rng = rng
        if source == "eval":
            t7 = pd.read_parquet(CACHE / "v2_T7_eval_A.parquet")
            marg = pd.read_parquet(CACHE / "v2_T7_marginal_A.parquet")
        else:
            t7 = pd.read_parquet(GDIR / "T7_theme_by_cell_release.parquet")
            marg = pd.read_parquet(GDIR / "T7_theme_marginal_release.parquet")
        m = marg.set_index("theme").share.reindex(range(K)).fillna(0.0)
        m[OTHER] = (
            0.0  # 기타 is not drawn as an anchor (it is not a theme to write into the narrative)
        )
        self.pm = (m / m.sum()).to_numpy()  # 8 cells, 기타 = 0
        self.by_cell = {c: g for c, g in t7.groupby("cell")}

    def expected(self, cell: str) -> np.ndarray:
        """Expected theme distribution drawn in this cell (THEMES order, 7 cells)."""
        g = self.by_cell.get(att_of(cell))
        if g is None:
            return self.pm[THEMES]
        p = g.set_index("theme").share.reindex(range(K)).fillna(0.0).to_numpy()
        p = p / p.sum()
        p = p + p[OTHER] * self.pm
        p[OTHER] = 0.0
        return p[THEMES] / p[THEMES].sum()

    def marginal(self) -> int:
        return int(self.rng.choice(K, p=self.pm))

    def cell(self, cell: str) -> int:
        g = self.by_cell.get(att_of(cell))
        if g is None:
            return self.marginal()
        p = g.share.to_numpy()
        t = int(self.rng.choice(g.theme.to_numpy(), p=p / p.sum()))
        return t if t != OTHER else self.marginal()


def persona_cell(q: dict) -> str:
    return f"{AGE_GROUP[q['age_band']]}|{q['attend_any']}"


def _truth():
    real = _real_coded()
    B = real[real.split == "B"]
    truth = {c: wdist(g) for c, g in B.groupby("cell")}
    wts = {c: int((g.code >= 0).sum()) for c, g in B.groupby("cell")}
    return truth, wts


def score(
    labels_by_pid: dict[str, int], cell_by_pid: dict[str, str], truth: dict, wts: dict
) -> float:
    tot = wsum = 0.0
    for c, p in truth.items():
        m = [t for pid, t in labels_by_pid.items() if cell_by_pid.get(pid) == c and t in THEMES]
        if not m:
            continue
        q = np.bincount(m, minlength=K)[THEMES] / len(m)
        tot += wts[c] * 0.5 * np.abs(p - q).sum()
        wsum += wts[c]
    return tot / wsum


def stage_pilot() -> int:
    rng = np.random.default_rng(SEED)
    smp = Sampler(rng, source="eval")
    bank = [json.loads(line) for line in EVAL_BANK.open(encoding="utf-8") if line.strip()]
    cells = {r["persona_id"]: persona_cell(r["quant"]) for r in bank}
    units = {"C1": [], "C2": []}
    for r in bank:
        card = anchor_card(r["quant"])
        c = cells[r["persona_id"]]
        units["C1"].append((r["persona_id"], c, card, smp.marginal()))
        units["C2"].append((r["persona_id"], c, card, smp.cell(c)))
    gen = {cond: gen_all(PILOT_DIR / f"{cond}.jsonl", units[cond]) for cond in ("C1", "C2")}
    codes = {
        "C0": code_all(
            "pilot_C0", {r["persona_id"]: r["narrative"].get("view_on_seoul", "") for r in bank}
        )
    }
    for cond in ("C1", "C2"):
        codes[cond] = code_all(f"pilot_{cond}", {pid: r["text"] for pid, r in gen[cond].items()})
    truth, wts = _truth()
    ev = pd.read_csv(CACHE / "v2_eval_cells.csv")
    w = ev.n_B.to_numpy(dtype=float)
    base = {k: float(np.average(ev[k], weights=w)) for k in ("marginal", "floor_AB", "oracle_T7")}
    rows = []
    for cond in ("C0", "C1", "C2"):
        rz = float("nan")
        if cond in gen:
            rz = float(
                np.mean(
                    [
                        codes[cond][pid] == r["wish_theme"]
                        for pid, r in gen[cond].items()
                        if pid in codes[cond]
                    ]
                )
            )
        rows.append(
            {"condition": cond, "tvd": score(codes[cond], cells, truth, wts), "realization": rz}
        )
    for k, v in base.items():
        rows.append({"condition": k, "tvd": v, "realization": float("nan")})
    # Floor from drawing the table without writing prose: mean of 200 draws on the same 300 people (separates the loss added by generation)
    draws = []
    for i in range(200):
        s_i = Sampler(np.random.default_rng(SEED + 100 + i), source="eval")
        draws.append(score({pid: s_i.cell(c) for pid, c in cells.items()}, cells, truth, wts))
    rows.append(
        {"condition": "draw_only", "tvd": float(np.mean(draws)), "realization": float("nan")}
    )
    d = pd.DataFrame(rows)
    d.to_csv(REPORTS / "era_theme_v2_pilot.csv", index=False)
    r = d.set_index("condition")
    ok = bool(r.loc["C2", "tvd"] < r.loc["C0", "tvd"] and r.loc["C2", "realization"] >= 0.7)
    lines = [
        "# E1 v2 파일럿 — 동결 뱅크 300 명, 코더 EXAONE, 코드북 v2 (B 절반 대비)",
        "",
        "| 조건 | 셀별 주제 TVD | realization |",
        "|---|---|---|",
        *[
            f"| {x.condition} | {x.tvd:.3f} | {'' if np.isnan(x.realization) else f'{x.realization:.2f}'} |"
            for x in d.itertuples()
        ],
        "",
        f"문지기 (ADR-033): C2 < C0 이고 realization ≥ 0.7 → **{'통과' if ok else '실패'}**.",
        "",
        "C0 = 앵커만으로 쓴 현행 view_on_seoul · C1 = 주변분포 주제 · C2 = 관람 여부 표 주제 (모두 A 절반 평가표). marginal/floor_AB/oracle_T7 은 실제 답 A 절반에서 만든 기준; draw_only 는 표에서 뽑기만 한 바닥.",
    ]
    (REPORTS / "era_theme_v2_pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0 if ok else 2


def stage_bank() -> int:
    rng = np.random.default_rng(SEED + 1)
    smp = Sampler(rng)
    bank = [json.loads(line) for line in BANK_PATH.open(encoding="utf-8") if line.strip()]
    units = []
    for r in bank:
        q = r["quant"]
        c = persona_cell(q)
        units.append((r["persona_id"], c, anchor_card(q), smp.cell(c)))
    gen_all(WISH_PATH, units)
    return 0


def stage_bank_eval() -> int:
    recs = {}
    for line in WISH_PATH.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["persona_id"]] = r
    df = pd.DataFrame(recs.values())
    n_empty = int((df.text.str.len() < 10).sum())
    n_digit = int(
        df.text.str.contains(r"\d+\s*세(?!대)").sum()
    )  # numeric ages only (anchor labels 「11회」「60대」「50만원」 are allowed, as in the 12 slots)
    n_first = int(df.text.str.contains("나는|나에게|내가|저는|제가").sum())
    n_sent = df.text.str.count(r"[.!?。]\s|[.!?。]$")
    rng = np.random.default_rng(SEED + 2)
    per = max(1, N_BANK_EVAL // df.cell.nunique())
    samp = pd.concat(
        [
            g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 31)))
            for _, g in df.groupby("cell")
        ],
        ignore_index=True,
    )
    codes = code_all("bank_sample", {r.persona_id: r.text for r in samp.itertuples()})
    cells = dict(zip(samp.persona_id, samp.cell))
    rz = float(
        np.mean([codes[p] == t for p, t in zip(samp.persona_id, samp.wish_theme) if p in codes])
    )
    # (a) Against the public table — does the prose reproduce the drawn table (attendance cells; 기타 mass put back onto the margin)
    t7 = pd.read_parquet(GDIR / "T7_theme_by_cell_release.parquet")
    smp = Sampler(np.random.default_rng(0))
    truth_rel = {c: smp.expected("x|" + c) for c in t7.cell.unique()}
    wts_rel = {c: int(g.n_cell.iloc[0]) for c, g in t7.groupby("cell")}
    att_cells = {p: att_of(c) for p, c in cells.items()}
    tvd_rel = score(codes, att_cells, truth_rel, wts_rel)
    # (b) Reference: against half B (optimistic, because the public table also used B)
    truth, wts = _truth()
    tvd_B = score(codes, cells, truth, wts)
    theme_share = df.wish_theme.value_counts(normalize=True).sort_index()
    pd.DataFrame(
        [
            {
                "n_bank": len(df),
                "n_sample": len(samp),
                "tvd_release": tvd_rel,
                "tvd_B": tvd_B,
                "realization": rz,
                "n_empty": n_empty,
                "n_digit": n_digit,
                "n_first_person": n_first,
                "sent_p95": float(n_sent.quantile(0.95)),
            }
        ]
    ).to_csv(REPORTS / "era_theme_v2_bank.csv", index=False)
    lines = [
        f"# 13 번째 칸 wish_for_seoul — 2024 뱅크 {len(df):,} 명 (층화 표본 {len(samp):,} 명을 EXAONE 이 코딩)",
        "",
        f"- 공개표 대비 주제 TVD **{tvd_rel:.3f}** (뽑은 표를 글이 재현하는 정도; realization {rz:.2f})",
        f"- 참고: B 절반 대비 {tvd_B:.3f} (공개표는 전체 응답자로 맞췄으므로 이 값은 낙관적 — 정직한 out-of-sample 수치는 파일럿)",
        f"- 자동 검사: 빈 글 {n_empty} · 숫자 나이 {n_digit} · 1 인칭 {n_first} · 문장 수 95 분위 {n_sent.quantile(0.95):.0f}",
        "- 뱅크의 wish_theme 분포: "
        + ", ".join(f"{CODEBOOK[int(t)][1]} {s:.2f}" for t, s in theme_share.items()),
    ]
    (REPORTS / "era_theme_v2_bank.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n%s", "\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["code-real", "table", "pilot", "bank", "bank-eval"])
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    return {
        "code-real": stage_code_real,
        "table": stage_table,
        "pilot": stage_pilot,
        "bank": stage_bank,
        "bank-eval": stage_bank_eval,
    }[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
