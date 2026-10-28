"""Era holdout at full scale — ADR-010 (wave replication).

Items: intersection canonical 4 + domain primary 5.
Sample: release narrative n=300; twin anchors n=300 (internal).
SSR → embed → cell (age_band×attend_any) weighted TVD · baselines.
τ: reuse `tau.json` frozen on 2024 (do not re-freeze per wave).

Run:
  uv run python scripts/era_holdout_run.py --wave 2022
  uv run python scripts/era_holdout_run.py --wave 2024 --skip-answer
Output: data/eval/era_holdout/w{wave}/ … · outputs/reports/era_holdout_{wave}.md
2024 legacy (flat) answers/cos are a read fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.generate import load_run
from sfac_persona.grounding.official import (
    build_official_anchor_frame,
    load_general_master,
    load_official_value_labels,
)
from sfac_persona.llm_client import get_client
from sfac_persona.narrative import anchor_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_run")

OUT = settings.data_root / "eval" / "era_holdout"
REPORTS = settings.outputs_dir / "reports"
SEED = 20260910
N_RELEASE = 300
N_TWIN = 300
CELL = ["age_band", "attend_any"]
ALLOWED_WAVES = {2020, 2022, 2024}
CONCURRENCY = 4

DOMAIN_PRIMARY = [
    ("attitude", "other", "arts_importance"),
    ("satisfaction", "satisfaction", "attendance_overall"),
    ("satisfaction", "satisfaction", "cultural_life_overall"),
    ("attitude", "frequency", "effect_overall_happiness"),
    ("attitude", "frequency", "effect_stress_relieved"),
]

QUESTIONS: dict[str, str] = {
    "leisure_activity_1st": "주말이나 휴일에는 주로 무엇을 하며 시간을 보내십니까?",
    "leisure_activity_wanted_1st": "여건이 된다면 앞으로 가장 하고 싶은 여가활동은 무엇입니까?",
    "leisure_time_weekday": "평일에는 하루에 여가시간이 대략 몇 시간 있습니까?",
    "leisure_time_weekend": "주말·휴일에는 하루에 여가시간이 대략 몇 시간 있습니까?",
    "arts_importance": "문화예술은 당신의 삶에서 얼마나 중요합니까?",
    "attendance_overall": "지난 1년 문화예술 관람에 전반적으로 얼마나 만족하십니까?",
    "cultural_life_overall": "본인의 문화생활에 전반적으로 얼마나 만족하십니까?",
    "effect_overall_happiness": "문화예술 활동이 전반적인 행복감에 얼마나 도움이 됩니까?",
    "effect_stress_relieved": "문화예술 활동이 스트레스 완화에 얼마나 도움이 됩니까?",
}

# Reference sentences by scale length.
#
# Fix 2026-09-11: previously a single 7-point sentence set was used, and only the codes
# present in the wave were selected from it.
# On a wave that uses 1–5 and 1–4 scales, such as 2020, **the scale maximum is then
# worded as 「보통」 or 「다소 높음」**,
# and there is no reference sentence for 「매우 그렇다」, so SSR cannot hit the top end.
# If the observed codes are contiguous 1..k, the anchors for that k are now used.
# k=7 is unchanged from before, so the 2022 and 2024 results do not change.
LIKERT_REFS_BY_K: dict[int, dict[str, list[str]]] = {
    4: {
        "1": ["전혀 그렇지 않다. 거의 해당되지 않는다.", "1점 매우 낮음"],
        "2": ["그렇지 않은 편이다. 낮은 편이다.", "2점 낮음"],
        "3": ["그런 편이다. 높은 편이다.", "3점 높음"],
        "4": ["매우 그렇다. 아주 높다.", "4점 매우 높음"],
    },
    5: {
        "1": ["전혀 그렇지 않다. 거의 해당되지 않는다.", "1점 매우 낮음"],
        "2": ["그렇지 않은 편이다. 낮은 편이다.", "2점 낮음"],
        "3": ["보통이다. 어느 쪽도 아니다.", "3점 보통"],
        "4": ["그런 편이다. 높은 편이다.", "4점 높음"],
        "5": ["매우 그렇다. 아주 높다.", "5점 매우 높음"],
    },
    7: {
        "1": ["전혀 그렇지 않다. 거의 해당되지 않는다.", "1점 매우 낮음"],
        "2": ["대체로 그렇지 않다. 낮은 편이다.", "2점 낮음"],
        "3": ["약간 부족한 편이다.", "3점 다소 낮음"],
        "4": ["보통이다. 어느 쪽도 아니다.", "4점 보통"],
        "5": ["어느 정도 그렇다. 괜찮은 편이다.", "5점 다소 높음"],
        "6": ["대체로 그렇다. 높은 편이다.", "6점 높음"],
        "7": ["매우 그렇다. 아주 높다.", "7점 매우 높음"],
    },
}
LIKERT7_REFS = LIKERT_REFS_BY_K[7]  # backward compatible

PROMPT = """아래 인물의 처지에서, 보기 목록 없이 각 질문에 한두 문장으로만 답하십시오. JSON 객체 하나만 출력.
키: {keys}

[인물]
{persona}

[질문]
{questions}
"""

ANCHOR_SKIP = {
    "persona_id",
    "w",
    "width",
    "depth",
    "survey_year",
    "survey_era",
    "weighting",
    "instrument_regime",
    "ablation",
    "row_uid",
}


def _read_jsonl(path: Path) -> list[dict]:
    """Read a jsonl file in full. Do not leak the file handle."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wave_dir(wave: int) -> Path:
    return OUT / f"w{wave}"


def resolve_answers_path(wave: int, name: str) -> Path:
    """Prefer w{wave}/name; for 2024 fall back to legacy flat OUT/name."""
    primary = wave_dir(wave) / name
    if primary.exists():
        return primary
    legacy = OUT / name
    if wave == 2024 and legacy.exists():
        return legacy
    return primary


def _tvd(p: pd.Series, q: pd.Series) -> float:
    idx = sorted(set(map(str, p.index)) | set(map(str, q.index)))
    pm = {str(k): float(v) for k, v in p.items()}
    qm = {str(k): float(v) for k, v in q.items()}
    pv = np.array([pm.get(k, 0.0) for k in idx])
    qv = np.array([qm.get(k, 0.0) for k in idx])
    if pv.sum():
        pv /= pv.sum()
    if qv.sum():
        qv /= qv.sum()
    return 0.5 * float(np.abs(pv - qv).sum())


def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = (x - np.nanmax(x)) / tau
    e = np.exp(z)
    return e / e.sum()


def _leisure_label_keys(wave: int) -> tuple[str, str]:
    """2024: A3_1/A5_1 · 2022: A3x1/A5x1."""
    labs = load_official_value_labels(wave)
    a3 = next((k for k in ("A3_1", "A3x1") if labs.get(k)), "A3_1")
    a5 = next((k for k in ("A5_1", "A5x1") if labs.get(k)), a3)
    return a3, a5


def _activity_options(wave: int) -> list[str]:
    a3_key, _ = _leisure_label_keys(wave)
    labs = load_official_value_labels(wave).get(a3_key, {})
    items: dict[int, str] = {}
    for k, v in labs.items():
        try:
            items[int(float(k))] = v
        except ValueError:
            continue
    return [items[i] for i in sorted(items)]


def _hour_options(wave: int, var: str) -> list[str]:
    """Cover observed hour codes for the wave (2022 weekday≤12, weekend≤15)."""
    defaults = {"leisure_time_weekday": 8, "leisure_time_weekend": 12}
    mx = defaults.get(var, 12)
    canon = pd.read_parquet(settings.official_person_dir / "official_person_canonical.parquet")
    sub = canon[(canon["wave"] == wave) & (canon["canonical_var"] == var)]["value"]
    if len(sub):
        mx = max(mx, int(float(sub.max())))
    return [str(i) for i in range(mx + 1)]


def _domain_likert_refs(wave: int, item: str) -> dict[str, list[str]]:
    """Observed Likert codes for the wave, with anchors matched to the scale LENGTH.

    In 2020, importance and satisfaction are 1–5 and effect is 1–4. Matching only the
    code count and reusing 7-point sentences words the scale maximum as 「보통」, so SSR
    structurally misses the top end (fix 2026-09-11).
    """
    canon_or_dom = pd.read_parquet(settings.official_person_dir / "official_person_domain.parquet")
    # map item → family/measure from DOMAIN_PRIMARY
    fam = meas = None
    for f, m, it in DOMAIN_PRIMARY:
        if it == item:
            fam, meas = f, m
            break
    if fam is None:
        return {k: list(v) for k, v in LIKERT_REFS_BY_K[7].items()}
    sub = canon_or_dom[
        (canon_or_dom["wave"] == wave)
        & (canon_or_dom["family"] == fam)
        & (canon_or_dom["measure"] == meas)
        & (canon_or_dom["canonical_item"] == item)
    ]["value"]
    codes = sorted({int(float(v)) for v in sub.dropna()})
    if not codes:
        return {k: list(v) for k, v in LIKERT_REFS_BY_K[7].items()}

    k = max(codes)
    contiguous = codes == list(range(1, k + 1))
    scale = LIKERT_REFS_BY_K.get(k) if contiguous else None
    if scale is None:
        # If observed codes are not contiguous 1..k, or the scale is unsupported,
        # linearly interpolate from the nearest scale.
        scale = _interpolated_scale(codes)
        log.warning(
            "wave=%d %s: nonstandard scale codes=%s — using linearly interpolated anchors",
            wave,
            item,
            codes,
        )
    else:
        log.info("wave=%d %s: %d-point scale anchors", wave, item, k)
    return {str(c): list(scale[str(c)]) for c in codes}


def _interpolated_scale(codes: list[int]) -> dict[str, list[str]]:
    """Assign 7-point anchors to a non-contiguous code set, proportional to position."""
    seven = LIKERT_REFS_BY_K[7]
    lo, hi = min(codes), max(codes)
    out: dict[str, list[str]] = {}
    for c in codes:
        frac = 0.0 if hi == lo else (c - lo) / (hi - lo)
        slot = str(round(frac * 6) + 1)
        out[str(c)] = list(seven[slot])
    return out


def write_refs(wave: int) -> dict[str, dict[str, list[str]]]:
    d = wave_dir(wave)
    d.mkdir(parents=True, exist_ok=True)
    acts = _activity_options(wave)
    if not acts:
        raise SystemExit(f"wave={wave}: leisure activity labels empty")
    refs: dict[str, dict[str, list[str]]] = {}
    for var in ("leisure_activity_1st", "leisure_activity_wanted_1st"):
        refs[var] = {o: [f"나는 주로 {o}을(를) 한다.", o[:40]] for o in acts}
    for var in ("leisure_time_weekday", "leisure_time_weekend"):
        hours = _hour_options(wave, var)
        refs[var] = {h: [f"하루 여가시간은 대략 {h}시간이다.", f"약 {h}시간"] for h in hours}
    for _, _, item in DOMAIN_PRIMARY:
        refs[item] = _domain_likert_refs(wave, item)
    path = d / "ssr_refs.json"
    path.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    # keep root copy for tooling that expects flat path (2024 legacy tools)
    if wave == 2024:
        (OUT / "ssr_refs.json").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return refs


def _embed(wave: int, selftest: bool = False, pattern: str | None = None) -> str | None:
    """Embedding step. In selftest mode, return the G4-refs result line
    (so the report is written by code, not by hand).
    """
    root = Path(__file__).resolve().parents[1]
    py = root / ".venv-embed" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)  # `uv sync --extra embed` installs the embedding stack here
    d = wave_dir(wave)
    cmd = [str(py), str(root / "scripts" / "ssr_embed.py"), "--dir", str(d)]
    if selftest:
        cmd.append("--selftest")
    else:
        cmd.extend(["--pattern", pattern or "answers_*_ssr.jsonl"])
    log.info("embed: %s", " ".join(cmd))
    if not selftest:
        subprocess.check_call(cmd, cwd=str(root))
        return None
    res = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=True)
    line = next(
        (ln.strip() for ln in res.stdout.splitlines() if ln.startswith("GATE G4-refs")), None
    )
    if line:
        log.info("%s", line)
    return line


def _persona_blob(quant: dict, narrative: dict | None) -> str:
    bits = [anchor_block(quant)]
    if narrative:
        if narrative.get("summary"):
            bits.append(f"한 줄: {narrative['summary']}")
        if narrative.get("weekend_routine"):
            bits.append(f"주말: {narrative['weekend_routine']}")
    return "\n".join(bits)


def _answer_one(client, model: str, pid: str, quant: dict, narrative: dict | None) -> dict:
    keys = list(QUESTIONS)
    prompt = PROMPT.format(
        keys=", ".join(keys),
        persona=_persona_blob(quant, narrative),
        questions="\n".join(f"- {k}: {QUESTIONS[k]}" for k in keys),
    )
    texts = {k: None for k in keys}
    attempt = -1
    for attempt in range(3):
        res = client.chat(
            model=model,
            system=None,
            user=prompt,
            temperature=0.7,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        try:
            raw = res.text[res.text.find("{") : res.text.rfind("}") + 1]
            obj = json.loads(raw)
            texts = {k: str(obj.get(k, "")).strip() or None for k in keys}
            if sum(1 for v in texts.values() if v) >= len(keys) - 2:
                break
        except Exception:  # noqa: BLE001, S112 — if the response is not JSON, try again
            continue
    return {
        "persona_id": pid,
        "condition": "narrative" if narrative else "anchors",
        "mode": "ssr",
        "texts": texts,
        "quant": {k: str(v) for k, v in quant.items() if k in CELL or k in ("sex",)},
        "attempt": attempt,
    }


def eval_bank_path(wave: int) -> Path:
    """Frozen eval bank. It holds the n=300 narratives that holdout actually scored.

    So that holdout stays reproducible even if full release generation (B1) overwrites
    `data/synthetic/era_*_narrative/`, pin the personas used for evaluation inside the
    eval directory (ADR-012).
    """
    return wave_dir(wave) / "eval_bank.jsonl"


def load_eval_bank(wave: int, n: int = N_RELEASE) -> list[dict]:
    """If a frozen bank exists, use it; otherwise draw from the release run and freeze."""
    path = eval_bank_path(wave)
    if path.exists():
        recs = _read_jsonl(path)
        log.info("eval bank (frozen) wave=%d n=%d ← %s", wave, len(recs), path.name)
        return recs
    run_name = f"era_{wave}_narrative"
    recs = load_run(run_name)
    if len(recs) < n:
        raise SystemExit(f"{run_name} n={len(recs)} < {n}. Generate narratives up to {n} first.")
    rng = np.random.default_rng(SEED + wave)
    if len(recs) > n:
        idx = rng.choice(len(recs), size=n, replace=False)
        recs = [recs[i] for i in sorted(idx)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("eval bank frozen wave=%d n=%d → %s", wave, len(recs), path.name)
    return recs


def run_release_answers(wave: int, n: int = N_RELEASE) -> Path:
    recs = load_eval_bank(wave, n)
    path = wave_dir(wave) / "answers_release_narrative_ssr.jsonl"
    return _run_answer_file(path, [(r["persona_id"], r["quant"], r.get("narrative")) for r in recs])


def _attach_row_uid(frame: pd.DataFrame, wave: int) -> pd.DataFrame:
    """build_official_anchor_frame is a positional join that assumes the same filter and
    the same order as the master.

    If that assumption breaks, anchors and holdout labels silently misalign — stop that here.
    """
    master = load_general_master(wave, allow_unweighted=False)
    if len(master) != len(frame):
        raise SystemExit(
            f"wave={wave}: anchor frame n={len(frame)} != master n={len(master)}. "
            "Filters in build_official_anchor_frame and load_general_master diverged — "
            "positional join refused."
        )
    out = frame.copy()
    out["row_uid"] = master["row_uid"].to_numpy()
    return out


def _twin_units(wave: int, n: int = N_TWIN) -> list[tuple[str, dict, None]]:
    real = _attach_row_uid(build_official_anchor_frame(wave), wave)
    rng = np.random.default_rng(SEED + wave + 1)
    take = real.sample(
        n=min(n, len(real)), random_state=int(rng.integers(0, 1_000_000))
    ).reset_index(drop=True)
    units = []
    for _, row in take.iterrows():
        q = {k: str(v) for k, v in row.items() if k not in ANCHOR_SKIP and pd.notna(v)}
        pid = f"twin-{row['row_uid']}"
        units.append((pid, q, None))
    return units


def run_twin_answers(wave: int, n: int = N_TWIN) -> Path:
    path = wave_dir(wave) / "answers_twin_anchors_ssr.jsonl"
    return _run_answer_file(path, _twin_units(wave, n))


def _run_answer_file(path: Path, units: list[tuple[str, dict, dict | None]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if path.exists():
        done = {r["persona_id"] for r in _read_jsonl(path)}
    todo = [u for u in units if u[0] not in done]
    log.info("%s: %d done, %d todo", path, len(done), len(todo))
    if not todo:
        return path
    client = get_client()
    model = settings.llm_default_model
    with (
        open(path, "a", encoding="utf-8") as f,
        ThreadPoolExecutor(max_workers=CONCURRENCY) as pool,
    ):
        futs = {pool.submit(_answer_one, client, model, pid, q, n): pid for pid, q, n in todo}
        t0 = time.time()
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 25 == 0 or k == len(todo):
                log.info("  %d/%d (%.1f/min)", k, len(todo), k / max(time.time() - t0, 1e-6) * 60)
    return path


def _keys_weights(wave: int) -> pd.DataFrame:
    keys = pd.read_parquet(settings.official_person_dir / "official_person_keys.parquet")
    return keys[(keys["wave"] == wave) & (keys["frame_group"] == "general")]


def real_item_frame(wave: int) -> pd.DataFrame:
    """anchors + holdout labels + w + cell. row_uid aligns with official master order."""
    out = _attach_row_uid(build_official_anchor_frame(wave), wave)
    k = _keys_weights(wave)
    wmap = k.set_index("row_uid")["weight"]
    canon = pd.read_parquet(settings.official_person_dir / "official_person_canonical.parquet")
    a3_key, a5_key = _leisure_label_keys(wave)
    all_labs = load_official_value_labels(wave)
    labs_a3 = all_labs.get(a3_key, {})
    labs_a5 = all_labs.get(a5_key, labs_a3)

    def lab_map(src_labs: dict, x: object) -> str:
        try:
            key = str(round(float(x)))
        except (TypeError, ValueError):
            return str(x)
        return src_labs.get(key, key)

    for var, labs in (
        ("leisure_activity_1st", labs_a3),
        ("leisure_activity_wanted_1st", labs_a5),
        ("leisure_time_weekday", None),
        ("leisure_time_weekend", None),
    ):
        sub = canon[(canon["wave"] == wave) & (canon["canonical_var"] == var)]
        m = sub.set_index("row_uid")["value"]

        def _map_one(u: object, _m=m, _labs=labs) -> str | None:
            if u not in _m.index:
                return None
            val = _m.loc[u]
            if pd.isna(val):
                return None
            if _labs is None:
                return str(int(float(val)))
            return lab_map(_labs, val)

        out[var] = out["row_uid"].map(_map_one)

    dom = pd.read_parquet(settings.official_person_dir / "official_person_domain.parquet")
    for fam, meas, item in DOMAIN_PRIMARY:
        sub = dom[
            (dom["wave"] == wave)
            & (dom["family"] == fam)
            & (dom["measure"] == meas)
            & (dom["canonical_item"] == item)
        ].drop_duplicates("row_uid")
        m = sub.set_index("row_uid")["value"]

        def _dom(u: object, _m=m) -> str | None:
            if u not in _m.index or pd.isna(_m.loc[u]):
                return None
            return str(int(float(_m.loc[u])))

        out[item] = out["row_uid"].map(_dom)
    out["w"] = out["row_uid"].map(wmap).astype(float)
    return out


def _cos_path_for(answers_path: Path) -> Path:
    """Match scripts/ssr_embed.py naming: answers_foo_ssr.jsonl → cos_foo.jsonl (same dir)."""
    name = answers_path.name
    parent = answers_path.parent
    if name.startswith("answers_") and name.endswith("_ssr.jsonl"):
        return parent / ("cos_" + name[len("answers_") : -len("_ssr.jsonl")] + ".jsonl")
    return parent / f"cos_{answers_path.stem}.jsonl"


def load_cos_probs(
    answers_path: Path, refs: dict, tau: float
) -> tuple[list[dict], dict[str, list[str]]]:
    cos_path = _cos_path_for(answers_path)
    if not cos_path.exists():
        raise FileNotFoundError(cos_path)
    cos = {r["persona_id"]: r for r in _read_jsonl(cos_path)}
    ans = _read_jsonl(answers_path)
    opt_order = {var: list(opts.keys()) for var, opts in refs.items()}
    for r in ans:
        pid = r["persona_id"]
        cdict = cos.get(pid, {}).get("cos", {})
        probs = {}
        for var, order in opt_order.items():
            c = cdict.get(var)
            if c and len(c) == len(order):
                probs[var] = dict(
                    zip(order, _softmax(np.asarray(c, dtype=float), tau).tolist(), strict=True)
                )
            else:
                probs[var] = None
        r["probs"] = probs
    return ans, opt_order


def cell_mean_tvd(
    ans: list[dict],
    real: pd.DataFrame,
    item: str,
    opt_order: list[str],
) -> float:
    rows = []
    for r in ans:
        q = r.get("quant") or {}
        cell = tuple(q.get(c, "") for c in CELL)
        p = r.get("probs", {}).get(item)
        if not p:
            continue
        rows.append({"cell": cell, "p": p})
    if not rows:
        return float("nan")
    adf = pd.DataFrame(rows)
    tvs = []
    weights = []
    for cell, g in adf.groupby("cell"):
        age, att = cell
        rw = real[(real["age_band"] == age) & (real["attend_any"] == att) & real[item].notna()]
        if len(rw) < 5 or len(g) < 1:
            continue
        real_d = rw.groupby(item)["w"].sum()
        real_d = real_d / real_d.sum()
        acc = {o: 0.0 for o in opt_order}
        for p in g["p"]:
            for o in opt_order:
                acc[o] += float(p.get(o, 0.0))
        syn_d = pd.Series({o: acc[o] / len(g) for o in opt_order})
        tvs.append(_tvd(real_d, syn_d))
        weights.append(len(rw))
    if not tvs:
        return float("nan")
    w = np.asarray(weights, dtype=float)
    return float(np.average(tvs, weights=w))


def baselines(real: pd.DataFrame, items: list[str], n_syn: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for item in items:
        ok = real[item].notna()
        y = real.loc[ok, item].astype(str)
        w = real.loc[ok, "w"].astype(float)
        tab = pd.DataFrame({"y": y, "w": w}).groupby("y")["w"].sum()
        tab = tab / tab.sum()
        cats = tab.index.to_numpy()
        probs = tab.to_numpy(dtype=float)
        floor_tvs = []
        for _ in range(15):
            draws = rng.choice(cats, size=n_syn, p=probs)
            syn = pd.Series(draws).value_counts(normalize=True)
            floor_tvs.append(_tvd(tab, syn))
        floor = float(np.mean(floor_tvs))

        cell_lookup_tvd = []
        for (_, _), g in real.loc[ok].groupby(["sex", "age_band"]):
            if len(g) < 8:
                continue
            d = g.groupby(item)["w"].sum()
            d = d / d.sum()
            cell_lookup_tvd.append(_tvd(tab, d))
        lookup_div = float(np.mean(cell_lookup_tvd)) if cell_lookup_tvd else float("nan")

        # Learned baseline.
        #
        # Fix 2026-09-11: the previous version (1) fit without sample_weight, (2) used the
        # argmax hard-label distribution, and (3) predicted in-sample. The target `tab` is
        # the weighted marginal and the SSR side is a mean of soft probabilities, so all
        # three unfairly disadvantage the baseline.
        # Now: weighted fit + OOF + both soft and argmax.
        X = real.loc[ok, ["sex", "age_band", "attend_any"]].astype(str)
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xm = enc.fit_transform(X)
        yv = y.to_numpy()
        wv = w.to_numpy(dtype=float)
        if len(set(yv)) < 2 or len(yv) < 30:
            learned_soft = learned_argmax = float("nan")
        else:
            cats_all = sorted(set(yv))
            soft = np.zeros((len(yv), len(cats_all)))
            hard = np.empty(len(yv), dtype=object)
            kf = KFold(5, shuffle=True, random_state=SEED)
            for tr, te in kf.split(Xm):
                if len(set(yv[tr])) < 2:
                    continue
                clf = LogisticRegression(max_iter=2000)
                clf.fit(Xm[tr], yv[tr], sample_weight=wv[tr])
                pr = (
                    pd.DataFrame(clf.predict_proba(Xm[te]), columns=clf.classes_.astype(str))
                    .reindex(columns=[str(c) for c in cats_all])
                    .fillna(0.0)
                    .to_numpy()
                )
                soft[te] = pr / np.clip(pr.sum(1, keepdims=True), 1e-12, None)
                hard[te] = clf.predict(Xm[te])
            wsum = wv.sum()
            soft_d = pd.Series(
                {str(c): float((soft[:, j] * wv).sum() / wsum) for j, c in enumerate(cats_all)}
            )
            learned_soft = _tvd(tab, soft_d)
            got = pd.notna(hard)
            hard_d = (
                pd.DataFrame({"y": hard[got].astype(str), "w": wv[got]}).groupby("y")["w"].sum()
                if got.any()
                else pd.Series(dtype=float)
            )
            learned_argmax = _tvd(tab, hard_d) if len(hard_d) else float("nan")

        rows.append(
            {
                "item": item,
                "noise_floor_tvd": floor,
                "lookup_vs_marginal_tvd": lookup_div,
                "learned_oof_soft_tvd": learned_soft,
                "learned_oof_argmax_tvd": learned_argmax,
                "n_real": int(ok.sum()),
                "n_cats": len(tab),
            }
        )
    return pd.DataFrame(rows)


def load_frozen_tau() -> tuple[float, dict]:
    path = OUT / "tau.json"
    if not path.exists():
        raise SystemExit(f"frozen τ missing: {path}. Run the 2024 full scale first.")
    meta = json.loads(path.read_text(encoding="utf-8"))
    return float(meta["tau"]), meta


def entropy_gap_at_tau(wave: int, refs: dict, real: pd.DataFrame, tau: float) -> float | None:
    """Log |mean personal entropy - mean cell entropy| at frozen tau (no re-freeze)."""
    path = resolve_answers_path(wave, "answers_twin_anchors_ssr.jsonl")
    if not path.exists():
        path = resolve_answers_path(wave, "answers_release_narrative_ssr.jsonl")
    if not path.exists():
        return None
    cos_path = _cos_path_for(path)
    if not cos_path.exists():
        return None
    cos = {r["persona_id"]: r["cos"] for r in _read_jsonl(cos_path)}
    ans_meta = {r["persona_id"]: r for r in _read_jsonl(path)}
    pers_ents: list[float] = []
    cell_ents: list[float] = []
    for pid, cdict in cos.items():
        meta = ans_meta.get(pid, {})
        q = meta.get("quant") or {}
        age, att = q.get("age_band"), q.get("attend_any")
        for var, order in ((v, list(refs[v])) for v in refs):
            c = cdict.get(var)
            if not c or len(c) != len(order):
                continue
            p = _softmax(np.asarray(c, dtype=float), tau)
            p = p[p > 0]
            pers_ents.append(float(-(p * np.log(p)).sum()))
            if age and att and var in real.columns:
                rw = real[
                    (real["age_band"] == age) & (real["attend_any"] == att) & real[var].notna()
                ]
                if len(rw) >= 5:
                    d = rw.groupby(var)["w"].sum()
                    d = (d / d.sum()).to_numpy(dtype=float)
                    d = d[d > 0]
                    cell_ents.append(float(-(d * np.log(d)).sum()))
    if not pers_ents or not cell_ents:
        return None
    gap = abs(float(np.mean(pers_ents)) - float(np.mean(cell_ents)))
    log.info(
        "wave=%d frozen tau=%.3f gap=%.4f (pers_ent=%.3f cell_ent=%.3f)",
        wave,
        tau,
        gap,
        float(np.mean(pers_ents)),
        float(np.mean(cell_ents)),
    )
    return gap


TAU_GRID = (0.02, 0.03, 0.05, 0.07, 0.1)
TAU_RULE = "min |mean personal entropy - mean cell entropy|"


def freeze_tau(refs: dict, write: bool = False) -> float:
    """Freeze the SSR softmax temperature once, on the 2024 twin subsample (ADR-010).

    For each tau on TAU_GRID, compare the mean entropy of the personas' option distributions with the mean
    entropy of the real answer distribution in their cell (age band x attendance), and keep the tau with the
    smallest gap. Other waves reuse the 2024 value. Without `write`, only check against the stored value.
    """
    real = real_item_frame(2024)
    gaps = {t: entropy_gap_at_tau(2024, refs, real, t) for t in TAU_GRID}
    if any(g is None for g in gaps.values()):
        raise SystemExit("2024 twin answers or cosines missing; cannot freeze tau")
    tau = min(gaps, key=gaps.get)
    for t, g in gaps.items():
        log.info("tau=%.2f gap=%.4f%s", t, g, "  <- chosen" if t == tau else "")
    path = OUT / "tau.json"
    if write:
        meta = {
            "tau": tau,
            "grid": list(TAU_GRID),
            "rule": TAU_RULE,
            "wave": 2024,
            "adr": "ADR-010",
        }
        path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        log.info("wrote %s", path)
    elif path.exists():
        stored = float(json.loads(path.read_text(encoding="utf-8"))["tau"])
        log.info(
            "stored tau=%.2f; recomputed tau=%.2f; %s",
            stored,
            tau,
            "match" if stored == tau else "MISMATCH",
        )
    return tau


def evaluate(wave: int, tau: float, refs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    real = real_item_frame(wave)
    items = list(QUESTIONS)
    base = baselines(real, items, N_RELEASE)
    rows = []
    for tag, name in (
        ("release_narrative", "answers_release_narrative_ssr.jsonl"),
        ("twin_anchors", "answers_twin_anchors_ssr.jsonl"),
    ):
        path = resolve_answers_path(wave, name)
        if not path.exists():
            continue
        ans, opt_order = load_cos_probs(path, refs, tau)
        for item in items:
            tvd = cell_mean_tvd(ans, real, item, opt_order[item])
            rows.append(
                {
                    "wave": wave,
                    "tag": tag,
                    "item": item,
                    "cell_mean_tvd": tvd,
                    "tau": tau,
                    "n": len(ans),
                }
            )
    return pd.DataFrame(rows), base


def write_report(
    wave: int,
    scores: pd.DataFrame,
    base: pd.DataFrame,
    tau: float,
    tau_meta: dict,
    entropy_gap: float | None,
    g4_line: str | None = None,
) -> Path:
    wd_path = REPORTS / "era_holdout_scores.csv"
    sha = tau_meta.get("sha256", "")
    lines = [
        f"# Era holdout {wave} — ADR-010 복제",
        "",
        f"- τ 재사용: `{tau}` · sha256=`{sha}` (`data/eval/era_holdout/tau.json`, 2024 동결)",
        f"- {wave} twin entropy gap at frozen τ: "
        + (f"{entropy_gap:.4f}" if entropy_gap is not None else "n/a"),
        (f"- {g4_line}" if g4_line else "- G4-refs: selftest 생략"),
        f"- release narrative n={N_RELEASE}; twin anchors n={N_TWIN} (내부)",
        "- 문항: canonical 4 + domain primary 5",
        "- 셀: `age_band × attend_any` · 모드 SSR",
        f"- 산출: `data/eval/era_holdout/w{wave}/`",
        f"- 스크립트: `scripts/era_holdout_run.py --wave {wave}`",
        "",
        "## 셀 가중 mean TVD (SSR)",
        "",
    ]
    if not scores.empty:
        lines.append(
            scores.to_markdown(index=False)
            if hasattr(scores, "to_markdown")
            else scores.to_string(index=False)
        )
    lines += ["", "## 기준선 (실측)", ""]
    if not base.empty:
        lines.append(
            base.to_markdown(index=False)
            if hasattr(base, "to_markdown")
            else base.to_string(index=False)
        )
    if wd_path.exists():
        wd = pd.read_csv(wd_path)
        wsub = wd[(wd["wave"] == wave) & (wd["layer"] == "width_depth")]
        if not wsub.empty:
            lines += [
                "",
                "## 폭·깊이 (quant, 참고)",
                "",
                wsub.to_markdown(index=False)
                if hasattr(wsub, "to_markdown")
                else wsub.to_string(index=False),
            ]
    lines += ["", "H1/H2는 본 표만으로 확정하지 않는다. Atlas-first: 측정 결과 보고.", ""]
    path = REPORTS / f"era_holdout_{wave}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    scores.to_csv(REPORTS / f"era_holdout_{wave}_scores.csv", index=False)
    base.to_csv(REPORTS / f"era_holdout_{wave}_baselines.csv", index=False)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2022)
    ap.add_argument("--skip-answer", action="store_true")
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--n-release", type=int, default=N_RELEASE)
    ap.add_argument("--n-twin", type=int, default=N_TWIN)
    ap.add_argument(
        "--freeze-tau",
        action="store_true",
        help="recompute the 2024 SSR temperature and check it against tau.json (with --write, overwrite it)",
    )
    ap.add_argument("--write", action="store_true", help="with --freeze-tau: write tau.json")
    args = ap.parse_args()
    if args.wave not in ALLOWED_WAVES:
        raise SystemExit(f"--wave must be one of {sorted(ALLOWED_WAVES)} (got {args.wave})")

    wave_dir(args.wave).mkdir(parents=True, exist_ok=True)
    if args.freeze_tau:
        freeze_tau(write_refs(2024), write=args.write)
        return
    refs = write_refs(args.wave)
    g4_line: str | None = None
    if not args.skip_selftest:
        try:
            g4_line = _embed(args.wave, selftest=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("selftest: %s", exc)

    if not args.skip_answer:
        run_release_answers(args.wave, args.n_release)
        run_twin_answers(args.wave, args.n_twin)

    if not args.skip_embed:
        _embed(args.wave, selftest=False, pattern="answers_*_ssr.jsonl")

    tau, tau_meta = load_frozen_tau()
    log.info("reusing frozen tau=%.3f sha=%s", tau, tau_meta.get("sha256"))
    real = real_item_frame(args.wave)
    gap = entropy_gap_at_tau(args.wave, refs, real, tau)
    scores, base = evaluate(args.wave, tau, refs)
    path = write_report(args.wave, scores, base, tau, tau_meta, gap, g4_line)
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
