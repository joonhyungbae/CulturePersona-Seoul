"""Positive control for the narrative-layer holdout.

Rewrite the frozen eval bank's 300 twelve-slot narratives with the same generator,
temperature, and prompt skeleton as release.
Holdout-item answers are not in the bank (only synthetic anchors). Within a cell
(age band × attendance), draw one person's worth from the weighted distribution attached
by `era_holdout_run.real_item_frame` and place it as a fact in the prompt's
[정량 정보]. Person identifiers are not stored; public outputs are aggregates only.

  dry-run   Check that the 300 prompts carry the facts and that labels match the SSR
            options. No LLM calls.
  generate  Oracle narratives with Qwen3.8 → data/synthetic/era_2024_narrative_oracle/
  answer    Answer twin / release / oracle with Qwen or EXAONE (existing _run_answer_file)
  embed     SSR cosine (.venv-embed, CPU)
  report    Mean cell TVD, paired bootstrap 95% CI, baselines
  all       generate → answer qwen → answer exaone → embed → report
            One model at a time. On a switch, set keep_alive 0 for that model only.
            Resume from the checkpoint.

GPU: SFP_LLM_NUM_CTX=4096. Do not touch other users' processes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_holdout_run as H

from sfac_persona.config import settings
from sfac_persona.generate import GenerateConfig, load_run, run
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.narrative import render_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("positive_control")

WAVE = 2024
TAG = "oracle"
N_BOOT = 2000
BOOT_SEED = 20260926
FACT_SEED = 20260926
CONCURRENCY = 2
FACT_MARK = "〔설문 사실〕"
ARMS = ("twin", "release", "oracle")
CONTRASTS = ("release-twin", "oracle-twin", "oracle-release")
EXAONE_ID = "hf.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF:IQ4_XS"

# Fact names shown in the prompt. They do not collide with anchor keys.
FACT_LABELS: dict[str, str] = {
    "leisure_activity_1st": "주말·휴일에 주로 하는 여가",
    "leisure_activity_wanted_1st": "앞으로 가장 하고 싶은 여가",
    "leisure_time_weekday": "평일 하루 여가시간",
    "leisure_time_weekend": "주말·휴일 하루 여가시간",
    "arts_importance": "삶에서 문화예술의 중요도",
    "attendance_overall": "지난 1년 문화예술 관람 만족",
    "cultural_life_overall": "본인 문화생활 만족",
    "effect_overall_happiness": "문화예술이 행복감에 도움이 되는 정도",
    "effect_stress_relieved": "문화예술이 스트레스 완화에 도움이 되는 정도",
}

_REAL: pd.DataFrame | None = None


def run_name() -> str:
    return f"era_{WAVE}_narrative_{TAG}"


def out_dir() -> Path:
    return settings.synthetic_dir / run_name()


def model_id(key: str) -> str:
    if key == "qwen":
        return settings.llm_default_model
    if key == "exaone":
        return EXAONE_ID
    raise SystemExit(f"unknown answer model: {key}")


def answers_path(arm: str, model_key: str) -> Path:
    return H.wave_dir(WAVE) / f"answers_{arm}_{model_key}_ssr.jsonl"


def ollama_base() -> str:
    return settings.llm_base_url.rstrip("/").removesuffix("/v1")


def _same_model(loaded: str, wanted: str) -> bool:
    return loaded == wanted or loaded.endswith("/" + wanted) or wanted.endswith("/" + loaded)


def loaded_models() -> list[str]:
    url = f"{ollama_base()}/api/ps"
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    return [str(m.get("name", "")) for m in resp.json().get("models", []) if m.get("name")]


def unload(model: str) -> None:
    """Unload only models this experiment loaded. Do not touch other processes."""
    hits = [n for n in loaded_models() if _same_model(n, model)]
    if not hits:
        log.info("unload skip (not loaded): %s", model)
        return
    url = f"{ollama_base()}/api/generate"
    for name in hits:
        resp = httpx.post(
            url,
            json={
                "model": name,
                "keep_alive": 0,
                "prompt": " ",
                "options": {"num_predict": 1, "num_ctx": 4096},
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        log.info("unloaded %s", name)
    time.sleep(2)


def unload_except(keep: str | None) -> None:
    for key in ("qwen", "exaone"):
        mid = model_id(key)
        if keep is not None and _same_model(mid, keep):
            continue
        unload(mid)


def ssr_refs() -> dict[str, dict[str, list[str]]]:
    path = H.wave_dir(WAVE) / "ssr_refs.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[str]]] = {}
    for item, opts in raw.items():
        out[item] = {str(k): (v if isinstance(v, list) else [str(v)]) for k, v in opts.items()}
    return out


def real_frame() -> pd.DataFrame:
    global _REAL
    if _REAL is None:
        _REAL = H.real_item_frame(WAVE)
    return _REAL


def _squash(text: str) -> str:
    return "".join(text.split())


def resolve_option(
    item: str, raw: object, refs: dict[str, dict[str, list[str]]]
) -> tuple[str, str] | None:
    """Option key and SSR reference sentence. None if the key is absent from the codebook."""
    if pd.isna(raw):
        return None
    code = str(raw).strip()
    try:
        number = float(code)
        if number.is_integer():
            code = str(int(number))
    except ValueError:
        pass
    opts = refs[item]
    hit = opts.get(code)
    if hit:
        return code, str(hit[0])
    norm = {_squash(k): k for k in opts}
    key = norm.get(_squash(code))
    alt = opts.get(key) if key else None
    if key and alt:
        return key, str(alt[0])
    return None


def _weights(frame: pd.DataFrame) -> np.ndarray:
    w = frame["w"].to_numpy(dtype=float) if "w" in frame.columns else np.ones(len(frame))
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(w.sum()) <= 0:
        w = np.ones(len(frame), dtype=float)
    return w / w.sum()


def _draw_value(frame: pd.DataFrame, item: str, rng: np.random.Generator) -> object | None:
    sub = frame[frame[item].notna()]
    if sub.empty:
        return None
    sub = sub.reset_index(drop=True)
    pick = int(rng.choice(len(sub), p=_weights(sub)))
    return sub.iloc[pick][item]


def assign_facts(
    bank: list[dict], refs: dict[str, dict[str, list[str]]]
) -> dict[str, dict[str, dict[str, str]]]:
    """Weighted draw within the cell. row_uid is not retained."""
    real = real_frame()
    rng = np.random.default_rng(FACT_SEED)
    items = list(H.QUESTIONS)
    out: dict[str, dict[str, dict[str, str]]] = {}
    unmatched: list[str] = []
    empty = 0
    for rec in bank:
        q = rec["quant"]
        cell = real[
            (real["age_band"] == q.get("age_band")) & (real["attend_any"] == q.get("attend_any"))
        ].reset_index(drop=True)
        if cell.empty:
            raise SystemExit(f"empty cell for persona {rec['persona_id'][:8]}")
        slot: dict[str, dict[str, str]] = {}
        for item in items:
            raw = _draw_value(cell, item, rng)
            if raw is None:
                empty += 1
                continue
            hit = resolve_option(item, raw, refs)
            if hit is None:
                unmatched.append(item)
                continue
            key, text = hit
            slot[item] = {"key": key, "text": text}
        out[str(rec["persona_id"])] = slot
    log.info(
        "facts n=%d filled=%d structurally_empty=%d unmatched=%d",
        len(out),
        sum(len(v) for v in out.values()),
        empty,
        len(unmatched),
    )
    if unmatched:
        log.error("unmatched option labels on items: %s", sorted(set(unmatched)))
        raise SystemExit("holdout options and answer labels are misaligned")
    return out


def facts_path() -> Path:
    return out_dir() / "oracle_facts.json"


def load_or_assign(
    bank: list[dict], refs: dict[str, dict[str, list[str]]]
) -> dict[str, dict[str, dict[str, str]]]:
    path = facts_path()
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        log.info("oracle facts loaded n=%d ← %s", len(data), path.name)
        return data
    facts = assign_facts(bank, refs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    log.info("oracle facts written n=%d", len(facts))
    return facts


def quant_with_facts(quant: dict, item_facts: dict[str, dict[str, str]]) -> dict[str, str]:
    q = {
        k: str(v)
        for k, v in quant.items()
        if k not in H.ANCHOR_SKIP and k not in FACT_LABELS.values() and pd.notna(v)
    }
    for item, label in FACT_LABELS.items():
        text = (item_facts.get(item) or {}).get("text")
        if text:
            q[label] = text
    return q


def strip_oracle_quant(quant: dict) -> dict:
    ban = set(FACT_LABELS.values())
    return {k: v for k, v in quant.items() if k not in ban}


def fact_sentence(item_facts: dict[str, dict[str, str]]) -> str:
    parts = []
    for item, label in FACT_LABELS.items():
        text = (item_facts.get(item) or {}).get("text")
        if text:
            parts.append(f"{label}: {text}")
    return " ".join(parts)


def weave_facts(narr: dict | None, sentence: str) -> dict:
    """Attach facts to the summary and weekend slots that the SSR path (_persona_blob) reads."""
    out = dict(narr or {})
    if not sentence:
        return out
    clause = f"{FACT_MARK} {sentence}"
    for key in ("summary", "weekend_routine"):
        cur = str(out.get(key) or "").strip()
        out[key] = cur if FACT_MARK in cur else f"{cur} {clause}".strip()
    return out


def prompt_for(quant: dict[str, str]) -> str:
    return render_prompt(quant)


def prepare() -> tuple[list[dict], dict[str, dict[str, dict[str, str]]], pd.DataFrame]:
    bank = H.load_eval_bank(WAVE)
    if len(bank) != H.N_RELEASE:
        raise SystemExit(f"eval bank n={len(bank)} != {H.N_RELEASE}")
    refs = ssr_refs()
    facts = load_or_assign(bank, refs)
    rows = []
    for rec in bank:
        q = quant_with_facts(rec["quant"], facts[rec["persona_id"]])
        q["persona_id"] = rec["persona_id"]
        rows.append(q)
    frame = pd.DataFrame(rows)
    dest = out_dir()
    dest.mkdir(parents=True, exist_ok=True)
    parquet = dest / "quant_from_eval_bank.parquet"
    frame.to_parquet(parquet, index=False)
    log.info("quant parquet n=%d cols=%d → %s", len(frame), frame.shape[1], parquet.name)
    return bank, facts, frame


def check_prompts(frame: pd.DataFrame) -> None:
    lengths: list[int] = []
    misses = 0
    for row in frame.to_dict(orient="records"):
        pid = row.pop("persona_id")
        quant = {k: str(v) for k, v in row.items() if pd.notna(v)}
        prompt = prompt_for(quant)
        if not prompt.startswith("당신은 서울에 사는"):
            raise SystemExit("prompt skeleton drifted")
        for label in FACT_LABELS.values():
            text = quant.get(label)
            if text and text not in prompt:
                misses += 1
                log.error("fact missing from prompt persona=%s label=%s", str(pid)[:8], label)
        lengths.append(len(prompt))
    if misses:
        raise SystemExit(f"prompts missing facts: {misses}")
    arr = np.asarray(lengths)
    log.info(
        "dry-run prompts n=%d chars min=%d median=%d max=%d",
        len(lengths),
        int(arr.min()),
        int(np.median(arr)),
        int(arr.max()),
    )


def twin_units() -> list[tuple[str, dict, None]]:
    """Reattach persona_ids from the published twin answer file onto the anchor frame."""
    frozen = H.wave_dir(WAVE) / "answers_twin_anchors_ssr.jsonl"
    if not frozen.exists():
        log.info("twin: frozen answers missing, resampling via _twin_units")
        return [(pid, q, None) for pid, q, _ in H._twin_units(WAVE)]
    seen: list[str] = []
    have: set[str] = set()
    for rec in H._read_jsonl(frozen):
        pid = str(rec["persona_id"])
        if pid not in have:
            have.add(pid)
            seen.append(pid)
    frame = H._attach_row_uid(build_official_anchor_frame(WAVE), WAVE)
    frame = frame.copy()
    frame["_pid"] = "twin-" + frame["row_uid"].astype(str)
    frame = frame.drop_duplicates("_pid").set_index("_pid")
    missing = [pid for pid in seen if pid not in frame.index]
    if missing:
        raise SystemExit(f"twin frozen ids missing from anchor frame: {len(missing)}")
    ban = set(H.ANCHOR_SKIP) | set(H.QUESTIONS) | set(FACT_LABELS.values())
    units: list[tuple[str, dict, None]] = []
    for pid in seen:
        row = frame.loc[pid]
        quant = {k: str(v) for k, v in row.items() if k not in ban and pd.notna(v)}
        units.append((pid, quant, None))
    log.info("twin units from frozen answers: %d", len(units))
    return units


def oracle_narratives() -> dict[str, dict]:
    path = out_dir() / "personas.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for rec in load_run(run_name()):
        qid = rec.get("quant_id")
        if qid and rec.get("narrative"):
            out[str(qid)] = rec["narrative"]
    return out


def units_for(
    arm: str,
    bank: list[dict],
    facts: dict[str, dict[str, dict[str, str]]],
) -> list[tuple[str, dict, dict | None]]:
    if arm == "twin":
        return twin_units()
    if arm == "release":
        return [(r["persona_id"], strip_oracle_quant(r["quant"]), r.get("narrative")) for r in bank]
    if arm == "oracle":
        narr = oracle_narratives()
        units: list[tuple[str, dict, dict | None]] = []
        for rec in bank:
            pid = rec["persona_id"]
            base = narr.get(pid)
            if not base:
                continue
            sentence = fact_sentence(facts[pid])
            units.append((pid, strip_oracle_quant(rec["quant"]), weave_facts(base, sentence)))
        log.info("oracle units %d/%d", len(units), len(bank))
        return units
    raise SystemExit(f"unknown arm: {arm}")


def generate() -> None:
    os.environ["SFP_LLM_NUM_CTX"] = "4096"
    unload_except(model_id("qwen"))
    _bank, _facts, frame = prepare()
    cfg = GenerateConfig(
        run_name=run_name(),
        n=len(frame),
        seed=20260924 + WAVE,
        model=model_id("qwen"),
        concurrency=CONCURRENCY,
        quant_parquet=out_dir() / "quant_from_eval_bank.parquet",
    )
    run(cfg)
    recs = load_run(run_name())
    ok = sum(1 for x in recs if x.get("ok"))
    log.info("%s n=%d ok=%d", run_name(), len(recs), ok)


def answer(model_key: str) -> None:
    os.environ["SFP_LLM_NUM_CTX"] = "4096"
    mid = model_id(model_key)
    unload_except(mid)
    bank, facts, _frame = prepare()
    prev = settings.llm_default_model
    prev_conc = H.CONCURRENCY
    settings.llm_default_model = mid
    H.CONCURRENCY = CONCURRENCY
    try:
        for arm in ARMS:
            path = answers_path(arm, model_key)
            units = units_for(arm, bank, facts)
            if arm == "oracle" and len(units) < len(bank):
                log.info("oracle answers wait: narratives %d/%d", len(units), len(bank))
            log.info("answer %s %s → %s", model_key, arm, path.name)
            H._run_answer_file(path, units)
    finally:
        settings.llm_default_model = prev
        H.CONCURRENCY = prev_conc


def embed() -> None:
    unload_except(None)
    py = ROOT / ".venv-embed" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)  # `uv sync --extra embed` installs the embedding stack here
        log.info("embed python fallback %s", py)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    script = str(ROOT / "scripts" / "ssr_embed.py")
    for model_key in ("qwen", "exaone"):
        for arm in ARMS:
            path = answers_path(arm, model_key)
            if not path.exists():
                raise SystemExit(f"answers missing: {path}")
            subprocess.check_call(
                [str(py), script, "--dir", str(H.wave_dir(WAVE)), "--pattern", path.name],
                cwd=str(ROOT),
                env=env,
            )


def mean_cell_tvd(
    ans: list[dict],
    real: pd.DataFrame,
    items: list[str],
    opt_order: dict[str, list[str]],
) -> float:
    vals = [H.cell_mean_tvd(ans, real, item, opt_order[item]) for item in items]
    return float(np.nanmean(vals))


def _bias_adjusted_ci(draws: np.ndarray, theta: float) -> tuple[float, float, float]:
    """95% interval reattached to the point estimate, and the bootstrap bias.

    Returns ``(ci_lo, ci_hi, bias)``, ``bias = mean(draws) - theta``.
    Cell TVD is a convex function of the probability vector, so the resampled
    distribution sits above the point estimate.
    The basic (reverse-percentile) interval ``[2θ̂ - q97.5, 2θ̂ - q2.5]`` reflects
    that percentile interval about θ̂, so if the percentile interval misses the point,
    the basic interval misses it too.
    Subtracting the bias from the percentile endpoints brings the interval back onto
    the point estimate.
    """
    if len(draws) == 0 or not np.isfinite(theta):
        return float("nan"), float("nan"), float("nan")
    q_lo, q_hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    bias = float(draws.mean() - theta)
    return q_lo - bias, q_hi - bias, bias


class _CellTvdIndex:
    """Same weighted mean as `cell_mean_tvd`. The observed distribution is built once."""

    def __init__(
        self,
        real: pd.DataFrame,
        items: list[str],
        opt_order: dict[str, list[str]],
    ) -> None:
        self.items = items
        self.opt_order = opt_order
        self._row_cache: dict[tuple[int, str], np.ndarray] = {}
        self.targets: dict[str, dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]]] = {}
        for item in items:
            spec: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]] = {}
            ok = real[item].notna()
            sub = real.loc[ok, ["age_band", "attend_any", item, "w"]]
            for (age, att), g in sub.groupby(["age_band", "attend_any"], sort=False):
                if len(g) < 5:
                    continue
                real_d = g.groupby(item)["w"].sum()
                real_d = real_d / real_d.sum()
                idx = sorted(set(map(str, real_d.index)) | set(map(str, opt_order[item])))
                pm = {str(k): float(v) for k, v in real_d.items()}
                pv = np.array([pm.get(k, 0.0) for k in idx], dtype=float)
                total = float(pv.sum())
                if total:
                    pv = pv / total
                pos = {lab: i for i, lab in enumerate(idx)}
                opt_pos = np.array([pos[str(o)] for o in opt_order[item]], dtype=int)
                spec[(str(age), str(att))] = (pv, opt_pos, float(len(g)))
            self.targets[item] = spec

    def mean(self, ans: list[dict]) -> float:
        if not ans:
            return float("nan")
        cells = [
            (
                str((r.get("quant") or {}).get("age_band", "")),
                str((r.get("quant") or {}).get("attend_any", "")),
            )
            for r in ans
        ]
        probs = [r.get("probs") or {} for r in ans]
        vals: list[float] = []
        for item in self.items:
            order = self.opt_order[item]
            spec = self.targets[item]
            buckets: dict[tuple[str, str], list[np.ndarray]] = {}
            width = len(order)
            for cell, prob in zip(cells, probs, strict=True):
                if cell not in spec:
                    continue
                one = prob.get(item)
                if not one:
                    continue
                key = (id(one), item)
                row = self._row_cache.get(key)
                if row is None:
                    row = np.empty(width, dtype=float)
                    for j, opt in enumerate(order):
                        row[j] = float(one.get(opt, 0.0))
                    self._row_cache[key] = row
                buckets.setdefault(cell, []).append(row)
            tvs: list[float] = []
            weights: list[float] = []
            for cell, rows in buckets.items():
                pv, opt_pos, weight = spec[cell]
                mean = np.mean(np.stack(rows, axis=0), axis=0)
                qv = np.zeros(pv.shape[0], dtype=float)
                qv[opt_pos] = mean
                total = float(qv.sum())
                if total:
                    qv = qv / total
                tvs.append(0.5 * float(np.abs(pv - qv).sum()))
                weights.append(weight)
            if tvs:
                vals.append(
                    float(np.average(np.asarray(tvs), weights=np.asarray(weights, dtype=float)))
                )
        if not vals:
            return float("nan")
        return float(np.mean(vals))


def resample_paired(
    shared_ids: list[str],
    cell_of: dict[str, tuple[str, str]],
    twin_ids_by_cell: dict[tuple[str, str], list[str]],
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """Resample with replacement, keeping cell sizes. release and oracle share the same ids."""
    ids_by_cell: dict[tuple[str, str], list[str]] = {}
    for pid in shared_ids:
        ids_by_cell.setdefault(cell_of[pid], []).append(pid)
    shared_out: list[str] = []
    twin_out: list[str] = []
    seen: set[tuple[str, str]] = set()
    for cell, ids in ids_by_cell.items():
        seen.add(cell)
        ix = rng.integers(0, len(ids), size=len(ids))
        shared_out.extend(ids[int(j)] for j in ix)
        tw = twin_ids_by_cell.get(cell, [])
        if tw:
            jx = rng.integers(0, len(tw), size=len(tw))
            twin_out.extend(tw[int(j)] for j in jx)
    for cell, tw in twin_ids_by_cell.items():
        if cell in seen or not tw:
            continue
        jx = rng.integers(0, len(tw), size=len(tw))
        twin_out.extend(tw[int(j)] for j in jx)
    return shared_out, twin_out


def _cell_of(rec: dict) -> tuple[str, str]:
    q = rec.get("quant") or {}
    return (str(q.get("age_band", "")), str(q.get("attend_any", "")))


def bootstrap_diffs(
    arms: dict[str, list[dict]],
    real: pd.DataFrame,
    items: list[str],
    opt_order: dict[str, list[str]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> list[dict]:
    """Cell-stratified persona bootstrap. release and oracle share the same indices.

    Intervals for all three contrasts use `_bias_adjusted_ci`. The point estimate itself
    is the difference of cell TVDs on the original sample.
    """
    rel = {r["persona_id"]: r for r in arms["release"]}
    ora = {r["persona_id"]: r for r in arms["oracle"]}
    shared = [i for i in rel if i in ora]
    cell_of = {i: _cell_of(rel[i]) for i in shared}
    twin_ids_by_cell: dict[tuple[str, str], list[str]] = {}
    twin_by_id = {r["persona_id"]: r for r in arms["twin"]}
    for rec in arms["twin"]:
        twin_ids_by_cell.setdefault(_cell_of(rec), []).append(rec["persona_id"])

    index = _CellTvdIndex(real, items, opt_order)

    def stat(recs: list[dict]) -> float:
        return index.mean(recs)

    # The point estimate is the same `cell_mean_tvd` as the published level table.
    # The fast path must match it.
    point = {
        arm: mean_cell_tvd(arms[arm], real, items, opt_order)
        for arm in ("twin", "release", "oracle")
    }
    for arm, theta in point.items():
        got = stat(arms[arm])
        if not np.isfinite(theta) or abs(got - theta) > 1e-8:
            raise AssertionError(f"fast cell TVD drifted on {arm}: {got} vs {theta}")
    point_diff = {
        "release-twin": point["release"] - point["twin"],
        "oracle-twin": point["oracle"] - point["twin"],
        "oracle-release": point["oracle"] - point["release"],
    }
    rng = np.random.default_rng(seed)
    bags: dict[str, list[float]] = {c: [] for c in CONTRASTS}
    for _ in range(n_boot):
        shared_b, twin_b = resample_paired(shared, cell_of, twin_ids_by_cell, rng)
        sr = stat([rel[i] for i in shared_b])
        so = stat([ora[i] for i in shared_b])
        st = stat([twin_by_id[i] for i in twin_b]) if twin_b else float("nan")
        if not all(np.isfinite(x) for x in (sr, st, so)):
            continue
        bags["release-twin"].append(sr - st)
        bags["oracle-twin"].append(so - st)
        bags["oracle-release"].append(so - sr)
    rows = []
    for contrast in CONTRASTS:
        draws = np.asarray(bags[contrast], dtype=float)
        theta = point_diff[contrast]
        ci_lo, ci_hi, bias = _bias_adjusted_ci(draws, theta)
        rows.append(
            {
                "contrast": contrast,
                "diff": theta,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "boot_bias": bias,
                "n_boot": len(draws),
                "n_shared": len(shared),
            }
        )
    return rows


def _opt_order(refs: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    return {item: list(opts) for item, opts in refs.items()}


def label_ceiling(
    bank: list[dict],
    facts: dict[str, dict[str, dict[str, str]]],
    real: pd.DataFrame,
    opt_order: dict[str, list[str]],
) -> float:
    """Cell TVD of hard labels that put 1 on the answer key. The floor SSR can reach."""
    items = list(H.QUESTIONS)
    ans = []
    for rec in bank:
        slot = facts[rec["persona_id"]]
        probs: dict[str, dict[str, float] | None] = {}
        for item in items:
            order = opt_order[item]
            key = (slot.get(item) or {}).get("key")
            if key not in order:
                probs[item] = None
                continue
            probs[item] = {o: (1.0 if o == key else 0.0) for o in order}
        q = rec["quant"]
        ans.append(
            {
                "persona_id": rec["persona_id"],
                "quant": {"age_band": q.get("age_band"), "attend_any": q.get("attend_any")},
                "probs": probs,
            }
        )
    return mean_cell_tvd(ans, real, items, opt_order)


def load_scored(arm: str, model_key: str, refs: dict, tau: float) -> list[dict]:
    path = answers_path(arm, model_key)
    if not H._cos_path_for(path).exists():
        raise SystemExit(f"embeddings missing: {H._cos_path_for(path)}")
    ans, _order = H.load_cos_probs(path, refs, tau)
    return ans


def baselines_from_csv() -> tuple[float, float]:
    path = settings.outputs_dir / "reports" / "era_holdout_baselines.csv"
    frame = pd.read_csv(path)
    sub = frame[frame["wave"] == WAVE]
    if sub.empty:
        raise SystemExit(f"no wave {WAVE} in {path}")
    return float(sub["marginal_lookup_tvd"].mean()), float(sub["real_sample_tvd"].mean())


def interpretation(
    level: dict[tuple[str, str], float],
    contrast: dict[tuple[str, str], dict],
    lookup: float,
    ceiling: float,
    label_floor: float,
) -> list[str]:
    """Three lines. Whether the instrument catches oracle, and whether twin≈release still
    holds when the answer model changes.
    """

    def lv(arm: str, model: str) -> float:
        return level[(arm, model)]

    def caught(model: str) -> bool:
        oracle = lv("oracle", model)
        base = min(lv("release", model), lv("twin", model))
        return oracle <= label_floor + 0.05 and oracle + 0.03 < base

    oq, oe = lv("oracle", "qwen"), lv("oracle", "exaone")
    if caught("qwen") and caught("exaone"):
        line1 = (
            f"도구는 oracle 을 잡는다. Qwen {oq:.3f} · EXAONE {oe:.3f} 이 "
            f"lookup {lookup:.3f} · 실측 n=300 {ceiling:.3f} · 라벨 바닥 {label_floor:.3f} 쪽에 있다."
        )
    elif caught("qwen") or caught("exaone"):
        line1 = (
            f"oracle 은 한 답변 모델에서 기준선 쪽으로 간다 "
            f"(Qwen {oq:.3f} · EXAONE {oe:.3f}, lookup {lookup:.3f} · "
            f"실측 n=300 {ceiling:.3f} · 라벨 바닥 {label_floor:.3f})."
        )
    else:
        line1 = (
            f"도구는 oracle 을 못 잡는다. Qwen {oq:.3f} · EXAONE {oe:.3f} 이 "
            f"lookup {lookup:.3f} · 실측 n=300 {ceiling:.3f} · 라벨 바닥 {label_floor:.3f} 까지 내려가지 않는다."
        )

    bits = []
    both_close = True
    for model in ("qwen", "exaone"):
        row = contrast[("release-twin", model)]
        gap = lv("release", model) - lv("twin", model)
        covers = row["ci_lo"] <= 0 <= row["ci_hi"]
        both_close = both_close and covers
        bits.append(
            f"{model} {lv('twin', model):.3f} 대 {lv('release', model):.3f} "
            f"(차이 {gap:+.3f} [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}])"
        )
    if both_close:
        line2 = "답변 모델을 바꿔도 twin ≈ release 이다. " + " · ".join(bits) + "."
    else:
        line2 = (
            "답변 모델에 따라 twin 과 release 의 차이가 0 을 벗어난다. " + " · ".join(bits) + "."
        )

    if caught("qwen") and caught("exaone") and both_close:
        line3 = "같은 지표가 정답 서사는 기준선 쪽으로 옮기고, 일반 서사와 앵커만은 두 답변 모델에서 같이 머문다."
    elif not caught("qwen") and not caught("exaone"):
        line3 = "정답을 넣은 서사도 release 와 같은 자리에 있으므로, 이 셀 TVD 로는 서사 효과를 가리기 어렵다."
    else:
        line3 = "oracle 이동과 twin–release 간격이 답변 모델마다 달라, 한 숫자로 묶지 않는다."
    return [line1, line2, line3]


def report(n_boot: int = N_BOOT) -> None:
    tau, _meta = H.load_frozen_tau()
    refs = ssr_refs()
    order = _opt_order(refs)
    real = real_frame()
    items = list(H.QUESTIONS)
    lookup, sample_ceiling = baselines_from_csv()
    bank = H.load_eval_bank(WAVE)
    facts = load_or_assign(bank, refs)
    floor = label_ceiling(bank, facts, real, order)
    log.info("oracle label ceiling %.4f lookup %.4f real_n %.4f", floor, lookup, sample_ceiling)

    level_rows: list[dict] = []
    levels: dict[tuple[str, str], float] = {}
    contrast_map: dict[tuple[str, str], dict] = {}
    for model_key in ("qwen", "exaone"):
        arms = {arm: load_scored(arm, model_key, refs, tau) for arm in ARMS}
        for arm, ans in arms.items():
            per_item = {item: H.cell_mean_tvd(ans, real, item, order[item]) for item in items}
            for item, val in per_item.items():
                level_rows.append(
                    {
                        "record": "item",
                        "arm": arm,
                        "answer_model": model_key,
                        "item": item,
                        "cell_mean_tvd": val,
                        "contrast": "",
                        "diff": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                        "boot_bias": np.nan,
                        "n": len(ans),
                        "n_boot": 0,
                    }
                )
            mean = float(np.nanmean(list(per_item.values())))
            levels[(arm, model_key)] = mean
            level_rows.append(
                {
                    "record": "level",
                    "arm": arm,
                    "answer_model": model_key,
                    "item": "ALL",
                    "cell_mean_tvd": mean,
                    "contrast": "",
                    "diff": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "boot_bias": np.nan,
                    "n": len(ans),
                    "n_boot": 0,
                }
            )
        for row in bootstrap_diffs(arms, real, items, order, n_boot=n_boot):
            assert row["ci_lo"] - 1e-9 <= row["diff"] <= row["ci_hi"] + 1e-9, row
            contrast_map[(row["contrast"], model_key)] = row
            level_rows.append(
                {
                    "record": "contrast",
                    "arm": "",
                    "answer_model": model_key,
                    "item": "ALL",
                    "cell_mean_tvd": np.nan,
                    "contrast": row["contrast"],
                    "diff": row["diff"],
                    "ci_lo": row["ci_lo"],
                    "ci_hi": row["ci_hi"],
                    "boot_bias": row["boot_bias"],
                    "n": row["n_shared"],
                    "n_boot": row["n_boot"],
                }
            )
    for name, val in (
        ("marginal_lookup", lookup),
        ("real_sample_n300", sample_ceiling),
        ("oracle_label_ceiling", floor),
    ):
        level_rows.append(
            {
                "record": "baseline",
                "arm": "",
                "answer_model": "",
                "item": "ALL",
                "cell_mean_tvd": val,
                "contrast": name,
                "diff": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "boot_bias": np.nan,
                "n": H.N_RELEASE,
                "n_boot": 0,
            }
        )
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "era_holdout_positive_control.csv"
    pd.DataFrame(level_rows).to_csv(csv_path, index=False)

    lines_i = interpretation(levels, contrast_map, lookup, sample_ceiling, floor)
    method = (
        "oracle 서사는 동결 뱅크 300명을 release 와 같은 Qwen3.8·온도 0.6·프롬프트 골격으로 다시 쓴 것이다. "
        "홀드아웃 정답은 셀(연령대×관람 여부)의 가중 분포에서 뽑아 [정량 정보]에 넣었고, "
        "SSR 이 읽는 요약·주말 칸에 같은 문장을 붙였다. "
        "짝 부트스트랩은 셀 안에서 페르소나를 복원추출하며 release·oracle 은 같은 인덱스를 쓴다 "
        f"(B={n_boot}). 셀별 TVD 는 재표집에서 위로 치우치므로, 퍼센타일 끝점에서 "
        "부트스트랩 편향(재표집 평균−점추정치)을 뺀 95% 구간을 보고한다. "
        "셀별 TVD 의 문항 평균. 낮을수록 좋다."
    )
    header = [
        "# 서사 holdout 양성 대조군 (2024)",
        "",
        method,
        "",
        "## 셀별 TVD 평균",
        "",
        "| arm | Qwen3.8 | EXAONE 4.0 32B |",
        "|---|---:|---:|",
    ]
    for arm in ARMS:
        header.append(f"| {arm} | {levels[(arm, 'qwen')]:.3f} | {levels[(arm, 'exaone')]:.3f} |")
    header += [
        f"| lookup (주변) | {lookup:.3f} | {lookup:.3f} |",
        f"| 실측 n=300 | {sample_ceiling:.3f} | {sample_ceiling:.3f} |",
        f"| oracle 라벨 바닥 | {floor:.3f} | {floor:.3f} |",
        "",
        "## 차이 (95% CI)",
        "",
        "| contrast | Qwen3.8 | EXAONE 4.0 32B |",
        "|---|---|---|",
    ]
    for contrast in CONTRASTS:
        cells = []
        for model in ("qwen", "exaone"):
            row = contrast_map[(contrast, model)]
            cells.append(f"{row['diff']:+.3f} [{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]")
        header.append(f"| {contrast} | " + " | ".join(cells) + " |")
    header += ["", "## 해석", ""]
    header.extend(lines_i)
    header.append("")
    md_path = reports / "era_holdout_positive_control.md"
    md_path.write_text("\n".join(header), encoding="utf-8")
    log.info("wrote %s and %s", csv_path.name, md_path.name)
    log.info("\n%s", "\n".join(lines_i))


def dry_run() -> None:
    os.environ["SFP_LLM_NUM_CTX"] = "4096"
    _bank, facts, frame = prepare()
    check_prompts(frame)
    units = twin_units()
    if len(units) != H.N_RELEASE:
        raise SystemExit(f"twin n={len(units)} != {H.N_RELEASE}")
    for _pid, quant, narr in units[:3]:
        leaked = set(quant) & (set(H.QUESTIONS) | set(FACT_LABELS.values()))
        if leaked or narr is not None:
            raise SystemExit("twin unit carries holdout facts or a narrative")
    refs = ssr_refs()
    order = _opt_order(refs)
    floor = label_ceiling(_bank, facts, real_frame(), order)
    log.info("dry-run ok n=%d oracle_label_ceiling=%.4f", len(frame), floor)


def all_stages(n_boot: int) -> None:
    os.environ["SFP_LLM_NUM_CTX"] = "4096"
    log.info("GPU models now: %s", loaded_models())
    generate()
    answer("qwen")
    answer("exaone")
    embed()
    report(n_boot)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dry-run", "all", "generate", "answer", "embed", "report"])
    ap.add_argument("--answer-model", choices=["qwen", "exaone"], default="qwen")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    os.environ["SFP_LLM_NUM_CTX"] = "4096"
    if args.cmd == "dry-run":
        dry_run()
    elif args.cmd == "all":
        all_stages(args.n_boot)
    elif args.cmd == "generate":
        generate()
    elif args.cmd == "answer":
        answer(args.answer_model)
    elif args.cmd == "embed":
        embed()
    else:
        report(args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
