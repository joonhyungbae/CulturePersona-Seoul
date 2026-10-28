"""A second elicitation for the held-out items: option log-probabilities instead of SSR.

Every held-out level in the paper comes from semantic similarity rating (SSR): the reader writes a sentence
and the sentence is mapped onto the options by embedding similarity. The release arm sits far above the
marginal lookup, and the paper attributes that level to the answering step. This script tests the attribution
with a second elicitation that has no sentence and no embedding: the reader sees the persona, the question and
lettered options, and the answer distribution is the model's next-token probability over the option letters
(renormalised over the listed options). Same personas, twin respondents, items, cells and metric as the SSR run.

Design fixed before running (2026-10-10):
  arms        release (anchors + two fields), synthanchors (the same personas' anchor lists), twin (real
              respondents' anchor lists, the SSR twin sample), prior (no persona: the reader's own answer
              distribution, applied to every cell)
  readers     the generator (qwen3.8:27b-mtp) primary; EXAONE 4.0 32B secondary if time allows
  elicitation one call per persona and item, think off, one token, top-20 log-probabilities; the probability
              of each option is exp(logprob of its letter) renormalised over the listed letters; a letter
              outside the top 20 gets zero; coverage (mass of the listed letters within the top 20) is recorded
  primary     2024, generator as reader: mean cell TVD of the release arm under log-probabilities, read next to
              the SSR release level, the marginal lookup and the real-sample ceiling of the published table.
              If the level falls to the lookup's neighbourhood, the SSR step set the published level; if it
              stays near the SSR level, the persona content does
  secondary   twin and synthanchors under log-probabilities (release minus synthanchors paired by persona,
              cell-stratified bootstrap, 2,000 draws); the prior arm; mean within-person entropy under both
              elicitations; EXAONE as reader
Every result is reported, whatever its sign.

Stages:
  answer   uv run python scripts/era_holdout_logprob.py answer --reader qwen [--arms release twin synthanchors prior]
  report   uv run python scripts/era_holdout_logprob.py report
Output: data/eval/era_holdout/w2024/answers_{arm}_{reader}_logprob.jsonl (per persona; not released) and
        outputs/reports/era_holdout_logprob.csv (aggregates only)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_holdout_positive_control as P
import era_holdout_run as H

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_logprob")

WAVE = 2024
READERS = {"qwen": settings.llm_default_model, "exaone": P.EXAONE_ID}
ARMS = ("release", "synthanchors", "twin", "prior", "template", "oracle")
CONCURRENCY = 2
TOP_K = 20
LETTERS = string.ascii_uppercase
REPORT = settings.outputs_dir / "reports" / "era_holdout_logprob.csv"

PROMPT = """아래 인물의 처지에서 질문에 답하십시오. 보기 중 하나를 골라 그 알파벳 한 글자만 출력하십시오.

[인물]
{persona}

[질문] {question}
{options}

답:"""

PROMPT_PRIOR = """서울에 사는 한 성인의 처지에서 질문에 답하십시오. 보기 중 하나를 골라 그 알파벳 한 글자만 출력하십시오.

[질문] {question}
{options}

답:"""


def answers_path(arm: str, reader: str) -> Path:
    return H.wave_dir(WAVE) / f"answers_{arm}_{reader}_logprob.jsonl"


def option_text(item: str, key: str, refs: dict[str, dict[str, list[str]]]) -> str:
    """What the reader sees for one option: the survey label for activities, hours for the time items,
    and the SSR reference sentence for the Likert items (the same semantic targets SSR maps onto)."""
    if item.startswith("leisure_activity"):
        return key
    if item.startswith("leisure_time"):
        return f"{key}시간"
    return refs[item][key][0]


def option_block(item: str, refs: dict) -> tuple[str, list[str]]:
    keys = list(refs[item])
    if len(keys) > len(LETTERS):
        raise SystemExit(f"{item}: {len(keys)} options exceed the letter range")
    lines = [f"{LETTERS[i]}. {option_text(item, k, refs)}" for i, k in enumerate(keys)]
    return "\n".join(lines), keys


def units_for(arm: str) -> list[tuple[str, dict, dict | None]]:
    if arm == "release":
        return [(r["persona_id"], r["quant"], r.get("narrative")) for r in H.load_eval_bank(WAVE)]
    if arm == "synthanchors":
        return [(r["persona_id"], r["quant"], None) for r in H.load_eval_bank(WAVE)]
    if arm == "twin":
        return H._twin_units(WAVE)
    if arm == "prior":
        return [("prior", {}, None)]
    if arm == "template":
        # Anchors rendered as fixed sentences in the two slots the reader sees (era_template_prose.py).
        from era_template_prose import template_fields

        return [
            (r["persona_id"], r["quant"], template_fields(r["quant"]))
            for r in H.load_eval_bank(WAVE)
        ]
    if arm == "oracle":
        # The positive-control narratives (facts drawn from the real cell distributions, stated in the prose).
        bank, facts, _ = P.prepare()
        return P.units_for("oracle", bank, facts)
    raise SystemExit(f"unknown arm {arm}")


def _call(model: str, prompt: str) -> list[dict]:
    resp = httpx.post(
        f"{P.ollama_base()}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": TOP_K,
            "options": {"num_predict": 1, "temperature": 0, "num_ctx": 4096},
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    lp = resp.json().get("logprobs") or []
    return lp[0].get("top_logprobs", []) if lp else []


def letter_probs(top: list[dict], n_opt: int) -> tuple[list[float], float]:
    """Probabilities over the n listed letters, renormalised; coverage = their mass within the top-k list."""
    lp = {t["token"]: float(t["logprob"]) for t in top}
    raw = [math.exp(lp[LETTERS[i]]) if LETTERS[i] in lp else 0.0 for i in range(n_opt)]
    total_top = sum(math.exp(v) for v in lp.values()) or 1.0
    s = sum(raw)
    coverage = s / total_top
    if s <= 0:
        return [1.0 / n_opt] * n_opt, 0.0
    return [v / s for v in raw], coverage


def answer_unit(
    model: str, arm: str, pid: str, quant: dict, narrative: dict | None, refs: dict
) -> dict:
    probs: dict[str, dict[str, float]] = {}
    coverage: dict[str, float] = {}
    persona = H._persona_blob(quant, narrative) if arm != "prior" else ""
    for item, question in H.QUESTIONS.items():
        block, keys = option_block(item, refs)
        prompt = (
            PROMPT_PRIOR.format(question=question, options=block)
            if arm == "prior"
            else PROMPT.format(persona=persona, question=question, options=block)
        )
        top = _call(model, prompt)
        p, cov = letter_probs(top, len(keys))
        probs[item] = dict(zip(keys, p, strict=True))
        coverage[item] = cov
    return {
        "persona_id": pid,
        "quant": {"age_band": quant.get("age_band"), "attend_any": quant.get("attend_any")},
        "probs": probs,
        "coverage": coverage,
    }


def answer(reader: str, arms: tuple[str, ...]) -> None:
    model = READERS[reader]
    P.unload_except(model)
    refs = P.ssr_refs()
    for arm in arms:
        path = answers_path(arm, reader)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {r["persona_id"] for r in H._read_jsonl(path)} if path.exists() else set()
        todo = [u for u in units_for(arm) if u[0] not in done]
        log.info("%s/%s: %d done, %d todo", arm, reader, len(done), len(todo))
        if not todo:
            continue
        t0 = time.time()
        with (
            open(path, "a", encoding="utf-8") as f,
            ThreadPoolExecutor(max_workers=CONCURRENCY) as pool,
        ):
            futs = [pool.submit(answer_unit, model, arm, pid, q, n, refs) for pid, q, n in todo]
            for k, fut in enumerate(as_completed(futs), 1):
                f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                f.flush()
                if k % 25 == 0 or k == len(todo):
                    log.info(
                        "  %s %d/%d (%.1f/min)", arm, k, len(todo), 60 * k / (time.time() - t0)
                    )


def _entropy(recs: list[dict], items: list[str]) -> float:
    vals = []
    for r in recs:
        for item in items:
            p = np.array([v for v in (r.get("probs") or {}).get(item, {}).values()], dtype=float)
            p = p[p > 0]
            if len(p):
                vals.append(float(-(p * np.log(p)).sum()))
    return float(np.mean(vals)) if vals else float("nan")


def prior_as_lookup(prior: dict, bank: list[dict]) -> list[dict]:
    """The reader's own answer distribution applied to every release persona, so cell_mean_tvd scores it
    exactly as a lookup that ignores the cells."""
    return [
        {"persona_id": r["persona_id"], "quant": r["quant"], "probs": prior["probs"]} for r in bank
    ]


def report() -> None:
    refs = P.ssr_refs()
    order = P._opt_order(refs)
    items = list(H.QUESTIONS)
    real = P.real_frame()
    tau, _ = H.load_frozen_tau()
    marginal, real_sample = P.baselines_from_csv()
    rows: list[dict] = []
    rows.append({"record": "baseline", "arm": "marginal_lookup", "reader": "", "value": marginal})
    rows.append({"record": "baseline", "arm": "real_sample", "reader": "", "value": real_sample})
    for reader in READERS:
        arms: dict[str, list[dict]] = {}
        for arm in ("release", "synthanchors", "twin", "template", "oracle"):
            path = answers_path(arm, reader)
            if not path.exists():
                continue
            recs = H._read_jsonl(path)
            arms[arm] = recs
            level = P.mean_cell_tvd(recs, real, items, order)
            cov = float(np.mean([np.mean(list(r["coverage"].values())) for r in recs]))
            rows.append(
                {
                    "record": "level",
                    "arm": arm,
                    "reader": reader,
                    "method": "logprob",
                    "value": level,
                    "n": len(recs),
                    "entropy": _entropy(recs, items),
                    "coverage": cov,
                }
            )
            for item in items:
                rows.append(
                    {
                        "record": "level_item",
                        "arm": arm,
                        "reader": reader,
                        "method": "logprob",
                        "item": item,
                        "value": H.cell_mean_tvd(recs, real, item, order[item]),
                    }
                )
        prior_path = answers_path("prior", reader)
        if prior_path.exists() and "release" in arms:
            prior = H._read_jsonl(prior_path)[0]
            recs = prior_as_lookup(prior, arms["release"])
            rows.append(
                {
                    "record": "level",
                    "arm": "prior",
                    "reader": reader,
                    "method": "logprob",
                    "value": P.mean_cell_tvd(recs, real, items, order),
                    "entropy": _entropy([prior], items),
                }
            )
        # SSR levels on the same personas, for the side-by-side reading.
        ssr_names = {
            "release": {"qwen": "release_narrative", "exaone": "release_exaone"},
            "twin": {"qwen": "twin_anchors", "exaone": "twin_exaone"},
            "synthanchors": {
                "qwen": "synthanchors_reader_qwen",
                "exaone": "synthanchors_reader_exaone",
            },
        }
        for arm, names in ssr_names.items():
            path = H.resolve_answers_path(WAVE, f"answers_{names[reader]}_ssr.jsonl")
            if not path.exists() or not H._cos_path_for(path).exists():
                continue
            ans, _ = H.load_cos_probs(path, refs, tau)
            rows.append(
                {
                    "record": "level",
                    "arm": arm,
                    "reader": reader,
                    "method": "ssr",
                    "value": P.mean_cell_tvd(ans, real, items, order),
                    "n": len(ans),
                    "entropy": _entropy(ans, items),
                }
            )
        for extra, label in (("template", "release-template"), ("oracle", "oracle-release")):
            if extra in arms and "release" in arms and "twin" in arms:
                pair = {"release": arms[extra], "oracle": arms["release"], "twin": arms["twin"]}
                if extra == "oracle":
                    pair = {
                        "release": arms["release"],
                        "oracle": arms["oracle"],
                        "twin": arms["twin"],
                    }
                for row in P.bootstrap_diffs(pair, real, items, order):
                    if row["contrast"] != "oracle-release":
                        continue
                    rows.append(
                        {
                            "record": "contrast",
                            "arm": label,
                            "reader": reader,
                            "method": "logprob",
                            "value": row["diff"],
                            "ci_lo": row["ci_lo"],
                            "ci_hi": row["ci_hi"],
                            "n": row["n_shared"],
                        }
                    )
        if {"release", "synthanchors", "twin"} <= set(arms):
            # bootstrap_diffs pairs "release" with "oracle": oracle-release here is release minus synthanchors.
            for row in P.bootstrap_diffs(
                {"release": arms["synthanchors"], "oracle": arms["release"], "twin": arms["twin"]},
                real,
                items,
                order,
            ):
                name = {
                    "oracle-release": "release-synthanchors",
                    "release-twin": "synthanchors-twin",
                    "oracle-twin": "release-twin",
                }[row["contrast"]]
                rows.append(
                    {
                        "record": "contrast",
                        "arm": name,
                        "reader": reader,
                        "method": "logprob",
                        "value": row["diff"],
                        "ci_lo": row["ci_lo"],
                        "ci_hi": row["ci_hi"],
                        "n": row["n_shared"],
                    }
                )
    out = pd.DataFrame(rows)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT, index=False)
    print(out[out.record != "level_item"].round(4).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["answer", "report"])
    ap.add_argument("--reader", choices=list(READERS), default="qwen")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    if a.stage == "answer":
        answer(a.reader, tuple(a.arms))
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
