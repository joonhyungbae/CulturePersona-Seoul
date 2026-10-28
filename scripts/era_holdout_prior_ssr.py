"""The reader's own answer distribution under SSR, with no persona: the prior arm of the held-out test.

Under option log-probabilities a reader shown no persona sits close to the persona arms. This script measures
the same prior under SSR, the elicitation every published level uses: the generator answers the nine items as
"an adult living in Seoul" with no anchors and no narrative, N times at the production temperature, the
sentences are mapped onto the options by the same embedding and frozen tau, and the pooled distribution is
applied to every cell of the 2024 evaluation personas, so that cell_mean_tvd scores it as a lookup.

Design fixed before running (2026-10-10):
  calls       N = 300 independent answers (one JSON with the nine items each), temperature as in era_holdout_run;
              readers: the generator (primary), EXAONE and Gemma (added 2026-10-10, before running, so that the
              no-persona prior is reported for the two readers the prose result rests on)
  scoring     SSR with the frozen tau and the 2024 reference statements; pooled distribution applied to every
              release persona; mean cell TVD, read against the SSR release level, the marginal lookup and the
              log-probability prior
Every result is reported, whatever its sign.

Stages:
  answer   uv run python scripts/era_holdout_prior_ssr.py answer
  embed    uv run python scripts/era_holdout_prior_ssr.py embed   (CPU)
  report   uv run python scripts/era_holdout_prior_ssr.py report
Output: outputs/reports/era_holdout_prior_ssr.csv (aggregates only)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import era_holdout_logprob as L
import era_holdout_positive_control as P
import era_holdout_run as H

from sfac_persona.config import settings
from sfac_persona.llm_client import get_client, parse_json_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_prior_ssr")

WAVE = 2024
N = 300
CONCURRENCY = 2
READERS = {
    "qwen": settings.llm_default_model,
    "exaone": P.EXAONE_ID,
    "gemma": "hf.co/bartowski/google_gemma-4-31b-it-GGUF:IQ4_XS",
}
READER = "qwen"
REPORT = settings.outputs_dir / "reports" / "era_holdout_prior_ssr.csv"


def path_for(reader: str):
    return H.wave_dir(WAVE) / f"answers_prior_{reader}_ssr.jsonl"


PROMPT = """서울에 사는 한 성인의 처지에서, 보기 목록 없이 각 질문에 한두 문장으로만 답하십시오. JSON 객체 하나만 출력.
키: {keys}

[질문]
{questions}
"""


def _answer_one(client, model: str, k: int) -> dict:
    keys = list(H.QUESTIONS)
    prompt = PROMPT.format(
        keys=", ".join(keys), questions="\n".join(f"- {q}: {H.QUESTIONS[q]}" for q in keys)
    )
    texts = {q: None for q in keys}
    for _ in range(3):
        res = client.chat(model=model, system=None, user=prompt, temperature=0.7)
        try:
            obj = parse_json_response(res.text)
        except Exception:  # noqa: BLE001, S112
            continue
        texts = {q: (str(obj.get(q)) if obj.get(q) else None) for q in keys}
        if all(texts.values()):
            break
    return {"persona_id": f"prior-{k:03d}", "quant": {}, "texts": texts}


def answer() -> None:
    PATH = path_for(READER)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    done = {r["persona_id"] for r in H._read_jsonl(PATH)} if PATH.exists() else set()
    todo = [k for k in range(N) if f"prior-{k:03d}" not in done]
    log.info("prior ssr: %d done, %d todo", len(done), len(todo))
    if not todo:
        return
    client = get_client()
    model = READERS[READER]
    t0 = time.time()
    with open(PATH, "a", encoding="utf-8") as f, ThreadPoolExecutor(CONCURRENCY) as pool:
        futs = [pool.submit(_answer_one, client, model, k) for k in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                log.info("  %d/%d (%.1f/min)", i, len(todo), 60 * i / (time.time() - t0))


def embed() -> None:
    H._embed(WAVE, pattern=path_for(READER).name)


def report() -> None:
    refs = P.ssr_refs()
    order = P._opt_order(refs)
    items = list(H.QUESTIONS)
    real = P.real_frame()
    tau, _ = H.load_frozen_tau()
    ans, _ = H.load_cos_probs(path_for(READER), refs, tau)
    pooled = {}
    for item in items:
        ps = [
            np.array([r["probs"][item][o] for o in order[item]])
            for r in ans
            if r["probs"].get(item)
        ]
        pooled[item] = dict(zip(order[item], np.mean(ps, axis=0).tolist(), strict=True))
    bank = H.load_eval_bank(WAVE)
    recs = L.prior_as_lookup({"probs": pooled}, bank)
    level = P.mean_cell_tvd(recs, real, items, order)
    marginal, ceiling = P.baselines_from_csv()
    rows = [
        {
            "record": "level",
            "arm": "prior",
            "reader": READER,
            "method": "ssr",
            "value": level,
            "n": len(ans),
        },
        {"record": "baseline", "arm": "marginal_lookup", "value": marginal},
        {"record": "baseline", "arm": "real_sample", "value": ceiling},
    ]
    for item in items:
        rows.append(
            {
                "record": "level_item",
                "arm": "prior",
                "reader": READER,
                "method": "ssr",
                "item": item,
                "value": H.cell_mean_tvd(recs, real, item, order[item]),
            }
        )
    out = pd.DataFrame(rows)
    if REPORT.exists():
        old = pd.read_csv(REPORT)
        if "reader" not in old.columns:
            old["reader"] = "qwen"
        old = old[~((old.reader == READER) & (old.record != "baseline"))]
        out = pd.concat([old[old.record != "baseline"], out], ignore_index=True)
    out.to_csv(REPORT, index=False)
    print(out[out.record != "level_item"].round(4).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["answer", "embed", "report"])
    ap.add_argument("--reader", choices=list(READERS), default="qwen")
    a = ap.parse_args()
    global READER
    READER = a.reader
    {"answer": answer, "embed": embed, "report": report}[a.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
