"""Era holdout sample-mode smoke (A3) — 2024 release narrative n=20.

Appendix for juxtaposition with the SSR main track. Single-option choice → one-hot
(within-person entropy = 0).

Run:
  uv run python scripts/era_holdout_sample_smoke.py --n 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.generate import load_run
from sfac_persona.llm_client import get_client
from sfac_persona.narrative import anchor_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_holdout_sample_smoke")

REPORTS = settings.outputs_dir / "reports"
OUT = settings.data_root / "eval" / "era_holdout"
WAVE = 2024
SEED = 20260910
CONCURRENCY = 4

PROMPT_SAMPLE = """아래 인물의 처지에서 각 문항의 보기 중 하나를 그대로 고르십시오. JSON 객체 하나만 출력.
키: {keys}
값은 보기 문자열을 그대로 씁니다.

[인물]
{persona}

[문항]
{questions}
"""


def _load_holdout_mod():
    path = Path(__file__).resolve().parent / "era_holdout_run.py"
    spec = importlib.util.spec_from_file_location("era_holdout_run", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _match(a: str, opts: list[str]) -> str | None:
    a = a.strip()
    return next((o for o in opts if o == a), None) or next(
        (o for o in opts if a and (a in o or o in a)), None
    )


def _persona_blob(quant: dict, narrative: dict | None) -> str:
    bits = [anchor_block(quant)]
    if narrative:
        if narrative.get("summary"):
            bits.append(f"한 줄: {narrative['summary']}")
        if narrative.get("weekend_routine"):
            bits.append(f"주말: {narrative['weekend_routine']}")
    return "\n".join(bits)


def _answer_sample(
    client,
    model: str,
    pid: str,
    quant: dict,
    narrative: dict | None,
    questions: dict[str, str],
    options: dict[str, list[str]],
) -> dict:
    keys = list(questions)
    qlines = []
    for k in keys:
        opts = options[k]
        qlines.append(f"- {k}: {questions[k]}\n  보기: {' / '.join(opts)}")
    prompt = PROMPT_SAMPLE.format(
        keys=", ".join(keys),
        persona=_persona_blob(quant, narrative),
        questions="\n".join(qlines),
    )
    texts: dict[str, str | None] = {k: None for k in keys}
    probs: dict[str, dict[str, float] | None] = {k: None for k in keys}
    attempt = -1
    for attempt in range(3):
        res = client.chat(
            model=model,
            system=None,
            user=prompt,
            temperature=0.3,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        try:
            raw = res.text[res.text.find("{") : res.text.rfind("}") + 1]
            obj = json.loads(raw)
            ok = 0
            for k in keys:
                m = _match(str(obj.get(k, "")), options[k])
                if m:
                    texts[k] = m
                    probs[k] = {o: (1.0 if o == m else 0.0) for o in options[k]}
                    ok += 1
                else:
                    texts[k] = None
                    probs[k] = None
            if ok >= len(keys) - 2:
                break
        except Exception:  # noqa: BLE001, S112
            continue
    return {
        "persona_id": pid,
        "condition": "narrative",
        "mode": "sample",
        "texts": texts,
        "probs": probs,
        "quant": {k: str(v) for k, v in quant.items() if k in ("sex", "age_band", "attend_any")},
        "attempt": attempt,
    }


def _options_from_refs(ehr, wave: int) -> dict[str, list[str]]:
    refs_path = ehr.resolve_answers_path(wave, "ssr_refs.json")
    if not refs_path.exists():
        refs_path = OUT / "ssr_refs.json"
    refs = json.loads(refs_path.read_text(encoding="utf-8"))
    return {k: list(v.keys()) for k, v in refs.items() if k in ehr.QUESTIONS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--wave", type=int, default=WAVE)
    args = ap.parse_args()
    if args.wave != 2024:
        raise SystemExit("A3 sample smoke is 2024-only")

    ehr = _load_holdout_mod()
    options = _options_from_refs(ehr, args.wave)
    questions = {k: ehr.QUESTIONS[k] for k in ehr.QUESTIONS if k in options}

    recs = load_run(f"era_{args.wave}_narrative")
    rng = np.random.default_rng(SEED + args.wave)
    if len(recs) > args.n:
        idx = rng.choice(len(recs), size=args.n, replace=False)
        recs = [recs[i] for i in sorted(idx)]
    else:
        recs = recs[: args.n]

    path = OUT / "answers_release_narrative_sample.jsonl"
    done: set[str] = set()
    if path.exists():
        done = {
            json.loads(l)["persona_id"]
            for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }
    todo = [
        (r["persona_id"], r["quant"], r.get("narrative"))
        for r in recs
        if r["persona_id"] not in done
    ]
    log.info("%s: %d done, %d todo", path.name, len(done), len(todo))

    if todo:
        client = get_client()
        model = settings.llm_default_model
        with (
            open(path, "a", encoding="utf-8") as f,
            ThreadPoolExecutor(max_workers=CONCURRENCY) as pool,
        ):
            futs = {
                pool.submit(_answer_sample, client, model, pid, q, n, questions, options): pid
                for pid, q, n in todo
            }
            t0 = time.time()
            for k, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if k % 5 == 0 or k == len(todo):
                    log.info(
                        "  %d/%d (%.1f/min)", k, len(todo), k / max(time.time() - t0, 1e-6) * 60
                    )

    ans = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # keep only the smoke ids
    keep = {r["persona_id"] for r in recs}
    ans = [r for r in ans if r["persona_id"] in keep]

    real = ehr.real_item_frame(args.wave)
    rows = []
    for item, opts in options.items():
        tvd = ehr.cell_mean_tvd(ans, real, item, opts)
        # within-person entropy: sample is one-hot → 0
        ents = []
        for r in ans:
            p = (r.get("probs") or {}).get(item)
            if not p:
                continue
            arr = np.array([float(p.get(o, 0.0)) for o in opts], dtype=float)
            if arr.sum() <= 0:
                continue
            arr = arr / arr.sum()
            ents.append(float(-(arr[arr > 0] * np.log(arr[arr > 0])).sum()))
        rows.append(
            {
                "item": item,
                "n": len(ans),
                "tvd_cell_mean": tvd,
                "mean_within_entropy": float(np.mean(ents)) if ents else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "era_holdout_sample_smoke.csv", index=False)

    lines = [
        "# Era holdout sample-mode smoke (A3)",
        "",
        f"- wave={args.wave}, release narrative n={len(ans)}, mode=sample (one-hot)",
        "- SSR remains primary (ADR-010). This is appendix contrast only.",
        "- Within-person entropy is 0 by construction under sample.",
        "",
        df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False),
        "",
        f"Answers: `{path}`",
        "",
    ]
    md = REPORTS / "era_holdout_sample_smoke.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
