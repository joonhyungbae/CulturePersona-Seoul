"""Generation entry point — draw quantitative anchors from the grounding-table
chain and attach narratives with an LLM.

- Checkpoint: append one line at a time to data/synthetic/<run>/personas.jsonl.
  A rerun continues from there.
- Checks: schema (parse) → consistency → cliche. A failed persona is regenerated
  up to `retries` times; if it still fails, it is kept with flags (not dropped —
  the rate must be reported).
- Person-level microdata is never read here.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sfac_persona.config import settings
from sfac_persona.grounding.chain import ChainSampler, load_spec
from sfac_persona.llm_client import get_client
from sfac_persona.narrative import FIELD_KEYS, branches, parse_narrative, render_prompt
from sfac_persona.validators.cliche import detect_cliches
from sfac_persona.validators.consistency import check_all

logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    run_name: str
    n: int
    seed: int = 20260907
    model: str | None = None
    temperature: float = 0.6
    max_tokens: int = 2048
    concurrency: int | None = None
    retries: int = 2
    spec_name: str = "sampler_specs.json"
    # ADR-007: resume from parquet quant (None means sample from the spec)
    quant_parquet: Path | None = None
    ablation: str = "structure_T"

    @property
    def out_dir(self) -> Path:
        return settings.synthetic_dir / self.run_name


def _quant_dicts_from_parquet(
    path: Path, n: int, seed: int
) -> tuple[list[dict[str, str]], list[str]]:
    """quant row → prompt dict, plus the **persona_id that points at the row**.

    The id is returned because resume must not be by position (index). This id is
    recorded as `quant_id`, which points at the quant row, and is distinct from the
    persona's opaque identifier (`persona_id`). When `n` is smaller than the full
    set, this function samples at random, so the row at index 0 depends on `n`.
    Resuming an n=300 checkpoint at n=15000 skips 300 rows that are not the ones
    already done (found 2026-09-12).
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    ids = [str(v) for v in df["persona_id"]] if "persona_id" in df.columns else []
    skip = {
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
    out: list[dict[str, str]] = []
    for _, row in df.iterrows():
        q = {k: str(v) for k, v in row.items() if k not in skip and pd.notna(v)}
        out.append(q)
    return out, ids


def _generate_one(
    client: Any, model: str, quant: dict[str, str], cfg: GenerateConfig
) -> dict[str, Any]:
    prompt = render_prompt(quant)
    attempts: list[dict[str, Any]] = []
    narr: dict[str, str] | None = None
    issues: list = []
    usage_total = {"input_tokens": 0, "output_tokens": 0}
    t0 = time.time()
    for attempt in range(cfg.retries + 1):
        res = client.chat(
            model=model,
            system=None,  # role and rules are in the user prompt; the client turns think off
            user=prompt,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            response_format={"type": "json_object"},
        )
        usage_total["input_tokens"] += res.usage.input_tokens
        usage_total["output_tokens"] += res.usage.output_tokens
        try:
            narr = parse_narrative(res.text)
        except Exception as exc:  # noqa: BLE001
            attempts.append({"attempt": attempt, "error": f"parse: {exc}"})
            narr = None
            continue
        issues = check_all(quant, narr)
        errors = [i for i in issues if i.severity == "error"]
        attempts.append({"attempt": attempt, "errors": [f"{i.check}:{i.field}" for i in errors]})
        if not errors:
            break
    cliches = detect_cliches(narr) if narr else []
    return {
        "persona_id": str(uuid.uuid4()),
        "model": model,
        "branches": branches(quant),
        "quant": quant,
        "narrative": narr or {k: "" for k in FIELD_KEYS},
        "issues": [i.__dict__ for i in issues],
        "cliches": [c.__dict__ if hasattr(c, "__dict__") else str(c) for c in cliches],
        "attempts": attempts,
        "ok": narr is not None and not any(i.severity == "error" for i in issues),
        "usage": usage_total,
        "seconds": round(time.time() - t0, 2),
    }


def run(cfg: GenerateConfig) -> Path:
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "personas.jsonl"
    done_idx: set[int] = set()
    done_ids: set[str] = set()
    n_legacy = 0
    if path.exists():
        bad = 0
        # A crash can leave a truncated last line (null bytes, etc.). Drop unreadable
        # lines and rewrite the file.
        keep: list[str] = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        for line in lines:
            try:
                rec = json.loads(line)
                done_idx.add(rec["index"])
                qid = rec.get("quant_id")
                if qid:
                    done_ids.add(str(qid))
                else:
                    n_legacy += 1
                keep.append(line if line.endswith("\n") else line + "\n")
            except Exception:  # noqa: BLE001
                bad += 1
        if bad:
            logger.warning("%s: dropping %d unparseable line(s) from checkpoint", cfg.run_name, bad)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(keep)
        logger.info("resuming %s: %d already generated", cfg.run_name, len(done_idx))

    ids: list[str] = []
    if cfg.quant_parquet is not None:
        quants, ids = _quant_dicts_from_parquet(cfg.quant_parquet, cfg.n, cfg.seed)
        cfg.n = len(quants)
    else:
        sampler = ChainSampler(load_spec(cfg.spec_name))
        rng = np.random.default_rng(cfg.seed)
        quants = [sampler.sample_one(rng) for _ in range(cfg.n)]

    if ids and n_legacy:
        raise SystemExit(
            f"{cfg.run_name}: {n_legacy} checkpoint rows have no quant_id. Index-based "
            "resume skips the wrong rows when `n` changes — set the old bank aside or "
            "backfill the ids, then rerun."
        )
    if ids:
        todo_idx = [i for i, pid in enumerate(ids) if pid not in done_ids]
    else:
        todo_idx = [i for i in range(cfg.n) if i not in done_idx]
    todo = [quants[i] for i in todo_idx]
    model = cfg.model or settings.llm_default_model
    conc = cfg.concurrency or settings.llm_max_concurrent
    (out / "generation_metadata.json").write_text(
        json.dumps(
            {
                "run": cfg.run_name,
                "n": cfg.n,
                "seed": cfg.seed,
                "model": model,
                "temperature": cfg.temperature,
                "spec": cfg.spec_name,
                "quant_parquet": str(cfg.quant_parquet) if cfg.quant_parquet else None,
                "ablation": cfg.ablation,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "concurrency": conc,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("generating %d personas with %s (concurrency %d)", len(todo), model, conc)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=conc) as pool:
        client = get_client()
        futures = {
            pool.submit(_generate_one, client, model, q, cfg): i for i, q in zip(todo_idx, todo)
        }
        n_ok = 0
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            i = futures[fut]
            rec["index"] = i
            if ids:
                # persona_id is an opaque UUID (public contract). The resume key is a
                # separate field that points at the quant row.
                rec["quant_id"] = ids[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_ok += rec["ok"]
            if k % 50 == 0 or k == len(todo):
                rate = k / (time.time() - t0)
                logger.info(
                    "%d/%d ok=%.1f%% %.2f/s eta %.0f min",
                    k,
                    len(todo),
                    100 * n_ok / k,
                    rate,
                    (len(todo) - k) / rate / 60,
                )
    return path


def load_run(run_name: str) -> list[dict[str, Any]]:
    path = settings.synthetic_dir / run_name / "personas.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8")]
