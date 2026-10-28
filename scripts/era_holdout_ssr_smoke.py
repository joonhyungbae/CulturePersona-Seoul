"""Era holdout SSR smoke — 2024 narrative n=20 (ADR-009 Step 2).

1) write data/eval/era_holdout/ssr_refs.json
2) Ollama SSR sentence answers
3) cosine via .venv-embed → softmax → TVD against the observed margin

Run:
  uv run python scripts/era_holdout_ssr_smoke.py
  .venv-embed/bin/python scripts/ssr_embed.py --dir data/eval/era_holdout --selftest
  .venv-embed/bin/python scripts/ssr_embed.py --dir data/eval/era_holdout --pattern 'smoke_answers.jsonl'
  # (the script also calls the embed subprocess itself)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.generate import load_run
from sfac_persona.grounding.official import load_official_value_labels
from sfac_persona.llm_client import get_client
from sfac_persona.narrative import anchor_block

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_ssr_smoke")

OUT = settings.data_root / "eval" / "era_holdout"
N_SMOKE = 20
TAU = float(json.loads((OUT / "tau.json").read_text(encoding="utf-8"))["tau"])  # frozen on 2024
SEED = 20260910

QUESTIONS = {
    "leisure_activity_1st": "주말이나 휴일에는 주로 무엇을 하며 시간을 보내십니까?",
    "leisure_activity_wanted_1st": "여건이 된다면 앞으로 가장 하고 싶은 여가활동은 무엇입니까?",
    "leisure_time_weekday": "평일에는 하루에 여가시간이 대략 몇 시간 있습니까?",
    "leisure_time_weekend": "주말·휴일에는 하루에 여가시간이 대략 몇 시간 있습니까?",
}

TIME_WEEKDAY = [str(i) for i in range(9)]  # 0–8 hours
TIME_WEEKEND = [str(i) for i in range(13)]  # 0–12 hours


def _activity_options() -> list[str]:
    labs = load_official_value_labels(2024).get("A3_1", {})
    items: dict[int, str] = {}
    for k, v in labs.items():
        try:
            items[int(float(k))] = v
        except ValueError:
            continue
    return [items[i] for i in sorted(items)]


def write_refs(options_act: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    refs: dict[str, dict[str, list[str]]] = {}
    for var, opts in (
        ("leisure_activity_1st", options_act),
        ("leisure_activity_wanted_1st", options_act),
    ):
        refs[var] = {o: [f"나는 주로 {o}을(를) 한다.", o[:40]] for o in opts}
        refs[var]["기타"] = refs[var].get(
            "기타", ["여가 때 하는 일이 위에 없는 다른 것이다.", "기타"]
        )
    for var, hours in (
        ("leisure_time_weekday", TIME_WEEKDAY),
        ("leisure_time_weekend", TIME_WEEKEND),
    ):
        refs[var] = {h: [f"하루 여가시간은 대략 {h}시간이다.", f"약 {h}시간"] for h in hours}
    path = OUT / "ssr_refs.json"
    path.write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


PROMPT = """아래 인물의 처지에서, 보기 목록 없이 각 질문에 한두 문장으로만 답하십시오. JSON 객체 하나만 출력.
키: {keys}

[인물]
{persona}

[질문]
{questions}
"""


def _persona_blob(rec: dict) -> str:
    q = rec.get("quant") or {}
    narr = rec.get("narrative") or {}
    bits = [anchor_block(q)]
    if narr.get("summary"):
        bits.append(f"한 줄: {narr['summary']}")
    if narr.get("weekend_routine"):
        bits.append(f"주말: {narr['weekend_routine']}")
    return "\n".join(bits)


def run_answers(n: int) -> Path:
    recs = load_run("era_2024_narrative")
    rng = np.random.default_rng(SEED)
    if len(recs) > n:
        idx = rng.choice(len(recs), size=n, replace=False)
        recs = [recs[i] for i in sorted(idx)]
    path = OUT / "smoke_answers.jsonl"
    done = set()
    if path.exists():
        done = {
            json.loads(l)["persona_id"]
            for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        }
    client = get_client()
    model = settings.llm_default_model
    keys = list(QUESTIONS)
    qtext = "\n".join(f"- {k}: {QUESTIONS[k]}" for k in keys)
    todo = [r for r in recs if r.get("persona_id") not in done]
    log.info("SSR smoke answers: %d done, %d todo", len(done), len(todo))
    with open(path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, 1):
            prompt = PROMPT.format(
                keys=", ".join(keys),
                persona=_persona_blob(rec),
                questions=qtext,
            )
            texts = {k: None for k in keys}
            for attempt in range(3):
                res = client.chat(
                    model=model,
                    system=None,
                    user=prompt,
                    temperature=0.7,
                    max_tokens=1200,
                    response_format={"type": "json_object"},
                )
                try:
                    raw = res.text[res.text.find("{") : res.text.rfind("}") + 1]
                    obj = json.loads(raw)
                    texts = {k: str(obj.get(k, "")).strip() or None for k in keys}
                    if sum(1 for v in texts.values() if v) >= 3:
                        break
                except Exception:  # noqa: BLE001, S112 — if the response is not JSON, retry
                    continue
            row = {
                "persona_id": rec["persona_id"],
                "condition": "narrative",
                "mode": "ssr",
                "texts": texts,
                "attempt": attempt,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            log.info("  %d/%d", i, len(todo))
            time.sleep(0.05)
    return path


def _embed_subprocess(selftest: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    py = root / ".venv-embed" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)  # `uv sync --extra embed` installs the embedding stack here
    cmd = [str(py), str(root / "scripts" / "ssr_embed.py"), "--dir", str(OUT)]
    if selftest:
        cmd.append("--selftest")
    else:
        cmd.extend(["--pattern", "smoke_answers.jsonl"])
    log.info("embed: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root))


def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum()


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


def _real_activity_dist(var: str) -> pd.Series:
    from sfac_persona.config import settings as S

    keys = pd.read_parquet(S.official_person_dir / "official_person_keys.parquet")
    k = keys[(keys["wave"] == 2024) & (keys["frame_group"] == "general")]
    wmap = k.set_index("row_uid")["weight"]
    canon = pd.read_parquet(S.official_person_dir / "official_person_canonical.parquet")
    sub = canon[(canon["wave"] == 2024) & (canon["canonical_var"] == var)]
    sub = sub[sub["row_uid"].isin(wmap.index)]
    labs = load_official_value_labels(2024).get("A3_1" if "activity" in var else "A5_1", {})
    # A5_1 same labels as A3 for wanted
    if "wanted" in var:
        labs = load_official_value_labels(2024).get("A5_1", labs)

    def lab(x: object) -> str:
        try:
            key = str(int(float(x)))
        except (TypeError, ValueError):
            return str(x)
        return labs.get(key, key)

    vals = sub["value"].map(lab)
    ww = sub["row_uid"].map(wmap).astype(float)
    tab = pd.DataFrame({"v": vals, "w": ww}).groupby("v")["w"].sum()
    return tab / tab.sum()


def _real_time_dist(var: str) -> pd.Series:
    from sfac_persona.config import settings as S

    keys = pd.read_parquet(S.official_person_dir / "official_person_keys.parquet")
    k = keys[(keys["wave"] == 2024) & (keys["frame_group"] == "general")]
    wmap = k.set_index("row_uid")["weight"]
    canon = pd.read_parquet(S.official_person_dir / "official_person_canonical.parquet")
    sub = canon[(canon["wave"] == 2024) & (canon["canonical_var"] == var)]
    sub = sub[sub["row_uid"].isin(wmap.index)]
    vals = sub["value"].map(lambda x: str(int(float(x))) if pd.notna(x) else "NA")
    ww = sub["row_uid"].map(wmap).astype(float)
    tab = pd.DataFrame({"v": vals, "w": ww}).groupby("v")["w"].sum()
    return tab / tab.sum()


def evaluate(tau: float = TAU) -> Path:
    refs = json.loads((OUT / "ssr_refs.json").read_text(encoding="utf-8"))
    answers = OUT / "smoke_answers.jsonl"
    expected = OUT / f"cos_{answers.stem}.jsonl"
    if not expected.exists():
        raise FileNotFoundError(f"missing cos file {expected}")

    cos = {
        r["persona_id"]: r["cos"]
        for r in (
            json.loads(l) for l in expected.read_text(encoding="utf-8").splitlines() if l.strip()
        )
    }
    opt_order = {var: list(opts.keys()) for var, opts in refs.items()}

    mean_probs: dict[str, pd.Series] = {}
    for var, order in opt_order.items():
        acc = np.zeros(len(order))
        n = 0
        for cdict in cos.values():
            c = cdict.get(var)
            if not c or len(c) != len(order):
                continue
            acc += _softmax(np.asarray(c, dtype=float), tau)
            n += 1
        if n == 0:
            continue
        mean_probs[var] = pd.Series(acc / n, index=order)

    rows = []
    for var, syn_p in mean_probs.items():
        if "time" in var:
            real = _real_time_dist(var)
        else:
            real = _real_activity_dist(var)
        rows.append(
            {
                "item": var,
                "tvd_ssr_vs_real": _tvd(real, syn_p),
                "n_personas": len(cos),
                "tau": tau,
            }
        )
    df = pd.DataFrame(rows)
    report = settings.outputs_dir / "reports" / "era_holdout_ssr_smoke.md"
    lines = [
        "# Era holdout SSR smoke (2024 narrative n≤20)",
        "",
        "배관 검증용. 본규모 평가·기준선 풀셋 아님.",
        "",
        df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False),
        "",
        "- refs: `data/eval/era_holdout/ssr_refs.json`",
        "- answers: `data/eval/era_holdout/smoke_answers.jsonl`",
        f"- cos: `{expected.name}`",
        "",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(settings.outputs_dir / "reports" / "era_holdout_ssr_smoke.csv", index=False)
    log.info("wrote %s", report)
    return report


def main() -> int:
    opts = _activity_options()
    write_refs(opts)
    log.info("refs written (%d activity options)", len(opts))
    try:
        _embed_subprocess(selftest=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("selftest failed/skipped: %s", exc)
    run_answers(N_SMOKE)
    try:
        _embed_subprocess(selftest=False)
    except Exception as exc:  # noqa: BLE001
        log.error("embed failed: %s", exc)
        return 1
    evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
