"""Downstream 4 — can an LLM use this resource's **prose** (ADR-022).

The first three tasks (genre portfolio, imputation, sample design) used numeric
anchors only. A language-resource reviewer will first ask what the narrative
(the prose) adds. The manuscript's only answer is the SSR holdout, where
release (narrative) 0.301 and twin (anchors only) 0.297 were the same.

So the narrative is given to the LLM as **examples**. Real respondents have no
narrative, so the only place prose can enter is the example slot. The task is
the same as the genre portfolio in §6: given the rest of a real attender's
cultural profile, predict one held-out genre (probability 0–100 → AUC).

Conditions — **the example people are the same across conditions.** What
changes is only how those people are shown.

  demo          Target demographics only. No examples. Upper bound of published tables.
  zero          Target cultural profile. No examples. The model's prior knowledge.
  shot_anchor   Show k synthetic personas as an **anchor list**.
  shot_narr     Show the same k people as **narrative prose** (summary, taste, companion, barrier).
  knn_narr      **Retrieve** the k people with the closest profile from the bank and show their narratives.
                Using the bank as a retrieval corpus — the most natural LLM use of the resource.

shot_narr − shot_anchor is "what the prose adds". Same targets and same example
people, so the interval on the difference comes from a paired bootstrap.

Examples use **synthetic personas only**. Real respondents' profiles
are used only inside the prompt, and person-level results stay in gitignored
`outputs/logs/`. `outputs/reports/` gets condition-level aggregates only.

Run: uv run python scripts/era_llm_genre.py --limit 300
Outputs: outputs/reports/era_llm_genre.csv · .md   (person-level: outputs/logs/era_llm_genre_rows.csv)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import ANCHOR_BY_NAME, GENRE_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("llm_genre")

SEED = 20260922
BASE_URL = settings.llm_base_url
MODEL = os.environ.get("SFP_LLM_MODEL", "qwen3.8:27b-mtp-q4_K_M")
K_SHOT = 8
TIMEOUT = 300
N_BOOT = 2000
CONDITIONS = ("demo", "zero", "shot_anchor", "shot_narr", "knn_narr")
DEMOGRAPHIC = ["sex", "age_band", "living_area", "education", "occupation", "household_income_band"]
PROFILE_EXTRA = ["attend_count_band", "paid_spend_band", "viewing_companion"]
NARR_FIELDS = ["summary", "cultural_taste", "companions", "barriers"]


# --------------------------------------------------------------------------- rendering


def genre_label(name: str) -> str:
    spec = ANCHOR_BY_NAME.get(name)
    return (spec.label_ko.replace(" 관람", "") if spec else name).strip()


def demo_text(q: dict) -> str:
    return "인구 정보: " + ", ".join(
        f"{ANCHOR_BY_NAME[a].label_ko}={q[a]}" for a in DEMOGRAPHIC if a in q
    )


def culture_text(q: dict, held_out: str) -> str:
    seen = [genre_label(g) for g in GENRE_ANCHORS if g != held_out and str(q.get(g)) == "예"]
    unseen = [genre_label(g) for g in GENRE_ANCHORS if g != held_out and str(q.get(g)) == "아니오"]
    extra = ", ".join(str(q.get(a)) for a in PROFILE_EXTRA if a in q)
    return (
        "지난 1년 관람한 장르: "
        + (", ".join(seen) or "없음")
        + "\n관람하지 않은 장르: "
        + (", ".join(unseen) or "없음")
        + "\n관람 횟수·지출·동반자: "
        + extra
    )


def narrative_text(narr: dict) -> str:
    return "\n".join(narr[f] for f in NARR_FIELDS if narr.get(f))


def label_line(q: dict, held_out: str) -> str:
    return (
        f"정답: {genre_label(held_out)} 관람 = {'예' if str(q.get(held_out)) == '예' else '아니오'}"
    )


def build_prompt(target: dict, held_out: str, cond: str, shots: list[dict]) -> str:
    name = genre_label(held_out)
    head = (
        "당신은 문화향유 조사 자료를 다루는 분석가다. 한 사람의 프로필을 보고 "
        f"지난 1년간 {name} 을(를) 관람했을 확률을 0 에서 100 사이 정수로만 답한다. "
        "설명 없이 숫자만 출력한다.\n\n"
    )
    body = ""
    if cond in ("shot_anchor", "shot_narr", "knn_narr") and shots:
        body += "다음은 같은 조사를 바탕으로 만든 가상 인물들이다.\n\n"
        for sh in shots:
            q = sh["quant"]
            if cond == "shot_anchor":
                body += demo_text(q) + "\n" + culture_text(q, held_out)
            else:
                body += narrative_text(sh["narrative"])
            body += "\n" + label_line(q, held_out) + "\n\n"
    body += "이제 이 사람을 판단하라.\n" + demo_text(target)
    if cond != "demo":
        body += "\n" + culture_text(target, held_out)
    body += f"\n{name} 관람 확률(0-100):"
    return head + body


# --------------------------------------------------------------------------- choosing examples


def balanced_shots(bank: list[dict], held_out: str, rng: np.random.Generator) -> list[dict]:
    """k people with yes/no answers balanced. Stops sparse genres from being all '아니오' in the examples."""
    yes = [p for p in bank if str(p["quant"].get(held_out)) == "예"]
    no = [p for p in bank if str(p["quant"].get(held_out)) == "아니오"]
    half = K_SHOT // 2
    pick = [yes[i] for i in rng.choice(len(yes), half, replace=False)] + [
        no[i] for i in rng.choice(len(no), K_SHOT - half, replace=False)
    ]
    return [pick[i] for i in rng.permutation(len(pick))]


def profile_vector(q: dict, held_out: str) -> tuple:
    return tuple(str(q.get(g)) for g in [*GENRE_ANCHORS, *PROFILE_EXTRA] if g != held_out)


def knn_shots(
    bank: list[dict], target: dict, held_out: str, rng: np.random.Generator
) -> list[dict]:
    """The k people whose cultural profile overlaps the target most (yes/no balance kept). Ties broken at random."""
    tv = profile_vector(target, held_out)

    def score(p: dict) -> float:
        pv = profile_vector(p["quant"], held_out)
        return sum(a == b for a, b in zip(tv, pv, strict=True)) + rng.random() * 1e-3

    half = K_SHOT // 2
    yes = sorted(
        (p for p in bank if str(p["quant"].get(held_out)) == "예"), key=score, reverse=True
    )
    no = sorted(
        (p for p in bank if str(p["quant"].get(held_out)) == "아니오"), key=score, reverse=True
    )
    pick = yes[:half] + no[: K_SHOT - half]
    return [pick[i] for i in rng.permutation(len(pick))]


# --------------------------------------------------------------------------- calls


def ask(prompt: str) -> float | None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 8},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("call failed: %s", exc)
        return None
    text = (out.get("message") or {}).get("content", "")
    m = re.search(r"\d{1,3}", text)
    if not m:
        log.warning("unparsed answer: %r", text[:40])
        return None
    return min(max(int(m.group()), 0), 100) / 100.0


# --------------------------------------------------------------------------- run


BANK_PATH = os.environ.get(
    "SFP_BANK_PATH"
)  # second-generator experiment: a different narrative bank as examples (era_second_generator)


def load_bank() -> list[dict]:
    p = (
        Path(BANK_PATH)
        if BANK_PATH
        else settings.data_root / "synthetic" / "era_2024_narrative" / "personas.jsonl"
    )
    bank = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    bank = [b for b in bank if b.get("ok") and str(b["quant"].get("attend_any")) == "예"]
    log.info("exemplar bank (2024 narrative, attenders, ok) n=%d", len(bank))
    return bank


def run(waves: list[int], limit: int, conditions: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    bank = load_bank()
    rows = []
    for wave in waves:
        real = build_official_anchor_frame(wave)
        real = real[real.attend_any.astype(str) == "예"].reset_index(drop=True)
        idx = rng.permutation(len(real))[:limit]
        for j, i in enumerate(idx):
            target = real.iloc[i].to_dict()
            held_out = GENRE_ANCHORS[j % len(GENRE_ANCHORS)]
            truth = int(str(target[held_out]) == "예")
            # For the same target, shot_anchor and shot_narr use the **same example people**.
            shots = balanced_shots(bank, held_out, rng)
            knn = knn_shots(bank, target, held_out, rng) if "knn_narr" in conditions else []
            for cond in conditions:
                use = knn if cond == "knn_narr" else shots
                p = ask(build_prompt(target, held_out, cond, use))
                if p is None:
                    continue
                rows.append(
                    {
                        "wave": wave,
                        "target_idx": j,
                        "condition": cond,
                        "held_out": held_out,
                        "truth": truth,
                        "p_hat": p,
                        "weight": float(target["w"]),
                    }
                )
            if (j + 1) % 25 == 0:
                log.info("wave=%d %d/%d", wave, j + 1, len(idx))
    return pd.DataFrame(rows)


def _wauc(g: pd.DataFrame) -> float:
    y = g.truth.to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, g.p_hat.to_numpy(), sample_weight=g.weight.to_numpy()))


def summarise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = pd.DataFrame(
        [
            {"condition": c, "n": len(g), "positives": int(g.truth.sum()), "auc": _wauc(g)}
            for c, g in df.groupby("condition")
        ]
    )
    # Paired bootstrap: resample targets with replacement for the AUC-difference interval.
    rng = np.random.default_rng(SEED)
    keys = df[["wave", "target_idx"]].drop_duplicates().to_numpy()
    pairs = [
        ("shot_narr", "shot_anchor"),
        ("knn_narr", "shot_narr"),
        ("shot_anchor", "zero"),
        ("zero", "demo"),
    ]
    diffs = []
    for a, b in pairs:
        if a not in set(df.condition) or b not in set(df.condition):
            continue
        both = df[df.condition.isin([a, b])]
        point = _wauc(both[both.condition == a]) - _wauc(both[both.condition == b])
        boots = []
        for _ in range(N_BOOT):
            pick = keys[rng.integers(0, len(keys), len(keys))]
            sel = pd.DataFrame(pick, columns=["wave", "target_idx"]).merge(
                both, on=["wave", "target_idx"]
            )
            boots.append(_wauc(sel[sel.condition == a]) - _wauc(sel[sel.condition == b]))
        boots = np.array([x for x in boots if np.isfinite(x)])
        diffs.append(
            {
                "contrast": f"{a} - {b}",
                "auc_diff": point,
                "lo": float(np.percentile(boots, 2.5)) if len(boots) else float("nan"),
                "hi": float(np.percentile(boots, 97.5)) if len(boots) else float("nan"),
            }
        )
    return per, pd.DataFrame(diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, action="append", default=None)
    ap.add_argument("--limit", type=int, default=300, help="target attenders per wave")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args()
    waves = args.wave or [2022, 2020]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")

    df = run(waves, args.limit, conditions)
    if df.empty:
        raise SystemExit("no LLM answers — the model did not respond")

    # Person-level rows go only to gitignored logs (not a public output).
    logs = settings.outputs_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("SFP_OUT_TAG", "")
    suffix = f"_{tag}" if tag else ""
    df.drop(columns=["weight"]).to_csv(logs / f"era_llm_genre_rows{suffix}.csv", index=False)

    per, diffs = summarise(df)
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    per.to_csv(reports / f"era_llm_genre{suffix}.csv", index=False)
    diffs.to_csv(reports / f"era_llm_genre_contrasts{suffix}.csv", index=False)
    (reports / f"era_llm_genre{suffix}.md").write_text(
        "# Downstream 4 — LLM 이 서사를 쓸 수 있는가 (ADR-022)\n\n"
        f"모델 {MODEL}, 파동 {waves}, 파동당 {args.limit} 명, k={K_SHOT}, 예시는 2024 합성 페르소나.\n\n"
        "## 조건별 AUC\n\n"
        + per.round(4).to_markdown(index=False)
        + "\n\n## 짝지은 차이 (대상 단위 부트스트랩)\n\n"
        + diffs.round(4).to_markdown(index=False)
        + "\n\n`shot_narr - shot_anchor` 가 「같은 인물을 글로 보여줄 때 더해지는 것」이다.\n"
        "구간이 0 을 포함하면 서사는 앵커 이상의 쓸 수 있는 정보를 주지 않는 것이다.\n",
        encoding="utf-8",
    )
    log.info("wrote %s", reports / f"era_llm_genre{suffix}.csv")
    print(per.round(4).to_string(index=False))
    print(diffs.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
