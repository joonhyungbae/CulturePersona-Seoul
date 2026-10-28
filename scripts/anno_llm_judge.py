"""LLM rater — scores the same items as humans, on the same rubric (preregistration S5, ADR-023).

**The prompt is frozen before the first human score.** This file is in the preregistration commit.

The model must be a **different family from the generator (Qwen)** — the same family is lenient
toward its own text (Panickssery et al., 2024). The local Gemma is used.

The rubric copies the contradiction definition and the four scales from the rater guide
(`annotation/src/guide.md`) as written. The LLM is not given stage 1 (guessing from masked
text) — the human guessing task is measured separately as S4 (S3 was withdrawn).

Input: annotation/data_private/items_full.json (full source; no stratum or key), practice
items excluded
Output: outputs/anno/llm_judge.jsonl (gitignore — hidden from raters until human scoring ends)

Run: uv run python scripts/anno_llm_judge.py [--limit N]
     uv run python scripts/anno_llm_judge.py --bank   # every clean persona in the bank
                                                   # (for PPI, ~16 h)
     Bank mode judges **only personas that passed the validator**, so the population matches the
     PPI correction term (consensus − LLM on the clean sample).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("llm_judge")

BASE_URL = settings.llm_base_url
MODEL = os.environ.get("SFP_JUDGE_MODEL", "gemma4:12b")
TIMEOUT = 300
ITEMS = ROOT / "annotation" / "data_private" / "items_full.json"
OUT = settings.outputs_dir / "anno" / "llm_judge.jsonl"
OUT_BANK = (
    settings.outputs_dir / "anno" / "llm_bank.jsonl"
)  # full-bank decisions (PPI, preregistration §6.4)
SCALES = ("grounded", "coherent", "plausible", "fluent")

RUBRIC = """당신은 가상 인물 프로필의 품질을 검수하는 평가자다. 아래 기준을 그대로 따른다.

[모순]
서사가 앵커(정해진 사실)와 양립할 수 없는 사실을 말하면 모순이다.
- 모순인 예: 앵커는 지난 1년 관람 안 함인데 서사는 지난달 뮤지컬을 봤다 / 앵커는 40대인데 서사는 고등학생 /
  관람 장르에 연극이 없는데 올해 연극을 봤다 / 앵커는 혼자 관람인데 늘 가족과 관람한다.
- 모순이 아닌 예: 앵커의 사실을 서사가 언급하지 않음 / 어울리지 않아 보여도 가능한 내용 /
  과거 이야기("젊었을 때는 연극을 자주 봤다") / 관람하지 않은 장르에 관심이 있다는 말.

[앵커 부합 1-5]
5 앵커와 어긋나는 내용이 전혀 없다 / 4 애매한 표현이 한 곳 / 3 사소한 불일치 한 곳 /
2 명백한 모순 한 곳 / 1 명백한 모순 두 곳 이상

[일관성 1-5] 12개 항목이 한 사람의 이야기로 읽히는가 (앵커가 아니라 서사 항목끼리)
5 자연스럽게 이어진다 / 4 한 곳이 조금 겉돈다 / 3 두세 곳이 안 맞는다 / 2 여러 항목이 다른 사람 같다 / 1 한 사람으로 안 읽힌다

[개연성 1-5] 이런 사람이 서울에 실제로 있을 법한가. 드물어도 가능하면 3 이상. 고정관념으로 판단하지 않는다.
5 매우 그럴듯 / 4 그럴듯 / 3 드물지만 가능 / 2 있을 법하지 않다 / 1 거의 있을 수 없다

[한국어 자연스러움 1-5] 내용이 아니라 문장을 본다. 한자나 외국어가 섞이면 2 이하.
5 원어민이 쓴 글 / 4 어색한 곳 하나 / 3 어색한 곳 여러 곳 / 2 번역투·기계적 문장이 많다 / 1 읽기 어렵다

다음 JSON 하나만 출력한다. 설명은 쓰지 않는다.
{"contradiction": "yes" 또는 "no", "contra_fields": [모순이 있는 항목 키 목록], "grounded": 정수, "coherent": 정수, "plausible": 정수, "fluent": 정수}"""


def render(item: dict) -> str:
    anchors = "\n".join(f"- {a['label']}: {a['value']}" for a in item["anchors"])
    seen = ", ".join(item["genres_attended"]) or "없음"
    unseen = ", ".join(item["genres_not"]) or "없음"
    narr = "\n".join(
        f"[{f['key']}] {f['label']}: {item['narrative'][f['key']]}" for f in item["fields"]
    )
    return (
        f"{RUBRIC}\n\n=== 앵커 ===\n{anchors}\n- 지난 1년 관람한 장르: {seen}\n- 관람하지 않은 장르: {unseen}\n\n"
        f"=== 서사 ===\n{narr}\n\n=== 평가 (JSON) ==="
    )


def parse(text: str) -> dict | None:
    try:
        obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    if obj.get("contradiction") not in ("yes", "no"):
        return None
    for k in SCALES:
        try:
            v = int(obj.get(k))
        except (TypeError, ValueError):
            return None
        if not 1 <= v <= 5:
            return None
        obj[k] = v
    obj["contra_fields"] = [str(x) for x in obj.get("contra_fields") or []]
    return obj


def ask(prompt: str) -> dict | None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        # gemma4 is a thinking model; left on, it spends the 200 tokens on reasoning and the answer
        # (JSON) comes back empty. As designed — "JSON only, no explanation" — thinking is off
        # (2026-09-23, a technical fix before looking at decisions, preregistration §5).
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 200},
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
    return parse((out.get("message") or {}).get("content", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="for a trial: only the first N items")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt only")
    ap.add_argument("--bank", action="store_true", help="every clean persona in the bank (PPI)")
    args = ap.parse_args()

    out = OUT_BANK if args.bank else OUT
    if args.bank:
        # Build the same shape as audit items (anchor card, genres, 12 fields) so the same prompt
        # sees the same input.
        sys.path.insert(0, str(ROOT / "scripts"))
        from anno_build_sample import FIELDS, WAVES, anchor_card, genre_lists, is_clean, load_bank

        items = []
        for w in WAVES:
            for r in load_bank(w):
                if not is_clean(r):
                    continue
                yes, no = genre_lists(r["quant"])
                items.append(
                    {
                        "item_id": r["quant_id"],
                        "wave": w,
                        "anchors": anchor_card(r["quant"]),
                        "genres_attended": yes,
                        "genres_not": no,
                        "fields": [{"key": f.key, "label": f.label_ko} for f in FIELDS],
                        "narrative": {f.key: r["narrative"].get(f.key, "") for f in FIELDS},
                    }
                )
    else:
        items = [i for i in json.loads(ITEMS.read_text(encoding="utf-8")) if not i["practice"]]
    if args.limit:
        items = items[: args.limit]
    if args.dry_run:
        print(render(items[0]))
        return 0

    done = set()
    if out.exists():
        done = {json.loads(line)["item_id"] for line in out.open(encoding="utf-8") if line.strip()}
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [i for i in items if i["item_id"] not in done]
    log.info("model=%s items=%d (already %d)", MODEL, len(todo), len(done))
    for n, it in enumerate(todo, 1):
        res = None
        for _ in range(3):  # if the format breaks, retry twice under the same conditions
            res = ask(render(it))
            if res:
                break
        if res is None:
            log.warning("unparsed after retries: %s", it["item_id"])
            continue
        res.update(item_id=it["item_id"], model=MODEL, judged_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        if n % (1000 if args.bank else 25) == 0:
            log.info("%d/%d", n, len(todo))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
