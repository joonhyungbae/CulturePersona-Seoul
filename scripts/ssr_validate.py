"""Validate the SSR mapping itself — measure only **matching by meaning**, apart from persona quality.

  .venv-embed/bin/python scripts/ssr_validate.py [--answers <file>]

That reference sentences separate from each other (`ssr_embed.py --selftest`) is only a
necessary condition. "The reference sentences differ" and "a real answer sentence finds the
right reference sentence" are different claims. Both are measured.

  A discrimination  Distribution of the cosine gap between the 1st and 2nd options
                     on actually generated answers.
                     A gap near 0 means the mapping does not discriminate. Read it with τ —
                     gap d becomes a probability ratio exp(d/τ) under softmax
                     (τ=0.05, d=0.066 → 3.7×).
  B round-trip       Does a sentence with the same meaning and different wording return
                     to the right option.
                     The reference sentence itself has cosine 1.0, so it is not a test;
                     a rewritten sentence is required.
                     **Limit: the rewritten sentences were written by the same person
                     who wrote the references.**
                     They came from the same head, so this is a necessary-condition check —
                     a failure is a real problem, and a pass is not a full guarantee.
                     The decisive evidence is person-level accuracy in twin mode (the only
                     condition where the real respondent's true answer is known).
                     Only the 16 scale items are looked at. That is where the discrimination
                     gap is thinnest and where SSR does the real work.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

logging.disable(logging.WARNING)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util

from sentence_transformers import SentenceTransformer

_sp = importlib.util.spec_from_file_location("se", ROOT / "scripts" / "ssr_embed.py")
se = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(se)

# Constants used in the report. Exported from here rather than typed into the paper by hand.
TAU_REPORT = (
    0.05  # τ used to turn a gap into a probability ratio (the baseline before choosing at G4)
)
UNDECIDED_GAP = 0.01  # a smaller gap is counted as effectively undecided

L = ("낮음", "보통", "높음")
O4 = ("전혀 그렇지 않다", "대체로 그렇지 않다", "대체로 그렇다", "매우 그렇다")


def _t(a: str, b: str, c: str) -> dict[str, str]:
    return dict(zip(L, (a, b, c)))


# Rewritten sentences — wording was kept from overlapping the reference sentences on purpose.
PARA = {
    "E2_3": _t(
        "서울 문화행사는 기대에 못 미쳐요. 별로였습니다.",
        "서울 문화생활은 그럭저럭이에요. 나쁘지도 좋지도 않습니다.",
        "서울은 볼 게 정말 많아요. 아주 마음에 듭니다.",
    ),
    "F2": _t(
        "작년엔 문화생활을 거의 못 했어요. 아쉽습니다.",
        "작년 문화생활은 그냥 평범했어요.",
        "작년엔 보고 싶은 걸 원 없이 봤습니다. 뿌듯해요.",
    ),
    "F3": _t(
        "예술 없어도 사는 데 아무 지장 없어요.",
        "예술이 있으면 좋긴 한데 꼭 필요하진 않아요.",
        "예술은 제 인생에서 빼놓을 수 없는 부분이에요.",
    ),
    "F4": _t(
        "요즘 사는 게 너무 힘들어요. 웃을 일이 없습니다.",
        "그냥저냥 살아요. 특별할 것 없습니다.",
        "요즘 정말 좋아요. 사는 게 즐겁습니다.",
    ),
    "F5": _t(
        "예술이 동네를 바꾼다는 말은 잘 모르겠어요.",
        "예술이 지역에 조금은 보탬이 되겠죠.",
        "예술이 동네 분위기를 확 살린다고 봅니다.",
    ),
    "F9": _t(
        "사람 사귀려고 문화행사 갈 생각은 없어요.",
        "그런 자리가 있으면 한 번쯤은 가볼 수도 있죠.",
        "사람들과 어울리려고 꼭 그런 활동을 해보고 싶어요.",
    ),
    "F11": _t(
        "AI가 만든 작품 전시는 별로 보고 싶지 않아요.",
        "AI 작품 전시도 기회 되면 한번 보죠.",
        "AI 작품 전시 꼭 가보고 싶어요. 궁금합니다.",
    ),
    "F12": _t(
        "AI 그림 보러 돈 낼 생각은 전혀 없어요.",
        "값이 싸면 AI 작품에도 돈 낼 수 있죠.",
        "AI 작품이라도 좋으면 돈 아깝지 않아요.",
    ),
    "G1_1": _t(
        "같이 갈 사람이 생겨도 제 생활은 그대로일 거예요.",
        "같이 갈 사람 있으면 조금은 더 가겠죠.",
        "같이 갈 친구만 있으면 훨씬 자주 다닐 거예요.",
    ),
    "G1_2": _t(
        "취향에 맞는 게 생겨도 딱히 더 가진 않을 것 같아요.",
        "제 취향에 맞으면 좀 더 가긴 하겠죠.",
        "제 취향에 맞는 것만 있으면 훨씬 많이 갈 거예요.",
    ),
    "G1_3": _t(
        "동네에 시설이 늘어도 제가 더 갈 것 같진 않아요.",
        "가까우면 조금은 더 가게 되겠죠.",
        "동네에 생기면 훨씬 자주 갈 거예요. 멀어서 못 갑니다.",
    ),
    "G1_4": _t(
        "싸진다고 더 가진 않아요. 돈 문제가 아니에요.",
        "좀 싸지면 조금 더 가긴 하겠죠.",
        "가격만 내려가면 훨씬 많이 볼 거예요. 비싸서 못 봅니다.",
    ),
    "G1_5": _t(
        "시간이 남아도 그런 데 쓰진 않을 것 같아요.",
        "여유가 생기면 조금은 더 하겠죠.",
        "시간만 나면 훨씬 많이 다닐 거예요. 바빠서 못 갑니다.",
    ),
    "F6_1": dict(
        zip(
            O4,
            (
                "외롭다고 느낀 적 없어요.",
                "외로운 편은 아니에요. 가끔 그럴 뿐입니다.",
                "외롭다는 생각이 자주 듭니다.",
                "늘 외로워요. 견디기 힘들 정도입니다.",
            ),
        )
    ),
    "F6_2": dict(
        zip(
            O4,
            (
                "소외감 같은 건 느껴본 적 없어요.",
                "딱히 겉돈다고 느끼진 않아요.",
                "사람들 사이에서 겉도는 느낌이 자주 들어요.",
                "늘 저만 빠져 있는 것 같아요. 많이 힘듭니다.",
            ),
        )
    ),
    "F7": {
        "0명": "연락하고 지내는 사람이 아예 없어요.",
        "1~2명": "친한 사람은 한두 명이 다예요.",
        "3~6명": "친하게 지내는 사람이 네댓 명쯤 돼요.",
        "6명 이상": "친한 사람이 열 명 가까이 돼요.",
    },
}


def load_refs() -> dict:
    return json.loads(
        (ROOT / "data" / "eval" / "holdout" / "ssr_refs.json").read_text(encoding="utf-8")
    )


def check_margin(m, refs: dict, path: Path, out: dict) -> None:
    with open(path, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh]
    R = {v: se.option_embeddings(m, o, 32) for v, o in refs.items()}
    opts = {v: list(o) for v, o in refs.items()}
    texts, keys = [], []
    for r in recs:
        for v, s in (r.get("texts") or {}).items():
            if s and v in R:
                texts.append(s)
                keys.append(v)
    if not texts:
        print("A discrimination: no sentences in the answer file — skipped")
        return
    E = m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    marg, top = defaultdict(list), defaultdict(list)
    for e, v in zip(E, keys):
        c = R[v] @ e
        o = np.sort(c)[::-1]
        marg[v].append(float(o[0] - o[1]))
        top[v].append(opts[v][int(np.argmax(c))])
    allm = np.concatenate([np.array(x) for x in marg.values()])
    print(f"\n== A discrimination ({len(recs)} people · {len(texts)} sentences, {path.name}) ==")
    print(
        f"1st−2nd cosine gap: median {np.median(allm):.3f} · 5th percentile {np.quantile(allm, 0.05):.3f} "
        f"· answers with gap<{UNDECIDED_GAP}: {100 * (allm < UNDECIDED_GAP).mean():.1f}%"
    )
    out.update(
        margin_n_personas=len(recs),
        margin_n_sentences=len(texts),
        margin_median=float(np.median(allm)),
        margin_p05=float(np.quantile(allm, 0.05)),
        margin_undecided_pct=float(100 * (allm < UNDECIDED_GAP).mean()),
        tau_report=TAU_REPORT,
        undecided_gap=UNDECIDED_GAP,
    )
    rows = sorted(
        (float(np.median(marg[v])), max(top[v].count(o) for o in opts[v]) / len(top[v]), v)
        for v in refs
        if v in marg
    )
    print(f"{'item':<9}{'gap_med':>9}{'mode_share':>9}   ← thinnest 5")
    for md, sh, v in rows[:5]:
        print(f"{v:<9}{md:>9.3f}{sh:>8.0%}")
    degen = [v for md, sh, v in rows if sh > 0.95]
    print("items collapsed to one option:", degen or "none")
    out["margin_degenerate_items"] = len(degen)
    out["margin_thinnest"] = [{"item": v, "median": round(md, 4)} for md, _, v in rows[:5]]


def check_roundtrip(m, refs: dict, out: dict) -> None:
    ok = tot = 0
    bad, margins, adjacent = [], [], 0
    for v, pp in PARA.items():
        opts = list(refs[v])
        R = se.option_embeddings(m, refs[v], 32)
        E = m.encode(
            list(pp.values()), batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        for (want, sent), e in zip(pp.items(), E):
            c = R @ e
            got = opts[int(np.argmax(c))]
            tot += 1
            s = np.sort(c)[::-1]
            margins.append(float(s[0] - s[1]))
            if got == want:
                ok += 1
            else:
                bad.append((v, want, got, sent))
                if abs(opts.index(got) - opts.index(want)) == 1:
                    adjacent += 1
    chance = float(np.mean([1 / len(refs[v]) for v in PARA for _ in PARA[v]]))
    print(f"\n== B round-trip (scale items {len(PARA)}, options {tot}) ==")
    print(
        f"accuracy {ok}/{tot} = {ok / tot:.1%}   (chance {chance:.1%})   gap median {np.median(margins):.3f}"
    )
    print(
        f"of {len(bad)} errors, adjacent cell {adjacent} · two or more cells off {len(bad) - adjacent}"
    )
    for v, w, g, s in bad:
        print(f"  {v:<7} intended={w:<12} → chosen={g:<12} | {s[:44]}")
    out.update(
        rt_items=len(PARA),
        rt_options=tot,
        rt_correct=ok,
        rt_acc=ok / tot,
        rt_chance=chance,
        rt_adjacent=adjacent,
        rt_far=len(bad) - adjacent,
        rt_margin_median=float(np.median(margins)),
        rt_errors=[{"item": v, "want": w, "got": g} for v, w, g, _ in bad],
    )
    print(
        "\nHow to read this: an error of two or more cells means that item's "
        "reference sentences should be revised. "
        "Confusion with the adjacent cell is the most benign failure on an ordinal scale."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="data/eval/holdout/answers_release_narrative_ssr.jsonl")
    ap.add_argument("--model", default=se.DEFAULT_MODEL)
    a = ap.parse_args()
    refs = load_refs()
    m = SentenceTransformer(a.model, device="cpu")
    m.max_seq_length = min(m.max_seq_length, 256)
    out: dict = {"model": a.model, "answers": a.answers}
    # Size and separation of the reference sentences themselves (gate G4 bar: max cosine < 0.90)
    mx = 0.0
    for o in refs.values():
        E = se.option_embeddings(m, o, 32)
        C = E @ E.T
        iu = np.triu_indices(len(o), 1)
        mx = max(mx, float(C[iu].max()))
    out.update(
        n_items=len(refs),
        n_options=sum(len(o) for o in refs.values()),
        n_reference_sentences=sum(len(x) for o in refs.values() for x in o.values()),
        ref_max_cos=mx,
    )
    print(
        f"reference sentences {out['n_reference_sentences']} · "
        f"options {out['n_options']} · max cosine {mx:.3f}"
    )
    p = ROOT / a.answers
    if p.exists():
        check_margin(m, refs, p, out)
    else:
        print(f"A discrimination: {p} missing — skipped")
    check_roundtrip(m, refs, out)
    dst = ROOT / "outputs" / "reports" / "ssr_validate.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {dst}")


if __name__ == "__main__":
    main()
