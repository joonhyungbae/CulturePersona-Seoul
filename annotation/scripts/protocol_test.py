"""Protocol test for the annotation backend — the same checks against the local server and the Apps Script web app.

  default (smoke)  read-only: login, progress, own state, reject a bad code. **Safe to run against the real spreadsheet.**
  --full           both raters do stage 1 (reveal) and scoring on every item → lock → adjudication. Fake scores accumulate, so
                   **local or test sheets (the emulator) only.**

Run:
  python3 scripts/protocol_test.py --endpoint http://127.0.0.1:5180/api --codes R1=..,R2=..,ADJ=..,OWNER=.. --full
  python3 scripts/protocol_test.py --endpoint https://script.google.com/macros/s/.../exec --codes R1=..,OWNER=..
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import urllib.request
from pathlib import Path

ITEMS = json.loads(
    (Path(__file__).resolve().parents[1] / "public" / "data" / "items.json").read_text("utf-8")
)
MAIN = [i["item_id"] for i in ITEMS if not i["practice"]]
PRACTICE = [i["item_id"] for i in ITEMS if i["practice"]]
failures: list[str] = []

try:  # Also check that a received key actually unseals the item (skip only that check if unavailable)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:  # pragma: no cover
    AESGCM = None


def unseal(item_id: str, key_b64: str) -> dict:
    it = next(i for i in ITEMS if i["item_id"] == item_id)
    iv, ct = (base64.b64decode(it["sealed"][k]) for k in ("iv", "ct"))
    return json.loads(AESGCM(base64.b64decode(key_b64)).decrypt(iv, ct, None))


def call(endpoint: str, body: dict) -> dict:
    req = urllib.request.Request(
        endpoint, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "text/plain"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def check(name: str, cond: bool, detail: object = "") -> None:
    print(
        ("  ok  " if cond else "  FAIL")
        + f" {name}"
        + (f" — {detail}" if not cond and detail else "")
    )
    if not cond:
        failures.append(name)


STAGE1 = {"age_guess": "30대", "attend_guess": "예", "secs": 5}


def annotation(rng: random.Random, contra: bool) -> dict:
    return {
        "stage2": {
            "contradiction": "yes" if contra else "no",
            "contra_fields": ["summary"] if contra else [],
            "contra_quote": "근거" if contra else "",
            "grounded": 2 if contra else rng.choice([4, 5]),
            "coherent": rng.choice([4, 5]),
            "plausible": rng.choice([4, 5]),
            "fluent": 5,
            "note": "",
            "secs": 20,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--codes", required=True, help="R1=..,R2=..,ADJ=..,OWNER=..")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    codes = dict(kv.split("=", 1) for kv in args.codes.split(","))
    E = args.endpoint

    print("[읽기]")
    r = call(E, {"action": "login", "code": "WRONGCODE000"})
    check("틀린 코드는 거부", not r.get("ok"), r)
    for role, code in codes.items():
        r = call(E, {"action": "login", "code": code})
        check(f"{role} 로그인", r.get("ok") and r.get("role") == role, r)
        r = call(E, {"action": "login", "code": code.lower()})
        check(f"{role} 소문자 코드도 받음", r.get("ok"), r)
    check(
        "공개 파일에 앵커·원문이 평문으로 없다",
        all(set(i) == {"item_id", "practice", "fields", "masked", "sealed"} for i in ITEMS),
    )
    p = {}
    if "OWNER" in codes:
        p = call(E, {"action": "progress", "code": codes["OWNER"]})
        check("관리자 진행 조회", p.get("ok") and p["R1"]["total"] == len(MAIN) and "R2" in p, p)
    if "R1" in codes:
        pr = call(E, {"action": "progress", "code": codes["R1"]})
        check(
            "평가자 진행은 자기 것만",
            pr.get("ok") and "R1" in pr and "R2" not in pr and pr["adjudication"]["total"] is None,
            pr,
        )
        s = call(E, {"action": "state", "code": codes["R1"]})
        check(
            "R1 상태 조회",
            s.get("ok")
            and isinstance(s.get("annotations"), dict)
            and isinstance(s.get("stage1"), dict),
            s,
        )
        check(
            "상태에 연습 해설 칸 (저장한 연습만)",
            isinstance(s.get("explain"), dict)
            and set(s.get("explain") or {}) <= set(PRACTICE)
            and set(s.get("explain") or {}) <= set(s.get("annotations") or {}),
            s.get("explain"),
        )
    if "ADJ" in codes and not p.get("adjudication", {}).get("available"):
        r = call(E, {"action": "adjudication", "code": codes["ADJ"]})
        check("평가 전 판정 목록은 닫힘", not r.get("ok"), r)
    if not args.full:
        print("\n읽기 점검 끝 (--full 없이 실행: 시트에 아무것도 쓰지 않았다)")
        return 1 if failures else 0

    print("\n[쓰기 — 시험용 저장소에서만]")
    rng = random.Random(7)
    iid = MAIN[0]
    r = call(
        E, {"action": "annotate", "code": codes["R1"], "item_id": iid, **annotation(rng, False)}
    )
    check("1 단계 전 채점 거부", not r.get("ok"), r)
    r = call(
        E,
        {
            "action": "reveal",
            "code": codes["R1"],
            "item_id": iid,
            "stage1": {**STAGE1, "age_guess": ""},
        },
    )
    check("빈 1 단계 거부 (열쇠 없음)", not r.get("ok") and "key" not in r, r)
    r = call(E, {"action": "reveal", "code": codes["R1"], "item_id": iid})
    check("1 단계 없이 열쇠 요청 거부", not r.get("ok") and "key" not in r, r)
    r = call(E, {"action": "reveal", "code": codes["ADJ"], "item_id": iid, "stage1": STAGE1})
    check("판정 코드로 열쇠 요청 거부", not r.get("ok") and "key" not in r, r)
    r = call(E, {"action": "reveal", "code": codes["R1"], "item_id": iid, "stage1": STAGE1})
    check(
        "1 단계 내면 열쇠",
        r.get("ok") and r.get("first") is True and len(base64.b64decode(r["key"])) == 32,
        r,
    )
    if AESGCM:
        inner = unseal(iid, r["key"])
        check(
            "열쇠로 봉인이 풀린다",
            {"anchors", "narrative", "genres_attended", "genres_not"} <= set(inner),
        )
    r2 = call(
        E,
        {
            "action": "reveal",
            "code": codes["R1"],
            "item_id": iid,
            "stage1": {**STAGE1, "age_guess": "70대 이상"},
        },
    )
    check(
        "1 단계는 처음 답이 유지 (고칠 수 없음)",
        r2.get("ok") and r2["first"] is False and r2["stage1"]["age_guess"] == "30대",
        r2,
    )
    st = call(E, {"action": "state", "code": codes["R2"]})
    check("R1 의 1 단계가 R2 에 안 보인다", iid not in st["stage1"], st.get("stage1"))
    a = annotation(rng, True)
    a["stage2"]["contra_quote"] = ""
    r = call(E, {"action": "annotate", "code": codes["R1"], "item_id": iid, **a})
    check("모순 있음인데 근거 없음 거부", not r.get("ok"), r)
    a = annotation(rng, False)
    a["stage2"]["contra_fields"] = "summary"
    r = call(E, {"action": "annotate", "code": codes["R1"], "item_id": iid, **a})
    check("모순 항목이 목록이 아니면 거부", not r.get("ok"), r)
    r = call(
        E, {"action": "annotate", "code": codes["ADJ"], "item_id": iid, **annotation(rng, False)}
    )
    check("판정 코드로 채점 거부", not r.get("ok"), r)
    r = call(
        E,
        {"action": "annotate", "code": codes["R1"], "item_id": "NOSUCH", **annotation(rng, False)},
    )
    check("없는 문항 거부", not r.get("ok"), r)
    a = annotation(rng, False)
    a["stage1"] = {
        **STAGE1,
        "age_guess": "70대 이상",
    }  # Stage 1 carried in the request must be ignored
    r = call(E, {"action": "annotate", "code": codes["R1"], "item_id": iid, **a})
    st = call(E, {"action": "state", "code": codes["R1"]})
    check(
        "채점 행의 1 단계는 서버가 기억한 답",
        r.get("ok") and st["annotations"][iid]["stage1"]["age_guess"] == "30대",
        st["annotations"].get(iid),
    )
    ex0 = call(E, {"action": "export", "code": codes["OWNER"]})
    check("채점 중에는 관리자도 내려받기 막힘", not ex0.get("ok"), ex0)
    pid = PRACTICE[0]
    call(E, {"action": "reveal", "code": codes["R1"], "item_id": pid, "stage1": STAGE1})
    r = call(
        E, {"action": "annotate", "code": codes["R1"], "item_id": pid, **annotation(rng, False)}
    )
    check("연습 문항 저장하면 해설 칸이 온다", r.get("ok") and "explain" in r, r)
    st1 = call(E, {"action": "state", "code": codes["R1"]})
    st2 = call(E, {"action": "state", "code": codes["R2"]})
    check(
        "저장한 연습 해설은 상태로 다시 받는다 (해설이 있을 때)",
        pid in st1["explain"] or r.get("explain") is None,
        st1.get("explain"),
    )
    check(
        "저장 안 한 연습 해설은 오지 않는다",
        all(k == pid for k in st1["explain"]) and pid not in st2["explain"],
        (list(st1["explain"]), list(st2["explain"])),
    )
    r = call(
        E, {"action": "annotate", "code": codes["R1"], "item_id": MAIN[0], **annotation(rng, False)}
    )
    check("본 문항 저장에는 해설이 없다", r.get("ok") and "explain" not in r, r)
    for _ in range(
        35
    ):  # There must be no global lock — a flood of bad codes must still let a valid code through
        call(E, {"action": "login", "code": "WRONGCODE000"})
    r = call(E, {"action": "login", "code": codes["R2"]})
    check("틀린 코드가 많이 와도 맞는 코드는 통과", r.get("ok"), r)

    for rater in ("R1", "R2"):
        for i, iid in enumerate(MAIN + PRACTICE):
            contra = (i % 11 == 0) if rater == "R1" else (i % 13 == 0)
            r = call(
                E, {"action": "reveal", "code": codes[rater], "item_id": iid, "stage1": STAGE1}
            )
            r = (
                r
                if not r.get("ok")
                else call(
                    E,
                    {
                        "action": "annotate",
                        "code": codes[rater],
                        "item_id": iid,
                        **annotation(rng, contra),
                    },
                )
            )
            if not r.get("ok"):
                check(f"{rater} 채점 {iid}", False, r)
                break
    s1 = call(E, {"action": "state", "code": codes["R1"]})
    check("R1 은 자기 채점만 받는다", all(v["rater"] == "R1" for v in s1["annotations"].values()))
    check(
        "R1 상태 문항 수",
        len(s1["annotations"]) == len(MAIN) + len(PRACTICE),
        len(s1["annotations"]),
    )
    e1 = call(E, {"action": "export", "code": codes["R1"]})
    check("R1 내보내기는 자기 것만", {a["rater"] for a in e1["annotations"]} == {"R1"})

    p = call(E, {"action": "progress", "code": codes["OWNER"]})
    check("둘 다 끝나면 판정 열림", p["adjudication"]["available"], p)
    pr = call(E, {"action": "progress", "code": codes["R2"]})
    check(
        "잠긴 뒤에도 평가자는 불일치 수를 모른다",
        pr["adjudication"]["total"] is None and "R1" not in pr,
        pr,
    )
    r = call(
        E, {"action": "annotate", "code": codes["R1"], "item_id": MAIN[1], **annotation(rng, False)}
    )
    check("둘 다 끝나면 독립 점수 잠김", not r.get("ok"), r)
    r = call(
        E,
        {
            "action": "annotate",
            "code": codes["R1"],
            "item_id": PRACTICE[0],
            **annotation(rng, False),
        },
    )
    check("연습 문항은 잠기지 않음", r.get("ok"), r)

    lst = call(E, {"action": "adjudication", "code": codes["ADJ"]})
    check("판정 목록", lst.get("ok") and len(lst["list"]) > 0, lst.get("error"))
    check("판정 목록에 문항 열쇠", all(len(base64.b64decode(x["key"])) == 32 for x in lst["list"]))
    d = lst["list"][0]
    base = {
        "action": "adjudicate",
        "code": codes["ADJ"],
        "item_id": d["item_id"],
        "consensus": d["R1"],
        "changed_by": "R2",
        "first_speaker": "R1",
    }
    r = call(E, {**base, "reason": ""})
    check("근거 없는 판정 거부", not r.get("ok"), r)
    r = call(E, {**base, "reason": "x", "first_speaker": "ADJ"})
    check("먼저 말한 사람이 R1·R2 가 아니면 거부", not r.get("ok"), r)
    r = call(E, {**base, "reason": "x", "changed_by": "someone"})
    check("누가 바꿨는지가 잘못되면 거부", not r.get("ok"), r)
    r = call(E, {**base, "reason": "요약 문장 기준"})
    check("판정 저장", r.get("ok"), r)
    r = call(E, {**base, "code": codes["R1"], "reason": "x"})
    check("평가자 코드로 판정 거부", not r.get("ok"), r)
    ex = call(E, {"action": "export", "code": codes["OWNER"]})
    check(
        "관리자 내보내기",
        ex.get("ok")
        and len(ex["annotations"]) == 2 * (len(MAIN) + len(PRACTICE))
        and len(ex["adjudication"]) == 1,
        (len(ex.get("annotations", [])), len(ex.get("adjudication", []))),
    )

    print(f"\n{'모두 통과' if not failures else f'실패 {len(failures)}: {failures}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
