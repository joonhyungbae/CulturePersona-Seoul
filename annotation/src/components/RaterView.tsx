import { useCallback, useEffect, useRef, useState } from "react";
import { api, items as loadItems, openItem, STALE_HELP } from "../api";
import { AGE_OPTIONS, ATTEND_OPTIONS, SCALES } from "../constants";
import type { Annotation, Item, PracticeExplain, PublicItem, Rater, Stage1, Stage2 } from "../types";
import { AnchorCard } from "./AnchorCard";
import { DEV_GOTO, DEV_ITEM } from "./DevBar";
import { Choices, Likert, QRow } from "./Likert";
import { Narrative, scrollPanesTop } from "./Narrative";

// Blinding is enforced by the server: the public file has only the masked narrative in plaintext; anchors and the original are sealed.
// The item key arrives only after stage 1 is submitted (reveal), and the server keeps only the first stage-1 answer.

// Whether they read the practice explanation through and pressed confirm — stored only in this browser (if lost, the explanation is shown once more).
const seenKey = (rater: string) => `anno.explainSeen.${rater}`;
function readSeen(rater: string): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(seenKey(rater)) ?? "[]") as string[]);
  } catch {
    return new Set();
  }
}
function markSeen(rater: string, id: string): void {
  try {
    const s = readSeen(rater);
    s.add(id);
    localStorage.setItem(seenKey(rater), JSON.stringify([...s]));
  } catch {
    /* If storage is blocked, the explanation just appears once more next time */
  }
}

const EMPTY1: Stage1 = { age_guess: "", attend_guess: "", secs: 0 };
const EMPTY2: Stage2 = {
  contradiction: "",
  contra_fields: [],
  contra_quote: "",
  grounded: 0,
  coherent: 0,
  plausible: 0,
  fluent: 0,
  note: "",
  secs: 0,
};

/** Same checks as the server — block once more before saving. */
function missing2(s: Stage2): string | null {
  if (!s.contradiction) return "모순 여부를 고르세요";
  if (s.contradiction === "yes" && !s.contra_fields.length) return "모순이 있는 항목을 하나 이상 고르세요 (서사 칸을 누르면 됩니다)";
  if (s.contradiction === "yes" && !s.contra_quote.trim()) return "모순의 근거 문장을 그대로 옮겨 적으세요";
  for (const sc of SCALES) if (!s[sc.key]) return `${sc.label} 점수를 고르세요`;
  return null;
}

/** A combination that disagrees with the guide — not blocked, only confirmed once. */
function softWarning(s: Stage2): string | null {
  if (s.contradiction === "yes" && s.grounded >= 4)
    return "모순이 있다고 했는데 앵커 부합을 4점 이상으로 주셨습니다. 이대로 저장할까요?";
  if (s.contradiction === "no" && s.grounded <= 2)
    return "모순이 없다고 했는데 앵커 부합을 2점 이하로 주셨습니다. 이대로 저장할까요?";
  return null;
}

export function RaterView({ rater, onSaved }: { rater: Rater; onSaved: () => void }) {
  const [order, setOrder] = useState<string[]>([]);
  const [catalog, setCatalog] = useState<Map<string, PublicItem>>(new Map());
  const [mineAll, setMineAll] = useState<Record<string, Annotation>>({});
  const [stage1All, setStage1All] = useState<Record<string, Stage1>>({});
  const [practice, setPractice] = useState<Set<string>>(new Set());
  const [idx, setIdx] = useState(0);
  const [pub, setPub] = useState<PublicItem | null>(null);
  const [opened, setOpened] = useState<Item | null>(null);
  const [stage, setStage] = useState<1 | 2>(1);
  const [s1, setS1] = useState<Stage1>(EMPTY1);
  const [s2, setS2] = useState<Stage2>(EMPTY2);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [memoOpen, setMemoOpen] = useState(false);
  const [explain, setExplain] = useState<PracticeExplain | null>(null);
  const [stale, setStale] = useState(false);
  const [jump, setJump] = useState("");
  // Explanations for saved practice items (the server sends only those already saved). Used to review and to resume.
  const [explainAll, setExplainAll] = useState<Record<string, PracticeExplain>>({});
  const explainRef = useRef(explainAll);
  explainRef.current = explainAll;
  const keys = useRef(new Map<string, string>());
  // mineAll changes on every save — if load depended on it, the same item would reload right after save and wipe the explanation and the input.
  const mineRef = useRef(mineAll);
  mineRef.current = mineAll;
  const t0 = useRef(Date.now());
  const done = new Set(order.filter((id) => mineAll[id]));

  // On start: load the queue and go to the first unfinished item (resume).
  useEffect(() => {
    // Items and order come from static files; my scores and stage-1 answers come from the server (spreadsheet). The other rater's scores are not sent.
    Promise.all([api.order(rater), loadItems(), api.state()]).then(([ord, cat, st]) => {
      setCatalog(cat);
      setMineAll(st.annotations);
      setStage1All(st.stage1);
      setOrder(ord);
      setPractice(new Set(ord.filter((id) => cat.get(id)?.practice)));
      setExplainAll(st.explain ?? {});
      // If they saved a practice item and left before confirming the explanation, return to that item and show the explanation first.
      const seen = readSeen(rater);
      const unread = ord.findIndex((id) => st.explain?.[id] && !seen.has(id));
      const first = ord.findIndex((id) => !st.annotations[id]);
      // The confirm-explanation flag lives only in this browser — switching browser or device clears it and opens the first practice item,
      // and the rater had to press Next over a hundred times. If any main item is already saved, go straight to the first unfinished item.
      const startedMain = ord.some((id) => !cat.get(id)?.practice && st.annotations[id]);
      const resume = first === -1 ? ord.length - 1 : first;
      setIdx(!startedMain && unread !== -1 && (first === -1 || unread < first) ? unread : resume);
    }, (e) => setError(String(e.message ?? e)));
  }, [rater]);

  // Check whether the open screen is stale: if my order file or the script bundle differs from the deploy, suggest a refresh.
  // Check when the window regains focus and every 5 minutes. If the check fails, skip it quietly (do not block scoring).
  useEffect(() => {
    if (!order.length) return;
    const mineBundle = [...document.querySelectorAll("script[src]")]
      .map((s) => (s as HTMLScriptElement).src.match(/assets\/index-[A-Za-z0-9_-]+\.js/)?.[0])
      .find(Boolean);
    let off = false;
    const check = async () => {
      try {
        const [live, bundle] = await Promise.all([api.order(rater), api.liveBundle()]);
        if (off) return;
        const orderChanged = live.length !== order.length || live.some((id, i) => id !== order[i]);
        const bundleChanged = Boolean(mineBundle && bundle && bundle !== mineBundle);
        if (orderChanged || bundleChanged) setStale(true);
      } catch {
        /* If the connection dropped, check again next time */
      }
    };
    const onFocus = () => document.visibilityState === "visible" && check();
    const timer = window.setInterval(check, 5 * 60 * 1000);
    document.addEventListener("visibilitychange", onFocus);
    return () => {
      off = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onFocus);
    };
  }, [order, rater]);

  /** Get the key from the server (without stage1 if stage 1 was already submitted) and open the item. */
  const reveal = useCallback(async (id: string, answer?: Stage1) => {
    const r = await api.reveal(id, answer);
    keys.current.set(id, r.key);
    setStage1All((m) => ({ ...m, [id]: r.stage1 }));
    return r.stage1;
  }, []);

  const load = useCallback(async (i: number) => {
    const id = order[i];
    const p = id ? catalog.get(id) : undefined;
    if (!p) return;
    setError(null);
    setPub(p);
    setOpened(null);
    setMemoOpen(false);
    setExplain(null);
    const mine = mineRef.current[id] ?? null;
    const given = stage1All[id] ?? null;
    if (mine || given) {
      // Stage 1 already submitted: it cannot be changed, so go straight to stage 2.
      if (!keys.current.has(id)) await reveal(id);
      setOpened(await openItem(p, keys.current.get(id)!));
      setS1(given ?? mine!.stage1);
      setS2(mine ? { ...mine.stage2 } : EMPTY2);
      setStage(2);
      // Saved practice item: show the explanation immediately if not yet confirmed (if confirmed, the show-again button).
      const ex = mine ? explainRef.current[id] : undefined;
      const startedMain = order.some((x) => !catalog.get(x)?.practice && mineRef.current[x]);
      if (ex && !startedMain && !readSeen(rater).has(id)) setExplain(ex);
    } else {
      setS1(EMPTY1);
      setS2(EMPTY2);
      setStage(1);
    }
    t0.current = Date.now();
    scrollPanesTop();
    // stage1All is omitted from the dependencies because reveal updates it (avoid reloading on every change).
  }, [order, catalog, reveal, rater]);

  useEffect(() => {
    if (order.length) load(idx).catch((e) => setError(String(e.message ?? e)));
  }, [idx, order, load]);

  // Dev mode: report the current item to the dev bar and accept its jump requests (omitted from the production build).
  useEffect(() => {
    if (!import.meta.env.DEV || !pub) return;
    window.dispatchEvent(new CustomEvent(DEV_ITEM, { detail: { idx, total: order.length, item_id: pub.item_id, stage } }));
  }, [idx, pub, stage, order.length]);
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const on = (e: Event) => {
      const d = (e as CustomEvent<{ idx?: number; item_id?: string }>).detail;
      const i = d.idx ?? order.indexOf(d.item_id ?? "");
      if (i >= 0 && i < order.length) setIdx(i);
    };
    window.addEventListener(DEV_GOTO, on);
    return () => window.removeEventListener(DEV_GOTO, on);
  }, [order]);

  const toStage2 = async () => {
    if (!s1.age_guess || !s1.attend_guess) {
      setError("두 가지를 모두 고르세요. 모르면 가장 그럴듯한 것을 고르면 됩니다.");
      return;
    }
    if (!pub) return;
    setSaving(true);
    try {
      const fixed = await reveal(pub.item_id, { ...s1, secs: Math.round((Date.now() - t0.current) / 1000) });
      setOpened(await openItem(pub, keys.current.get(pub.item_id)!));
      setS1(fixed); // The answer the server stored (that answer if another window submitted first)
      setStage(2);
      setError(null);
      t0.current = Date.now();
      scrollPanesTop();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setSaving(false);
    }
  };

  const advance = (d: Set<string>) => {
    const next = order.findIndex((id, i) => i > idx && !d.has(id));
    if (next !== -1) setIdx(next);
  };

  const save = async () => {
    const m = missing2(s2);
    if (m) return setError(m);
    const w = softWarning(s2);
    if (w && !window.confirm(w)) return;
    const id = order[idx];
    const wasDone = done.has(id);
    setSaving(true);
    try {
      const secs = s2.secs || Math.round((Date.now() - t0.current) / 1000);
      const r = await api.annotate(id, { ...s2, secs });
      const nextMine = { ...mineAll, [id]: { rater, item_id: id, stage1: s1, stage2: { ...s2, secs }, saved_at: new Date().toISOString() } };
      setMineAll(nextMine);
      onSaved();
      // For a practice item, show the written explanation first; move on when the rater confirms it.
      if (r.explain) {
        const ex = r.explain;
        setExplainAll((m) => ({ ...m, [id]: ex }));
        setExplain(ex);
        scrollPanesTop();
      }
      else if (!wasDone) advance(new Set(order.filter((x) => nextMine[x])));
    } catch (e) {
      // The response may have been lost while the save succeeded — sync to the server state.
      const msg = String((e as Error).message);
      try {
        const st = await api.state();
        if (!wasDone && st.annotations[id]) {
          setMineAll(st.annotations);
          onSaved();
          advance(new Set(order.filter((x) => st.annotations[x])));
          return;
        }
      } catch {
        /* If the server is unreachable, show the error below */
      }
      setError(msg);
    } finally {
      setSaving(false);
    }
  };

  const toggleField = (key: string) => {
    if (s2.contradiction !== "yes") return;
    const has = s2.contra_fields.includes(key);
    setS2({ ...s2, contra_fields: has ? s2.contra_fields.filter((k) => k !== key) : [...s2.contra_fields, key] });
  };

  if (!order.length) return <div className="card">{error ?? "불러오는 중…"}</div>;
  const allDone = order.every((id) => done.has(id));
  // First unfinished item (where to resume). Navigation is limited to saved items and this point — do not skip ahead to unsolved items.
  const firstOpen = order.findIndex((id) => !done.has(id));
  const resumeIdx = firstOpen === -1 ? order.length - 1 : firstOpen;
  // Cannot leave an item whose stage 2 is open and unsaved (same rule as the previous button).
  const locked = stage === 2 && !done.has(order[idx]);
  const canGo = (i: number) => i >= 0 && i < order.length && (done.has(order[i]) || i === resumeIdx);
  const goJump = () => {
    const n = Number(jump);
    if (!Number.isInteger(n) || !canGo(n - 1)) {
      setError(`1 – ${resumeIdx + 1} 사이의 번호를 넣으세요. 아직 풀지 않은 문항으로는 건너뛸 수 없습니다.`);
      return;
    }
    setJump("");
    setIdx(n - 1);
  };
  const isStale = stale || error === STALE_HELP;
  const item = pub;
  if (!item || (stage === 2 && !opened))
    return (
      <div className="card">
        {error ?? "불러오는 중…"}
        {error === STALE_HELP && (
          <p>
            <button className="btn primary" onClick={() => window.location.reload()}>지금 새로 고침</button>
          </p>
        )}
      </div>
    );

  // Notify once when practice ends and the first main item opens.
  const firstMain = !item.practice && idx > 0 && practice.has(order[idx - 1]);
  const notices = (
    <>
      {firstMain && <div className="notice warn">연습이 끝났습니다. 지침서를 한 번 더 읽고 본 문항을 시작하세요. 여기부터는 분석에 쓰입니다.</div>}
      {isStale && (
        <div className="notice warn stale">
          검수 화면이 새 버전으로 바뀌었습니다. 새로 고침하면 이어서 할 문항이 바로 열립니다. 저장한 채점은 그대로입니다.
          <button className="btn small primary" onClick={() => window.location.reload()}>지금 새로 고침</button>
        </div>
      )}
      {allDone && <div className="notice ok">모든 문항을 끝냈습니다. 고치고 싶은 문항은 번호를 넣어 이동하거나 이전 버튼으로 돌아가 고칠 수 있습니다.</div>}
      {error && <div className="notice error">{error}</div>}
    </>
  );
  const fieldLabel = (k: string) => item.fields.find((f) => f.key === k)?.label ?? k;

  // Split the screen vertically: narrative (the text to read) on the left, anchors (facts to compare) and the rating on the right. Packed to fit one screen.
  return (
    <div className="work">
      <div className="workhead">
        <span className="badge step">{stage === 1 ? "1단계 · 가린 글로 추측" : "2단계 · 앵커와 대조해 채점"}</span>
        {item.practice && <span className="badge practice">연습 · 분석에 쓰지 않음</span>}
        <span className="muted">
          문항 {idx + 1} / {order.length} · {item.item_id}
        </span>
        <span style={{ flex: 1 }} />
        {idx !== resumeIdx && !allDone && (
          <button
            className="btn small primary"
            disabled={locked}
            title={locked ? "이 문항을 저장한 뒤 이동할 수 있습니다" : `아직 풀지 않은 첫 문항(${resumeIdx + 1}번)으로 갑니다`}
            onClick={() => setIdx(resumeIdx)}
          >
            이어서 하기 ({resumeIdx + 1}번) ⇥
          </button>
        )}
        <form
          className="jump"
          onSubmit={(e) => {
            e.preventDefault();
            if (!locked) goJump();
          }}
        >
          <input
            type="text"
            inputMode="numeric"
            aria-label="이동할 문항 번호"
            placeholder="번호"
            value={jump}
            disabled={locked}
            onChange={(e) => setJump(e.target.value.replace(/[^0-9]/g, ""))}
          />
          <button className="btn small" type="submit" disabled={locked || !jump} title={`저장한 문항과 이어서 할 문항(1 – ${resumeIdx + 1}번)으로 갈 수 있습니다`}>
            이동
          </button>
        </form>
        <button
          className="btn small"
          disabled={idx === 0 || (stage === 2 && !done.has(order[idx]))}
          title={stage === 2 && !done.has(order[idx]) ? "이 문항을 저장한 뒤 이동할 수 있습니다" : undefined}
          onClick={() => setIdx(idx - 1)}
        >
          ← 이전
        </button>
        <button className="btn small" disabled={idx >= order.length - 1 || !done.has(order[idx])} onClick={() => setIdx(idx + 1)}>
          다음 →
        </button>
      </div>

      <div className="workspace">
        <section className="pane pane-read">
          <div className="card">
            {stage === 1 ? (
              <>
                <div className="pane-title">서사 <span className="muted">정답을 드러내는 단어는 ○○로 가렸습니다</span></div>
                <Narrative fields={item.fields} text={item.masked} masked />
              </>
            ) : (
              <>
                <div className="pane-title">
                  서사
                  {s2.contradiction === "yes" ? (
                    <span className="pick-hint">모순이 있는 문단을 눌러 표시하세요</span>
                  ) : (
                    <span className="muted">앵커(오른쪽)와 어긋나는 곳이 있는지 읽어 보세요</span>
                  )}
                </div>
                <Narrative fields={item.fields} text={opened!.narrative} picked={s2.contra_fields} onToggle={s2.contradiction === "yes" ? toggleField : undefined} />
              </>
            )}
          </div>
        </section>

        <section className="pane pane-form">
          {notices}
          {explain && (
            <div className="card explain">
              <div className="explain-head">
                연습 정답 · 모순 <b>{explain.contradiction === "yes" ? "있음" : "없음"}</b>
                {explain.fields?.length ? <> · {explain.fields.map(fieldLabel).join(", ")}</> : null}
                {explain.grounded ? <> · 앵커 부합 {explain.grounded}</> : null}
                <span className={`badge ${explain.contradiction === s2.contradiction ? "ok" : "warn"}`}>
                  {explain.contradiction === s2.contradiction ? "내 판단과 같음" : "내 판단과 다름"}
                </span>
              </div>
              <p>{explain.text}</p>
              <button
                className="btn primary"
                onClick={() => {
                  markSeen(rater, order[idx]);
                  setExplain(null);
                  advance(new Set(order.filter((x) => mineAll[x])));
                }}
              >
                해설 확인 · 다음 문항 →
              </button>
            </div>
          )}
          {stage === 1 ? (
            <div className="card qcard">
              <QRow label="연령대" hint="글만 보고 추측">
                <Choices options={AGE_OPTIONS} value={s1.age_guess} onChange={(v) => setS1({ ...s1, age_guess: v })} />
              </QRow>
              <QRow label="지난 1년 현장 관람" hint="영화·공연·전시·축제 등 (영화관 포함, 온라인 시청 제외)">
                <Choices options={ATTEND_OPTIONS} value={s1.attend_guess} onChange={(v) => setS1({ ...s1, attend_guess: v })} />
              </QRow>
              <p className="muted" style={{ margin: "10px 0 0" }}>모르겠으면 가장 그럴듯한 것을 고르세요. 다음으로 넘어가면 앵커가 열리고, 1단계는 고칠 수 없습니다.</p>
            </div>
          ) : (
            <>
              <AnchorCard item={opened!} />
              <div className="card qcard">
                <QRow n={1} label="앵커와 모순" hint="빠진 내용·과거·관심은 아님">
                  <div className="row tight">
                    <Choices
                      options={["없음", "있음"]}
                      value={s2.contradiction === "yes" ? "있음" : s2.contradiction === "no" ? "없음" : ""}
                      onChange={(v) =>
                        setS2({
                          ...s2,
                          contradiction: v === "있음" ? "yes" : "no",
                          ...(v === "없음" ? { contra_fields: [], contra_quote: "" } : {}),
                        })
                      }
                    />
                    {s2.contradiction === "yes" && (
                      <span className="muted">
                        {s2.contra_fields.length ? <b className="bad">{s2.contra_fields.map(fieldLabel).join(", ")}</b> : "← 왼쪽에서 모순 문단을 누르세요"}
                      </span>
                    )}
                  </div>
                  {s2.contradiction === "yes" && (
                    <textarea
                      className="quote-input"
                      rows={1}
                      placeholder="근거 문장을 그대로 옮겨 적으세요"
                      value={s2.contra_quote}
                      onChange={(e) => setS2({ ...s2, contra_quote: e.target.value })}
                    />
                  )}
                </QRow>
                {SCALES.map((sc, i) => (
                  <QRow key={sc.key} n={i + 2} label={sc.label} hint={sc.hint}>
                    <Likert levels={sc.levels} value={s2[sc.key]} onChange={(v) => setS2({ ...s2, [sc.key]: v })} />
                  </QRow>
                ))}
                {memoOpen || s2.note ? (
                  <QRow label="메모" hint="선택">
                    <textarea rows={2} value={s2.note} onChange={(e) => setS2({ ...s2, note: e.target.value })} placeholder="지침서로 판단하기 어려웠던 점" />
                  </QRow>
                ) : (
                  <button className="linkbtn" onClick={() => setMemoOpen(true)}>+ 메모 남기기 (선택)</button>
                )}
              </div>
            </>
          )}
          <div className="pane-actions" hidden={Boolean(explain)}>
            {stage === 1 ? (
              <button className="btn primary" onClick={toStage2} disabled={saving}>
                {saving ? "보내는 중…" : "다음: 앵커 열고 채점하기 →"}
              </button>
            ) : (
              <>
                {explainAll[order[idx]] && (
                  <button className="btn" onClick={() => { setExplain(explainAll[order[idx]]); scrollPanesTop(); }}>
                    연습 해설 다시 보기
                  </button>
                )}
                <button className="btn primary" onClick={save} disabled={saving}>
                  {saving ? "저장 중…" : done.has(order[idx]) ? "고친 내용 저장" : "저장하고 다음 문항 →"}
                </button>
              </>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
