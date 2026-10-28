import { useEffect, useState } from "react";
import { api, items as loadItems, openItem } from "../api";
import { SCALES } from "../constants";
import type { Disagreement, Rater, Stage2 } from "../types";
import { AnchorCard } from "./AnchorCard";
import { Guide, guideSections } from "./Guide";
import { Choices } from "./Likert";
import { Narrative, scrollPanesTop } from "./Narrative";

type Consensus = Omit<Stage2, "secs">;

// The consensus fields start blank — starting from one rater's judgment pulls the decision that way (the registered protocol v2 §6.5).
const BLANK: Consensus = { contradiction: "", contra_fields: [], contra_quote: "", grounded: 0, coherent: 0, plausible: 0, fluent: 0, note: "" };

/** Whether the two raters disagreed on the contradiction (otherwise the item differs only on the groundedness score). */
const contraSplit = (d: Pick<Disagreement, "R1" | "R2">) => d.R1.contradiction !== d.R2.contradiction;

// Short phrases appended on click so the adjudication reason can be written quickly (they can be edited)
const reasonChips = (n: Record<Rater, string>) => [
  `${n.R1}이 그 문장을 놓침`,
  `${n.R2}이 그 문장을 놓침`,
  "어긋나는 앵커를 함께 짚을 수 없어 모순 아님",
  "서사 문장과 앵커가 둘 다 참일 수 없어 모순",
  "빈도·가끔 표현이라 모순 아님",
];
const DEFAULT_NAME: Record<Rater, string> = { R1: "평가자 1", R2: "평가자 2" };
// The section both raters read together before adjudicating — the matching section of the registered guide **verbatim** (no new rules).
// Only the "1. 모순이 있는가" section, up to the genre table (the genre table is rarely used in adjudication).
const ADJ_GUIDE = guideSections(["1. 모순이 있는가"]).split("**장르 대응**")[0].trimEnd();
const INTRO_KEY = "anno.adj.intro.v1";
const introSeen = () => {
  try {
    return localStorage.getItem(INTRO_KEY) === "1";
  } catch {
    return false;
  }
};
const MIN_PER_ITEM = 2; // Rough minutes left (when only the contradiction is discussed)

export function Adjudicate({ onSaved }: { onSaved: () => void }) {
  const [list, setList] = useState<Disagreement[] | null>(null);
  const [i, setI] = useState(0);
  const [c, setC] = useState<Consensus>(BLANK);
  const [changedBy, setChangedBy] = useState("");
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [NAME, setNames] = useState<Record<Rater, string>>(DEFAULT_NAME);
  const [showGuide, setShowGuide] = useState(!introSeen());

  const reload = () =>
    Promise.all([api.adjudication(), loadItems()])
      // The adjudication list includes a per-item key (only after both raters have finished). Rater names exist only on the server.
      .then(([res, cat]) => {
        // If a name is present, show it as "name (rater 1)" — the stored values (R1/R2) stay as they are
        if (res.names) setNames({ R1: res.names.R1 ? `${res.names.R1}(평가자 1)` : "평가자 1", R2: res.names.R2 ? `${res.names.R2}(평가자 2)` : "평가자 2" });
        return [res.list, cat] as const;
      })
      .then(([raw, cat]) => Promise.all(raw.map(async (d) => ({ ...d, item: await openItem(cat.get(d.item_id)!, d.key) }))))
      .then(
      (raw: Disagreement[]) => {
        // Put items that disagree on the contradiction first — the primary claim (P1) depends only on this adjudication. Items that differ by 2+ only on groundedness go last.
        // Stable sort, so the order inside each group stays as the server sent it.
        // Only contradiction disagreements are adjudicated. Items that differ only on groundedness are dropped: their consensus score is unused in any analysis (ADR-039; the count is recorded in the report).
        const full = raw.filter(contraSplit);
        setList(full);
        const first = full.findIndex((d) => !d.decided);
        setI((cur) => (full[cur] && !full[cur].decided ? cur : first === -1 ? 0 : first));
      },
      (e) => setError(String(e.message ?? e)),
    );

  useEffect(() => {
    reload();
  }, []);

  const d = list?.[i];
  // The first speaker alternates — so one person does not lead every adjudication (the registered protocol).
  const first: Rater = i % 2 === 0 ? "R1" : "R2";

  useEffect(() => {
    if (!d) return;
    if (d.decided) {
      setC({ ...d.decided.consensus });
      setChangedBy(d.decided.changed_by);
      setReason(d.decided.reason);
    } else {
      setC(BLANK);
      setChangedBy("");
      setReason("");
    }
    setError(null);
    scrollPanesTop();
  }, [i, d]);

  if (error && !list) return <div className="notice error">{error}</div>;
  if (!list) return <div className="card">불러오는 중…</div>;
  if (!list.length) return <div className="notice ok">두 평가자의 판단이 크게 다른 문항이 없습니다. 판정할 것이 없습니다.</div>;
  if (!d) return null;

  if (showGuide) {
    const first = !introSeen();
    return (
      <div className="work">
        <div className="card adj-intro">
          <div className="notice warn">
            {first ? (
              <>
                <b>판정을 시작하기 전에 두 분이 함께 아래 지침을 다시 읽어 주세요.</b> 처음 받은 지침서의 해당 부분을 그대로 옮긴 것입니다.
                <br />
                <b>모순으로 합의하려면</b> 서사의 문장과 그것이 어긋나는 앵커 항목을 <b>함께 짚을 수 있어야</b> 합니다. 짚지 못하면 모순이 아닙니다.
              </>
            ) : (
              <>지침서의 판정 관련 부분입니다 (원문 그대로).</>
            )}
          </div>
          <Guide text={ADJ_GUIDE} />
          <div className="row" style={{ justifyContent: "flex-end", padding: "8px 0 4px" }}>
            <button
              className="btn primary"
              onClick={() => {
                try {
                  localStorage.setItem(INTRO_KEY, "1");
                } catch {
                  /* Continue even if saving is blocked */
                }
                setShowGuide(false);
              }}
            >
              {first ? "함께 읽었습니다 — 판정 시작" : "판정으로 돌아가기"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const save = async () => {
    if (!c.contradiction) return setError("합의한 모순 여부를 고르세요");
    if (c.contradiction === "yes" && !c.contra_fields.length) return setError("합의한 모순 문단을 왼쪽 서사에서 누르세요");
    if (c.contradiction === "yes" && !c.contra_quote.trim()) return setError("합의한 모순의 근거 문장을 적으세요");
    if (!changedBy) return setError("누가 의견을 바꿨는지 고르세요");
    if (!reason.trim()) return setError("판정 근거를 적으세요");
    setSaving(true);
    try {
      // The four score fields are not adjudicated — analysis uses only the independent scores. To keep the save format, record the lower of the two raters' scores (the stricter one).
      const scores = Object.fromEntries(SCALES.map((sc) => [sc.key, Math.min(d.R1[sc.key], d.R2[sc.key])]));
      await api.adjudicate({ item_id: d.item_id, consensus: { ...c, ...scores }, changed_by: changedBy, reason, first_speaker: first });
      onSaved();
      await reload();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setSaving(false);
    }
  };

  const nContra = list.filter(contraSplit).length;
  const contraLeft = list.filter((x) => contraSplit(x) && !x.decided).length;

  const fieldLabel = (k: string) => d.item.fields.find((f) => f.key === k)?.label ?? k;
  const contraText = (x: Stage2) => (x.contradiction === "yes" ? `있음 · ${x.contra_fields.map(fieldLabel).join(", ")}` : "없음");
  const differs = (k: "contradiction" | (typeof SCALES)[number]["key"]) => d.R1[k] !== d.R2[k];

  return (
    <div className="work">
      <div className="workhead">
        <span className="badge step">합의 판정</span>
        <span className="badge practice">모순 여부가 갈림</span>
        <span className="muted">
          {i + 1}번째 / {nContra}개 · 끝낸 판정 {nContra - contraLeft}/{nContra}
          {contraLeft > 0 && <> · 남은 {contraLeft}개 · 약 {contraLeft * MIN_PER_ITEM}분</>} · {d.item_id}
        </span>
        <span style={{ flex: 1 }} />
        <button className="btn small" onClick={() => setShowGuide(true)}>판정 지침</button>
        <button className="btn small" disabled={i === 0} onClick={() => setI(i - 1)}>← 이전</button>
        <button className="btn small" disabled={i >= list.length - 1} onClick={() => setI(i + 1)}>다음 →</button>
      </div>

      <div className="workspace">
        <section className="pane pane-read">
          <div className="card">
            <div className="pane-title">
              서사
              {c.contradiction === "yes" ? <span className="pick-hint">합의한 모순 문단을 눌러 표시하세요</span> : <span className="muted">두 평가자의 근거 문장을 함께 찾아 보세요</span>}
            </div>
            <Narrative
              fields={d.item.fields}
              text={d.item.narrative}
              picked={c.contra_fields}
              marks={Object.fromEntries(
                d.item.fields.map((f) => [f.key, (["R1", "R2"] as const).filter((r) => d[r].contradiction === "yes" && d[r].contra_fields.includes(f.key)).map((r) => NAME[r])]),
              )}
              quotes={[d.R1.contra_quote, d.R2.contra_quote].filter(Boolean) as string[]}
              onToggle={c.contradiction === "yes" ? (k) => {
                const has = c.contra_fields.includes(k);
                setC({ ...c, contra_fields: has ? c.contra_fields.filter((x) => x !== k) : [...c.contra_fields, k] });
              } : undefined}
            />
          </div>
        </section>

        <section className="pane pane-form">
          <div className="notice warn">
            <b>{NAME[first]}</b>이 먼저 설명합니다. <b>모순 여부만</b> 함께 정합니다.
            합의가 안 되면 더 엄격한 쪽(모순 있음)으로 입력하고 「합의 불가」.
          </div>
          {nContra > 0 && contraLeft === 0 && (
            <div className="notice ok">
              <b>판정 {nContra}개를 모두 끝냈습니다. 수고하셨습니다.</b>
            </div>
          )}
          {error && <div className="notice error">{error}</div>}
          <div className="card qcard rater-says">
            {(["R1", "R2"] as const).map((r) => (
              <p key={r} className="quote">
                <b>{NAME[r]}</b> 모순 {contraText(d[r])}
                {d[r].contra_quote && <> · 근거 <span className="qtext">“{d[r].contra_quote}”</span></>}
                {d[r].note && <span className="muted"> · 메모 {d[r].note}</span>}
              </p>
            ))}
            <p className="muted small">근거 문장은 왼쪽 서사에 노란색으로, 지목한 문단에는 평가자 표시가 붙어 있습니다.</p>
          </div>
          <AnchorCard item={d.item} />
          <div className="card qcard">
            <table className="cmp">
              <thead>
                <tr>
                  <th />
                  <th>{NAME.R1}</th>
                  <th>{NAME.R2}</th>
                  <th>합의</th>
                </tr>
              </thead>
              <tbody>
                <tr className={differs("contradiction") ? "diff" : ""}>
                  <th>모순</th>
                  <td>{contraText(d.R1)}</td>
                  <td>{contraText(d.R2)}</td>
                  <td>
                    {(
                      <Choices
                        options={["없음", "있음"]}
                        value={c.contradiction === "yes" ? "있음" : c.contradiction === "no" ? "없음" : ""}
                        onChange={(v) => setC({ ...c, contradiction: v === "있음" ? "yes" : "no", ...(v === "없음" ? { contra_fields: [], contra_quote: "" } : {}) })}
                      />
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
            {(
              <div className="adj-form">
                {c.contradiction === "yes" && (
                  <textarea className="quote-input" rows={1} placeholder="합의한 모순의 근거 문장" value={c.contra_quote} onChange={(e) => setC({ ...c, contra_quote: e.target.value })} />
                )}
                <div className="row">
                  <b style={{ fontSize: 14 }}>누가 의견을 바꿨나</b>
                  <Choices
                    options={[NAME.R1, NAME.R2, "둘 다", "합의 불가"]}
                    value={{ R1: NAME.R1, R2: NAME.R2, both: "둘 다", none: "합의 불가" }[changedBy] ?? ""}
                    onChange={(v) => setChangedBy({ [NAME.R1]: "R1", [NAME.R2]: "R2", "둘 다": "both", "합의 불가": "none" }[v] ?? "")}
                  />
                </div>
                {changedBy === "none" && c.contradiction === "no" && (
                  <div className="notice warn">합의 불가면 지침대로 더 엄격한 쪽, 즉 모순 「있음」으로 입력합니다.</div>
                )}
                <div className="rchips">
                  {reasonChips(NAME).map((t) => (
                    <button key={t} className="rchip" onClick={() => setReason(reason.trim() ? `${reason.trim()} · ${t}` : t)}>
                      {t}
                    </button>
                  ))}
                </div>
                <textarea className="quote-input" rows={1} placeholder="판정 근거 — 위 문구를 누르거나 한 줄로 적기" value={reason} onChange={(e) => setReason(e.target.value)} />
              </div>
            )}
          </div>
          <div className="pane-actions">
            <button className="btn primary" onClick={save} disabled={saving}>
              {saving ? "저장 중…" : d.decided ? "판정 고쳐 저장" : "판정 저장"}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
