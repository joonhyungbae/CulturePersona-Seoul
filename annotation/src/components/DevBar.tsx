import { useEffect, useState } from "react";
import { api } from "../api";
import type { Who } from "../types";

/** In dev mode, the current item reported by RaterView and the jump request sent by DevBar. */
export const DEV_ITEM = "anno-dev-item";
export const DEV_GOTO = "anno-dev-goto";
export type DevItem = { idx: number; total: number; item_id: string; stage: 1 | 2 };

const ROLES: Who[] = ["R1", "R2", "ADJ", "OWNER"];

/**
 * Dev bar for checking UX changes (dev_ux.sh). Visible only on the local sandbox server —
 * omitted entirely from the production build because import.meta.env.DEV is false.
 */
export function DevBar({ who }: { who: Who }) {
  const [codes, setCodes] = useState<Record<Who, string> | null>(null);
  const [cur, setCur] = useState<DevItem | null>(null);
  const [jump, setJump] = useState("");
  const [msg, setMsg] = useState("");
  const [open, setOpen] = useState(true);

  useEffect(() => {
    api.dev.info().then((r) => setCodes(r.codes), () => setCodes(null));
    const on = (e: Event) => setCur((e as CustomEvent<DevItem>).detail);
    window.addEventListener(DEV_ITEM, on);
    return () => window.removeEventListener(DEV_ITEM, on);
  }, []);

  if (!codes) return null; // Hide when the server is not in dev mode

  const run = async (label: string, fn: () => Promise<unknown>, reload = true) => {
    setMsg(`${label}…`);
    try {
      await fn();
      if (reload) location.reload();
      else setMsg(`${label} 끝`);
    } catch (e) {
      setMsg(String((e as Error).message));
    }
  };
  const goto = () => {
    const n = Number(jump);
    const detail = Number.isInteger(n) && n > 0 ? { idx: n - 1 } : { item_id: jump.trim().toUpperCase() };
    window.dispatchEvent(new CustomEvent(DEV_GOTO, { detail }));
  };
  const isRater = who === "R1" || who === "R2";

  if (!open)
    return (
      <button className="devbar-tab" onClick={() => setOpen(true)}>
        DEV
      </button>
    );
  return (
    <div className="devbar">
      <b>DEV</b>
      <span className="muted">로컬 모래상자 · 실제 시트와 무관</span>
      <span className="sep" />
      {ROLES.map((r) => (
        <button key={r} className={`btn small ${r === who ? "primary" : ""}`} onClick={() => run(`${r} 로 전환`, () => api.login(codes[r]))}>
          {r}
        </button>
      ))}
      {isRater && (
        <>
          <span className="sep" />
          <span className="muted">{cur ? `문항 ${cur.idx + 1}/${cur.total} · ${cur.item_id} · ${cur.stage}단계` : ""}</span>
          <input value={jump} onChange={(e) => setJump(e.target.value)} onKeyDown={(e) => e.key === "Enter" && goto()} placeholder="번호·코드" />
          <button className="btn small" onClick={goto}>이동</button>
          <button className="btn small" disabled={!cur} onClick={() => cur && run("이 문항 초기화", () => api.dev.resetItem(cur.item_id))} title="1단계부터 다시 봅니다">
            이 문항 다시
          </button>
          <button className="btn small" onClick={() => window.confirm("내 1·2단계를 모두 지울까요?") && run("내 채점 초기화", api.dev.resetRater)}>내 채점 지우기</button>
        </>
      )}
      <span className="sep" />
      <button className="btn small" onClick={() => run("두 평가자 자동 채점", api.dev.fill)} title="판정·관리자 화면을 보려고 두 평가자가 모두 끝낸 상태를 만듭니다">
        자동 채점(판정 열기)
      </button>
      <button className="btn small" onClick={() => window.confirm("모래상자의 모든 채점·판정을 지울까요?") && run("전체 초기화", api.dev.wipe)}>전체 초기화</button>
      {msg && <span className="muted">{msg}</span>}
      <span style={{ flex: 1 }} />
      <button className="btn small ghost" onClick={() => setOpen(false)}>접기</button>
    </div>
  );
}
