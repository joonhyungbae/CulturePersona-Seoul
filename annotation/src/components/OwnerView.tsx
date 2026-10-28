import { useState } from "react";
import { api } from "../api";
import { download, toCsv } from "../csv";
import type { Progress } from "../types";

/** Owner only — view progress and download. Scores are not shown (they may share a screen with a rater). */
export function OwnerView({ progress, onRefresh }: { progress: Progress | null; onRefresh: () => void }) {
  const [msg, setMsg] = useState<string | null>(null);
  const pull = async () => {
    setMsg("내려받는 중…");
    try {
      const r = await api.exportRows();
      download(`annotations_all_${new Date().toISOString().slice(0, 10)}.csv`, toCsv(r.annotations, r.adjudication));
      setMsg(`독립 채점 ${r.annotations.length} 행 · 합의 판정 ${r.adjudication.length} 행을 내려받았습니다.`);
    } catch (e) {
      setMsg(String((e as Error).message));
    }
  };
  return (
    <div className="card">
      <h2>진행 현황 (관리자)</h2>
      {progress ? (
        <p>
          평가자 1: <b>{progress.R1?.done}</b> / {progress.R1?.total} · 평가자 2: <b>{progress.R2?.done}</b> / {progress.R2?.total}
          <br />
          합의 판정: {progress.adjudication.available ? `${progress.adjudication.done} / ${progress.adjudication.total}` : "두 평가자가 끝나면 열립니다"}
        </p>
      ) : (
        <p className="muted">불러오는 중…</p>
      )}
      <div className="row">
        <button className="btn" onClick={onRefresh}>새로고침</button>
        <button className="btn primary" onClick={pull}>전체 채점 CSV 내려받기</button>
      </div>
      {msg && <p className="muted" style={{ marginTop: 10 }}>{msg}</p>}
      <p className="muted" style={{ marginTop: 10 }}>
        원본은 Google 스프레드시트의 annotations · adjudication 시트에 그대로 쌓입니다. 분석은 연구 저장소의 <code>scripts/anno_pull.py</code> 로 내려받아 돌립니다.
      </p>
    </div>
  );
}
