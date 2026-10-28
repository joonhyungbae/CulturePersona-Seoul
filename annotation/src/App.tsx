import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { ENDPOINT } from "./config";
import { download, toCsv } from "./csv";
import { Adjudicate } from "./components/Adjudicate";
import { DevBar } from "./components/DevBar";
import { Guide } from "./components/Guide";
import { Login } from "./components/Login";
import { OwnerView } from "./components/OwnerView";
import { RaterView } from "./components/RaterView";
import type { Progress, Rater, Who } from "./types";

const NAME: Record<Who, string> = { R1: "평가자 1", R2: "평가자 2", ADJ: "합의 판정", OWNER: "관리자" };

export default function App() {
  const [who, setWho] = useState<Who | null | undefined>(undefined);
  const [tab, setTab] = useState<"work" | "guide">("work");
  const [progress, setProgress] = useState<Progress | null>(null);

  useEffect(() => {
    if (!api.hasCode()) return setWho(null);
    api.whoami().then(setWho, () => {
      api.logout();
      setWho(null);
    });
  }, []);

  const refresh = useCallback(() => {
    api.progress().then(setProgress, () => undefined);
  }, []);

  useEffect(() => {
    if (who) refresh();
  }, [who, refresh]);

  // If this is a deployed build and the backend URL was not set, say so before login.
  if (import.meta.env.PROD && ENDPOINT === "/api")
    return (
      <main>
        <div className="card login">
          <h1>설정이 끝나지 않았습니다</h1>
          <p className="muted">
            이 검수 화면은 아직 스프레드시트와 연결되지 않았습니다. 관리자가 저장소 변수 <code>APPS_SCRIPT_URL</code> 을 넣고 다시
            배포하면 열립니다.
          </p>
        </div>
      </main>
    );
  if (who === undefined) return <main><div className="card">불러오는 중…</div></main>;
  if (who === null)
    return (
      <main>
        <Login
          onLogin={(w) => {
            setWho(w);
            setTab(w === "R1" || w === "R2" ? "guide" : "work");
          }}
        />
      </main>
    );

  const isRater = who === "R1" || who === "R2";
  const mine = isRater ? progress?.[who] : null;
  const adj = progress?.adjudication;
  const pct = mine ? (100 * mine.done) / Math.max(1, mine.total) : adj?.total ? (100 * (adj.done ?? 0)) / adj.total : 0;

  const exportMine = async () => {
    const r = await api.exportRows();
    download(`annotations_${who}.csv`, toCsv(r.annotations, r.adjudication));
  };

  return (
    <>
      <header className="topbar">
        <div className="brand">
          페르소나 검수<small>{NAME[who]}</small>
        </div>
        <div className="progress">
          <div className="bar">
            <span style={{ width: `${pct}%` }} />
          </div>
          <div className="num">
            {mine && <>본 문항 {mine.done} / {mine.total}</>}
            {!isRater && progress && (
              <>
                평가자 1 {progress.R1?.done}/{progress.R1?.total} · 평가자 2 {progress.R2?.done}/{progress.R2?.total}
                {adj?.available ? <> · 판정 {adj.done}/{adj.total}</> : " · 판정 대기"}
              </>
            )}
          </div>
        </div>
        <button className={`btn small ${tab === "work" ? "primary" : ""}`} onClick={() => setTab("work")}>
          {who === "ADJ" ? "판정" : who === "OWNER" ? "현황" : "검수"}
        </button>
        <button className={`btn small ${tab === "guide" ? "primary" : ""}`} onClick={() => setTab("guide")}>
          지침서
        </button>
        {isRater && (
          <button className="btn small ghost" onClick={exportMine}>
            내 채점 CSV
          </button>
        )}
        <button
          className="btn small ghost"
          onClick={() => {
            api.logout();
            setWho(null);
          }}
        >
          나가기
        </button>
      </header>
      <main>
        {tab === "guide" ? (
          <Guide />
        ) : who === "OWNER" ? (
          <OwnerView progress={progress} onRefresh={refresh} />
        ) : who === "ADJ" ? (
          adj?.available ? (
            <Adjudicate onSaved={refresh} />
          ) : (
            <div className="card">
              <h2>합의 판정은 아직 열리지 않았습니다</h2>
              <p className="muted">두 평가자가 본 문항을 모두 끝내야 열립니다. 그 전에는 서로의 점수를 볼 수 없습니다.</p>
            </div>
          )
        ) : (
          <RaterView rater={who as Rater} onSaved={refresh} />
        )}
      </main>
      {import.meta.env.DEV && <DevBar who={who} />}
    </>
  );
}
