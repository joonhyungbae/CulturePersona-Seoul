import { useState } from "react";
import { api } from "../api";
import type { Who } from "../types";

export function Login({ onLogin }: { onLogin: (w: Who) => void }) {
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!code.trim()) return setError("받은 접속 코드를 넣으세요");
    setBusy(true);
    try {
      onLogin(await api.login(code));
    } catch (err) {
      setError(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card login" onSubmit={submit}>
      <h1>페르소나 검수</h1>
      <p className="muted">받은 접속 코드로 들어갑니다. 코드가 누구인지(평가자 1·2, 합의 판정)를 알려주므로 따로 고를 필요가 없습니다.</p>
      <label htmlFor="code">접속 코드</label>
      <input
        id="code"
        type="password"
        autoComplete="off"
        autoCapitalize="characters"
        spellCheck={false}
        style={{ width: "100%", letterSpacing: 2 }}
        value={code}
        onChange={(e) => setCode(e.target.value)}
      />
      {error && <div className="notice error" style={{ marginTop: 12 }}>{error}</div>}
      <div style={{ marginTop: 16 }}>
        <button className="btn primary" type="submit" disabled={busy}>
          {busy ? "확인 중…" : "들어가기"}
        </button>
      </div>
    </form>
  );
}
