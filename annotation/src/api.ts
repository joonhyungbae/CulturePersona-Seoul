import { DATA, ENDPOINT } from "./config";
import type { Annotation, Disagreement, Item, PracticeExplain, Progress, PublicItem, Stage1, Stage2, Who } from "./types";

const CODE_KEY = "anno.code";
const STALE_ERROR = "없는 문항";
export const STALE_HELP = "문항 목록이 바뀌었습니다. 화면을 새로 고침하면 이어서 할 문항이 열립니다. 지금까지 저장한 채점은 그대로입니다.";

export function savedCode(): string | null {
  try {
    return localStorage.getItem(CODE_KEY);
  } catch {
    return null;
  }
}
function storeCode(code: string | null): void {
  try {
    if (code) localStorage.setItem(CODE_KEY, code);
    else localStorage.removeItem(CODE_KEY);
  } catch {
    /* If storage is blocked, the code is entered again on refresh */
  }
}

let code: string | null = savedCode();

/**
 * Every request is one POST {action, code, ...}. No Content-Type is set — as text/plain the
 * browser sends no preflight, so it talks to the Apps Script web app directly.
 */
async function post<T>(action: string, payload: Record<string, unknown> = {}, withCode = code): Promise<T> {
  let res: Response;
  try {
    res = await fetch(ENDPOINT, { method: "POST", body: JSON.stringify({ action, code: withCode, ...payload }) });
  } catch {
    throw new Error("서버에 닿지 않습니다. 인터넷 연결을 확인하고 다시 시도하세요.");
  }
  const body = (await res.json().catch(() => ({ ok: false, error: `응답을 읽지 못했습니다 (HTTP ${res.status})` }))) as T & {
    ok: boolean;
    error?: string;
  };
  if (!body.ok) {
    // After the item list changes, a request from an old screen is rejected as '없는 문항' — tell the rater what to do.
    if (body.error === STALE_ERROR) throw new Error(STALE_HELP);
    throw new Error(body.error ?? "알 수 없는 오류");
  }
  return body;
}

let itemsCache: Promise<Map<string, PublicItem>> | null = null;
export function items(): Promise<Map<string, PublicItem>> {
  itemsCache ??= fetch(`${DATA}items.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`문항 파일을 받지 못했습니다 (HTTP ${r.status})`);
      return r.json() as Promise<PublicItem[]>;
    })
    .then((arr) => new Map(arr.map((i) => [i.item_id, i])));
  itemsCache.catch(() => (itemsCache = null)); // On failure, fetch again next time
  return itemsCache;
}

const b64 = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

/** Unseal (AES-256-GCM) with the per-item key from the server. Paired with seal in build_public.mjs. */
export async function openItem(pub: PublicItem, keyB64: string): Promise<Item> {
  if (!globalThis.crypto?.subtle)
    throw new Error("이 브라우저 주소에서는 암호 해제를 쓸 수 없습니다. https 주소(또는 localhost)로 여세요.");
  const key = await crypto.subtle.importKey("raw", b64(keyB64), "AES-GCM", false, ["decrypt"]);
  const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv: b64(pub.sealed.iv) }, key, b64(pub.sealed.ct));
  const inner = JSON.parse(new TextDecoder().decode(plain)) as Pick<Item, "anchors" | "genres_attended" | "genres_not" | "narrative">;
  const { sealed: _sealed, ...rest } = pub;
  void _sealed;
  return { ...rest, ...inner };
}

export const api = {
  hasCode: () => Boolean(code),
  login: async (c: string) => {
    const r = await post<{ role: Who }>("login", {}, c.trim());
    code = c.trim();
    storeCode(code);
    return r.role;
  },
  whoami: () => post<{ role: Who }>("login").then((r) => r.role),
  logout: () => {
    code = null;
    storeCode(null);
  },
  // The order file changes when the sample is adjusted — always fetch it fresh, bypassing browser and CDN caches (raters kept working an old order, ADR-039).
  order: (rater: string) =>
    fetch(`${DATA}order_${rater}.json?t=${Date.now()}`, { cache: "no-store" }).then((r) => {
      if (!r.ok) throw new Error(`순서 파일을 받지 못했습니다 (HTTP ${r.status})`);
      return r.json() as Promise<string[]>;
    }),
  /** The currently deployed screen (script bundle name). Used to see if the open screen is stale. null if it cannot be fetched. */
  liveBundle: () =>
    fetch(`${import.meta.env.BASE_URL}index.html?t=${Date.now()}`, { cache: "no-store" })
      .then((r) => (r.ok ? r.text() : ""))
      .then((html) => html.match(/assets\/index-[A-Za-z0-9_-]+\.js/)?.[0] ?? null)
      .catch(() => null),
  /** My stage-2 scores and the stage-1 answer the server stored (the first one submitted). */
  state: () =>
    post<{ annotations: Record<string, Annotation>; stage1: Record<string, Stage1>; explain?: Record<string, PracticeExplain> }>(
      "state",
    ),
  progress: () => post<Progress & { ok: boolean }>("progress"),
  /** Submit the stage-1 answer and receive that item's key. If already submitted, stage1 is ignored and the first answer is returned. */
  reveal: (item_id: string, stage1?: Stage1) => post<{ key: string; stage1: Stage1; first: boolean }>("reveal", { item_id, stage1 }),
  /** Stage 1 uses what the server stored, so only stage 2 is sent. */
  /** For a practice item, the written answer and explanation (explain) come back with the response. */
  annotate: (item_id: string, stage2: Stage2) =>
    post<{ explain?: PracticeExplain | null }>("annotate", { item_id, stage2 }),
  adjudication: () => post<{ list: Omit<Disagreement, "item">[]; names?: Partial<Record<"R1" | "R2", string>> }>("adjudication"),
  adjudicate: (body: Record<string, unknown>) => post("adjudicate", body),
  /** Dev mode only (dev_ux.sh) — exists only on the local sandbox server. */
  dev: {
    info: () => post<{ codes: Record<Who, string>; store: string }>("dev_info"),
    resetItem: (item_id: string) => post("dev_reset_item", { item_id }),
    resetRater: () => post("dev_reset_rater"),
    fill: () => post<{ filled: number }>("dev_fill"),
    wipe: () => post("dev_wipe"),
  },
  exportRows: () =>
    post<{ annotations: Annotation[]; adjudication: Array<Record<string, unknown>> }>("export"),
};
