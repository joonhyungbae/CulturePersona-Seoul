/**
 * Local test backend (attached to the Vite dev server). The deployed backend is Google Apps Script
 * (apps_script/Code.template.js), and this file follows **the same protocol and the same rules** —
 * POST /api with {action, code, ...} returns {ok, ...}. If a rule changes, change both and run
 * scripts/protocol_test.py --full against the local server and the Apps Script emulator.
 *
 * Storage is outputs/anno/ in the main repository (gitignore). Override with ANNO_STORE (for tests).
 *   annotations/R1.jsonl · R2.jsonl                 stage-2 scores (last line per item is valid)
 *   annotations/R1.stage1.jsonl · R2.stage1.jsonl   stage 1 (the **first** line per item is valid — it cannot be changed)
 *   adjudication.jsonl                              adjudication
 *   raters.json                                     {R1, R2, ADJ, OWNER} → access codes
 * Per-item keys are data_private/item_keys.json (written by build_public.mjs).
 * Row format of annotations/*.jsonl is read as-is by the registered-protocol analysis (anno_analysis.py).
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, "..");
const REPO = path.resolve(APP, "..");
const STORE = process.env.ANNO_STORE ? path.resolve(process.env.ANNO_STORE) : path.join(REPO, "outputs", "anno");
const ANN_DIR = path.join(STORE, "annotations");
const ADJ_FILE = path.join(STORE, "adjudication.jsonl");
const CODE_FILE = path.join(STORE, "raters.json");

/** Dev mode (dev_ux.sh): dev_* actions enabled only on the sandbox store. Absent from the deploy (Apps Script). */
const DEV = process.env.ANNO_DEV === "1";
const RATERS = ["R1", "R2"] as const;
type Rater = (typeof RATERS)[number];
const SCALES = ["grounded", "coherent", "plausible", "fluent"] as const;

type Body = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

function append(file: string, row: Body): void {
  mkdirSync(path.dirname(file), { recursive: true }); // Recreate the store if it was removed mid-run
  appendFileSync(file, JSON.stringify(row) + "\n", "utf-8");
}
function readJsonl(file: string): Body[] {
  if (!existsSync(file)) return [];
  return readFileSync(file, "utf-8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
}
function latest(rows: Body[]): Record<string, Body> {
  const m: Record<string, Body> = {};
  for (const r of rows) m[r.item_id] = r;
  return m;
}
function first(rows: Body[]): Record<string, Body> {
  const m: Record<string, Body> = {};
  for (const r of rows) if (!m[r.item_id]) m[r.item_id] = r;
  return m;
}

function validStage1(s: Body | undefined): string | null {
  if (!s || typeof s !== "object") return "1 단계가 비었다";
  for (const k of ["age_guess", "attend_guess"]) {
    if (typeof s[k] !== "string" || !s[k]) return `1 단계 ${k} 를 고르지 않았다`;
  }
  return null;
}
function validStage2(s: Body | undefined): string | null {
  if (!s || typeof s !== "object") return "2 단계가 비었다";
  if (s.contradiction !== "yes" && s.contradiction !== "no") return "모순 여부를 고르지 않았다";
  if (!Array.isArray(s.contra_fields)) return "모순 항목 형식이 잘못됐다";
  if (s.contradiction === "yes") {
    if (!s.contra_fields.length) return "모순이 있는 항목을 하나 이상 고르라";
    if (!String(s.contra_quote ?? "").trim()) return "모순의 근거 문장을 적어라";
  }
  for (const k of SCALES) {
    const v = Number(s[k]);
    if (!Number.isInteger(v) || v < 1 || v > 5) return `${k} 점수를 고르지 않았다`;
  }
  return null;
}
/** Disagreement rule from the registered protocol — same as disagrees_ in Apps Script. */
export function disagrees(a: Body, b: Body): boolean {
  if (a.contradiction !== b.contradiction) return true;
  return Math.abs(Number(a.grounded) - Number(b.grounded)) >= 2;
}
/** Normalize to the same shape Apps Script rebuilds from a sheet row (so the format does not differ by backend). */
function normStage1(s: Body): Body {
  return { age_guess: String(s.age_guess), attend_guess: String(s.attend_guess), secs: Number(s.secs) || 0 };
}
function normStage2(s: Body, withSecs = true): Body {
  const yes = s.contradiction === "yes";
  const out: Body = {
    contradiction: s.contradiction,
    contra_fields: yes ? s.contra_fields.map(String) : [],
    contra_quote: yes ? String(s.contra_quote) : "",
    grounded: Number(s.grounded),
    coherent: Number(s.coherent),
    plausible: Number(s.plausible),
    fluent: Number(s.fluent),
    note: String(s.note ?? ""),
  };
  if (withSecs) out.secs = Number(s.secs) || 0;
  return out;
}

export function annoPlugin(): Plugin {
  let mainIds: string[] = [];
  let practiceIds: string[] = [];
  let keys: Record<string, string> = {};
  let explain: Record<string, unknown> = {};

  const ann = () => ({
    R1: latest(readJsonl(path.join(ANN_DIR, "R1.jsonl"))),
    R2: latest(readJsonl(path.join(ANN_DIR, "R2.jsonl"))),
  });
  const stage1s = (r: Rater) => first(readJsonl(path.join(ANN_DIR, `${r}.stage1.jsonl`)));
  const known = (id: string) => mainIds.includes(id) || practiceIds.includes(id);
  const doneMain = (a: ReturnType<typeof ann>, r: Rater) => mainIds.filter((id) => a[r][id]).length;
  const bothComplete = (a: ReturnType<typeof ann>) => RATERS.every((r) => doneMain(a, r) === mainIds.length);
  const disagreementList = (a: ReturnType<typeof ann>) =>
    mainIds
      .filter((id) => a.R1[id] && a.R2[id] && disagrees(a.R1[id].stage2, a.R2[id].stage2))
      .map((id) => ({ item_id: id, R1: a.R1[id].stage2, R2: a.R2[id].stage2 }));

  // No global lock — same as Apps Script (removed because bad codes could lock everyone out).
  function roleOf(code: unknown): string {
    if (!existsSync(CODE_FILE)) throw new Error(`코드 파일이 없다: ${CODE_FILE}`);
    const codes = JSON.parse(readFileSync(CODE_FILE, "utf-8")) as Record<string, string>;
    const want = String(code ?? "").trim().toUpperCase();
    const role = Object.keys(codes).find((k) => codes[k].trim().toUpperCase() === want);
    if (!role) throw new Error("접속 코드가 맞지 않습니다");
    return role;
  }

  /** Delete matching lines from a file (dev-mode reset). */
  function dropLines(file: string, keep: (r: Body) => boolean): void {
    if (!existsSync(file)) return;
    const rows = readJsonl(file).filter(keep);
    writeFileSync(file, rows.map((r) => JSON.stringify(r) + "\n").join(""), "utf-8");
  }

  function devRoute(b: Body): Body {
    const codes = JSON.parse(readFileSync(CODE_FILE, "utf-8")) as Record<string, string>;
    if (b.action === "dev_info") return { ok: true, dev: true, codes, store: STORE };
    if (b.action === "dev_reset_item" || b.action === "dev_reset_rater") {
      const role = roleOf(b.code);
      if (!(RATERS as readonly string[]).includes(role)) return { ok: false, error: "평가자 코드로만" };
      const hit = (r: Body) => b.action === "dev_reset_rater" || r.item_id === b.item_id;
      dropLines(path.join(ANN_DIR, `${role}.jsonl`), (r) => !hit(r));
      dropLines(path.join(ANN_DIR, `${role}.stage1.jsonl`), (r) => !hit(r));
      return { ok: true };
    }
    if (b.action === "dev_wipe") {
      rmSync(ANN_DIR, { recursive: true, force: true });
      rmSync(ADJ_FILE, { force: true });
      mkdirSync(ANN_DIR, { recursive: true });
      return { ok: true };
    }
    if (b.action === "dev_fill") {
      // Make both raters look finished on every item — to open the adjudication and owner screens. Force a few disagreements.
      const a = ann();
      const now = new Date().toISOString();
      let n = 0;
      for (const [j, r] of RATERS.entries()) {
        const s1 = stage1s(r);
        [...mainIds, ...practiceIds].forEach((id, i) => {
          const stage1 = s1[id]?.stage1 ?? { age_guess: "40대", attend_guess: "예", secs: 7 };
          if (!s1[id]) append(path.join(ANN_DIR, `${r}.stage1.jsonl`), { rater: r, item_id: id, stage1, saved_at: now });
          if (a[r][id]) return;
          const contra = i % (j === 0 ? 9 : 11) === 0;
          append(path.join(ANN_DIR, `${r}.jsonl`), {
            rater: r, item_id: id, stage1, saved_at: now,
            stage2: {
              contradiction: contra ? "yes" : "no", contra_fields: contra ? ["summary"] : [],
              contra_quote: contra ? "(개발 모드 자동 채점)" : "", grounded: contra ? 2 : 4 + (i % 2),
              coherent: 4, plausible: 4, fluent: 5, note: "", secs: 30,
            },
          });
          n += 1;
        });
      }
      return { ok: true, filled: n };
    }
    return { ok: false, error: `없는 개발 동작: ${b.action}` };
  }

  function route(b: Body): Body {
    if (typeof b.action === "string" && b.action.startsWith("dev_"))
      return DEV ? devRoute(b) : { ok: false, error: "개발 모드가 아니다" };
    const role = roleOf(b.code);
    const isRater = (RATERS as readonly string[]).includes(role);
    if (b.action === "login") return { ok: true, role };

    if (b.action === "progress") {
      const a = ann();
      const complete = bothComplete(a);
      const mine = (r: Rater) => ({ done: doneMain(a, r), total: mainIds.length });
      if (isRater) return { ok: true, [role]: mine(role as Rater), adjudication: { available: complete, done: null, total: null } };
      return {
        ok: true, R1: mine("R1"), R2: mine("R2"),
        adjudication: {
          available: complete,
          done: Object.keys(latest(readJsonl(ADJ_FILE))).length,
          total: complete ? disagreementList(a).length : null,
        },
      };
    }

    if (b.action === "state") {
      if (!isRater) return { ok: true, annotations: {}, stage1: {}, explain: {} };
      const s1 = stage1s(role as Rater);
      const mine = ann()[role as Rater];
      // Practice explanations only for practice items that rater has already saved (same as Apps Script).
      const ex = Object.fromEntries(practiceIds.filter((id) => mine[id] && explain[id]).map((id) => [id, explain[id]]));
      return {
        ok: true,
        annotations: mine,
        stage1: Object.fromEntries(Object.entries(s1).map(([k, v]) => [k, v.stage1])),
        explain: ex,
      };
    }

    if (b.action === "reveal") {
      if (!isRater) return { ok: false, error: "평가자 코드로만 쓸 수 있다" };
      if (!known(b.item_id)) return { ok: false, error: "없는 문항" };
      const prev = stage1s(role as Rater)[b.item_id];
      if (prev) return { ok: true, stage1: prev.stage1, key: keys[b.item_id], first: false };
      const err = validStage1(b.stage1);
      if (err) return { ok: false, error: err };
      const s1 = normStage1(b.stage1);
      append(path.join(ANN_DIR, `${role}.stage1.jsonl`), { rater: role, item_id: b.item_id, stage1: s1, saved_at: new Date().toISOString() });
      return { ok: true, stage1: s1, key: keys[b.item_id], first: true };
    }

    if (b.action === "annotate") {
      if (!isRater) return { ok: false, error: "평가자 코드로만 채점할 수 있다" };
      if (!known(b.item_id)) return { ok: false, error: "없는 문항" };
      const err = validStage2(b.stage2);
      if (err) return { ok: false, error: err };
      const s1 = stage1s(role as Rater)[b.item_id];
      if (!s1) return { ok: false, error: "1 단계를 먼저 끝내야 한다" };
      const practice = practiceIds.includes(b.item_id);
      if (!practice && bothComplete(ann()))
        return { ok: false, error: "두 평가자가 모두 끝나 독립 채점이 잠겼습니다. 이견은 합의 판정에서 다룹니다." };
      // Stage 1 is the first answer the server stored — stage 1 in the request is ignored.
      const row = { rater: role, item_id: b.item_id, stage1: s1.stage1, stage2: normStage2(b.stage2), saved_at: new Date().toISOString() };
      append(path.join(ANN_DIR, `${role}.jsonl`), row);
      return practice ? { ok: true, explain: explain[b.item_id] ?? null } : { ok: true };
    }

    if (b.action === "adjudication") {
      if (role !== "ADJ" && role !== "OWNER") return { ok: false, error: "판정 코드가 필요하다" };
      const a = ann();
      if (!bothComplete(a)) return { ok: false, error: "두 평가자가 모두 끝나야 판정할 수 있다" };
      const adj = latest(readJsonl(ADJ_FILE));
      const np = path.join(APP, "data_private", "rater_names.json");
      const names = existsSync(np) ? JSON.parse(readFileSync(np, "utf-8")) : {};
      return { ok: true, names, list: disagreementList(a).map((d) => ({ ...d, decided: adj[d.item_id] ?? null, key: keys[d.item_id] })) };
    }

    if (b.action === "adjudicate") {
      if (role !== "ADJ") return { ok: false, error: "판정 코드가 필요하다" };
      const err = validStage2(b.consensus);
      if (err) return { ok: false, error: err };
      if (!String(b.reason ?? "").trim()) return { ok: false, error: "판정 근거를 적어라" };
      if (!["R1", "R2", "both", "none"].includes(b.changed_by)) return { ok: false, error: "누가 바꿨는지 고르라" };
      if (!(RATERS as readonly string[]).includes(b.first_speaker)) return { ok: false, error: "먼저 말한 사람이 잘못됐다" };
      const a = ann();
      if (!bothComplete(a)) return { ok: false, error: "두 평가자가 모두 끝나야 판정할 수 있다" };
      if (!disagreementList(a).some((d) => d.item_id === b.item_id)) return { ok: false, error: "불일치 문항이 아니다" };
      const row = {
        item_id: b.item_id, consensus: normStage2(b.consensus, false), changed_by: b.changed_by,
        reason: String(b.reason), first_speaker: b.first_speaker, saved_at: new Date().toISOString(),
      };
      append(ADJ_FILE, row);
      return { ok: true };
    }

    if (b.action === "export") {
      const a = ann();
      if (isRater) return { ok: true, annotations: Object.values(a[role as Rater]), adjudication: [] };
      if (!bothComplete(a)) return { ok: false, error: "두 평가자가 모두 끝난 뒤 내보낼 수 있다 (그 전에는 진행 수만 봅니다)" };
      return {
        ok: true,
        annotations: [...Object.values(a.R1), ...Object.values(a.R2)],
        adjudication: Object.values(latest(readJsonl(ADJ_FILE))),
      };
    }
    return { ok: false, error: `없는 동작: ${b.action}` };
  }

  async function readBody(req: IncomingMessage): Promise<Body> {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    return JSON.parse(Buffer.concat(chunks).toString("utf-8") || "{}");
  }
  function send(res: ServerResponse, body: Body): void {
    res.statusCode = 200; // Always 200 with {ok}, like Apps Script
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.setHeader("Cache-Control", "no-store");
    res.end(JSON.stringify(body));
  }

  return {
    name: "anno-api",
    configureServer(server) {
      const full = JSON.parse(readFileSync(path.join(APP, "data_private", "items_full.json"), "utf-8")) as Body[];
      mainIds = full.filter((i) => !i.practice).map((i) => i.item_id);
      practiceIds = full.filter((i) => i.practice).map((i) => i.item_id);
      keys = JSON.parse(readFileSync(path.join(APP, "data_private", "item_keys.json"), "utf-8"));
      const ex = path.join(APP, "data_private", "practice_explain.json");
      explain = existsSync(ex) ? JSON.parse(readFileSync(ex, "utf-8")) : {};
      mkdirSync(ANN_DIR, { recursive: true });
      server.middlewares.use(async (req, res, next) => {
        if (req.url !== "/api" || req.method !== "POST") return next();
        let body: Body = {};
        let out: Body;
        try {
          body = await readBody(req);
          out = route(body);
        } catch (e) {
          out = { ok: false, error: String((e as Error).message ?? e) };
        }
        // Access log — same contents as the log sheet in Apps Script
        let role = "?";
        try {
          role = roleOf(body.code);
        } catch {
          role = "?";
        }
        append(path.join(STORE, "access_log.jsonl"), { at: new Date().toISOString(), role, action: body.action, item_id: body.item_id ?? "", ok: out.ok ? 1 : 0 });
        send(res, out);
      });
    },
  };
}
