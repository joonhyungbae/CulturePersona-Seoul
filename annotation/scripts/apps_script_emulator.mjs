// Apps Script emulator — runs apps_script/Code.gs **as-is** in Node.
//
// Built to test the backend end to end without a real Google account. It fakes SpreadsheetApp, CacheService, LockService,
// ContentService, and Utilities, and in particular reproduces **Google Sheets rewriting values on its own**:
//   - number-like text → number ("030666" → 30666, "90E588" → Infinity → #NUM!)
//   - starts with "=" → formula (here, #ERROR!)
//   - ISO-date-like text → date object
//   - a leading single quote (') keeps the text as written (the quote is not stored) — the sheet quote prefix
// Without this behavior, emulator tests miss bugs that break on a real sheet.
//
// Run: node scripts/apps_script_emulator.mjs [--port 5190] [--code apps_script/Code.gs]
// Then run protocol_test.py against http://127.0.0.1:5190/exec. codes are printed at startup.
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import http from "node:http";
import vm from "node:vm";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => (a.startsWith("--") ? [...acc, [a.slice(2), arr[i + 1]]] : acc), []),
);
const PORT = Number(args.port ?? 5190);
const CODE_PATH = args.code ?? new URL("../apps_script/Code.gs", import.meta.url);

// --------------------------------------------------------------------------- sheet value coercion

const NUM = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
const ISO = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/;

function sheetsCoerce(v) {
  if (typeof v !== "string") return v;
  if (v.startsWith("'")) return v.slice(1); // quote prefix: keep the text as written
  if (v.startsWith("=")) return "#ERROR!";
  if (NUM.test(v.trim())) {
    const n = Number(v);
    return Number.isFinite(n) ? n : "#NUM!";
  }
  if (ISO.test(v)) return new Date(v);
  return v;
}

class Range {
  constructor(sheet, r, c, nr, nc) {
    Object.assign(this, { sheet, r, c, nr, nc });
  }
  setValues(vals) {
    vals.forEach((row, i) =>
      row.forEach((v, j) => {
        const rr = this.r - 1 + i;
        while (this.sheet.rows.length <= rr) this.sheet.rows.push([]);
        this.sheet.rows[rr][this.c - 1 + j] = sheetsCoerce(v);
      }),
    );
    return this;
  }
  getValues() {
    const out = [];
    for (let i = 0; i < this.nr; i++) {
      const row = this.sheet.rows[this.r - 1 + i] ?? [];
      out.push(Array.from({ length: this.nc }, (_, j) => row[this.c - 1 + j] ?? ""));
    }
    return out;
  }
}

class Sheet {
  constructor(name) {
    this.name = name;
    this.rows = [];
  }
  getLastRow() {
    return this.rows.length;
  }
  getLastColumn() {
    return Math.max(0, ...this.rows.map((r) => r.length));
  }
  getRange(r, c, nr = 1, nc = 1) {
    return new Range(this, r, c, nr, nc);
  }
  getDataRange() {
    // Return 1×1 even for an empty sheet (same as the real behavior).
    return new Range(this, 1, 1, Math.max(1, this.getLastRow()), Math.max(1, this.getLastColumn()));
  }
  appendRow(row) {
    this.rows.push(row.map(sheetsCoerce));
    return this;
  }
  setFrozenRows() {
    return this;
  }
}

const sheets = new Map();
const spreadsheet = {
  getId: () => "EMU_SHEET",
  getUrl: () => "https://docs.google.com/spreadsheets/d/EMU_SHEET",
  getSheetByName: (n) => sheets.get(n) ?? null,
  insertSheet: (n) => {
    const s = new Sheet(n);
    sheets.set(n, s);
    return s;
  },
};

const cacheStore = new Map();
const CacheService = {
  getScriptCache: () => ({
    get: (k) => {
      const e = cacheStore.get(k);
      if (!e || e.until < Date.now()) return null;
      return e.v;
    },
    put: (k, v, ttl = 600) => {
      if (String(v).length > 100000) throw new Error("CacheService: 값이 100KB 를 넘는다");
      cacheStore.set(k, { v: String(v), until: Date.now() + ttl * 1000 });
    },
    remove: (k) => cacheStore.delete(k),
  }),
};

// --standalone: no bound sheet, as with a script created separately on script.google.com (getActiveSpreadsheet → null)
const STANDALONE = process.argv.includes("--standalone");
const props = new Map();
const context = vm.createContext({
  SpreadsheetApp: {
    getActiveSpreadsheet: () => (STANDALONE ? null : spreadsheet),
    create: () => spreadsheet,
    openById: (id) => {
      if (id !== "EMU_SHEET") throw new Error("없는 스프레드시트");
      return spreadsheet;
    },
  },
  PropertiesService: {
    getScriptProperties: () => ({ getProperty: (k) => props.get(k) ?? null, setProperty: (k, v) => props.set(k, v) }),
  },
  CacheService,
  LockService: { getScriptLock: () => ({ waitLock() {}, releaseLock() {} }) },
  ContentService: {
    MimeType: { JSON: "application/json" },
    createTextOutput: (s) => ({ s, setMimeType() { return this; } }),
  },
  Utilities: { getUuid: () => randomUUID() },
  console,
  Date,
  JSON,
  Math,
  Number,
  String,
  Object,
  Array,
  parseInt,
});
vm.runInContext(readFileSync(CODE_PATH, "utf-8"), context, { filename: "Code.gs" });
vm.runInContext("setup()", context);

const codes = Object.fromEntries(sheets.get("codes").rows.slice(1).map((r) => [r[0], String(r[1])]));
console.log("CODES " + JSON.stringify(codes));

http
  .createServer((req, res) => {
    if (req.method === "GET" && req.url === "/dump") {
      // For tests: the value that actually landed in the sheet (check coercion)
      res.setHeader("Content-Type", "application/json");
      return res.end(JSON.stringify(Object.fromEntries([...sheets].map(([k, s]) => [k, s.rows.slice(0, 4)]))));
    }
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const out = context.doPost({ postData: { contents: Buffer.concat(chunks).toString("utf-8") } });
      res.setHeader("Content-Type", "application/json; charset=utf-8");
      res.end(out.s);
    });
  })
  .listen(PORT, "127.0.0.1", () => console.log(`emulator http://127.0.0.1:${PORT}/exec`));
