/**
 * CulturePersona-Seoul human annotation — spreadsheet backend (ADR-023).
 *
 * `npm run build:data` generates this file from the template. Do not edit the item list or keys by hand.
 *
 * Setup (README.md):
 *   1. New Google spreadsheet → Extensions → Apps Script → paste this whole file.
 *   2. Pick setup from the function list and run it once (authorize). Creates the annotations, stage1, adjudication, and codes sheets.
 *   3. Deploy → New deployment → type: Web app · execute as: Me · access: Anyone → copy the web app URL.
 *
 * Blinding rules (same as the local server server/annoPlugin.ts — check both with scripts/protocol_test.py):
 *   - The public item file has only the masked narrative in plaintext; anchors and the original are sealed with a per-item key.
 *     The key is given only after the rater submits that item's stage-1 answer to this server (reveal). A submitted stage-1 answer cannot be changed.
 *   - A rater receives only their own scores. The other rater's scores and progress are not returned by any action.
 *   - When both raters finish the main items, independent scores lock (annotate is rejected) and adjudication opens.
 *   - Disagreement rule: contradiction differs, or anchor fit differs by 2 or more (the registered protocol §1).
 *   - ADJ is a code handed over only for adjudication; OWNER is the code the administrator uses to download. OWNER too receives
 *     only progress counts until both raters have finished (no peeking during scoring). Every request is written to the log sheet.
 */

const MAIN_IDS = /*__MAIN_IDS__*/ [];
const PRACTICE_IDS = /*__PRACTICE_IDS__*/ [];
const ITEM_KEYS = /*__ITEM_KEYS__*/ {};
// Written answer and explanation for practice items — returned when a practice item is saved (instead of an oral explanation, the registered protocol v2 §6.1).
const PRACTICE_EXPLAIN = /*__PRACTICE_EXPLAIN__*/ {};
// Rater names shown on the adjudication screen (data_private/rater_names.json). Not in the public repository or the deploy artifact.
const RATER_NAMES = /*__RATER_NAMES__*/ {};

const RATERS = ['R1', 'R2'];
const SCALES = ['grounded', 'coherent', 'plausible', 'fluent'];
const ANN_HEADER = ['rater', 'item_id', 'practice', 'age_guess', 'attend_guess', 'area_guess', 'stage1_secs',
  'contradiction', 'contra_fields', 'contra_quote', 'grounded', 'coherent', 'plausible', 'fluent', 'note',
  'stage2_secs', 'saved_at'];
const S1_HEADER = ['rater', 'item_id', 'age_guess', 'attend_guess', 'area_guess', 'stage1_secs', 'saved_at'];
// The area_guess column stays after the living-area guess was withdrawn (2026-09-23, the registered protocol change log) — to keep
// column positions aligned with sheets already created. Always empty and never read.
const ADJ_HEADER = ['item_id', 'contradiction', 'contra_fields', 'contra_quote', 'grounded', 'coherent',
  'plausible', 'fluent', 'note', 'changed_by', 'reason', 'first_speaker', 'saved_at'];
const LOG_HEADER = ['at', 'role', 'action', 'item_id', 'ok'];
const CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
const CODE_CACHE_SECS = 60; // A hand edit to the codes sheet takes effect within 1 minute

// --------------------------------------------------------------------------- sheet value protection

/**
 * Sheets turns number-like text into numbers ("030666" → 30666, "90E588" → scientific notation) and, if it starts with '=',
 * runs it as a formula. 35 item codes broke this way and adjudication never opened (reproduced in the emulator: scoring all 325 still left 311).
 * Prefix every text cell with a single quote so it stays text. The quote is not part of the cell value. This also blocks formula injection.
 */
function txt_(v) {
  return "'" + String(v === null || v === undefined ? '' : v);
}

// --------------------------------------------------------------------------- setup

/**
 * Spreadsheet to write to. A script created from the sheet menu (Extensions → Apps Script) uses that sheet;
 * a script created separately on script.google.com uses the sheet setup() created and remembered.
 */
function ss_() {
  const bound = SpreadsheetApp.getActiveSpreadsheet();
  if (bound) return bound;
  const id = PropertiesService.getScriptProperties().getProperty('SHEET_ID');
  if (!id) throw new Error('저장할 스프레드시트가 없다 — 편집기에서 setup 을 먼저 실행하라');
  return SpreadsheetApp.openById(id);
}

function setup() {
  let ss = SpreadsheetApp.getActiveSpreadsheet();
  if (!ss) {
    const props = PropertiesService.getScriptProperties();
    const id = props.getProperty('SHEET_ID');
    ss = id ? SpreadsheetApp.openById(id) : SpreadsheetApp.create('CulturePersona 검수');
    props.setProperty('SHEET_ID', ss.getId());
  }
  console.log('저장 스프레드시트: ' + ss.getUrl());
  ensureSheet_(ss, 'annotations', ANN_HEADER);
  ensureSheet_(ss, 'stage1', S1_HEADER);
  ensureSheet_(ss, 'adjudication', ADJ_HEADER);
  ensureSheet_(ss, 'log', LOG_HEADER);
  const codes = ensureSheet_(ss, 'codes', ['role', 'code']);
  if (codes.getLastRow() < 2) {
    const rows = ['R1', 'R2', 'ADJ', 'OWNER'].map((r) => [txt_(r), txt_(randomCode_())]);
    codes.getRange(2, 1, rows.length, 2).setValues(rows);
  }
  CacheService.getScriptCache().remove('codes');
}

function ensureSheet_(ss, name, header) {
  let sh = ss.getSheetByName(name);
  if (!sh) sh = ss.insertSheet(name);
  if (sh.getLastRow() === 0) {
    sh.getRange(1, 1, 1, header.length).setValues([header]);
    sh.setFrozenRows(1);
  }
  return sh;
}

function randomCode_() {
  // UUID v4 is cryptographic randomness. 12 characters × 5 bits = 60 bits (32^12 ≈ 10^18).
  const hex = Utilities.getUuid().replace(/-/g, '');
  let out = '';
  for (let i = 0; i < 12; i++) out += CODE_ALPHABET[parseInt(hex.substr(i * 2, 2), 16) % 32];
  return out;
}

// --------------------------------------------------------------------------- entry point

function doGet() {
  return json_({ ok: true, service: 'culturepersona-anno' });
}

function doPost(e) {
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    return json_({ ok: false, error: '요청 형식이 잘못됐다' });
  }
  let out;
  try {
    out = route_(body);
  } catch (err) {
    const msg = String(err && err.message ? err.message : err);
    out = /lock/i.test(msg) ? { ok: false, error: '저장이 몰렸습니다. 잠시 뒤 다시 저장하세요.' } : { ok: false, error: msg };
  }
  log_(body, out);
  return json_(out);
}

/** Access log — who did what, and when (so the order of reveal, export, and adjudication can be audited later). */
function log_(body, out) {
  try {
    let role = '?';
    try {
      role = roleOf_(body.code);
    } catch (e) {
      role = '?';
    }
    const sh = ensureSheet_(ss_(), 'log', LOG_HEADER);
    sh.appendRow([txt_(new Date().toISOString()), txt_(role), txt_(body.action), txt_(body.item_id || ''), out && out.ok ? 1 : 0]);
  } catch (e) {
    // A failed log write must not block scoring.
  }
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

// --------------------------------------------------------------------------- auth

/**
 * Do not lock everyone out on a bad code. A global lock would let bad codes sent to the public URL block everyone
 * (noted in independent review). Codes are 60 bits, so brute force is not practical.
 */
function roleOf_(code) {
  const cache = CacheService.getScriptCache();
  let table = cache.get('codes');
  if (!table) {
    const sh = ss_().getSheetByName('codes');
    if (!sh || sh.getLastRow() < 2) throw new Error('설정이 끝나지 않았습니다 (setup 을 실행하세요)');
    const rows = sh.getRange(2, 1, sh.getLastRow() - 1, 2).getValues();
    // Uppercase both sides so a code edited by hand in the sheet still matches.
    table = JSON.stringify(rows.reduce((m, r) => { m[String(r[1]).trim().toUpperCase()] = String(r[0]).trim(); return m; }, {}));
    cache.put('codes', table, CODE_CACHE_SECS);
  }
  const role = JSON.parse(table)[String(code || '').trim().toUpperCase()];
  if (!role) throw new Error('접속 코드가 맞지 않습니다');
  return role;
}

// --------------------------------------------------------------------------- reads

function sheet_(name) {
  return ss_().getSheetByName(name);
}

function rows_(name) {
  const values = sheet_(name).getDataRange().getValues();
  const head = values[0];
  const out = [];
  for (let i = 1; i < values.length; i++) out.push(rowObj_(head, values[i]));
  return out;
}

function rowObj_(head, row) {
  const o = {};
  head.forEach((h, j) => { o[h] = row[j] instanceof Date ? row[j].toISOString() : String(row[j]); });
  return o;
}

function splitList_(s) {
  return String(s || '').split('|').filter(Boolean);
}

/** Stage 1 per rater. **The first submitted answer is the valid one** — it cannot be changed after seeing the anchors. */
function stage1s_() {
  const out = { R1: {}, R2: {} };
  rows_('stage1').forEach((r) => {
    if (!out[r.rater] || out[r.rater][r.item_id]) return;
    out[r.rater][r.item_id] = {
      age_guess: r.age_guess, attend_guess: r.attend_guess, secs: Number(r.stage1_secs),
    };
  });
  return out;
}

function annotations_() {
  const byRater = { R1: {}, R2: {} };
  rows_('annotations').forEach((r) => {
    if (!byRater[r.rater]) return;
    byRater[r.rater][r.item_id] = { // For stage 2, the last save is valid
      rater: r.rater, item_id: r.item_id, saved_at: r.saved_at,
      stage1: { age_guess: r.age_guess, attend_guess: r.attend_guess, secs: Number(r.stage1_secs) },
      stage2: {
        contradiction: r.contradiction, contra_fields: splitList_(r.contra_fields), contra_quote: r.contra_quote,
        grounded: Number(r.grounded), coherent: Number(r.coherent), plausible: Number(r.plausible),
        fluent: Number(r.fluent), note: r.note, secs: Number(r.stage2_secs),
      },
    };
  });
  return byRater;
}

function adjudications_() {
  const out = {};
  rows_('adjudication').forEach((r) => {
    out[r.item_id] = {
      item_id: r.item_id,
      consensus: {
        contradiction: r.contradiction, contra_fields: splitList_(r.contra_fields), contra_quote: r.contra_quote,
        grounded: Number(r.grounded), coherent: Number(r.coherent), plausible: Number(r.plausible),
        fluent: Number(r.fluent), note: r.note,
      },
      changed_by: r.changed_by, reason: r.reason, first_speaker: r.first_speaker, saved_at: r.saved_at,
    };
  });
  return out;
}

// --------------------------------------------------------------------------- rules

function doneMain_(ann, r) {
  return MAIN_IDS.filter((id) => ann[r][id]).length;
}

function bothComplete_(ann) {
  return RATERS.every((r) => doneMain_(ann, r) === MAIN_IDS.length);
}

function disagrees_(a, b) {
  if (a.contradiction !== b.contradiction) return true;
  return Math.abs(a.grounded - b.grounded) >= 2;
}

function disagreementList_(ann) {
  return MAIN_IDS
    .filter((id) => ann.R1[id] && ann.R2[id] && disagrees_(ann.R1[id].stage2, ann.R2[id].stage2))
    .map((id) => ({ item_id: id, R1: ann.R1[id].stage2, R2: ann.R2[id].stage2 }));
}

function known_(id) {
  return MAIN_IDS.indexOf(id) >= 0 || PRACTICE_IDS.indexOf(id) >= 0;
}

function validStage1_(s) {
  if (!s || typeof s !== 'object') return '1 단계가 비었다';
  for (const k of ['age_guess', 'attend_guess']) {
    if (typeof s[k] !== 'string' || !s[k]) return '1 단계 ' + k + ' 를 고르지 않았다';
  }
  return null;
}

function validStage2_(s) {
  if (!s || typeof s !== 'object') return '2 단계가 비었다';
  if (s.contradiction !== 'yes' && s.contradiction !== 'no') return '모순 여부를 고르지 않았다';
  if (!Array.isArray(s.contra_fields)) return '모순 항목 형식이 잘못됐다';
  if (s.contradiction === 'yes') {
    if (!s.contra_fields.length) return '모순이 있는 항목을 하나 이상 고르라';
    if (!s.contra_quote || !String(s.contra_quote).trim()) return '모순의 근거 문장을 적어라';
  }
  for (const k of SCALES) {
    const v = Number(s[k]);
    if (!Number.isInteger(v) || v < 1 || v > 5) return k + ' 점수를 고르지 않았다';
  }
  return null;
}

function withLock_(fn) {
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    return fn();
  } finally {
    lock.releaseLock();
  }
}

// --------------------------------------------------------------------------- actions

function route_(b) {
  const role = roleOf_(b.code);
  const isRater = RATERS.indexOf(role) >= 0;

  if (b.action === 'login') return { ok: true, role: role };

  if (b.action === 'progress') {
    const ann = annotations_();
    const complete = bothComplete_(ann);
    const mine = (r) => ({ done: doneMain_(ann, r), total: MAIN_IDS.length });
    // A rater gets only their own progress — the other's progress becomes time pressure, and after the lock it would reveal the disagreement count (review note).
    if (isRater) {
      const out = { ok: true, adjudication: { available: complete, done: null, total: null } };
      out[role] = mine(role);
      return out;
    }
    return {
      ok: true, R1: mine('R1'), R2: mine('R2'),
      adjudication: { available: complete, done: Object.keys(adjudications_()).length, total: complete ? disagreementList_(ann).length : null },
    };
  }

  if (b.action === 'state') {
    if (!isRater) return { ok: true, annotations: {}, stage1: {}, explain: {} };
    const mine = annotations_()[role];
    // Practice explanations only for practice items **that rater has already saved** — never before saving (for review and resume).
    const explain = {};
    PRACTICE_IDS.forEach((id) => {
      if (mine[id] && PRACTICE_EXPLAIN[id]) explain[id] = PRACTICE_EXPLAIN[id];
    });
    return { ok: true, annotations: mine, stage1: stage1s_()[role], explain: explain };
  }

  if (b.action === 'reveal') {
    // Accept the stage-1 answer and return that item's key. If already submitted, return the first answer (it cannot be changed).
    if (!isRater) return { ok: false, error: '평가자 코드로만 쓸 수 있다' };
    if (!known_(b.item_id)) return { ok: false, error: '없는 문항' };
    return withLock_(() => {
      const prev = stage1s_()[role][b.item_id];
      if (prev) return { ok: true, stage1: prev, key: ITEM_KEYS[b.item_id], first: false };
      const err = validStage1_(b.stage1);
      if (err) return { ok: false, error: err };
      const s1 = b.stage1;
      sheet_('stage1').appendRow([txt_(role), txt_(b.item_id), txt_(s1.age_guess), txt_(s1.attend_guess),
        '', Number(s1.secs) || 0, txt_(new Date().toISOString())]);
      return {
        ok: true, key: ITEM_KEYS[b.item_id], first: true,
        stage1: { age_guess: s1.age_guess, attend_guess: s1.attend_guess, secs: Number(s1.secs) || 0 },
      };
    });
  }

  if (b.action === 'annotate') {
    if (!isRater) return { ok: false, error: '평가자 코드로만 채점할 수 있다' };
    if (!known_(b.item_id)) return { ok: false, error: '없는 문항' };
    const err = validStage2_(b.stage2);
    if (err) return { ok: false, error: err };
    return withLock_(() => {
      const s1 = stage1s_()[role][b.item_id];
      if (!s1) return { ok: false, error: '1 단계를 먼저 끝내야 한다' };
      const practice = PRACTICE_IDS.indexOf(b.item_id) >= 0;
      // When both raters finish, freeze the independent scores — editing after seeing the adjudication would break independence (the registered protocol §4).
      if (!practice && bothComplete_(annotations_())) {
        return { ok: false, error: '두 평가자가 모두 끝나 독립 채점이 잠겼습니다. 이견은 합의 판정에서 다룹니다.' };
      }
      const s2 = b.stage2;
      const yes = s2.contradiction === 'yes';
      // Stage 1 uses the first answer the server stored — stage 1 carried in the request is ignored.
      sheet_('annotations').appendRow([
        txt_(role), txt_(b.item_id), practice ? 1 : 0, txt_(s1.age_guess), txt_(s1.attend_guess), '',
        Number(s1.secs) || 0, txt_(s2.contradiction), txt_(yes ? s2.contra_fields.join('|') : ''),
        txt_(yes ? s2.contra_quote : ''), Number(s2.grounded), Number(s2.coherent), Number(s2.plausible),
        Number(s2.fluent), txt_(s2.note || ''), Number(s2.secs) || 0, txt_(new Date().toISOString()),
      ]);
      return practice ? { ok: true, explain: PRACTICE_EXPLAIN[b.item_id] || null } : { ok: true };
    });
  }

  if (b.action === 'adjudication') {
    if (role !== 'ADJ' && role !== 'OWNER') return { ok: false, error: '판정 코드가 필요하다' };
    const ann = annotations_();
    if (!bothComplete_(ann)) return { ok: false, error: '두 평가자가 모두 끝나야 판정할 수 있다' };
    const adj = adjudications_();
    return {
      ok: true,
      names: RATER_NAMES,
      list: disagreementList_(ann).map((d) => Object.assign(d, { decided: adj[d.item_id] || null, key: ITEM_KEYS[d.item_id] })),
    };
  }

  if (b.action === 'adjudicate') {
    if (role !== 'ADJ') return { ok: false, error: '판정 코드가 필요하다' };
    const c = b.consensus;
    const err = validStage2_(c);
    if (err) return { ok: false, error: err };
    if (!b.reason || !String(b.reason).trim()) return { ok: false, error: '판정 근거를 적어라' };
    if (['R1', 'R2', 'both', 'none'].indexOf(b.changed_by) < 0) return { ok: false, error: '누가 바꿨는지 고르라' };
    if (RATERS.indexOf(b.first_speaker) < 0) return { ok: false, error: '먼저 말한 사람이 잘못됐다' };
    return withLock_(() => {
      const ann = annotations_();
      if (!bothComplete_(ann)) return { ok: false, error: '두 평가자가 모두 끝나야 판정할 수 있다' };
      if (!disagreementList_(ann).some((d) => d.item_id === b.item_id)) return { ok: false, error: '불일치 문항이 아니다' };
      const yes = c.contradiction === 'yes';
      sheet_('adjudication').appendRow([
        txt_(b.item_id), txt_(c.contradiction), txt_(yes ? c.contra_fields.join('|') : ''), txt_(yes ? c.contra_quote : ''),
        Number(c.grounded), Number(c.coherent), Number(c.plausible), Number(c.fluent), txt_(c.note || ''),
        txt_(b.changed_by), txt_(b.reason), txt_(b.first_speaker), txt_(new Date().toISOString()),
      ]);
      return { ok: true };
    });
  }

  if (b.action === 'export') {
    const ann = annotations_();
    if (isRater) return { ok: true, annotations: Object.values(ann[role]), adjudication: [] };
    if (!bothComplete_(ann)) return { ok: false, error: '두 평가자가 모두 끝난 뒤 내보낼 수 있다 (그 전에는 진행 수만 봅니다)' };
    return {
      ok: true,
      annotations: Object.values(ann.R1).concat(Object.values(ann.R2)),
      adjudication: Object.values(adjudications_()),
    };
  }

  return { ok: false, error: '없는 동작: ' + b.action };
}
