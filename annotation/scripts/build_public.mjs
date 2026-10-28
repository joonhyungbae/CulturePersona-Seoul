// Build the public item file and Apps Script.
//
// So that stage-1 blinding is **enforced by the server**, the public file (public/data/items.json → GitHub Pages) contains
//   - only the masked narrative (masked) and field names in plaintext,
//   - anchors and the original narrative **encrypted per item with a different AES-256-GCM key** (sealed).
// Keys live only in apps_script/Code.gs, and a rater receives an item's key only after submitting that item's stage-1 answer (reveal).
// Opening the public file in developer tools therefore cannot show the anchors before stage 1.
//
// input  data_private/items_full.json   full originals (written by anno_build_sample.py in the main repository)
//        data_private/item_keys.json    per-item keys (created if missing — do not change them after a deploy)
// output public/data/items.json         plaintext (masked narrative) + seal (anchors and original)
//        apps_script/Code.gs            Code.template.js + item list + keys
// data_private/ stays in the private repository only and is not published to Pages (outside public/).
import { createCipheriv, randomBytes } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";

const at = (p) => new URL(`../${p}`, import.meta.url);
const full = JSON.parse(readFileSync(at("data_private/items_full.json"), "utf-8"));

let keys = {};
if (existsSync(at("data_private/item_keys.json"))) keys = JSON.parse(readFileSync(at("data_private/item_keys.json"), "utf-8"));
let added = 0;
for (const it of full) {
  if (!keys[it.item_id]) {
    keys[it.item_id] = randomBytes(32).toString("base64");
    added += 1;
  }
}
// Keep keys only for the current items (if the sample is redrawn, old item keys would be carried uselessly in Code.gs).
keys = Object.fromEntries(full.map((it) => [it.item_id, keys[it.item_id]]));
writeFileSync(at("data_private/item_keys.json"), JSON.stringify(keys, null, 1));

// Pad every plaintext to the same length so ciphertext length does not leak anchor length (leakage audit: attendance classification 0.747→0.831).
// JSON.parse ignores the spaces after the JSON.
const inner = (it) => JSON.stringify({ anchors: it.anchors, genres_attended: it.genres_attended, genres_not: it.genres_not, narrative: it.narrative });
const PAD_TO = Math.ceil(Math.max(...full.map((it) => Buffer.byteLength(inner(it), "utf-8"))) / 1024) * 1024 + 1024;

function seal(plain, keyB64) {
  const buf = Buffer.from(plain, "utf-8");
  const padded = Buffer.concat([buf, Buffer.alloc(PAD_TO - buf.length, 0x20)]);
  const iv = randomBytes(12);
  const c = createCipheriv("aes-256-gcm", Buffer.from(keyB64, "base64"), iv);
  const ct = Buffer.concat([c.update(padded), c.final(), c.getAuthTag()]); // Web Crypto expects ct‖tag
  return { iv: iv.toString("base64"), ct: ct.toString("base64") };
}

const pub = full.map((it) => ({
  item_id: it.item_id,
  practice: it.practice,
  fields: it.fields,
  masked: it.masked,
  sealed: seal(inner(it), keys[it.item_id]),
}));
writeFileSync(at("public/data/items.json"), JSON.stringify(pub));

// Check that original and anchor strings did not leak into the public file.
const text = readFileSync(at("public/data/items.json"), "utf-8");
// Unmasked fields match the original, so skip them and look only for **originals of masked fields**.
const leak = full.find(
  (it) =>
    Object.entries(it.narrative).some(([k, t]) => t && t !== it.masked[k] && text.includes(t)) ||
    text.includes(`"anchors"`) ||
    text.includes(`"genres_attended"`),
);
if (leak) throw new Error(`공개 파일에 원문이 샜다: ${leak.item_id}`);

const mainAll = full.filter((i) => !i.practice).map((i) => i.item_id);
const practice = full.filter((i) => i.practice).map((i) => i.item_id);
// Smaller annotation sample (ADR-039): if data_private/main_subset.json exists, shrink the server's main-item list to that subset.
// Completion, progress counts, and whether a save is allowed all follow MAIN_IDS, so finishing the subset on both sides locks independent scores and opens adjudication.
const subsetPath = at("data_private/main_subset.json");
const subset = existsSync(subsetPath) ? new Set(JSON.parse(readFileSync(subsetPath, "utf-8"))) : null;
if (subset) {
  const unknown = [...subset].filter((id) => !mainAll.includes(id));
  if (unknown.length) throw new Error(`부분집합에 본 문항이 아닌 id ${unknown.length} 개: ${unknown.slice(0, 5).join(", ")}`);
}
const main = subset ? mainAll.filter((id) => subset.has(id)) : mainAll;
const explainPath = at("data_private/practice_explain.json");
const explain = existsSync(explainPath) ? JSON.parse(readFileSync(explainPath, "utf-8")) : {};
const noExplain = practice.filter((id) => !explain[id]);
if (noExplain.length) console.warn(`연습 해설이 없는 문항 ${noExplain.length}: ${noExplain.join(", ")}`);
if (new Set(pub.map((p) => p.sealed.ct.length)).size !== 1) throw new Error("봉인문 길이가 문항마다 다르다");
const namesPath = at("data_private/rater_names.json");
const raterNames = existsSync(namesPath) ? JSON.parse(readFileSync(namesPath, "utf-8")) : {};
const tpl = readFileSync(at("apps_script/Code.template.js"), "utf-8");
const out = tpl
  .replace("/*__MAIN_IDS__*/ []", JSON.stringify(main))
  .replace("/*__PRACTICE_IDS__*/ []", JSON.stringify(practice))
  .replace("/*__ITEM_KEYS__*/ {}", JSON.stringify(keys))
  .replace("/*__PRACTICE_EXPLAIN__*/ {}", JSON.stringify(explain))
  .replace("/*__RATER_NAMES__*/ {}", JSON.stringify(raterNames));
if (/__(MAIN_IDS|PRACTICE_IDS|ITEM_KEYS|PRACTICE_EXPLAIN|RATER_NAMES)__/.test(out)) throw new Error("자리표시자가 남았다");
writeFileSync(at("apps_script/Code.gs"), out);
console.log(`public/data/items.json 봉인 ${pub.length} · Code.gs 본 문항 ${main.length}${subset ? ` (부분집합, 전체 ${mainAll.length})` : ""} · 연습 ${practice.length} · 새 열쇠 ${added}`);
