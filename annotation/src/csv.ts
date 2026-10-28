import type { Annotation } from "./types";

const HEADER = [
  "rater", "item_id", "age_guess", "attend_guess", "stage1_secs",
  "contradiction", "contra_fields", "contra_quote", "grounded", "coherent", "plausible", "fluent",
  "note", "stage2_secs", "saved_at",
];

function cell(v: unknown): string {
  const s = Array.isArray(v) ? v.join("|") : v == null ? "" : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** CSV for spreadsheets. A UTF-8 BOM is prepended so Excel does not garble Korean text. */
export function toCsv(rows: Annotation[], adjudication: Array<Record<string, unknown>> = []): string {
  const lines = [HEADER.join(",")];
  for (const a of rows) {
    const s1 = a.stage1;
    const s2 = a.stage2;
    lines.push(
      [a.rater, a.item_id, s1.age_guess, s1.attend_guess, s1.secs, s2.contradiction, s2.contra_fields,
        s2.contra_quote, s2.grounded, s2.coherent, s2.plausible, s2.fluent, s2.note, s2.secs, a.saved_at].map(cell).join(","),
    );
  }
  for (const j of adjudication) {
    const c = (j.consensus ?? {}) as Record<string, unknown>;
    lines.push(
      ["CONSENSUS", j.item_id, "", "", "", "", c.contradiction, c.contra_fields, c.contra_quote, c.grounded, c.coherent,
        c.plausible, c.fluent, `changed_by=${j.changed_by}; first=${j.first_speaker}; ${j.reason}`, "", j.saved_at]
        .map(cell).join(","),
    );
  }
  return "﻿" + lines.join("\r\n") + "\r\n";
}

export function download(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
