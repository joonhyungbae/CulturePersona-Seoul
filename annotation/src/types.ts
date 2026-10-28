export type Rater = "R1" | "R2";
export type Who = Rater | "ADJ" | "OWNER";

export interface Field {
  key: string;
  label: string;
}

/** One item in the public file (public/data/items.json) — anchors and the original text are sealed. */
export interface PublicItem {
  item_id: string;
  practice: boolean;
  fields: Field[];
  masked: Record<string, string>;
  sealed: { iv: string; ct: string };
}

/** An item unsealed with the key received from the server after submitting stage 1. */
export interface Item extends Omit<PublicItem, "sealed"> {
  anchors: Array<{ label: string; value: string }>;
  genres_attended: string[];
  genres_not: string[];
  narrative: Record<string, string>;
}

export interface Stage1 {
  age_guess: string;
  attend_guess: string;
  secs: number;
}

export interface Stage2 {
  contradiction: "yes" | "no" | "";
  contra_fields: string[];
  contra_quote: string;
  grounded: number;
  coherent: number;
  plausible: number;
  fluent: number;
  note: string;
  secs: number;
}

export interface Annotation {
  rater: Rater;
  item_id: string;
  stage1: Stage1;
  stage2: Stage2;
  saved_at: string;
}

/** A rater receives only their own progress (the other rater's progress and the adjudication count are not sent). */
export interface Progress {
  R1?: { done: number; total: number };
  R2?: { done: number; total: number };
  adjudication: { available: boolean; done: number | null; total: number | null };
}

export interface Disagreement {
  item_id: string;
  item: Item;
  key: string;
  R1: Stage2;
  R2: Stage2;
  decided: null | {
    consensus: Omit<Stage2, "secs">;
    changed_by: string;
    reason: string;
    first_speaker: Rater;
  };
}

/** Written answer and explanation for a practice item (data_private/practice_explain.json → Code.gs). */
export interface PracticeExplain {
  contradiction: "yes" | "no";
  fields?: string[];
  grounded?: number;
  text: string;
}
