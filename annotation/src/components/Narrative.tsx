import type { Field } from "../types";

/** When the item changes, scroll the left and right panes (and the page on a narrow screen) to the top. */
export function scrollPanesTop(): void {
  window.scrollTo({ top: 0 });
  document.querySelectorAll<HTMLElement>(".pane").forEach((el) => (el.scrollTop = 0));
}

/** Show a masked spot (○○) as a gray chip — do not reveal what was removed. */
function withMasks(text: string) {
  const parts = text.split("○○");
  return parts.flatMap((p, i) => (i === 0 ? [p] : [<span key={i} className="mask">○○</span>, p]));
}

/** Find the rater's evidence sentence in the paragraph and highlight it yellow (only if it appears verbatim). */
function withHighlights(text: string, quotes: string[]) {
  const qs = quotes.map((q) => q.trim().replace(/^["“”']+|["“”']+$/g, "")).filter((q) => q.length >= 4 && text.includes(q));
  if (!qs.length) return [text];
  const out: (string | JSX.Element)[] = [];
  let rest = text;
  let n = 0;
  while (rest) {
    const hits = qs.map((q) => [rest.indexOf(q), q] as const).filter(([at]) => at >= 0).sort((a, b) => a[0] - b[0]);
    if (!hits.length) break;
    const [at, q] = hits[0];
    out.push(rest.slice(0, at), <mark key={n++} className="qmark">{q}</mark>);
    rest = rest.slice(at + q.length);
  }
  out.push(rest);
  return out;
}

/**
 * The 12 narrative fields as flowing text in two columns, without boxes. The field name is a small label before the paragraph.
 * In contradiction-marking mode (onToggle), click a paragraph to select it.
 */
export function Narrative({
  fields,
  text,
  masked = false,
  picked = [],
  onToggle,
  marks = {},
  quotes = [],
}: {
  fields: Field[];
  text: Record<string, string>;
  masked?: boolean;
  picked?: string[];
  onToggle?: (key: string) => void;
  /** Adjudication: paragraph a rater marked as a contradiction → show which rater marked it */
  marks?: Record<string, string[]>;
  /** Adjudication: evidence sentences the raters wrote — find them in the narrative and highlight */
  quotes?: string[];
}) {
  return (
    <div className={`narr ${onToggle ? "pickable" : ""}`}>
      {fields.map((f) => (
        <p
          key={f.key}
          className={`para ${picked.includes(f.key) ? "picked" : ""}`}
          onClick={onToggle ? () => onToggle(f.key) : undefined}
          title={onToggle ? "누르면 모순 항목으로 표시됩니다" : undefined}
        >
          <span className="plabel">{f.label}</span>
          {marks[f.key]?.map((who) => (
            <span key={who} className="rmark">{who} 지목</span>
          ))}
          {masked ? withMasks(text[f.key] ?? "") : withHighlights(text[f.key] ?? "", quotes)}
        </p>
      ))}
    </div>
  );
}
