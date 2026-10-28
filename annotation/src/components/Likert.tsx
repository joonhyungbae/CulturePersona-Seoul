/** 1–5 scale. Number and criterion wording on one button — the rater chooses while seeing the criterion. */
export function Likert({
  levels,
  value,
  onChange,
  compact = false,
}: {
  levels: string[];
  value: number;
  onChange: (v: number) => void;
  /** Numbers only (adjudication table). The criterion text shows on hover. */
  compact?: boolean;
}) {
  return (
    <div className={`likert ${compact ? "compact" : ""}`} role="radiogroup">
      {levels.map((lab, i) => {
        const v = i + 1;
        return (
          <button
            key={v}
            type="button"
            role="radio"
            aria-checked={value === v}
            className={value === v ? "on" : ""}
            onClick={() => onChange(v)}
            title={compact ? `${v} · ${lab}` : undefined}
          >
            <b>{v}</b>
            {!compact && <span>{lab}</span>}
          </button>
        );
      })}
    </div>
  );
}

export function Choices({ options, value, onChange }: { options: string[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="choices" role="radiogroup">
      {options.map((o) => (
        <button key={o} type="button" role="radio" aria-checked={value === o} className={`chip ${value === o ? "on" : ""}`} onClick={() => onChange(o)}>
          {o}
        </button>
      ))}
    </div>
  );
}

/** One question per row: the question (plus a short note) on the left, the choices on the right. */
export function QRow({ n, label, hint, children }: { n?: number; label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="qrow">
      <div className="qhead">
        <b>
          {n !== undefined && `${n}. `}
          {label}
        </b>
        {hint && <small>{hint}</small>}
      </div>
      <div className="qbody">{children}</div>
    </div>
  );
}
