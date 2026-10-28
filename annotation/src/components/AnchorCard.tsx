import { ANCHOR_GROUPS, ANCHOR_SHORT, GENRES } from "../constants";
import type { Item } from "../types";

/** Anchors — two groups side by side; the 10 genres toggle on one line. Packed so they fit at a glance. */
export function AnchorCard({ item }: { item: Item }) {
  const byLabel = new Map(item.anchors.map((a) => [a.label, a.value]));
  const placed = new Set(ANCHOR_GROUPS.flatMap((g) => g.labels));
  const extra = item.anchors.filter((a) => !placed.has(a.label)); // Keep anchors that are not in the layout list
  const groups = ANCHOR_GROUPS.map((g, i) => ({
    title: g.title,
    rows: [...g.labels.filter((l) => byLabel.has(l)).map((l) => ({ label: l, value: byLabel.get(l)! })), ...(i === 1 ? extra : [])],
  }));
  const attended = new Set(item.genres_attended);
  const genres = [...GENRES, ...[...item.genres_attended, ...item.genres_not].filter((g) => !GENRES.includes(g))];
  return (
    <div className="card anchor-card">
      <div className="ac-head">
        <h2>앵커</h2>
        <span className="muted">정해진 사실 — 서사가 이것과 어긋나면 모순</span>
      </div>
      <div className="ac-groups">
        {groups.map((g) => (
          <dl key={g.title} className="ac-group">
            <div className="ac-title">{g.title}</div>
            {g.rows.map((r) => (
              <div key={r.label} className="ac-row">
                <dt title={r.label}>{ANCHOR_SHORT[r.label] ?? r.label}</dt>
                <dd>{r.value}</dd>
              </div>
            ))}
          </dl>
        ))}
      </div>
      <div className="ac-genres">
        <span className="ac-title">지난 1년 관람 장르</span>
        {genres.map((g) => (
          <span key={g} className={`gtag ${attended.has(g) ? "yes" : "no"}`} title={attended.has(g) ? "관람함" : "관람 안 함"}>
            {attended.has(g) ? "✓ " : ""}
            {g}
          </span>
        ))}
        {!attended.size && <span className="muted">— 관람한 장르 없음</span>}
      </div>
    </div>
  );
}
