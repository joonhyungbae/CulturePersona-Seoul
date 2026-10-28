import guideText from "../guide.md?raw";

/** Minimal markdown for the guide: headings, lists, tables, paragraphs, **bold**. No external dependency. */
function inline(s: string) {
  return s.split(/(\*\*[^*]+\*\*)/g).map((p, i) => (p.startsWith("**") ? <b key={i}>{p.slice(2, -2)}</b> : p));
}

/** Pull only sections that start with "## heading" from the guide, verbatim (intro before adjudication — do not write new wording). */
export function guideSections(prefixes: string[]): string {
  const parts = guideText.split(/\n(?=## )/);
  return prefixes
    .map((p) => parts.find((s) => s.startsWith(`## ${p}`)))
    .filter(Boolean)
    .join("\n\n");
}

export function Guide({ text = guideText }: { text?: string } = {}) {
  const lines = text.split("\n");
  const out: JSX.Element[] = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("### ")) out.push(<h3 key={i}>{inline(l.slice(4))}</h3>);
    else if (l.startsWith("## ")) out.push(<h2 key={i}>{inline(l.slice(3))}</h2>);
    else if (l.startsWith("# ")) out.push(<h1 key={i}>{inline(l.slice(2))}</h1>);
    else if (l.startsWith("- ")) {
      const items: string[] = [];
      while (i < lines.length && lines[i].startsWith("- ")) items.push(lines[i++].slice(2));
      out.push(<ul key={i}>{items.map((t, k) => <li key={k}>{inline(t)}</li>)}</ul>);
      continue;
    } else if (/^\d+\. /.test(l)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\. /.test(lines[i])) items.push(lines[i++].replace(/^\d+\. /, ""));
      out.push(<ol key={i}>{items.map((t, k) => <li key={k}>{inline(t)}</li>)}</ol>);
      continue;
    } else if (l.startsWith("|")) {
      const rows: string[][] = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = lines[i].split("|").slice(1, -1).map((c) => c.trim());
        if (!cells.every((c) => /^-+$/.test(c))) rows.push(cells);
        i++;
      }
      out.push(
        <table key={i}>
          <thead><tr>{rows[0].map((c, k) => <th key={k}>{inline(c)}</th>)}</tr></thead>
          <tbody>{rows.slice(1).map((r, k) => <tr key={k}>{r.map((c, j) => <td key={j}>{inline(c)}</td>)}</tr>)}</tbody>
        </table>,
      );
      continue;
    } else if (l.trim()) out.push(<p key={i}>{inline(l)}</p>);
    i++;
  }
  return <div className="card guide">{out}</div>;
}
