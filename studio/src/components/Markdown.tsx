import type { ReactNode } from "react";

// A small, safe Markdown renderer (no HTML pass-through): headings, paragraphs, lists, block quotes, fenced code,
// tables, rules, links (http/https only), bold, italic, inline code. Enough for research reports.

function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const rx = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\s][^*]*\*|_[^_\s][^_]*_)|(\[[^\]]+\]\([^)\s]+\))|(https?:\/\/[^\s)<]+)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = rx.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    const k = `${key}-${i++}`;
    if (m[1]) out.push(<code key={k} className="rounded-[2px] bg-raised px-1 text-[0.92em] text-sodium">{t.slice(1, -1)}</code>);
    else if (m[2]) out.push(<strong key={k} className="font-semibold text-ink">{inline(t.slice(2, -2), k)}</strong>);
    else if (m[3]) out.push(<em key={k}>{inline(t.slice(1, -1), k)}</em>);
    else if (m[4]) {
      const mm = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(t)!;
      const href = /^https?:\/\//.test(mm[2]) ? mm[2] : null;
      out.push(href ? <a key={k} href={href} target="_blank" rel="noopener noreferrer" className="text-sodium hover:underline">{inline(mm[1], k)}</a> : <span key={k}>{mm[1]}</span>);
    } else out.push(<a key={k} href={t} target="_blank" rel="noopener noreferrer" className="break-all text-sodium hover:underline">{t}</a>);
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

const cells = (l: string) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

export function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let n = 0;
  while (i < lines.length) {
    const l = lines[i];
    const k = `b${n++}`;
    if (/^```/.test(l)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) body.push(lines[i++]);
      i++;
      blocks.push(<pre key={k} className="scroll-thin overflow-x-auto rounded-[3px] border border-rule bg-bg p-3 text-[0.75rem]">{body.join("\n")}</pre>);
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(l);
    if (h) {
      const lvl = h[1].length;
      const cls = lvl === 1 ? "display text-[1.75rem] font-bold uppercase" : lvl === 2 ? "display mt-2 text-[1.3rem] font-bold uppercase text-sodium" : "mt-1 text-[0.9375rem] font-semibold text-ink";
      blocks.push(<div key={k} role="heading" aria-level={lvl} className={cls}>{inline(h[2], k)}</div>);
      i++;
      continue;
    }
    if (/^\s*(-\s*){3,}$|^\s*(\*\s*){3,}$|^\s*(_\s*){3,}$/.test(l)) {
      blocks.push(<hr key={k} className="border-rule" />);
      i++;
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      const head = cells(l);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      blocks.push(
        <div key={k} className="scroll-thin overflow-x-auto">
          <table className="w-full text-left text-[0.75rem]">
            <thead><tr className="border-b border-rule-strong">{head.map((c, j) => <th key={j} className="px-2 py-1.5 font-semibold text-ink">{inline(c, `${k}h${j}`)}</th>)}</tr></thead>
            <tbody>{rows.map((r, ri) => <tr key={ri} className="border-b border-rule/60">{r.map((c, j) => <td key={j} className="px-2 py-1.5 align-top text-muted">{inline(c, `${k}r${ri}c${j}`)}</td>)}</tr>)}</tbody>
          </table>
        </div>,
      );
      continue;
    }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(l)) {
      const ordered = /^\s*\d/.test(l);
      const items: string[] = [];
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
        let item = lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, "");
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) item += " " + lines[i++].trim();
        items.push(item);
      }
      const Tag = ordered ? "ol" : "ul";
      blocks.push(<Tag key={k} className={`ml-5 flex flex-col gap-1 ${ordered ? "list-decimal" : "list-disc"} marker:text-sodium-deep`}>{items.map((it, j) => <li key={j}>{inline(it, `${k}i${j}`)}</li>)}</Tag>);
      continue;
    }
    if (/^>\s?/.test(l)) {
      const q: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) q.push(lines[i++].replace(/^>\s?/, ""));
      blocks.push(<blockquote key={k} className="border-l-2 border-sodium-deep pl-3 text-muted">{inline(q.join(" "), k)}</blockquote>);
      continue;
    }
    if (!l.trim()) {
      i++;
      continue;
    }
    const p: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|>|\s*([-*+]|\d+[.)])\s+|\s*\|.*\|\s*$)/.test(lines[i])) p.push(lines[i++]);
    if (!p.length) p.push(lines[i++]);
    blocks.push(<p key={k} className="leading-relaxed">{inline(p.join(" "), k)}</p>);
  }
  return <div className="flex max-w-[78ch] flex-col gap-3 text-[0.8125rem] text-muted">{blocks}</div>;
}
