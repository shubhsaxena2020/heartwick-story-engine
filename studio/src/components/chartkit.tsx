"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

// Hand-rolled SVG charts drawn in real pixels (measured width), so text never stretches. Bars grow in and lines
// draw in on mount; every mark has a <title> with its exact value.

export function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((e) => setW(Math.round(e[0].contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export interface Series {
  key: string;
  label: string;
  color: string;
}

export function Legend({ series }: { series: Series[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[0.6875rem] text-muted">
      {series.map((s) => (
        <span key={s.key} className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2 rounded-[1px]" style={{ background: s.color }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

const AX = "var(--color-faint)";
const GRID = "var(--color-rule)";

/** Vertical stacked bars over categories. */
export function Bars({ rows, series, height = 200, unit = "", empty }: {
  rows: { label: string; values: Record<string, number>; title?: string }[]; series: Series[]; height?: number; unit?: string; empty?: ReactNode;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  if (!rows.length) return <div className="text-[0.75rem] text-muted">{empty ?? "No data yet."}</div>;
  const L = 40, R = 8, T = 10, B = 34;
  const max = niceMax(Math.max(...rows.map((r) => series.reduce((a, s) => a + (r.values[s.key] ?? 0), 0))));
  const iw = Math.max(0, w - L - R);
  const ih = height - T - B;
  const step = iw / rows.length;
  const bw = Math.max(4, Math.min(46, step * 0.62));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((k) => k * max);
  return (
    <div ref={ref} className="w-full">
      {w > 0 && (
        <svg width={w} height={height} role="img" aria-label="bar chart">
          {ticks.map((t) => {
            const y = T + ih - (t / max) * ih;
            return (
              <g key={t}>
                <line x1={L} x2={w - R} y1={y} y2={y} stroke={GRID} strokeDasharray={t ? "2 3" : undefined} />
                <text x={L - 6} y={y + 3} textAnchor="end" fontSize="10" fill={AX} className="tnum">{Math.round(t * 10) / 10}</text>
              </g>
            );
          })}
          {rows.map((r, i) => {
            let acc = 0;
            const x = L + step * i + (step - bw) / 2;
            return (
              <g key={r.label}>
                {series.map((s) => {
                  const v = r.values[s.key] ?? 0;
                  if (!v) return null;
                  const h = (v / max) * ih;
                  const y = T + ih - ((acc + v) / max) * ih;
                  acc += v;
                  return (
                    <rect key={s.key} x={x} y={y} width={bw} height={Math.max(1, h)} fill={s.color} rx="1">
                      <title>{`${r.title ?? r.label} · ${s.label}: ${Math.round(v * 10) / 10}${unit}`}</title>
                      {/* grow from the axis (SMIL: CSS transforms on SVG groups are unreliable) */}
                      <animate attributeName="height" from="0" to={Math.max(1, h)} dur="0.8s" begin={`${i * 0.05}s`} fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1" keyTimes="0;1" />
                      <animate attributeName="y" from={T + ih - (acc - v) / max * ih} to={y} dur="0.8s" begin={`${i * 0.05}s`} fill="freeze" calcMode="spline" keySplines="0.2 0.8 0.2 1" keyTimes="0;1" />
                    </rect>
                  );
                })}
              </g>
            );
          })}
          {rows.map((r, i) => (
            <text key={r.label} x={L + step * i + step / 2} y={height - B + 14} textAnchor="middle" fontSize="10" fill={AX}>
              {r.label.length > Math.max(4, Math.floor(step / 6.5)) ? r.label.slice(0, Math.max(3, Math.floor(step / 6.5) - 1)) + "…" : r.label}
            </text>
          ))}
        </svg>
      )}
    </div>
  );
}

/** Horizontal stacked bars, one row per item. */
export function HBars({ rows, series, unit = "" }: { rows: { label: string; values: Record<string, number> }[]; series: Series[]; unit?: string }) {
  if (!rows.length) return <div className="text-[0.75rem] text-muted">No data yet.</div>;
  const max = Math.max(...rows.map((r) => series.reduce((a, s) => a + (r.values[s.key] ?? 0), 0)), 1e-9);
  return (
    <ul className="flex flex-col gap-2">
      {rows.map((r, i) => {
        const total = series.reduce((a, s) => a + (r.values[s.key] ?? 0), 0);
        return (
          <li key={r.label} className="grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)_3.75rem] items-center gap-2 text-[0.6875rem]">
            <span className="truncate text-muted" title={r.label}>{r.label}</span>
            <span className="flex h-3 overflow-hidden rounded-[1px] bg-rule-strong/50">
              {series.map((s) => {
                const v = r.values[s.key] ?? 0;
                return v ? <span key={s.key} className="grow-x h-full" title={`${s.label}: ${Math.round(v * 10) / 10}${unit}`} style={{ width: `${(v / max) * 100}%`, background: s.color, ["--i" as string]: i }} /> : null;
              })}
            </span>
            <span className="tnum text-right text-ink">{Math.round(total * 10) / 10}{unit}</span>
          </li>
        );
      })}
    </ul>
  );
}

/** Points per category joined by a line, with an optional target band. */
export function Line({ points, height = 180, band, unit = "", color = "var(--color-sodium)", min, max, fmt }: {
  points: { label: string; value: number | null; title?: string }[]; height?: number; band?: [number, number]; unit?: string; color?: string;
  min?: number; max?: number; fmt?: (v: number) => string;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const vals = points.map((p) => p.value).filter((v): v is number => v != null);
  if (!vals.length) return <div className="text-[0.75rem] text-muted">No data yet.</div>;
  const f = fmt ?? ((v: number) => String(Math.round(v * 100) / 100));
  const lo = min ?? Math.min(...vals, ...(band ?? []));
  const hi = max ?? Math.max(...vals, ...(band ?? []));
  const pad = (hi - lo) * 0.15 || 1;
  const y0 = lo - pad, y1 = hi + pad;
  const L = 44, R = 10, T = 10, B = 34;
  const iw = Math.max(0, w - L - R), ih = height - T - B;
  const x = (i: number) => L + (points.length === 1 ? iw / 2 : (i / (points.length - 1)) * iw);
  const y = (v: number) => T + ih - ((v - y0) / (y1 - y0)) * ih;
  const d = points.map((p, i) => (p.value == null ? null : `${x(i).toFixed(1)},${y(p.value).toFixed(1)}`)).filter(Boolean).map((s, i) => `${i ? "L" : "M"}${s}`).join(" ");
  const ticks = [y0 + pad, (y0 + y1) / 2, y1 - pad];
  return (
    <div ref={ref} className="w-full">
      {w > 0 && (
        <svg width={w} height={height} role="img" aria-label="line chart">
          {band && <rect x={L} width={iw} y={y(band[1])} height={Math.max(1, y(band[0]) - y(band[1]))} fill="var(--color-ok)" opacity="0.08" />}
          {band && band.map((b) => <line key={b} x1={L} x2={w - R} y1={y(b)} y2={y(b)} stroke="var(--color-ok)" strokeOpacity="0.45" strokeDasharray="3 3" />)}
          {ticks.map((t) => (
            <text key={t} x={L - 6} y={y(t) + 3} textAnchor="end" fontSize="10" fill={AX} className="tnum">{f(t)}</text>
          ))}
          <path d={d} fill="none" stroke={color} strokeWidth="2" className="draw-line" />
          {points.map((p, i) =>
            p.value == null ? null : (
              <circle key={i} cx={x(i)} cy={y(p.value)} r="3.5" fill="var(--color-panel)" stroke={color} strokeWidth="2" className="rise-in" style={{ ["--i" as string]: i }}>
                <title>{`${p.title ?? p.label}: ${f(p.value)}${unit}`}</title>
              </circle>
            ),
          )}
          {points.map((p, i) => (
            <text key={p.label + i} x={x(i)} y={height - B + 14} textAnchor="middle" fontSize="10" fill={AX}>
              {p.label.length > 14 ? p.label.slice(0, 13) + "…" : p.label}
            </text>
          ))}
        </svg>
      )}
    </div>
  );
}

/** Cumulative step lines over time (e.g. how many videos had reached each stage). */
export function Steps({ series, height = 220 }: { series: (Series & { times: number[] })[]; height?: number }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const all = series.flatMap((s) => s.times);
  if (!all.length) return <div className="text-[0.75rem] text-muted">No data yet.</div>;
  const t0 = Math.min(...all) - 15 * 60_000;
  const t1 = Math.max(Date.now(), Math.max(...all) + 15 * 60_000);
  const maxN = niceMax(Math.max(...series.map((s) => s.times.length)));
  const L = 34, R = 10, T = 10, B = 28;
  const iw = Math.max(0, w - L - R), ih = height - T - B;
  const x = (t: number) => L + ((t - t0) / (t1 - t0)) * iw;
  const y = (n: number) => T + ih - (n / maxN) * ih;
  const ticksX = Array.from({ length: 5 }, (_, i) => t0 + ((t1 - t0) * i) / 4);
  const span = t1 - t0;
  const fmtT = (t: number) =>
    new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", ...(span > 2 * 86400_000 ? { day: "2-digit", month: "short" } : { hour: "2-digit", minute: "2-digit", hourCycle: "h23" }) }).format(t);
  return (
    <div ref={ref} className="w-full">
      {w > 0 && (
        <svg width={w} height={height} role="img" aria-label="videos per stage over time">
          {Array.from({ length: maxN + 1 }, (_, n) => n).filter((n) => maxN <= 10 || n % Math.ceil(maxN / 5) === 0).map((n) => (
            <g key={n}>
              <line x1={L} x2={w - R} y1={y(n)} y2={y(n)} stroke={GRID} strokeDasharray={n ? "2 3" : undefined} />
              <text x={L - 6} y={y(n) + 3} textAnchor="end" fontSize="10" fill={AX}>{n}</text>
            </g>
          ))}
          {ticksX.map((t, i) => (
            <text key={t} x={x(t)} y={height - 8} textAnchor={i === 0 ? "start" : i === ticksX.length - 1 ? "end" : "middle"} fontSize="10" fill={AX}>{fmtT(t)}</text>
          ))}
          {series.map((s) => {
            const ts = [...s.times].sort((a, b) => a - b);
            let d = `M${x(t0)},${y(0)}`;
            ts.forEach((t, i) => {
              d += ` L${x(t).toFixed(1)},${y(i).toFixed(1)} L${x(t).toFixed(1)},${y(i + 1).toFixed(1)}`;
            });
            d += ` L${x(t1)},${y(ts.length)}`;
            return (
              <path key={s.key} d={d} fill="none" stroke={s.color} strokeWidth="2" className="draw-line">
                <title>{`${s.label}: ${ts.length}`}</title>
              </path>
            );
          })}
        </svg>
      )}
    </div>
  );
}
