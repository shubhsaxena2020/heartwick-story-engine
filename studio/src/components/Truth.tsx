"use client";

import { useEffect, useRef, useState } from "react";
import { clock, dur, toMs } from "@/lib/format";
import type { Truth, TruthCheck } from "@/lib/types";
import { Lamp, type Tone } from "./ui";

const OVERDUE_MS = 3 * 3600_000; // the laptop re-checks on every change; the stored copy follows the heartbeat

function show(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  return String(v);
}

/** "data verified hh:mm ✓" / "N mismatches": the laptop recomputes key numbers a second way (ffprobe, ffmpeg
 *  loudness, file counts, Kaggle's own quota) on every rebuild; the drill-down lists every check. */
export function TruthBadge({ truth, now }: { truth: Truth | null | undefined; now: number }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const down = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("keydown", key);
    window.addEventListener("mousedown", down);
    return () => {
      window.removeEventListener("keydown", key);
      window.removeEventListener("mousedown", down);
    };
  }, [open]);
  const at = toMs(truth?.at);
  const fails = truth?.n_fail ?? 0;
  let tone: Tone = "faint";
  let text = "data not verified yet";
  if (truth && at != null) {
    const old = now - at > OVERDUE_MS;
    tone = fails ? "err" : old ? "warn" : "ok";
    text = fails ? `${fails} mismatch${fails === 1 ? "" : "es"} · ${clock(at, false)}` : `data verified ${clock(at, false)} ✓${old ? " (old)" : ""}`;
  }
  const rank = (c: TruthCheck) => (c.ok === false ? 0 : c.ok == null ? 1 : 2);
  const rows = [...(truth?.checks ?? [])].sort((a, b) => rank(a) - rank(b));
  return (
    <div ref={box} className="relative">
      <button
        type="button"
        data-testid="truth-badge"
        data-fails={fails}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        title="Key numbers recomputed from the raw files a second, independent way"
        className={`flex items-center gap-2 rounded-[3px] border px-2 py-1 text-[0.6875rem] font-semibold tracking-[0.04em] ${
          tone === "err" ? "border-err/55 bg-err/10 text-err" : tone === "ok" ? "border-ok/45 text-ok" : tone === "warn" ? "border-warn/45 text-warn" : "border-rule-strong text-faint"
        }`}
      >
        <Lamp tone={tone} />
        <span className="tnum">{text}</span>
      </button>
      {open && (
        <div role="dialog" aria-label="Self-check results" className="slide-in absolute left-0 top-full z-40 mt-2 w-[min(46rem,calc(100vw-2rem))] rounded-[3px] border border-rule-strong bg-panel shadow-2xl">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-rule px-3 py-2 text-[0.6875rem]">
            <span className="label !text-ink">Self-check</span>
            {truth ? (
              <span className="text-muted">
                {truth.n_checks} checks · <span className={fails ? "text-err" : "text-ok"}>{fails} mismatches</span> · {truth.n_skip} skipped · ran{" "}
                {at != null ? `${dur(now - at)} ago` : "—"}
              </span>
            ) : (
              <span className="text-muted">The laptop has not reported a self-check yet.</span>
            )}
            <button type="button" onClick={() => setOpen(false)} className="ml-auto text-muted hover:text-ink">
              close
            </button>
          </div>
          <div className="scroll-thin max-h-[60vh] overflow-y-auto">
            <table className="w-full text-left text-[0.6875rem]">
              <thead className="sticky top-0 bg-panel text-faint">
                <tr>
                  <th className="px-3 py-1.5 font-semibold">Check</th>
                  <th className="px-2 py-1.5 text-right font-semibold">Recomputed</th>
                  <th className="px-2 py-1.5 text-right font-semibold">Shown</th>
                  <th className="hidden px-3 py-1.5 font-semibold sm:table-cell">How</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.key} className="border-t border-rule/60 align-top">
                    <td className="px-3 py-1">
                      <span className="flex items-center gap-1.5">
                        <Lamp tone={r.ok === false ? "err" : r.ok == null ? "faint" : "ok"} />
                        <span className={r.ok === false ? "font-semibold text-err" : "text-ink"}>{r.label}</span>
                      </span>
                      <span className="block pl-3.5 text-faint">
                        {r.page}
                        {r.note ? ` · ${r.note}` : ""}
                      </span>
                    </td>
                    <td className="tnum px-2 py-1 text-right">{show(r.expected)}</td>
                    <td className={`tnum px-2 py-1 text-right ${r.ok === false ? "text-err" : ""}`}>
                      {show(r.shown)}
                      {r.tol ? <span className="block text-faint">±{show(r.tol)}</span> : null}
                    </td>
                    <td className="hidden px-3 py-1 text-faint sm:table-cell">{r.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
