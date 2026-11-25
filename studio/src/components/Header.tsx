"use client";

import { useEffect, useState } from "react";
import { clock, dur, stamp } from "@/lib/format";
import type { LiveReading, Snapshot } from "@/lib/types";
import type { LiveStatus } from "@/lib/useLive";
import { Mark } from "./Mark";
import { TruthBadge } from "./Truth";
import { Lamp, useNow, type Tone } from "./ui";

export const TABS = [
  { key: "overview", label: "Overview" },
  { key: "videos", label: "Videos" },
  { key: "kaggle", label: "Kaggle" },
  { key: "charts", label: "Charts" },
  { key: "errors", label: "Errors" },
  { key: "research", label: "Research" },
  { key: "channel", label: "Channel" },
] as const;
export type TabKey = (typeof TABS)[number]["key"];

export const LIVE_FRESH_MS = 10_000;

export function liveness(status: LiveStatus, lastAt: number | null, lastSeen: number | null, now: number) {
  if (status === "live" && lastAt && now - lastAt < LIVE_FRESH_MS) return { tone: "live" as Tone, text: "Live", live: true };
  if (status === "connecting" && !lastSeen) return { tone: "faint" as Tone, text: "Connecting", live: false };
  if (!lastSeen) return { tone: "err" as Tone, text: "Laptop offline", live: false };
  return { tone: (now - lastSeen > 30 * 60_000 ? "err" : "warn") as Tone, text: `Laptop offline · seen ${dur(now - lastSeen)} ago`, live: false };
}

export function Header({ snapshot, status, lastAt, lastSeen, reading, tab, onTab, errCount }: {
  snapshot: Snapshot | null; status: LiveStatus; lastAt: number | null; lastSeen: number | null; reading: LiveReading | null;
  tab: TabKey; onTab: (t: TabKey) => void; errCount: number;
}) {
  const now = useNow(1000);
  const st = liveness(status, lastAt, lastSeen, now);
  const [motion, setMotion] = useState(true);
  useEffect(() => {
    const id = requestAnimationFrame(() => setMotion(document.documentElement.classList.contains("motion")));
    return () => cancelAnimationFrame(id);
  }, []);
  const toggleMotion = () => {
    const on = !motion;
    setMotion(on);
    document.documentElement.classList.toggle("motion", on);
    try {
      localStorage.setItem("hws_motion", on ? "1" : "0");
    } catch {
      /* private window */
    }
  };
  const cpu = st.live ? reading?.cpu_pct : null;
  return (
    <header className="relative z-30 border-b border-rule bg-bg/92 backdrop-blur-md sm:sticky sm:top-0">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:bg-sodium focus:px-2 focus:text-bg">
        Skip to content
      </a>
      <div className="mx-auto flex max-w-[80rem] flex-wrap items-center gap-x-5 gap-y-2 px-4 pt-3 sm:px-6">
        <div className="flex items-center gap-3">
          <Mark />
          <div className="flex items-baseline gap-2">
            <span className="display text-[1.375rem] font-extrabold uppercase tracking-[0.01em] sm:text-[1.625rem]">Heartwick</span>
            <span className="text-[0.75rem] font-semibold tracking-[0.18em] text-sodium">{"//"}STUDIO</span>
          </div>
        </div>
        <div
          role="status"
          aria-live="polite"
          data-testid="live-pill"
          data-live={st.live ? "1" : "0"}
          title={st.live ? "The laptop answers through its live tunnel (polled every 3 s)" : "Showing the last snapshot the laptop posted. Reconnects by itself."}
          className={`flex max-w-full items-center gap-2 rounded-[3px] border px-2 py-1 text-[0.6875rem] font-semibold uppercase tracking-[0.12em] ${
            st.live ? "border-live/60 bg-live/10 text-live" : st.tone === "warn" ? "border-warn/45 text-warn" : st.tone === "err" ? "border-err/50 bg-err/10 text-err" : "border-rule-strong text-faint"
          }`}
        >
          <Lamp tone={st.tone} pulse={st.live} />
          <span className="tnum truncate">{st.text}</span>
        </div>
        <TruthBadge truth={snapshot?.truth} now={now} />
        <div className="flex w-full items-center justify-between gap-x-4 gap-y-2 sm:ml-auto sm:w-auto sm:justify-end sm:gap-x-5">
          <div className="flex flex-col leading-tight">
            <span className="tnum whitespace-nowrap text-[0.8125rem] font-semibold" title="India (Kolkata)">
              <span className="label mr-1.5">IST</span>
              {clock(now)}
            </span>
            <span className="tnum hidden text-[0.625rem] text-faint sm:block" title="When the shown data was built on the laptop">
              data {snapshot ? stamp(snapshot.generated_at) : "—"}
            </span>
          </div>
          <div className="flex flex-col gap-0.5" title="Main laptop CPU (cap 75 %), read on each live poll">
            <span className="label">Laptop CPU</span>
            <span data-testid="hdr-cpu" className={`tnum text-[0.75rem] font-semibold ${cpu == null ? "text-faint" : cpu > 75 ? "text-err" : "text-ink"}`}>
              {cpu == null ? "—" : `${Math.round(cpu)}%`}
              <span className="font-normal text-faint"> / 75</span>
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={toggleMotion}
              aria-pressed={motion}
              title="Continuous animations (pulsing lamps, moving stripes). Off saves CPU on a slow laptop."
              className="rounded-[3px] border border-rule-strong px-2 py-1 text-[0.625rem] font-bold uppercase tracking-[0.12em] text-muted hover:text-ink"
            >
              Motion {motion ? "on" : "off"}
            </button>
            <form method="post" action="/api/logout">
              <button type="submit" className="rounded-[3px] px-2 py-1 text-[0.625rem] font-bold uppercase tracking-[0.12em] text-faint hover:text-ink">
                Log out
              </button>
            </form>
          </div>
        </div>
      </div>
      <nav aria-label="Sections" className="mx-auto max-w-[80rem] px-4 sm:px-6">
        <ul className="scroll-thin -mb-px mt-2 flex gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <li key={t.key}>
              <button
                type="button"
                onClick={() => onTab(t.key)}
                aria-current={tab === t.key ? "page" : undefined}
                data-tab={t.key}
                className={`whitespace-nowrap border-b-2 px-3 py-2 text-[0.6875rem] font-semibold uppercase tracking-[0.14em] transition-colors ${
                  tab === t.key ? "border-sodium text-ink" : "border-transparent text-muted hover:text-ink"
                }`}
              >
                {t.label}
                {t.key === "errors" && errCount > 0 && <span className="tnum ml-1.5 rounded-[2px] bg-err/15 px-1 text-err">{errCount}</span>}
              </button>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
