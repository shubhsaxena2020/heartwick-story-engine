"use client";

import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { fullStamp, num, stamp } from "@/lib/format";
import type { StudioAction } from "@/lib/types";

// ---------------------------------------------------------------- context

export interface StudioApi {
  /** queue a write on Vercel, then hand it to the laptop if it is online. */
  act: (a: Omit<StudioAction, "id" | "at">) => Promise<{ ok: boolean; queued: boolean; instant: boolean; error?: string; id?: string }>;
  /** tunnel URL for a media key (fallback), or null when the laptop is not reachable */
  tunnelMedia: (key: string) => string | null;
  pendingIds: Set<string>;
  go: (tab: string, anchor?: string) => void;
}
export const StudioCtx = createContext<StudioApi>({
  act: async () => ({ ok: false, queued: false, instant: false }),
  tunnelMedia: () => null,
  pendingIds: new Set(),
  go: () => {},
});
export const useStudio = () => useContext(StudioCtx);

export const InitialNowCtx = createContext<number>(0);

export function useNow(intervalMs = 1000): number {
  const initial = useContext(InitialNowCtx);
  const [now, setNow] = useState(initial);
  useEffect(() => {
    const first = setTimeout(() => setNow(Date.now()), 0);
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, [intervalMs]);
  return now;
}

// ---------------------------------------------------------------- motion

function reduced(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return true;
  }
}

/** Count toward `target` over `ms`. A timer lands the exact value at ms+100 even if frames are throttled. */
export function useTween(target: number | null | undefined, ms = 450): number | null {
  // starts at 0 on the server and the client alike (no hydration mismatch), then counts up to the value
  const [v, setV] = useState<number | null>(target == null ? null : 0);
  const from = useRef<number | null>(target == null ? null : 0);
  useEffect(() => {
    if (target == null || !Number.isFinite(target)) {
      from.current = null;
      const id = requestAnimationFrame(() => setV(null));
      return () => cancelAnimationFrame(id);
    }
    const start = from.current ?? 0;
    if (start === target || reduced()) {
      from.current = target;
      const id = requestAnimationFrame(() => setV(target));
      return () => cancelAnimationFrame(id);
    }
    const t0 = performance.now();
    let raf = 0;
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / ms);
      const cur = start + (target - start) * (1 - Math.pow(1 - k, 3));
      from.current = cur;
      setV(cur);
      if (k < 1) raf = requestAnimationFrame(step);
      else from.current = target;
    };
    raf = requestAnimationFrame(step);
    const land = setTimeout(() => {
      cancelAnimationFrame(raf);
      from.current = target;
      setV(target);
    }, ms + 100);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(land);
    };
  }, [target, ms]);
  return v;
}

/** A number that counts up to its value and flashes when it changes. */
export function Num({ value, digits = 0, className = "", testid }: { value: number | null | undefined; digits?: number; className?: string; testid?: string }) {
  const v = useTween(value);
  const [flash, setFlash] = useState(false);
  const prev = useRef(value);
  useEffect(() => {
    if (prev.current != null && value != null && prev.current !== value) {
      const on = requestAnimationFrame(() => setFlash(true));
      const off = setTimeout(() => setFlash(false), 900);
      prev.current = value;
      return () => {
        cancelAnimationFrame(on);
        clearTimeout(off);
      };
    }
    prev.current = value;
  }, [value]);
  const shown = v == null ? null : digits ? v : Math.round(v);
  return (
    <span data-testid={testid} data-value={value ?? ""} className={`tnum ${flash ? "num-flash" : ""} ${className}`}>
      {num(shown, digits)}
    </span>
  );
}

// ---------------------------------------------------------------- primitives

export type Tone = "ok" | "warn" | "err" | "live" | "sodium" | "faint" | "ink" | "cpu";
export const toneText: Record<Tone, string> = {
  ok: "text-ok", warn: "text-warn", err: "text-err", live: "text-live", sodium: "text-sodium", faint: "text-faint", ink: "text-ink", cpu: "text-cpu",
};
export const toneBg: Record<Tone, string> = {
  ok: "bg-ok", warn: "bg-warn", err: "bg-err", live: "bg-live", sodium: "bg-sodium", faint: "bg-faint", ink: "bg-ink", cpu: "bg-cpu",
};

export function Panel({ title, meta, right, children, className = "", bodyClass = "p-4", id }: {
  title: string; meta?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; bodyClass?: string; id?: string;
}) {
  return (
    <section id={id} aria-label={title} className={`panel bracket min-w-0 ${className}`}>
      <header className="flex min-h-[2.625rem] flex-wrap items-center gap-x-3 gap-y-1 border-b border-rule px-4 py-2">
        <h3 className="label !text-ink">{title}</h3>
        {meta != null && <span className="label !normal-case !tracking-[0.06em] text-faint">{meta}</span>}
        {right && <div className="ml-auto flex items-center gap-2">{right}</div>}
      </header>
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

/** A group of panels under one heading: the page's top level. */
export function Section({ title, hint, right, children, id }: { title: string; hint?: ReactNode; right?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section id={id} aria-label={title} className="flex min-w-0 flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 pt-1">
        <h2 className="display text-[1.25rem] font-bold uppercase tracking-[0.03em] text-ink">{title}</h2>
        {hint != null && <span className="text-[0.6875rem] text-faint">{hint}</span>}
        <span aria-hidden className="hidden h-px min-w-8 flex-1 self-center bg-rule sm:block" />
        {right}
      </div>
      {children}
    </section>
  );
}

export function Split({ children, side = "23.75rem" }: { children: ReactNode; side?: string }) {
  return (
    <div className="grid gap-4 lg:[grid-template-columns:minmax(0,1fr)_var(--side)]" style={{ ["--side" as string]: side }}>
      {children}
    </div>
  );
}

export function Lamp({ tone, pulse = false, label }: { tone: Tone; pulse?: boolean; label?: string }) {
  return (
    <span
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={`inline-block size-[0.4375rem] shrink-0 rounded-full ${toneBg[tone]} ${toneText[tone]} ${pulse ? "lamp-pulse" : ""}`}
      style={{ boxShadow: tone === "faint" ? undefined : "0 0 8px currentColor" }}
    />
  );
}

export function Pill({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex shrink-0 items-center gap-1.5 rounded-[2px] border border-current/35 px-1.5 py-[1px] text-[0.625rem] font-semibold uppercase leading-[1rem] tracking-[0.1em] ${toneText[tone]}`}>
      {children}
    </span>
  );
}

/** Thin tick meter. Ticks past `limit` are drawn as a no-go zone. */
export function TickMeter({ value, max, ticks = 40, limit, tone = "sodium", height = 14, label }: {
  value: number | null | undefined; max: number; ticks?: number; limit?: number | null; tone?: Tone; height?: number; label: string;
}) {
  const v = value == null || !Number.isFinite(value) ? null : Math.max(0, value);
  const filled = v == null ? 0 : Math.round(Math.min(1, v / Math.max(max, 1e-9)) * ticks);
  const limitTick = limit != null ? Math.round((limit / max) * ticks) : null;
  return (
    <div role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={max} aria-valuenow={v ?? undefined} className="flex w-full items-end gap-[2px]" style={{ height }}>
      {Array.from({ length: ticks }, (_, i) => {
        const on = i < filled;
        const beyond = limitTick != null && i >= limitTick;
        const cls = on ? (beyond ? toneBg.err : toneBg[tone]) : beyond ? "bg-err/25" : "bg-rule-strong/70";
        return <span key={i} className={`h-full min-w-[1px] flex-1 rounded-[1px] transition-colors duration-500 ${cls}`} style={{ transitionDelay: `${i * 12}ms` }} />;
      })}
    </div>
  );
}

/** Progress bar with a count, e.g. narration lines 12/16. Striped while `running`. */
export function Bar({ done, total, tone = "sodium", running = false, label }: { done: number; total: number; tone?: Tone; running?: boolean; label: string }) {
  const p = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  return (
    <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={total} aria-valuenow={done} className="h-[0.3125rem] w-full overflow-hidden rounded-[1px] bg-rule-strong/70">
      <div className={`grow-bar h-full ${toneBg[tone]} ${running ? "run-stripes" : ""}`} style={{ width: `${p}%` }} />
    </div>
  );
}

/** Ring gauge (e.g. GPU hours used of the weekly quota). */
export function Ring({ value, max, size = 64, tone = "sodium", label, children }: { value: number; max: number; size?: number; tone?: Tone; label: string; children?: ReactNode }) {
  const r = 15.9155;
  const v = useTween(value, 900) ?? 0;
  const p = Math.max(0, Math.min(100, (v / Math.max(max, 1e-9)) * 100));
  const stroke = { sodium: "var(--color-sodium)", ok: "var(--color-ok)", warn: "var(--color-warn)", err: "var(--color-err)", cpu: "var(--color-cpu)" }[tone as string] ?? "var(--color-sodium)";
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox="0 0 36 36" role="img" aria-label={label} className="-rotate-90">
        <circle cx="18" cy="18" r={r} fill="none" stroke="var(--color-rule-strong)" strokeWidth="3" />
        <circle cx="18" cy="18" r={r} fill="none" stroke={stroke} strokeWidth="3" strokeDasharray={`${p} ${100 - p}`} strokeLinecap="butt" />
      </svg>
      {children && <div className="absolute inset-0 grid place-items-center">{children}</div>}
    </div>
  );
}

export function Chips<T extends string>({ value, options, onChange, label, counts, names }: {
  value: T; options: readonly T[]; onChange: (v: T) => void; label: string; counts?: Partial<Record<T, number>>; names?: Partial<Record<T, string>>;
}) {
  return (
    <div role="group" aria-label={label} className="flex flex-wrap gap-1">
      {options.map((k) => (
        <button
          key={k}
          type="button"
          aria-pressed={value === k}
          onClick={() => onChange(k)}
          className={`whitespace-nowrap rounded-[2px] px-2 py-[0.1875rem] text-[0.65625rem] font-semibold uppercase tracking-[0.1em] transition-colors ${
            value === k ? "bg-sodium text-[#1a0d04]" : "text-muted hover:bg-raised hover:text-ink"
          }`}
        >
          {names?.[k] ?? k}
          {counts?.[k] != null && <span className="tnum ml-1 opacity-70">{counts[k]}</span>}
        </button>
      ))}
    </div>
  );
}

export function Empty({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex min-h-[5.5rem] flex-col items-start justify-center gap-2 rounded-[3px] border border-dashed border-rule-strong px-4 py-5 text-[0.75rem] text-muted">
      <div className="max-w-[62ch]">{children}</div>
      {action}
    </div>
  );
}

export function Time({ iso, className = "" }: { iso: string | null | undefined; className?: string }) {
  if (!iso) return <span className={className}>—</span>;
  return (
    <time dateTime={iso} title={fullStamp(iso)} className={`tnum whitespace-nowrap ${className}`}>
      {stamp(iso)}
    </time>
  );
}

/** One labelled figure: big display number with a unit and a small caption. */
export function Stat({ label, value, unit, digits = 0, sub, tone = "ink", testid }: {
  label: string; value: number | null | undefined; unit?: string; digits?: number; sub?: ReactNode; tone?: Tone; testid?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="label">{label}</div>
      <div className={`display mt-1 text-[2.25rem] font-bold ${toneText[tone]}`}>
        <Num value={value} digits={digits} testid={testid} />
        {unit && <span className="ml-1 font-mono text-[0.75rem] font-normal text-muted">{unit}</span>}
      </div>
      {sub != null && <div className="mt-1 text-[0.6875rem] text-faint">{sub}</div>}
    </div>
  );
}

/** Login-protected media from Blob (/api/media/<key>); falls back to the laptop's tunnel if the Blob copy is missing. */
export function useMediaSrc(key: string | null | undefined): [string | null, () => void] {
  const { tunnelMedia } = useStudio();
  const [fallback, setFallback] = useState(false);
  const src = !key ? null : fallback ? tunnelMedia(key) : `/api/media/${key}`;
  return [src, () => setFallback(true)];
}

export function Img({ mkey, alt, className = "" }: { mkey: string | null | undefined; alt: string; className?: string }) {
  const [src, fail] = useMediaSrc(mkey);
  const [broken, setBroken] = useState(false);
  if (!src || broken) {
    return <div className={`grid place-items-center bg-raised text-[0.625rem] uppercase tracking-[0.1em] text-faint ${className}`}>no image yet</div>;
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={alt}
      loading="lazy"
      decoding="async"
      onError={() => (src.startsWith("/api/") ? fail() : setBroken(true))}
      className={`bg-raised object-cover ${className}`}
    />
  );
}

export function Btn({ children, onClick, tone = "default", disabled, title, testid, type = "button" }: {
  children: ReactNode; onClick?: () => void; tone?: "default" | "primary" | "warn" | "ghost"; disabled?: boolean; title?: string; testid?: string; type?: "button" | "submit";
}) {
  const cls = {
    primary: "bg-sodium text-[#1a0d04] hover:brightness-110 border-sodium",
    warn: "border-warn/60 text-warn hover:bg-warn/10",
    ghost: "border-transparent text-muted hover:text-ink hover:bg-raised",
    default: "border-rule-strong text-ink hover:border-sodium hover:text-sodium",
  }[tone];
  return (
    <button
      type={type}
      data-testid={testid}
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-[3px] border px-2.5 py-1.5 text-[0.6875rem] font-bold uppercase tracking-[0.12em] transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${cls}`}
    >
      {children}
    </button>
  );
}
