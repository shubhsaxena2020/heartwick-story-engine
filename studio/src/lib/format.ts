// Formatting. Every time is shown in IST (the laptop and the family are in India); hover shows the full date.

const TZ = "Asia/Kolkata";

export function toMs(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const s = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$/.test(iso) ? iso.replace(" ", "T") + "+05:30" : iso;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
}

const cache = new Map<string, Intl.DateTimeFormat>();
function fmt(opts: Intl.DateTimeFormatOptions): Intl.DateTimeFormat {
  const k = JSON.stringify(opts);
  let f = cache.get(k);
  if (!f) {
    f = new Intl.DateTimeFormat("en-GB", { timeZone: TZ, hourCycle: "h23", ...opts });
    cache.set(k, f);
  }
  return f;
}

export function clock(ms: number, seconds = true): string {
  return fmt({ hour: "2-digit", minute: "2-digit", ...(seconds ? { second: "2-digit" } : {}) }).format(ms);
}

/** "21 NOV 16:19" */
export function stamp(iso: string | null | undefined): string {
  const ms = toMs(iso);
  if (ms == null) return "—";
  return `${fmt({ day: "2-digit", month: "short" }).format(ms).toUpperCase()} ${clock(ms, false)}`;
}

export function fullStamp(iso: string | null | undefined): string {
  const ms = toMs(iso);
  if (ms == null) return "";
  return fmt({ weekday: "short", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(ms) + " IST";
}

export function dayKey(ms: number): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit" }).format(ms);
}

export function dayLabel(key: string): string {
  const ms = Date.parse(key + "T12:00:00+05:30");
  return Number.isFinite(ms) ? fmt({ day: "2-digit", month: "short" }).format(ms).toUpperCase() : key;
}

export function dur(deltaMs: number, suffix = false): string {
  const future = deltaMs < 0;
  const s = Math.abs(deltaMs) / 1000;
  let out: string;
  if (s < 60) out = `${Math.floor(s)}s`;
  else if (s < 3600) out = `${Math.floor(s / 60)}m`;
  else if (s < 86400) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    out = m ? `${h}h ${m}m` : `${h}h`;
  } else out = `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
  if (!suffix) return out;
  return future ? `in ${out}` : `${out} ago`;
}

export function ago(iso: string | null | undefined, now: number): string {
  const ms = toMs(iso);
  return ms == null ? "—" : dur(now - ms, true);
}

/** 72.4 s -> "1:12" */
export function mmss(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  const t = Math.round(s);
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
}

export function minutes(m: number | null | undefined): string {
  if (m == null || !Number.isFinite(m)) return "—";
  if (m < 60) return `${m.toFixed(m < 10 ? 1 : 0)} min`;
  return `${Math.floor(m / 60)} h ${Math.round(m % 60)} min`;
}

export function num(v: number | null | undefined, digits = 0): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function bytes(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n} B`;
}

export const arr = <T,>(v: T[] | null | undefined): T[] => (Array.isArray(v) ? v : []);

export function signed(v: number | null | undefined, digits = 1): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return (v > 0 ? "+" : "") + v.toFixed(digits);
}
