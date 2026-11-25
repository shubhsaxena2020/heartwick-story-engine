"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { LiveReading, Snapshot, StudioAction } from "./types";

export type LiveStatus = "connecting" | "live" | "down" | "nourl";

const POLL_MS = 3_000; // while the tab is visible
const HIDDEN_STOP_MS = 60_000; // a hidden tab stops asking after a minute (no viewer = no work on the laptop)
const RETRY_MAX_MS = 20_000; // laptop away: re-read the tunnel URL this often (a reboot brings a NEW URL)
const TIMEOUT_MS = 8_000;

interface Token {
  token: string;
  exp: number;
  url: string | null;
}

/**
 * The laptop's live channel: a token + tunnel URL from /api/live-token, then short polls of <tunnel>/state.
 * The laptop answers with the full snapshot only when it changed (rev), otherwise with a small live reading.
 * Two failed polls in a row = down; reconnects with backoff, re-reading the URL every time.
 */
export function useLive(onSnapshot: (s: Snapshot) => void, onUnauthorized: () => void) {
  const [status, setStatus] = useState<LiveStatus>("connecting");
  const [lastAt, setLastAt] = useState<number | null>(null);
  const [reading, setReading] = useState<LiveReading | null>(null);
  const tok = useRef<Token | null>(null);
  const handlers = useRef({ onSnapshot, onUnauthorized });
  useEffect(() => {
    handlers.current = { onSnapshot, onUnauthorized };
  });

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    let hiddenTimer: number | undefined;
    let stopped = false;
    let rev = "";
    let fails = 0;
    let retry = 0;

    const later = (fn: () => void, ms: number) => {
      window.clearTimeout(timer);
      timer = window.setTimeout(fn, ms);
    };

    async function getToken(): Promise<boolean> {
      try {
        const r = await fetch("/api/live-token", { cache: "no-store" });
        if (r.status === 401) {
          handlers.current.onUnauthorized();
          return false;
        }
        if (!r.ok) throw new Error(String(r.status));
        tok.current = (await r.json()) as Token;
        return true;
      } catch {
        tok.current = null;
        return false;
      }
    }

    async function connect() {
      if (!alive || stopped) return;
      const ok = await getToken();
      if (!alive) return;
      if (!ok) {
        setStatus("down");
        retry += 1;
        return later(connect, Math.min(RETRY_MAX_MS, 2_000 * 2 ** Math.min(retry, 4)));
      }
      if (!tok.current?.url) {
        setStatus("nourl");
        return later(connect, RETRY_MAX_MS);
      }
      fails = 0;
      poll();
    }

    async function poll() {
      if (!alive || stopped) return;
      const t = tok.current;
      if (!t?.url) return connect();
      if (t.exp * 1000 - Date.now() < 60_000) return connect();
      const ctl = new AbortController();
      const kill = window.setTimeout(() => ctl.abort(), TIMEOUT_MS);
      try {
        const r = await fetch(`${t.url}/state?t=${encodeURIComponent(t.token)}&v=${encodeURIComponent(rev)}`, { cache: "no-store", signal: ctl.signal });
        if (r.status === 401) return connect();
        if (!r.ok) throw new Error(String(r.status));
        const j = (await r.json()) as { rev: string; same: boolean; live: LiveReading; snapshot?: Snapshot };
        if (!alive) return;
        fails = 0;
        retry = 0;
        if (j.snapshot) {
          rev = j.rev;
          handlers.current.onSnapshot(j.snapshot);
        }
        setReading(j.live);
        setLastAt(Date.now());
        setStatus("live");
        later(poll, POLL_MS);
      } catch {
        if (!alive) return;
        fails += 1;
        if (fails >= 2) {
          setStatus("down");
          retry += 1;
          later(connect, Math.min(RETRY_MAX_MS, 2_000 * 2 ** Math.min(retry, 4)));
        } else later(poll, 1_500);
      } finally {
        window.clearTimeout(kill);
      }
    }

    const onVis = () => {
      if (document.visibilityState === "hidden") {
        hiddenTimer = window.setTimeout(() => {
          stopped = true;
          window.clearTimeout(timer);
        }, HIDDEN_STOP_MS);
      } else {
        window.clearTimeout(hiddenTimer);
        if (stopped) {
          stopped = false;
          retry = 0;
          connect();
        }
      }
    };
    document.addEventListener("visibilitychange", onVis);
    connect();
    return () => {
      alive = false;
      window.clearTimeout(timer);
      window.clearTimeout(hiddenTimer);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  /** Hand a write straight to the laptop (instant when it is online). true = the laptop applied it. */
  const sendAction = useCallback(async (a: StudioAction): Promise<boolean> => {
    const t = tok.current;
    if (!t?.url) return false;
    try {
      const r = await fetch(`${t.url}/action?t=${encodeURIComponent(t.token)}`, {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: JSON.stringify(a),
      });
      return r.ok;
    } catch {
      return false;
    }
  }, []);

  /** The same media through the tunnel (fallback when the Blob copy is not there yet). */
  const tunnelMedia = useCallback((key: string): string | null => {
    const t = tok.current;
    return t?.url ? `${t.url}/media/${key}?t=${encodeURIComponent(t.token)}` : null;
  }, []);

  return { status, lastAt, reading, sendAction, tunnelMedia };
}
