"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { dur, stamp, toMs } from "@/lib/format";
import type { LiveInfo, Snapshot, SnapshotResponse, StudioAction } from "@/lib/types";
import { useLive } from "@/lib/useLive";
import { Channel } from "./Channel";
import { Charts } from "./Charts";
import { Errors } from "./Errors";
import { Header, liveness, TABS, type TabKey } from "./Header";
import { KagglePage } from "./Kaggle";
import { Overview } from "./Overview";
import { Research } from "./Research";
import { InitialNowCtx, StudioCtx, useNow, type StudioApi } from "./ui";
import { Videos } from "./Videos";

const FALLBACK_POLL_MS = 60_000;
const LAST_LIVE_KEY = "hws_last_live";

function tabFromHash(): TabKey {
  if (typeof window === "undefined") return "overview";
  const h = window.location.hash.replace(/^#/, "").split("/")[0];
  return (TABS.find((t) => t.key === h)?.key ?? "overview") as TabKey;
}

export function Console({ initial, serverNow }: { initial: SnapshotResponse | null; serverNow: number }) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(initial?.snapshot ?? null);
  const [liveInfo, setLiveInfo] = useState<LiveInfo>(initial?.live ?? { url: null, at: null });
  const [tab, setTab] = useState<TabKey>("overview");
  const [pending, setPending] = useState<Map<string, StudioAction>>(new Map());
  const [storedLastLive, setStoredLastLive] = useState<number | null>(null);
  const router = useRouter();
  const onUnauth = useCallback(() => router.push("/login"), [router]);
  const live = useLive(setSnapshot, onUnauth);

  useEffect(() => {
    const read = () => setTab(tabFromHash());
    const id = requestAnimationFrame(read);
    window.addEventListener("hashchange", read);
    try {
      const v = Number(localStorage.getItem(LAST_LIVE_KEY));
      if (v) requestAnimationFrame(() => setStoredLastLive(v));
    } catch {
      /* private window */
    }
    return () => {
      cancelAnimationFrame(id);
      window.removeEventListener("hashchange", read);
    };
  }, []);

  useEffect(() => {
    if (live.status === "live" && live.lastAt) {
      try {
        localStorage.setItem(LAST_LIVE_KEY, String(live.lastAt));
      } catch {
        /* ignore */
      }
    }
  }, [live.status, live.lastAt]);

  // while the live channel is down: the last snapshot stored on Vercel, every minute
  const isLive = live.status === "live";
  useEffect(() => {
    if (isLive) return;
    let alive = true;
    const pull = async () => {
      try {
        const r = await fetch("/api/snapshot", { cache: "no-store" });
        if (r.status === 401) return onUnauth();
        if (!r.ok) return;
        const j = (await r.json()) as SnapshotResponse;
        if (!alive) return;
        if (j.snapshot) {
          setSnapshot((cur) => (cur && toMs(cur.generated_at)! > toMs(j.snapshot!.generated_at)! ? cur : j.snapshot));
        }
        setLiveInfo(j.live);
      } catch {
        /* next round */
      }
    };
    const first = setTimeout(pull, 1500);
    const id = setInterval(pull, FALLBACK_POLL_MS);
    return () => {
      alive = false;
      clearTimeout(first);
      clearInterval(id);
    };
  }, [isLive, onUnauth]);

  // writes the laptop has confirmed (in its snapshot) leave the pending list
  const applied = useMemo(() => new Set((snapshot?.actions?.applied ?? []).map((a) => a.id)), [snapshot]);
  const visiblePending = useMemo(() => new Map([...pending].filter(([id]) => !applied.has(id))), [pending, applied]);

  const act = useCallback<StudioApi["act"]>(
    async (a) => {
      let action: StudioAction | null = null;
      let queued = false;
      let error: string | undefined;
      try {
        const r = await fetch("/api/actions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(a) });
        if (r.status === 401) onUnauth();
        const j = (await r.json().catch(() => ({}))) as { ok?: boolean; action?: StudioAction; error?: string };
        action = j.action ?? null;
        queued = r.ok && !!j.ok;
        if (!queued) error = j.error ?? `HTTP ${r.status}`;
      } catch {
        error = "network error";
      }
      if (!action) action = { ...a, id: crypto.randomUUID(), at: new Date().toISOString() } as StudioAction;
      const instant = await live.sendAction(action);
      if (queued || instant) setPending((m) => new Map(m).set(action!.id, action!));
      return { ok: queued || instant, queued, instant, error: queued || instant ? undefined : error, id: action.id };
    },
    [live, onUnauth],
  );

  const go = useCallback((t: string, anchor?: string) => {
    window.location.hash = anchor ? `${t}/${anchor}` : t;
    setTab(t as TabKey);
    if (anchor) setTimeout(() => document.getElementById(anchor)?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
  }, []);

  const api = useMemo<StudioApi>(() => ({ act, tunnelMedia: live.tunnelMedia, pendingIds: new Set(visiblePending.keys()), go }), [act, live.tunnelMedia, visiblePending, go]);

  const lastSeen = useMemo(() => {
    const c = [live.lastAt, storedLastLive, toMs(snapshot?.generated_at), toMs(liveInfo.at)].filter((x): x is number => !!x);
    return c.length ? Math.max(...c) : null;
  }, [live.lastAt, storedLastLive, snapshot?.generated_at, liveInfo.at]);

  const errCount = (snapshot?.events ?? []).filter((e) => e.level === "err").length;
  const onTab = (t: TabKey) => go(t);

  return (
    <InitialNowCtx.Provider value={serverNow}>
      <StudioCtx.Provider value={api}>
        <Header snapshot={snapshot} status={live.status} lastAt={live.lastAt} lastSeen={lastSeen} reading={live.reading} tab={tab} onTab={onTab} errCount={errCount} />
        <OfflineBanner status={live.status} lastAt={live.lastAt} lastSeen={lastSeen} snapshot={snapshot} pending={visiblePending.size} />
        <main id="main" className="relative z-10 mx-auto flex max-w-[80rem] flex-col gap-8 px-4 py-6 sm:px-6">
          {!snapshot ? (
            <NoData />
          ) : (
            <div key={tab} className="page-in flex flex-col gap-8">
              {tab === "overview" && <Overview s={snapshot} pending={visiblePending} />}
              {tab === "videos" && <Videos s={snapshot} pending={visiblePending} />}
              {tab === "kaggle" && <KagglePage s={snapshot} />}
              {tab === "charts" && <Charts s={snapshot} />}
              {tab === "errors" && <Errors s={snapshot} />}
              {tab === "research" && <Research s={snapshot} />}
              {tab === "channel" && <Channel s={snapshot} />}
            </div>
          )}
          <footer className="border-t border-rule pt-4 text-[0.6875rem] text-faint">
            Heartwick Studio · read from the engine&apos;s real files on the main laptop · nothing is uploaded to YouTube or Instagram from here
          </footer>
        </main>
      </StudioCtx.Provider>
    </InitialNowCtx.Provider>
  );
}

function OfflineBanner({ status, lastAt, lastSeen, snapshot, pending }: { status: ReturnType<typeof useLive>["status"]; lastAt: number | null; lastSeen: number | null; snapshot: Snapshot | null; pending: number }) {
  const now = useNow(1000);
  const st = liveness(status, lastAt, lastSeen, now);
  if (st.live || status === "connecting") return null;
  return (
    <div role="alert" data-testid="offline-banner" className="slide-in relative z-20 border-b border-warn/40 bg-warn/10">
      <div className="mx-auto flex max-w-[80rem] flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-2 text-[0.75rem] sm:px-6">
        <span className="font-bold uppercase tracking-[0.12em] text-warn">Laptop offline</span>
        <span className="text-ink">
          {lastSeen ? (
            <>
              since about <span className="tnum font-semibold">{stamp(new Date(lastSeen).toISOString())}</span> ({dur(now - lastSeen)} ago, last contact).
            </>
          ) : (
            "no contact yet."
          )}{" "}
          Showing its last snapshot{snapshot ? ` from ${stamp(snapshot.generated_at)}` : ""}. It reconnects by itself when the laptop is back.
        </span>
        {pending > 0 && <span className="text-warn">{pending} of your changes are queued on Vercel and will reach the laptop when it is on.</span>}
      </div>
    </div>
  );
}

function NoData() {
  return (
    <div className="panel bracket p-6 text-[0.8125rem] text-muted">
      <div className="display mb-2 text-[1.5rem] font-bold uppercase text-ink">No data yet</div>
      The main laptop has not posted a snapshot. It does that within a minute of starting (scheduled task &quot;Heartwick Studio&quot;).
    </div>
  );
}
