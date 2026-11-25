import { get, put } from "@vercel/blob";
import { promises as fs } from "node:fs";
import path from "node:path";
import type { LiveInfo, Snapshot, StudioAction } from "./types";

// Storage: the private Blob store "heartwick-studio-data" on Vercel. Without Blob credentials (local dev) the same
// calls use files in ./.local-store, and before any ingest the snapshot falls back to LOCAL_SNAPSHOT_PATH or the
// sample fixture. Blob operations are counted (Hobby: 2,000 uploads, 10,000 cache-miss reads a month for the
// account), so reads go through the CDN cache (snapshot max-age 60 s) and an instance cache, and writes happen
// only when the laptop posts something new or someone clicks a button.

const useBlob = Boolean(process.env.BLOB_READ_WRITE_TOKEN || process.env.BLOB_STORE_ID || process.env.VERCEL);
const LOCAL_DIR = path.join(/*turbopackIgnore: true*/ process.cwd(), ".local-store");

async function readBlob(pathname: string, fresh = false): Promise<{ text: string; etag: string | null } | null> {
  if (useBlob) {
    const r = await get(pathname, { access: "private", useCache: !fresh });
    if (!r || r.statusCode !== 200 || !r.stream) return null;
    return { text: await new Response(r.stream).text(), etag: r.blob.etag ?? null };
  }
  try {
    return { text: await fs.readFile(/*turbopackIgnore: true*/ path.join(LOCAL_DIR, pathname), "utf8"), etag: null };
  } catch {
    return null;
  }
}

async function writeBlob(pathname: string, body: string, opts: { maxAge?: number; ifMatch?: string | null; create?: boolean } = {}) {
  if (useBlob) {
    await put(pathname, body, {
      access: "private",
      allowOverwrite: !opts.create,
      addRandomSuffix: false,
      contentType: "application/json",
      cacheControlMaxAge: opts.maxAge ?? 60,
      ...(opts.ifMatch ? { ifMatch: opts.ifMatch } : {}),
    });
    return;
  }
  const file = path.join(LOCAL_DIR, pathname);
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, body, "utf8");
}

// ---------------------------------------------------------------- snapshot
const SNAP = "snapshot.json";
const SNAP_TTL_MS = 15_000;
let snapCache: { at: number; snapshot: Snapshot | null; storedAt: string | null } | null = null;

async function localFallback(): Promise<Snapshot | null> {
  if (useBlob) return null;
  for (const p of [process.env.LOCAL_SNAPSHOT_PATH, path.join(/*turbopackIgnore: true*/ process.cwd(), "fixtures", "sample-snapshot.json")]) {
    if (!p) continue;
    try {
      return JSON.parse(await fs.readFile(/*turbopackIgnore: true*/ p, "utf8")) as Snapshot;
    } catch {
      /* next */
    }
  }
  return null;
}

export async function readSnapshot(): Promise<{ snapshot: Snapshot | null; storedAt: string | null }> {
  if (snapCache && Date.now() - snapCache.at < SNAP_TTL_MS) return snapCache;
  let snapshot: Snapshot | null = null;
  let storedAt: string | null = null;
  const r = await readBlob(SNAP);
  if (r) {
    try {
      const w = JSON.parse(r.text) as Snapshot & { _stored_at?: string };
      storedAt = w._stored_at ?? null;
      delete w._stored_at;
      snapshot = w;
    } catch {
      snapshot = null;
    }
  }
  if (!snapshot) snapshot = await localFallback();
  snapCache = { at: Date.now(), snapshot, storedAt };
  return snapCache;
}

export async function writeSnapshot(snapshot: Snapshot): Promise<string> {
  const storedAt = new Date().toISOString();
  await writeBlob(SNAP, JSON.stringify({ ...snapshot, _stored_at: storedAt }), { maxAge: 60 });
  snapCache = { at: Date.now(), snapshot, storedAt };
  return storedAt;
}

// ---------------------------------------------------------------- live tunnel URL
const LIVE = "live.json";
let liveCache: { at: number; info: LiveInfo } | null = null;

export async function readLive(): Promise<LiveInfo> {
  if (liveCache && Date.now() - liveCache.at < 10_000) return liveCache.info;
  let info: LiveInfo = { url: null, at: null };
  // through the CDN cache (max-age 60 s): cache hits are not billed, and browsers ask often while the laptop is off
  const r = await readBlob(LIVE);
  if (r) {
    try {
      const j = JSON.parse(r.text) as Partial<LiveInfo>;
      info = { url: typeof j.url === "string" ? j.url : null, at: typeof j.at === "string" ? j.at : null };
    } catch {
      /* empty */
    }
  }
  liveCache = { at: Date.now(), info };
  return info;
}

/** false when that URL is already stored (no write). */
export async function writeLive(url: string, at: string): Promise<boolean> {
  liveCache = null;
  const cur = await readBlob(LIVE, true).then((r) => (r ? (JSON.parse(r.text) as LiveInfo) : null)).catch(() => null);
  if (cur?.url === url) return false;
  const info = { url, at };
  await writeBlob(LIVE, JSON.stringify(info), { maxAge: 60 });
  liveCache = { at: Date.now(), info };
  return true;
}

// ---------------------------------------------------------------- inbox (writes queued for the laptop)
const INBOX = "inbox.json";
const INBOX_MAX = 300;
const APPLIED_KEEP_MS = 3 * 86400_000;

export async function readInbox(fresh = true): Promise<{ actions: StudioAction[]; etag: string | null }> {
  const r = await readBlob(INBOX, fresh);
  if (!r) return { actions: [], etag: null };
  try {
    const j = JSON.parse(r.text) as { actions?: StudioAction[] };
    return { actions: Array.isArray(j.actions) ? j.actions : [], etag: r.etag };
  } catch {
    return { actions: [], etag: r.etag };
  }
}

/** Append one action (read, append, conditional write; retried when someone else wrote in between). Entries the
 *  laptop has confirmed applied are dropped after 3 days, so the file stays small. Never drops unapplied ones. */
export async function appendAction(a: StudioAction, appliedIds: Set<string>, nowMs = Date.now()): Promise<void> {
  for (let attempt = 0; attempt < 4; attempt++) {
    const { actions, etag } = await readInbox(true);
    if (actions.some((x) => x.id === a.id)) return;
    const kept = actions.filter((x) => !(appliedIds.has(x.id) && nowMs - Date.parse(x.at) > APPLIED_KEEP_MS));
    const unapplied = kept.filter((x) => !appliedIds.has(x.id));
    const applied = kept.filter((x) => appliedIds.has(x.id)).slice(-Math.max(0, INBOX_MAX - unapplied.length - 1));
    const next = [...applied, ...unapplied, a].sort((x, y) => Date.parse(x.at) - Date.parse(y.at));
    try {
      await writeBlob(INBOX, JSON.stringify({ actions: next }), { maxAge: 60, ifMatch: etag, create: etag == null && useBlob });
      return;
    } catch (e) {
      if (attempt === 3) throw e;
      await new Promise((r) => setTimeout(r, 150 * (attempt + 1)));
    }
  }
}

// ---------------------------------------------------------------- media
export const MEDIA_KEY = /^[0-9a-f]{12}\/[A-Za-z0-9._-]{1,120}$/;

/** Local dev: the laptop's media index maps a key to the real file. */
export async function localMediaPath(key: string): Promise<string | null> {
  const idx = process.env.LOCAL_MEDIA_INDEX;
  if (useBlob || !idx) return null;
  try {
    const j = JSON.parse(await fs.readFile(/*turbopackIgnore: true*/ idx, "utf8")) as { keys?: Record<string, { path?: string }> };
    return j.keys?.[key]?.path ?? null;
  } catch {
    return null;
  }
}

export const blobEnabled = useBlob;
