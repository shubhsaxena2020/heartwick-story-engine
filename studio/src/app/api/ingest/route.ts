import { gunzipSync } from "node:zlib";
import { json, laptopAuthorized } from "@/lib/laptop";
import { writeSnapshot } from "@/lib/store";
import type { Snapshot } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX = 4 * 1024 * 1024; // Vercel's request limit is 4.5 MB; snapshots are ~20-200 KB gzipped
const MAX_JSON = 12 * 1024 * 1024;

// The laptop's exporter posts the snapshot here (gzip or plain JSON, Bearer INGEST_TOKEN), only when it changed in
// a way that matters, or as a heartbeat. One Blob write per accepted post.
export async function POST(req: Request) {
  if (!laptopAuthorized(req.headers.get("authorization"))) return json(401, { error: "unauthorized" });
  if (Number(req.headers.get("content-length") ?? 0) > MAX) return json(413, { error: "body larger than 4 MB" });
  let raw: Buffer;
  try {
    raw = Buffer.from(await req.arrayBuffer());
  } catch {
    return json(400, { error: "could not read body" });
  }
  if (raw.length > MAX) return json(413, { error: "body larger than 4 MB" });
  let text: string;
  try {
    text = (raw[0] === 0x1f && raw[1] === 0x8b ? gunzipSync(raw, { maxOutputLength: MAX_JSON }) : raw).toString("utf8");
  } catch {
    return json(400, { error: "invalid gzip or too large once decompressed" });
  }
  let snap: Snapshot;
  try {
    snap = JSON.parse(text);
  } catch {
    return json(400, { error: "not JSON" });
  }
  if (!snap || typeof snap !== "object" || Array.isArray(snap) || snap.schema !== 1) return json(422, { error: "expected schema 1" });
  try {
    const storedAt = await writeSnapshot(snap);
    return json(200, { ok: true, stored_at: storedAt, bytes: text.length });
  } catch (e) {
    console.error("snapshot write failed", e instanceof Error ? e.message : e);
    return json(502, { error: "storage write failed" });
  }
}
