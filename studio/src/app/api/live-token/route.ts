import { hasSession, json, liveToken } from "@/lib/laptop";
import { readLive } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Logged-in browsers get a 10-minute token for the laptop's live channel and its current tunnel URL.
export async function GET() {
  if (!(await hasSession())) return json(401, { error: "unauthorized" });
  const t = liveToken();
  if (!t) return json(500, { error: "INGEST_TOKEN not set" });
  const live = await readLive().catch(() => ({ url: null, at: null }));
  return json(200, { ...t, url: live.url, url_at: live.at }, "private, no-store");
}
