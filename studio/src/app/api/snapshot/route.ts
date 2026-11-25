import { hasSession, json } from "@/lib/laptop";
import { readLive, readSnapshot } from "@/lib/store";
import type { SnapshotResponse } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The last snapshot the laptop posted (used while the live channel is down, e.g. the laptop is off).
export async function GET() {
  if (!(await hasSession())) return json(401, { error: "unauthorized" });
  try {
    const [{ snapshot, storedAt }, live] = await Promise.all([readSnapshot(), readLive().catch(() => ({ url: null, at: null }))]);
    const body: SnapshotResponse = { snapshot, stored_at: storedAt, live };
    return json(200, body, "private, no-store");
  } catch (e) {
    console.error("snapshot read failed", e instanceof Error ? e.message : e);
    return json(502, { error: "storage read failed" });
  }
}
