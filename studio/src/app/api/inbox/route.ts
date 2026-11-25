import { json, laptopAuthorized } from "@/lib/laptop";
import { readInbox } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The laptop's exporter pulls the queued writes (approvals, notes, queue / re-render requests) from here. It applies
// each id once and reports the applied ids in its next snapshot; nothing is deleted on read.
export async function GET(req: Request) {
  if (!laptopAuthorized(req.headers.get("authorization"))) return json(401, { error: "unauthorized" });
  try {
    const { actions } = await readInbox(true); // fresh: a write made just before the laptop woke must not be missed
    return json(200, { actions });
  } catch (e) {
    console.error("inbox read failed", e instanceof Error ? e.message : e);
    return json(502, { error: "storage read failed" });
  }
}
