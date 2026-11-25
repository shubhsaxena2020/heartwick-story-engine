import { randomUUID } from "node:crypto";
import { checkAction } from "@/lib/actions";
import { hasSession, json } from "@/lib/laptop";
import { appendAction, readSnapshot } from "@/lib/store";
import type { StudioAction } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// A write from the site (approve / needs changes + note / published / queue story / re-render). It is queued in
// Blob inbox.json first, so it is never lost while the laptop is off; the laptop's exporter applies it on its next
// pull, or at once when the browser also hands it to the live channel. Same id on both paths = applied once.
export async function POST(req: Request) {
  if (!(await hasSession())) return json(401, { error: "unauthorized" });
  if (Number(req.headers.get("content-length") ?? 0) > 8192) return json(413, { error: "too large" });
  const body = await req.json().catch(() => null);
  const c = checkAction(body);
  if (!c.ok) return json(422, { error: c.error });
  const { snapshot } = await readSnapshot().catch(() => ({ snapshot: null }));
  const a = c.action;
  if (snapshot) {
    if (a.video && !(snapshot.videos ?? []).some((v) => v.name === a.video)) return json(422, { error: "unknown video" });
    if (a.story && !(snapshot.stories ?? []).some((s) => s.slug === a.story)) return json(422, { error: "unknown story" });
  }
  const action: StudioAction = { id: randomUUID(), at: new Date().toISOString(), ...a };
  const applied = new Set((snapshot?.actions?.applied ?? []).map((x) => x.id));
  try {
    await appendAction(action, applied);
  } catch (e) {
    console.error("inbox write failed", e instanceof Error ? e.message : e);
    return json(502, { error: "could not queue it on Vercel", action });
  }
  return json(200, { ok: true, action });
}
