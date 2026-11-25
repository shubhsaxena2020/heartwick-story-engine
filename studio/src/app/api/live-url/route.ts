import { json, laptopAuthorized, TUNNEL_URL } from "@/lib/laptop";
import { writeLive } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// dashboard/live.py posts its cloudflared quick-tunnel URL here when it changes (a few times a day at most).
export async function POST(req: Request) {
  if (!laptopAuthorized(req.headers.get("authorization"))) return json(401, { error: "unauthorized" });
  const b = (await req.json().catch(() => null)) as { url?: unknown; at?: unknown } | null;
  const url = typeof b?.url === "string" ? b.url.trim().replace(/\/+$/, "") : "";
  if (!TUNNEL_URL.test(url)) return json(422, { error: "url must be https://<name>.trycloudflare.com" });
  const at = typeof b?.at === "string" && !Number.isNaN(Date.parse(b.at)) ? b.at : new Date().toISOString();
  try {
    return json(200, { ok: true, written: await writeLive(url, at) });
  } catch (e) {
    console.error("live.json write failed", e instanceof Error ? e.message : e);
    return json(502, { error: "storage write failed" });
  }
}
