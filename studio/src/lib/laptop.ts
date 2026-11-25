import { createHash, createHmac, timingSafeEqual } from "node:crypto";
import { cookies } from "next/headers";
import { SESSION_COOKIE, sessionValid } from "./session";

// Shared by the laptop-facing routes (/api/ingest, /api/live-url, /api/inbox) and the live-channel token.

/** Bearer INGEST_TOKEN check: both sides hashed first so the comparison is equal-length and constant time. */
export function laptopAuthorized(header: string | null): boolean {
  const want = process.env.INGEST_TOKEN;
  if (!want || !header?.startsWith("Bearer ")) return false;
  const a = createHash("sha256").update(header.slice(7)).digest();
  const b = createHash("sha256").update(want).digest();
  return timingSafeEqual(a, b);
}

export const LIVE_TOKEN_TTL_S = 10 * 60;

/** "hw1.<exp>.<hex HMAC(INGEST_TOKEN, 'hw1.<exp>')>", verified by dashboard/live.py with the same secret. */
export function liveToken(nowMs = Date.now()): { token: string; exp: number } | null {
  const secret = process.env.INGEST_TOKEN;
  if (!secret) return null;
  const exp = Math.floor(nowMs / 1000) + LIVE_TOKEN_TTL_S;
  const msg = `hw1.${exp}`;
  return { token: `${msg}.${createHmac("sha256", secret).update(msg).digest("hex")}`, exp };
}

export const TUNNEL_URL = /^https:\/\/[a-z0-9-]+\.trycloudflare\.com$/;

export async function hasSession(): Promise<boolean> {
  const jar = await cookies();
  return sessionValid(jar.get(SESSION_COOKIE)?.value);
}

export function json(status: number, body: unknown, cache = "no-store") {
  return Response.json(body, { status, headers: { "Cache-Control": cache } });
}
