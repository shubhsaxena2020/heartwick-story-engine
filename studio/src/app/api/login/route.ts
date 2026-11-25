import { NextResponse } from "next/server";
import { newSession, passwordOk, SESSION_COOKIE, SESSION_TTL_S } from "@/lib/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Per-instance throttle: 10 wrong passwords per IP per 10 minutes.
const misses = new Map<string, { n: number; first: number }>();
const WINDOW = 10 * 60_000;

function nextPath(v: FormDataEntryValue | null | undefined): string {
  const s = typeof v === "string" ? v : "/";
  return s.startsWith("/") && !s.startsWith("//") && !s.startsWith("/api/") ? s : "/";
}

export async function POST(req: Request) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "local";
  const now = Date.now();
  const rec = misses.get(ip);
  if (rec && now - rec.first < WINDOW && rec.n >= 10) return NextResponse.redirect(new URL("/login?e=throttled", req.url), 303);
  const form = await req.formData().catch(() => null);
  const pw = form?.get("password");
  const next = nextPath(form?.get("next"));
  if (typeof pw !== "string" || !(await passwordOk(pw))) {
    const r = rec && now - rec.first < WINDOW ? rec : { n: 0, first: now };
    r.n += 1;
    misses.set(ip, r);
    return NextResponse.redirect(new URL(`/login?e=wrong&next=${encodeURIComponent(next)}`, req.url), 303);
  }
  misses.delete(ip);
  const token = await newSession();
  if (!token) return NextResponse.redirect(new URL("/login?e=config", req.url), 303);
  const res = NextResponse.redirect(new URL(next, req.url), 303);
  res.cookies.set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_TTL_S,
  });
  return res;
}
