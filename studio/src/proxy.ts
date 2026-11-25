import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE, sessionValid } from "@/lib/session";

// Everything needs the dashboard login except the login page/handler and the laptop's own endpoints
// (/api/ingest, /api/live-url, /api/inbox), which check Bearer INGEST_TOKEN themselves.
export async function proxy(req: NextRequest) {
  if (await sessionValid(req.cookies.get(SESSION_COOKIE)?.value)) return NextResponse.next();
  const { pathname, search } = req.nextUrl;
  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401, headers: { "Cache-Control": "no-store" } });
  }
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = pathname === "/" ? "" : `?next=${encodeURIComponent(pathname + search)}`;
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!login|api/login|api/ingest|api/live-url|api/inbox|_next/static|_next/image|favicon.ico|icon.svg|robots.txt).*)"],
};
