import { get } from "@vercel/blob";
import { createReadStream, promises as fs } from "node:fs";
import { Readable } from "node:stream";
import { hasSession } from "@/lib/laptop";
import { blobEnabled, localMediaPath, MEDIA_KEY } from "@/lib/store";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Login-protected media (thumbnails, contact sheets, 480p previews, voice samples) from the private Blob store.
// Keys are content hashes, so a key never changes content: browsers may cache it for a year. Videos are served in
// ranges of at most 3 MB (a function response must stay under 4.5 MB); <video> asks for the rest by itself.
const CHUNK = 3 * 1024 * 1024;
const TYPES: Record<string, string> = { jpg: "image/jpeg", jpeg: "image/jpeg", png: "image/png", webp: "image/webp", mp4: "video/mp4",
  m4a: "audio/mp4", wav: "audio/wav", mp3: "audio/mpeg", ogg: "audio/ogg", flac: "audio/flac" };

function parseRange(h: string | null): { start: number; end: number | null } | null {
  const m = /^bytes=(\d+)-(\d*)$/.exec(h ?? "");
  return m ? { start: Number(m[1]), end: m[2] ? Number(m[2]) : null } : null;
}

export async function GET(req: Request, { params }: { params: Promise<{ key: string[] }> }) {
  if (!(await hasSession())) return new Response("unauthorized", { status: 401 });
  const key = (await params).key.join("/");
  if (!MEDIA_KEY.test(key)) return new Response("not found", { status: 404 });
  const type = TYPES[key.split(".").pop()!.toLowerCase()] ?? "application/octet-stream";
  const asked = parseRange(req.headers.get("range"));
  const big = type.startsWith("video/") || type.startsWith("audio/");
  const range = asked ?? (big ? { start: 0, end: null } : null);
  const base = { "Content-Type": type, "Cache-Control": "private, max-age=31536000, immutable", "Accept-Ranges": "bytes", "X-Content-Type-Options": "nosniff" };

  if (!blobEnabled) {
    const p = await localMediaPath(key);
    if (!p) return new Response("not found", { status: 404 });
    const size = (await fs.stat(p)).size;
    const start = range ? Math.min(range.start, size) : 0;
    const end = range ? Math.min(size - 1, range.end ?? start + CHUNK - 1, start + CHUNK - 1) : size - 1;
    if (start >= size) return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}` } });
    const stream = Readable.toWeb(createReadStream(p, { start, end })) as ReadableStream;
    return new Response(stream, { status: range ? 206 : 200, headers: { ...base, "Content-Length": String(end - start + 1), ...(range ? { "Content-Range": `bytes ${start}-${end}/${size}` } : {}) } });
  }

  const headers: Record<string, string> = {};
  if (range) headers.range = `bytes=${range.start}-${range.end != null ? Math.min(range.end, range.start + CHUNK - 1) : range.start + CHUNK - 1}`;
  const r = await get(`m/${key}`, { access: "private", headers }).catch(() => null);
  if (!r || !r.stream || r.statusCode !== 200) return new Response("not found", { status: 404 });
  const out: Record<string, string> = { ...base };
  for (const h of ["content-length", "content-range"]) {
    const v = r.headers.get(h);
    if (v) out[h] = v;
  }
  const partial = r.headers.get("content-range") != null;
  return new Response(r.stream, { status: partial ? 206 : 200, headers: out });
}
