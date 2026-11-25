// Laptop-side Blob uploader for dashboard/media.py (official @vercel/blob SDK, private store).
// stdin: JSON [{op: "put", file, pathname, contentType, maxAge, multipart} | {op: "del", pathnames: [...]}]
// stdout: JSON [{ok, pathname?, size?, error?}] in the same order. Needs BLOB_READ_WRITE_TOKEN in the environment.
import { readFile } from "node:fs/promises";
import { del, put } from "@vercel/blob";

const chunks = [];
for await (const c of process.stdin) chunks.push(c);
const ops = JSON.parse(Buffer.concat(chunks).toString("utf8") || "[]");
const out = [];
for (const op of ops) {
  try {
    if (op.op === "put") {
      const body = await readFile(op.file);
      const r = await put(op.pathname, body, {
        access: "private",
        allowOverwrite: true,
        addRandomSuffix: false,
        contentType: op.contentType,
        cacheControlMaxAge: op.maxAge ?? 31536000,
        multipart: Boolean(op.multipart),
      });
      out.push({ ok: true, pathname: r.pathname, size: body.length });
    } else if (op.op === "del") {
      if (op.pathnames?.length) await del(op.pathnames);
      out.push({ ok: true });
    } else out.push({ ok: false, error: `unknown op ${op.op}` });
  } catch (e) {
    out.push({ ok: false, error: String(e?.message ?? e).slice(0, 300) });
  }
}
process.stdout.write(JSON.stringify(out));
