// Signed session cookie (Web Crypto only, so it works in the proxy and in route handlers).
// Value: "hs1.<expiry epoch s>.<base64url HMAC-SHA256(key, 'hs1.<expiry>')>". The key comes from SESSION_SECRET, or
// from DASHBOARD_PASSWORD + INGEST_TOKEN, so changing the password signs every device out.

export const SESSION_COOKIE = "hws_session";
export const SESSION_TTL_S = 30 * 24 * 3600;

const enc = new TextEncoder();

function keyMaterial(): string | null {
  if (process.env.SESSION_SECRET) return process.env.SESSION_SECRET;
  const pw = process.env.DASHBOARD_PASSWORD;
  return pw ? `heartwick-studio:${pw}:${process.env.INGEST_TOKEN ?? ""}` : null;
}

function toB64url(buf: ArrayBuffer): string {
  let s = "";
  for (const b of new Uint8Array(buf)) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function sign(message: string): Promise<string | null> {
  const material = keyMaterial();
  if (!material) return null;
  const key = await crypto.subtle.importKey("raw", enc.encode(material), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  return toB64url(await crypto.subtle.sign("HMAC", key, enc.encode(message)));
}

/** Equal-time comparison (only the length can leak). */
export function sameText(a: string, b: string): boolean {
  const x = enc.encode(a);
  const y = enc.encode(b);
  let d = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) d |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return d === 0;
}

export async function newSession(nowMs = Date.now()): Promise<string | null> {
  const head = `hs1.${Math.floor(nowMs / 1000) + SESSION_TTL_S}`;
  const sig = await sign(head);
  return sig ? `${head}.${sig}` : null;
}

export async function sessionValid(token: string | null | undefined, nowMs = Date.now()): Promise<boolean> {
  if (!token) return false;
  const [v, exp, sig] = token.split(".");
  if (v !== "hs1" || !exp || !sig || token.split(".").length !== 3) return false;
  const e = Number(exp);
  if (!Number.isFinite(e) || e * 1000 < nowMs) return false;
  const want = await sign(`hs1.${exp}`);
  return !!want && sameText(sig, want);
}

/** Compare a submitted password with DASHBOARD_PASSWORD through HMAC digests. */
export async function passwordOk(candidate: string): Promise<boolean> {
  const real = process.env.DASHBOARD_PASSWORD;
  if (!real) return false;
  const [a, b] = await Promise.all([sign(`pw:${candidate}`), sign(`pw:${real}`)]);
  return !!a && !!b && sameText(a, b);
}
