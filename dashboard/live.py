"""Heartwick Studio live channel: laptop -> browser through a cloudflared quick tunnel (no Vercel in the loop).

  python dashboard/live.py                 # 127.0.0.1:8788 + tunnel (started by tools/start_heartwick.ps1)
  python dashboard/live.py --no-tunnel     # local only (dev)

Endpoints (all but /health need ?t=<token>, minted by the site's session-protected /api/live-token):
  GET  /health                     {"ok": true}, no data (tunnel watchdog)
  GET  /state?v=<rev>              {rev, same, live:{cpu,ram,...}, snapshot?}: the snapshot only when its rev changed.
                                   The browser polls every 3 s while its tab is visible. Inputs are re-stat'ed at most
                                   every 3 s and the snapshot is rebuilt only when a file changed.
  POST /action                     one write (approve / needs_changes / ... / queue_story / rerender), applied at once
                                   (same id rules as the exporter's inbox pull, so nothing is applied twice)
  GET  /media/<sha12>/<name>       a local media file by content key (Range supported), only keys the snapshot lists
Token: "hw1.<exp>.<hex HMAC-SHA256(INGEST_TOKEN, 'hw1.<exp>')>", at most 11 min ahead. CORS: the site's origin only.
No viewer = no work: the server blocks in accept(); nothing is rebuilt or measured until a request arrives.
The tunnel URL is POSTed to ${DASHBOARD_URL}/api/live-url only when it changes.
"""
import argparse
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import actions as A  # noqa: E402
import media as M  # noqa: E402
import studio_state as S  # noqa: E402
from studio_common import (LIVE_TUNNEL, PROD_ORIGIN, SNAPSHOT_LOCAL, iso, logger, lower_priority, read_env,  # noqa: E402
                           read_json, write_json)

HOST, PORT = "127.0.0.1", 8788
ORIGIN_RX = re.compile(r"^(https://heartwick-studio(-[a-z0-9-]+)?\.vercel\.app|http://(localhost|127\.0\.0\.1):3\d{3})$")
TOKEN_MAX_S = 11 * 60
CHECK_EVERY = 3.0
CLOUDFLARED = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
TUNNEL_RX = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
MAX_BODY = 8192
CHUNK = 1 << 16
log = logger("live")


def make_token(secret, exp):
    msg = f"hw1.{int(exp)}"
    return msg + "." + hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def verify_token(token, secret, now=None):
    if not token or not secret or not isinstance(token, str):
        return False
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "hw1" or not parts[1].isdigit() or len(parts[1]) > 12 or not re.fullmatch(r"[0-9a-f]{64}", parts[2]):
        return False
    now = time.time() if now is None else now
    exp = int(parts[1])
    if exp <= now or exp > now + TOKEN_MAX_S:
        return False
    want = hmac.new(secret.encode(), f"hw1.{exp}".encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, parts[2])


class State:
    """The snapshot as the live channel serves it: rebuilt only when the inputs' signature changes."""

    def __init__(self):
        self.lock = threading.Lock()
        self.index = M.Index()
        self.index.save = lambda: None          # the exporter owns media_index.json; this process only reads it
        self.checked = 0.0
        self.sig = None
        self.snap, self.rev, self.body = None, None, None
        self.cpu = None
        self.seen = {}                          # client ip+ua hash -> last request time (viewer count)
        self.stats = {"state": 0, "rebuilds": 0, "actions": 0, "media": 0, "rejected": 0}
        try:
            import psutil
            self.psutil = psutil
            psutil.cpu_percent(None)
        except ImportError:
            self.psutil = None

    def refresh(self, force=False):
        with self.lock:
            now = time.time()
            if not force and now - self.checked < CHECK_EVERY and self.snap is not None:
                return
            self.checked = now
            sig = S.signature()
            if sig == self.sig and not force and self.snap is not None:
                return
            self.index = M.Index()
            self.index.save = lambda: None
            snap, _ = S.build(self.index, now)
            snap["truth"] = S.self_check(snap, now)
            local = read_json(SNAPSHOT_LOCAL, {}) or {}
            snap["exporter"] = local.get("exporter")
            body = {k: v for k, v in snap.items() if k != "generated_at"}
            self.rev = hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]
            self.snap, self.sig = snap, sig
            self.stats["rebuilds"] += 1

    def live(self):
        cpu = ram = None
        if self.psutil:
            cpu = self.psutil.cpu_percent(None)        # since the previous call (one poll ago): no sampling wait
            ram = self.psutil.virtual_memory().percent
        now = time.time()
        self.seen = {k: t for k, t in self.seen.items() if now - t < 15}
        return {"at": iso(now), "cpu_pct": cpu, "ram_pct": ram, "viewers": len(self.seen), "server_pid": os.getpid()}


def make_handler(state, secret):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _origin(self):
            o = self.headers.get("Origin")
            return o if o and ORIGIN_RX.match(o) else None

        def _send(self, code, obj, extra=None):
            raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode()
            self.send_response(code)
            o = self._origin()
            if o:
                self.send_header("Access-Control-Allow-Origin", o)
                self.send_header("Vary", "Origin")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)

        def _auth(self, q):
            o = self.headers.get("Origin")
            if o is not None and not ORIGIN_RX.match(o):
                state.stats["rejected"] += 1
                self._send(403, {"error": "origin not allowed"})
                return False
            if not verify_token((q.get("t") or [""])[0], secret):
                state.stats["rejected"] += 1
                self._send(401, {"error": "unauthorized"})
                return False
            state.seen[hashlib.sha1(f"{self.client_address[0]}|{self.headers.get('User-Agent')}".encode()).hexdigest()] = time.time()
            return True

        def do_OPTIONS(self):
            self.send_response(204)
            o = self._origin()
            if o:
                self.send_header("Access-Control-Allow-Origin", o)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/health":
                return self._send(200, {"ok": True})
            if u.path == "/state":
                if not self._auth(q):
                    return
                state.stats["state"] += 1
                try:
                    state.refresh()
                except Exception as e:
                    log(f"rebuild failed: {type(e).__name__}: {e}")
                    if state.snap is None:
                        return self._send(500, {"error": "snapshot build failed"})
                same = (q.get("v") or [""])[0] == state.rev
                out = {"rev": state.rev, "same": same, "live": state.live()}
                if not same:
                    out["snapshot"] = state.snap
                return self._send(200, out)
            if u.path.startswith("/media/"):
                if not self._auth(q):
                    return
                return self._media(unquote(u.path[len("/media/"):]))
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path != "/action":
                return self._send(404, {"error": "not found"})
            if not self._auth(q):
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(413, {"error": "body too large"})
            try:
                a = json.loads(self.rfile.read(n).decode("utf-8"))
            except ValueError:
                return self._send(400, {"error": "invalid JSON"})
            why = A.validate(a)
            if why:
                return self._send(422, {"error": why})
            res = A.apply([a], "tunnel", log)
            state.stats["actions"] += 1
            try:
                state.refresh(force=True)
            except Exception:
                pass
            return self._send(200, {"ok": True, "result": res[0] if res else {"id": a["id"], "result": "already applied"}, "rev": state.rev})

        def _media(self, key):
            if not re.fullmatch(r"[0-9a-f]{12}/[A-Za-z0-9._-]{1,120}", key):
                return self._send(404, {"error": "not found"})
            if state.snap is None:
                state.refresh()
            path = state.index.local_path(key)
            if not path:
                return self._send(404, {"error": "not found"})
            size = os.path.getsize(path)
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(size - 1, int(m.group(2))) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            state.stats["media"] += 1
            self.send_response(206 if m else 200)
            o = self._origin()
            if o:
                self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Content-Type", M.content_type(key))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=3600")
            self.send_header("Content-Length", str(end - start + 1))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    b = fh.read(min(CHUNK, left))
                    if not b:
                        break
                    self.wfile.write(b)
                    left -= len(b)

    return Handler


# --------------------------------------------------------------------------- tunnel
def publish_url(url, env):
    base, token = env.get("DASHBOARD_URL"), env.get("INGEST_TOKEN")
    if not base or not token:
        return "no DASHBOARD_URL"
    req = urllib.request.Request(base.rstrip("/") + "/api/live-url", data=json.dumps({"url": url, "at": iso()}).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                                          "User-Agent": "heartwick-studio-live"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return f"POST {r.status}"
    except urllib.error.HTTPError as e:
        return f"POST failed: HTTP {e.code}"
    except Exception as e:
        return f"POST failed: {type(e).__name__}"


def kill_stale_tunnel():
    """A cloudflared left by a previous live.py (killed hard) still tunnels to our port: end it by its recorded pid."""
    old = read_json(LIVE_TUNNEL, {}) or {}
    pid = old.get("cloudflared_pid")
    if not pid:
        return
    try:
        import psutil
        p = psutil.Process(int(pid))
        if "cloudflared" in p.name().lower() and f"http://{HOST}:{PORT}" in " ".join(p.cmdline()):
            p.terminate()
            log(f"ended stale cloudflared pid {pid}")
    except Exception:
        pass


class Tunnel:
    def __init__(self, env):
        self.env = env
        self.proc, self.url, self.published = None, None, None
        self.started = 0.0
        self.fails = 0
        self.stop = False

    def _record(self):
        write_json(LIVE_TUNNEL, {"url": self.url, "cloudflared_pid": self.proc.pid if self.proc else None, "live_pid": os.getpid(),
                                 "since": iso(self.started)})

    def run(self):
        backoff = 5
        while not self.stop:
            try:
                self.proc = subprocess.Popen([CLOUDFLARED, "tunnel", "--url", f"http://{HOST}:{PORT}", "--no-autoupdate"],
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             text=True, encoding="utf-8", errors="replace",
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
            except OSError as e:
                log(f"cloudflared failed to start: {e}")
                time.sleep(60)
                continue
            self.started, self.fails, self.url = time.time(), 0, None
            log(f"cloudflared started pid {self.proc.pid}")
            self._record()
            for line in self.proc.stdout:                 # drains the pipe for the child's whole life
                m = TUNNEL_RX.search(line)
                if m and m.group(0) != self.url and "api.trycloudflare.com" not in m.group(0):
                    self.url = m.group(0)
                    log(f"tunnel url {self.url}")
                    self._record()
                    threading.Thread(target=self.publish, args=(self.url,), daemon=True).start()
            rc = self.proc.wait()
            ran = time.time() - self.started
            log(f"cloudflared exited code {rc} after {ran:.0f} s")
            self.url = None
            backoff = 5 if ran > 120 else min(120, backoff * 2)
            time.sleep(backoff)

    def publish(self, url):
        wait = 20
        while url == self.url and url != self.published and not self.stop:
            res = publish_url(url, self.env)
            log(f"live url -> site: {res}")
            if res.startswith("POST 2"):
                self.published = url
                return
            time.sleep(wait)
            wait = min(300, wait * 2)

    def watchdog(self):
        """Quick tunnels can go deaf while cloudflared runs: probe /health through it every 2 min, restart after 3 misses."""
        while not self.stop:
            time.sleep(120)
            url, p = self.url, self.proc
            if not url or not p or p.poll() is not None or time.time() - self.started < 90:
                continue
            try:
                with urllib.request.urlopen(urllib.request.Request(url + "/health", headers={"User-Agent": "heartwick-studio-live"}), timeout=15) as r:
                    ok = r.status == 200
            except Exception:
                ok = False
            self.fails = 0 if ok else self.fails + 1
            if self.fails >= 3:
                log("tunnel did not answer 3 times: restarting cloudflared")
                self.fails = 0
                try:
                    p.terminate()
                except OSError:
                    pass


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dashboard/live.py")
    ap.add_argument("--no-tunnel", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args(argv)
    lower_priority()
    env = read_env()
    secret = env.get("INGEST_TOKEN")
    if not secret:
        log("no INGEST_TOKEN in dashboard/data/studio.env: refusing to start")
        return 2
    state = State()
    srv = ThreadingHTTPServer((HOST, a.port), make_handler(state, secret))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"live server pid {os.getpid()} on http://{HOST}:{a.port}")
    tunnel = None
    if not a.no_tunnel:
        kill_stale_tunnel()
        tunnel = Tunnel(env)
        threading.Thread(target=tunnel.run, daemon=True).start()
        threading.Thread(target=tunnel.watchdog, daemon=True).start()
    c0, w0 = time.process_time(), time.time()
    try:
        while True:
            time.sleep(600)
            c1, w1 = time.process_time(), time.time()
            log(f"cpu {100 * (c1 - c0) / (w1 - w0) / (os.cpu_count() or 1):.3f}% of machine over {(w1 - w0) / 60:.0f} min; "
                f"viewers {len(state.seen)}; stats {state.stats}")
            c0, w0 = c1, w1
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel:
            tunnel.stop = True
            if tunnel.proc and tunnel.proc.poll() is None:
                tunnel.proc.terminate()
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
