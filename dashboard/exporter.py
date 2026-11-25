"""Heartwick Studio exporter: laptop -> Vercel (snapshot + media), Vercel -> laptop (queued writes).

  python dashboard/exporter.py                 # loop (started by tools/start_heartwick.ps1)
  python dashboard/exporter.py --once          # one pass, then exit

Each minute it stats the inputs (~100 files, no reads). Only when something changed does it rebuild the snapshot
(studio_state.build + self_check), encode a 480p preview for a new/changed mp4, upload new media to the private
Blob store, and POST the snapshot to /api/ingest. To keep Blob operations low (Hobby: 2,000 uploads a month for
the whole account) a snapshot is posted at once only for changes that matter (a stage, a job state, an approval,
new media, a new error); minor ones (quota minutes, timers) wait for the hourly heartbeat. The site's live view
comes through the tunnel (dashboard/live.py), not from these uploads.
Kaggle: `kernels status` for unfinished jobs every 5 min, `kaggle quota` every 30 min -> dashboard/data/kaggle_live.json.
Writes: GET /api/inbox at start, every 60 min, and at once when the live server drops dashboard/data/pull_now.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import actions as A  # noqa: E402
import media as M  # noqa: E402
import studio_state as S  # noqa: E402
from studio_common import (KAGGLE_EXE, KAGGLE_LIVE, KAGGLE_USER, PULL_NOW, SNAPSHOT_LOCAL, FileLock, iso,  # noqa: E402
                           local_iso_to_epoch, logger, lower_priority, read_env, read_json, write_json)

log = logger("exporter")
UA = "heartwick-studio-exporter"
KSTATUS_EVERY = 300
QUOTA_EVERY = 1800
PULL_EVERY = 3600         # s; writes made while the laptop is online arrive at once through the tunnel
HEARTBEAT = 7200
MIN_GAP = 120            # s between two snapshot uploads, even for important changes
MAX_EVENTS = 30


def _run_kaggle(*args, timeout=90):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    r = subprocess.run([KAGGLE_EXE, *args], capture_output=True, timeout=timeout, creationflags=flags)
    return (r.stdout + r.stderr).decode("utf-8", errors="replace")


def parse_quota(text):
    m = re.search(r"^GPU\s+([\d.]+)h\s+([\d.]+)h\s+([\d.]+)h\s+(\S+)", text, re.M)
    if not m:
        return None
    return {"used_h": float(m.group(1)), "remaining_h": float(m.group(2)), "total_h": float(m.group(3)), "refresh_at": m.group(4),
            "checked_at": iso(), "source": "kaggle quota"}


def parse_status(text):
    m = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', text) or re.search(r"KernelWorkerStatus\.([A-Z_]+)", text)
    if m:
        return m.group(1)
    t = text.lower()
    if "permission 'kernels.get' was denied" in t or "404" in t or "not found" in t:
        return "GONE"        # deleted on Kaggle (e.g. to free a GPU session): it will never run
    return None


class Exporter:
    def __init__(self, env, upload=True):
        self.env, self.upload = env, upload
        self.index = M.Index()
        self.events = []                          # exporter's own problems, shown in the errors feed
        self.sig = None
        self.important = None
        self.minor = None
        self.put_at = 0.0
        self.pending_put = False
        self.k_at = self.q_at = self.pull_at = 0.0
        self.snapshot = None
        self.started = time.time()
        self.stats = {"ingests": 0, "ingest_fail": 0, "pulls": 0, "applied": 0, "kaggle_calls": 0, "previews": 0}
        self.last_ingest = None
        self.media_stats = None

    def event(self, level, source, text):
        self.events.append((iso(), level, source, text))
        self.events = self.events[-MAX_EVENTS:]
        log(f"{level} {source}: {text}")

    # ---------------------------------------------------------------- Kaggle
    def poll_kaggle(self, now, force=False):
        kl = read_json(KAGGLE_LIVE, {}) or {}
        changed = False
        if force or now - self.q_at >= QUOTA_EVERY:
            self.q_at = now
            try:
                self.stats["kaggle_calls"] += 1
                q = parse_quota(_run_kaggle("quota"))
                if q:
                    kl["quota"] = q
                    kl.pop("error", None)
                else:
                    kl["error"] = "kaggle quota: unexpected output"
                changed = True
            except Exception as e:
                kl["error"] = f"kaggle quota failed: {type(e).__name__}"
                changed = True
        active = []
        for j in ((self.snapshot or {}).get("kaggle") or {}).get("jobs") or []:
            t = local_iso_to_epoch(j.get("pushed_at")) or 0
            if j.get("state") in ("unknown", "queued", "running") and now - t < 48 * 3600:
                active.append(j["kernel"])
        if active and (force or now - self.k_at >= KSTATUS_EVERY):
            self.k_at = now
            kern = kl.setdefault("kernels", {})
            tr = kl.setdefault("transitions", {})
            for k in dict.fromkeys(active):
                try:
                    self.stats["kaggle_calls"] += 1
                    st = parse_status(_run_kaggle("kernels", "status", f"{KAGGLE_USER}/{k}"))
                except Exception as e:
                    st = None
                    self.event("warn", "kaggle", f"status of {k} failed: {type(e).__name__}")
                if not st:
                    continue
                prev = (kern.get(k) or {}).get("status")
                kern[k] = {"status": st, "checked_at": iso()}
                if st != prev:
                    tr.setdefault(k, []).append({"state": st, "at": iso()})
                    tr[k] = tr[k][-20:]
                    log(f"kaggle {k}: {prev} -> {st}")
                changed = True
        if changed:
            kl["checked_at"] = iso()
            write_json(KAGGLE_LIVE, kl, indent=1)
        return changed

    # ---------------------------------------------------------------- Vercel
    def _req(self, method, path, body=None, headers=None, timeout=30):
        base, token = self.env.get("DASHBOARD_URL"), self.env.get("INGEST_TOKEN")
        if not base or not token:
            raise RuntimeError("DASHBOARD_URL / INGEST_TOKEN missing in dashboard/data/studio.env")
        h = {"Authorization": "Bearer " + token, "User-Agent": UA, **(headers or {})}
        req = urllib.request.Request(base.rstrip("/") + path, data=body, method=method, headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()

    def pull_inbox(self, now, reason):
        self.pull_at = now
        try:
            st, raw = self._req("GET", "/api/inbox")
            items = (json.loads(raw) or {}).get("actions") or []
        except Exception as e:
            self.event("warn", "inbox", f"could not read queued changes from Vercel ({reason}): {type(e).__name__} {getattr(e, 'code', '')}".strip())
            return 0
        self.stats["pulls"] += 1
        res = A.apply(items, "vercel inbox", log)
        self.stats["applied"] += len(res)
        if res:
            log(f"inbox ({reason}): {len(items)} queued on Vercel, {len(res)} new applied")
        return len(res)

    def ingest(self, snap):
        body = gzip.compress(json.dumps(snap, ensure_ascii=False, separators=(",", ":")).encode(), 6)
        try:
            st, raw = self._req("POST", "/api/ingest", body, {"Content-Type": "application/json", "Content-Encoding": "gzip"}, timeout=60)
            self.stats["ingests"] += 1
            self.last_ingest = {"at": iso(), "ok": True, "bytes": len(body)}
            return True
        except urllib.error.HTTPError as e:
            msg = f"ingest HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
        except Exception as e:
            msg = f"ingest failed: {type(e).__name__}"
        self.stats["ingest_fail"] += 1
        self.last_ingest = {"at": iso(), "ok": False, "error": msg}
        self.event("warn", "exporter", msg)
        return False

    # ---------------------------------------------------------------- build
    @staticmethod
    def fingerprints(snap):
        imp = {
            "counts": snap["counts"], "stories": [(s["slug"], s.get("error"), s.get("lines")) for s in snap["stories"]],
            "videos": [(v["name"], v["stage"], v["progress"], v["approval"], sorted((k, (r or {}).get("key")) for k, r in v["media"].items()),
                        v["errors"], v["warnings"], v["lufs"], v["duration_s"]) for v in snap["videos"]],
            "jobs": [(j["kernel"], j["pushed_at"], j["state"]) for j in snap["kaggle"]["jobs"]],
            "queue": [(snap["queue"]["current"] or {}).get("text"), snap["queue"]["runner"]["running"],
                      [(j["id"], j["status"]) for j in snap["queue"]["jobs"]]],
            "actions": [a.get("id") for a in snap["actions"]["applied"]],
            "truth": (snap.get("truth") or {}).get("n_fail"),
            # research/ is written by another agent, possibly often: its changes ride the hourly minor upload
            "events": [e for e in snap["events"] if e["level"] != "info"][:20],
        }
        h = lambda o: hashlib.sha1(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()  # noqa: E731
        minor = {k: v for k, v in snap.items() if k not in ("generated_at", "exporter", "truth")}
        return h(imp), h(minor)

    def encode_one_preview(self, previews):
        for name, mp4, pv in previews:
            if not os.path.isfile(pv):
                self.stats["previews"] += 1
                ok = M.encode_preview(mp4, pv, log)
                if not ok:
                    self.event("warn", "media", f"480p preview encode failed for {name}")
                return True
        return False

    def pass_once(self, now, force=False):
        kchanged = self.poll_kaggle(now, force=force)
        if os.path.exists(PULL_NOW):
            try:
                os.remove(PULL_NOW)
            except OSError:
                pass
            self.pull_inbox(now, "nudge")
        elif force or now - self.pull_at >= PULL_EVERY:
            self.pull_inbox(now, "start" if force else "timer")
        sig = S.signature()
        if sig == self.sig and not kchanged and not force and not self.pending_put and now - self.put_at < HEARTBEAT:
            return False
        rebuilt = False
        if sig != self.sig or kchanged or force or self.snapshot is None:
            extra = [e for e in self.events]
            snap, previews = S.build(self.index, now, extra_events=extra)
            if self.encode_one_preview(previews):
                return True                   # a new preview changes the inputs: rebuild on the next pass (soon)
            snap["truth"] = S.self_check(snap, now)
            self.snapshot = snap
            self.sig = sig
            rebuilt = True
        snap = self.snapshot
        imp, minor = self.fingerprints(snap)
        due = False
        if imp != self.important:
            due = now - self.put_at >= MIN_GAP
            self.pending_put = not due
        elif minor != self.minor or self.pending_put:
            due = now - self.put_at >= 3600
            self.pending_put = not due
        if now - self.put_at >= HEARTBEAT:
            due = True
        snap["exporter"] = {"pid": os.getpid(), "started_at": iso(self.started), "stats": self.stats, "last_ingest": self.last_ingest,
                            "media": self.media_stats, "heartbeat_s": HEARTBEAT, "upload": self.upload}
        with FileLock(SNAPSHOT_LOCAL):
            write_json(SNAPSHOT_LOCAL, snap)
        if not due or not self.upload:
            return rebuilt
        token = self.env.get("BLOB_READ_WRITE_TOKEN")
        if token:
            keys = media_keys(snap)
            try:
                self.media_stats = {**M.sync(self.index, keys, token, log), "at": iso()}
                if self.media_stats["failed"]:
                    self.event("warn", "media", f"{self.media_stats['failed']} media uploads failed (retried next pass)")
            except Exception as e:
                self.event("warn", "media", f"media sync failed: {type(e).__name__}: {e}"[:200])
        snap["exporter"]["media"] = self.media_stats
        snap["exporter"]["uploaded_keys"] = sorted(k for k, e in self.index.keys.items() if e.get("uploaded_at"))
        if self.ingest(snap):
            self.put_at, self.important, self.minor, self.pending_put = now, imp, minor, False
            log(f"snapshot posted ({'important' if imp else 'minor'} change, {len(json.dumps(snap)) // 1024} KB)")
        return True


def media_keys(snap):
    keys = []
    for v in snap["videos"]:
        keys += [r["key"] for r in v["media"].values() if r]
        keys += [s["key"] for s in v.get("samples") or [] if s.get("key")]
    keys += [s["key"] for s in snap["research"]["samples"] if s.get("key")]
    return list(dict.fromkeys(keys))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dashboard/exporter.py")
    ap.add_argument("--loop-seconds", type=float, default=60)
    ap.add_argument("--hours", type=float, default=0, help="stop after this many hours (0 = run until stopped)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-upload", action="store_true", help="build and write dashboard/data/snapshot.json only")
    a = ap.parse_args(argv)
    lower_priority()
    env = read_env()
    ex = Exporter(env, upload=not a.no_upload)
    log(f"exporter started pid {os.getpid()} (upload {'on' if ex.upload else 'off'}, blob token {'set' if env.get('BLOB_READ_WRITE_TOKEN') else 'missing'})")
    deadline = time.time() + a.hours * 3600 if a.hours else None
    c0, w0 = time.process_time(), time.time()
    first = True
    while True:
        try:
            busy = ex.pass_once(time.time(), force=first)
        except Exception as e:
            busy = False
            ex.event("err", "exporter", f"pass failed: {type(e).__name__}: {e}"[:300])
        first = False
        if a.once:
            if busy and not ex.snapshot:
                continue
            break
        if deadline and time.time() > deadline:
            break
        if time.time() - w0 >= 600:
            c1, w1 = time.process_time(), time.time()
            log(f"cpu {100 * (c1 - c0) / (w1 - w0) / (os.cpu_count() or 1):.3f}% of machine over {(w1 - w0) / 60:.0f} min "
                f"(own process; ffmpeg/kaggle children excluded); stats {ex.stats}")
            c0, w0 = c1, w1
        wake = time.time() + (5 if busy else a.loop_seconds)
        while time.time() < wake and not os.path.exists(PULL_NOW):     # a nudge cuts the wait short
            time.sleep(min(5, max(0.1, wake - time.time())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
