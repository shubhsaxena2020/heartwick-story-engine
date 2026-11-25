"""Media for Heartwick Studio: content-addressed keys, 480p previews, uploads to the private Vercel Blob store.

A media key is "<sha1[:12]>/<file name>". The site serves it at /api/media/<key> (login required) from the Blob
pathname "m/<key>"; the live server serves the same key from the local file through the tunnel. Keys change
only when a file's bytes change, so an upload happens only for new or changed files, and browsers can cache
a key forever. Index: dashboard/data/media_index.json {key: {path, mtime, size, uploaded_at}} plus a path cache.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio_common import MEDIA_INDEX, PREVIEWS, ROOT, FileLock, iso, read_json, write_json  # noqa: E402

FFMPEG = "ffmpeg"
NODE = r"C:\Program Files\nodejs\node.exe" if os.path.exists(r"C:\Program Files\nodejs\node.exe") else "node"
BLOB_PUT = os.path.join(ROOT, "studio", "scripts", "blob-put.mjs")
SAFE = re.compile(r"[^A-Za-z0-9._-]+")
MAX_TOTAL_BYTES = 300 * 1024 * 1024          # hard cap on what is kept in Blob (Hobby store: 1 GB)
PREVIEW_HEIGHT = 854                         # 480x854 for the 1080x1920 vertical videos


class Index:
    """sha cache by (path, mtime, size) and the set of uploaded keys. One instance per process."""

    def __init__(self):
        d = read_json(MEDIA_INDEX, {}) or {}
        self.paths = d.get("paths", {})       # abs path -> {mtime, size, sha}
        self.keys = d.get("keys", {})         # key -> {path, size, uploaded_at}
        self.dirty = False

    def save(self):
        if self.dirty:
            with FileLock(MEDIA_INDEX):
                write_json(MEDIA_INDEX, {"paths": self.paths, "keys": self.keys})
            self.dirty = False

    def sha(self, path):
        try:
            st = os.stat(path)
        except OSError:
            return None
        c = self.paths.get(path)
        if c and c["mtime"] == st.st_mtime and c["size"] == st.st_size:
            return c["sha"]
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        self.paths[path] = {"mtime": st.st_mtime, "size": st.st_size, "sha": h.hexdigest()}
        self.dirty = True
        return h.hexdigest()

    def ref(self, path, name=None):
        """{key, bytes} for a local file (no upload), or None when it does not exist."""
        s = self.sha(path)
        if not s:
            return None
        key = f"{s[:12]}/{SAFE.sub('_', name or os.path.basename(path))}"
        if key not in self.keys or self.keys[key].get("path") != path:
            self.keys.setdefault(key, {})
            self.keys[key].update(path=path, size=os.path.getsize(path))
            self.dirty = True
        return {"key": key, "bytes": os.path.getsize(path)}

    def local_path(self, key):
        """The local file behind a key (for the live server), only when its bytes still match."""
        k = self.keys.get(key)
        if not k:
            return None
        p = k.get("path")
        if not p or not os.path.isfile(p):
            return None
        s = self.sha(p)
        return p if s and s[:12] == key.split("/", 1)[0] else None


def preview_path(video, mp4_sha):
    return os.path.join(PREVIEWS, f"{video}_{mp4_sha[:12]}_480p.mp4")


def _limit(proc):
    try:
        import psutil
        p = psutil.Process(proc.pid)
        p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)   # CPUs: inherited from the exporter (start_heartwick.ps1 pins it)
    except Exception:
        pass


def encode_preview(src, dst, log=None):
    """480p H.264 CRF 30 + AAC 64 kb/s, faststart, 2 threads at idle priority (main-laptop CPU cap 75 %)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part.mp4"
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-vf", f"scale=-2:{PREVIEW_HEIGHT}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
           "-maxrate", "900k", "-bufsize", "1800k", "-pix_fmt", "yuv420p", "-threads", "2",
           "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", tmp]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _limit(p)
    _, err = p.communicate(timeout=1800)
    if p.returncode != 0 or not os.path.exists(tmp):
        if log:
            log(f"preview encode failed for {src}: {err.decode(errors='replace')[-300:]}")
        return False
    os.replace(tmp, dst)
    video = os.path.basename(dst).rsplit("_", 2)[0]
    for old in os.listdir(os.path.dirname(dst)):           # keep only the current preview of this video
        if old != os.path.basename(dst) and old.endswith("_480p.mp4") and old.rsplit("_", 2)[0] == video:
            try:
                os.remove(os.path.join(os.path.dirname(dst), old))
            except OSError:
                pass
    if log:
        log(f"preview {os.path.basename(dst)} {os.path.getsize(dst) / 1e6:.1f} MB in {time.time() - t0:.0f} s")
    return True


def blob_call(ops, token, log=None, timeout=1800):
    """Run studio/scripts/blob-put.mjs with a list of {op: put|del, file?, pathname, contentType?, maxAge?}."""
    if not ops:
        return []
    env = dict(os.environ, BLOB_READ_WRITE_TOKEN=token)
    env.pop("ELECTRON_RUN_AS_NODE", None)
    p = subprocess.Popen([NODE, BLOB_PUT], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=os.path.join(ROOT, "studio"), env=env,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    out, err = p.communicate(json.dumps(ops).encode(), timeout=timeout)
    try:
        res = json.loads(out.decode() or "[]")
    except ValueError:
        res = [{"ok": False, "error": (out + err).decode(errors="replace")[-300:]} for _ in ops]
    if p.returncode and log:
        log(f"blob-put exit {p.returncode}: {err.decode(errors='replace')[-300:]}")
    return res


def content_type(name):
    ext = name.rsplit(".", 1)[-1].lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "mp4": "video/mp4", "m4a": "audio/mp4",
            "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg", "webp": "image/webp"}.get(ext, "application/octet-stream")


def sync(index, wanted_keys, token, log=None):
    """Upload wanted keys not yet in Blob; delete uploaded keys nothing references any more.
    Returns {uploaded, deleted, failed, bytes_stored}."""
    want = [k for k in wanted_keys if k in index.keys]
    total = sum(index.keys[k].get("size", 0) for k in want)
    if total > MAX_TOTAL_BYTES:
        want.sort(key=lambda k: index.keys[k].get("size", 0))         # keep the small, most useful files first
        keep, acc = [], 0
        for k in want:
            acc += index.keys[k].get("size", 0)
            if acc <= MAX_TOTAL_BYTES:
                keep.append(k)
        want = keep
    ops = []
    for k in want:
        e = index.keys[k]
        if not e.get("uploaded_at") and e.get("path") and os.path.isfile(e["path"]):
            ops.append({"op": "put", "file": e["path"], "pathname": f"m/{k}", "contentType": content_type(k),
                        "maxAge": 31536000, "multipart": e.get("size", 0) > 8 * 1024 * 1024})
    gone = [k for k, e in index.keys.items() if e.get("uploaded_at") and k not in set(wanted_keys)]
    if gone:
        ops.append({"op": "del", "pathnames": [f"m/{k}" for k in gone]})
    res = blob_call(ops, token, log) if ops else []
    up = failed = 0
    for op, r in zip(ops, res):
        if op["op"] == "put":
            k = op["pathname"][2:]
            if r.get("ok"):
                index.keys[k]["uploaded_at"] = iso()
                up += 1
            else:
                failed += 1
                if log:
                    log(f"upload failed {k}: {r.get('error')}")
        elif r.get("ok"):
            for k in gone:
                index.keys.pop(k, None)
    for k in list(index.keys):                         # forget local keys that are neither wanted nor uploaded
        if k not in set(wanted_keys) and not index.keys[k].get("uploaded_at"):
            index.keys.pop(k, None)
    index.dirty = True
    index.save()
    stored = sum(e.get("size", 0) for e in index.keys.values() if e.get("uploaded_at"))
    return {"uploaded": up, "deleted": len(gone) if gone else 0, "failed": failed, "bytes_stored": stored}
