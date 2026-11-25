"""Shared helpers for Heartwick Studio's laptop side (exporter, live channel, actions).

Paths, the secrets file, logging, atomic JSON writes and a cross-process lock. Nothing here does work on import.
Secrets live in dashboard/data/studio.env (git-ignored): DASHBOARD_URL, INGEST_TOKEN, DASHBOARD_PASSWORD,
BLOB_READ_WRITE_TOKEN. Environment variables of the same names override the file.
"""
import datetime
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dashboard", "data")
ENV_FILE = os.path.join(DATA, "studio.env")
LOGS = os.path.join(DATA, "logs")
APPROVALS = os.path.join(DATA, "approvals.json")
APPLIED = os.path.join(DATA, "actions_applied.json")
SNAPSHOT_LOCAL = os.path.join(DATA, "snapshot.json")
KAGGLE_LIVE = os.path.join(DATA, "kaggle_live.json")
MEDIA_INDEX = os.path.join(DATA, "media_index.json")
PREVIEWS = os.path.join(DATA, "previews")
LIVE_TUNNEL = os.path.join(DATA, "live_tunnel.json")
PULL_NOW = os.path.join(DATA, "pull_now")
QUEUE = os.path.join(ROOT, "kaggle", "queue.json")
QUEUE_LOG = os.path.join(ROOT, "kaggle", "queue_runner.log")
USAGE = os.path.join(ROOT, "kaggle", "usage.jsonl")
KAGGLE_EXE = os.environ.get("KAGGLE_EXE", r"C:\Users\shubh\.kaggle-venv\Scripts\kaggle.exe")
KAGGLE_USER = os.environ.get("KAGGLE_USER", "shubhsaxena2008")
PROD_ORIGIN = "https://heartwick-studio.vercel.app"
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def read_env():
    env = {}
    try:
        for line in open(ENV_FILE, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"')
    except OSError:
        pass
    for k in ("DASHBOARD_URL", "INGEST_TOKEN", "DASHBOARD_PASSWORD", "BLOB_READ_WRITE_TOKEN"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def iso(t=None):
    """Epoch seconds (or now) -> ISO 8601 with the IST offset, the operator's zone."""
    return datetime.datetime.fromtimestamp(time.time() if t is None else t, IST).isoformat(timespec="seconds")


def local_iso_to_epoch(s):
    """Naive local timestamps in the engine's logs ("2026-11-21T13:35:09") are laptop time (IST)."""
    if not s:
        return None
    try:
        d = datetime.datetime.fromisoformat(str(s).replace(" ", "T"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=IST)
    return d.timestamp()


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write_json(path, data, indent=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
    for _ in range(20):  # Windows: a reader may hold the target open for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def logger(name):
    os.makedirs(LOGS, exist_ok=True)
    path = os.path.join(LOGS, f"{name}.log")

    def log(*parts):
        line = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(p) for p in parts)
        try:
            if os.path.exists(path) and os.path.getsize(path) > 2_000_000:   # bounded: keep one previous file
                os.replace(path, path + ".prev")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    return log


class FileLock:
    """Exclusive lock between the exporter, the live server and run_queue (O_EXCL lock file, stale after 60 s)."""

    def __init__(self, path, timeout=15):
        self.path, self.timeout = path + ".lock", timeout

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > 60:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"lock busy: {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass
