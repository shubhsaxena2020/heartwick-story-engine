"""Sequential Heartwick job queue (one Kaggle job at a time). Light: it only calls heartwick.py and waits.

Jobs live in kaggle/queue.json {"jobs": [{id, label, args, status, ...}]}; status is pending -> running -> done | failed.
Heartwick Studio's "queue story" and "re-render" buttons add pending jobs (dashboard/actions.py). This runner takes
the oldest pending job, runs `python heartwick.py <args>` at BelowNormal priority, records the result, and exits
when nothing is pending. Safe to start any time (tools/start_heartwick.ps1 does, every 15 min, only when a job is
pending and no runner is alive): a second runner exits at once, and a job left "running" by a runner that died is
marked "interrupted" instead of being pushed to Kaggle again. Log: kaggle/queue_runner.log (shown in the dashboard).

(Until 2026-11-21 this file held a fixed list: pilot fetch, pilot render, full Pip, Blue Mitten. That list is still
being worked through by the runner started then, which keeps its own copy in memory.)
"""
import datetime, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dashboard"))
from studio_common import QUEUE, FileLock, iso, read_json, write_json  # noqa: E402

LOG = os.path.join(ROOT, "kaggle", "queue_runner.log")
PY = sys.executable

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    psutil = None


def log(*a):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


def other_runner():
    if not psutil:
        return None
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] != me and (p.info["name"] or "").lower().startswith("python") and \
                    any(c.endswith("run_queue.py") for c in (p.info["cmdline"] or [])):
                return p.info["pid"]
        except Exception:
            pass
    return None


def alive(pid):
    try:
        return bool(pid) and psutil is not None and psutil.pid_exists(int(pid))
    except Exception:
        return False


def update(job_id, **fields):
    with FileLock(QUEUE):
        q = read_json(QUEUE, {"jobs": []}) or {"jobs": []}
        for j in q["jobs"]:
            if j.get("id") == job_id:
                j.update(fields)
        write_json(QUEUE, q, indent=1)


def take():
    """Mark dead 'running' jobs interrupted, then claim the oldest pending job."""
    with FileLock(QUEUE):
        q = read_json(QUEUE, {"jobs": []}) or {"jobs": []}
        for j in q["jobs"]:
            if j.get("status") == "running" and not alive(j.get("runner_pid")):
                j.update(status="interrupted", finished_at=iso())
                log("job", j.get("id"), "was running when its runner stopped: marked interrupted (not re-pushed)")
        job = next((j for j in q["jobs"] if j.get("status") == "pending"), None)
        if job:
            job.update(status="running", started_at=iso(), runner_pid=os.getpid())
        write_json(QUEUE, q, indent=1)
    return job


def run(args):
    log("RUN", " ".join(args))
    r = subprocess.run([PY, "-u", os.path.join(ROOT, "heartwick.py"), *args], cwd=ROOT, capture_output=True,
                       encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                       creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "%|" not in l)
    log(f"rc={r.returncode}\n" + out[-3000:])
    return r.returncode


if __name__ == "__main__":
    other = other_runner()
    if other:
        sys.exit(f"another queue runner is alive (pid {other})")
    log("queue runner started, pid", os.getpid())
    while True:
        job = take()
        if not job:
            break
        rc = run(job["args"])
        update(job["id"], status="done" if rc == 0 else "failed", rc=rc, finished_at=iso())
        time.sleep(2)
    log("queue runner finished")
