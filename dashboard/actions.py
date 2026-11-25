"""Writes from Heartwick Studio (the Vercel site) applied on this laptop.

An action is {id, kind, video?, story?, mode?, note?, url?, at}. Kinds:
  approve / needs_changes / reset / published   -> dashboard/data/approvals.json[video]
  queue_story {story, mode: short|full}         -> a pending job in kaggle/queue.json (run by tools/run_queue.py)
  rerender {video}                               -> a pending job that re-runs that video's make command
Every action id is applied at most once (dashboard/data/actions_applied.json), whichever path brings it first:
the tunnel (instant, while the laptop is online) or the exporter's pull of the Vercel inbox (queued while it is off).
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio_common import (APPLIED, APPROVALS, DATA, QUEUE, ROOT, FileLock, iso, read_json, write_json)  # noqa: E402

APPROVAL_KINDS = {"approve": "approved", "needs_changes": "needs_changes", "reset": "pending", "published": "published"}
KINDS = set(APPROVAL_KINDS) | {"queue_story", "rerender"}
NAME_RX = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
ID_RX = re.compile(r"^[A-Za-z0-9-]{8,64}$")
MAX_APPLIED = 500
RUNNER = os.path.join(ROOT, "tools", "run_queue.py")
PYTHON = r"C:\Program Files\Python313\python.exe" if os.path.exists(r"C:\Program Files\Python313\python.exe") else sys.executable


def _out_dir(name):
    return os.path.join(ROOT, "out", name)


def _story_file(slug):
    return os.path.join(ROOT, "stories", f"{slug}.md")


def make_args_for(video):
    """The make command that produced out/<video>, rebuilt from its plan.json (sections, pace, name)."""
    plan = read_json(os.path.join(_out_dir(video), "plan.json"))
    if not plan or not plan.get("slug"):
        return None
    slug = plan["slug"]
    if not os.path.exists(_story_file(slug)):
        return None
    args = ["make", f"stories/{slug}.md", "--name", video]
    speed = plan.get("speed")
    if speed == 1.0:
        args.append("--short")
    elif speed not in (None, 0.94):
        args += ["--speed", str(speed)]
    try:
        sys.path.insert(0, ROOT)
        from engine import story as S
        _, secs = S.parse_story(_story_file(slug))
        titles = [s.get("title") for s in secs]
        mine = [s.get("title") for s in plan.get("sections", [])]
        if mine and mine != titles:
            idx = [titles.index(t) + 1 for t in mine if t in titles]
            if idx and idx == list(range(idx[0], idx[-1] + 1)):
                args += ["--sections", f"{idx[0]}-{idx[-1]}"]
    except Exception:
        pass
    return args


def queue_story_args(slug, mode):
    name = f"{slug}_short" if mode == "short" else f"{slug}_full"
    args = ["make", f"stories/{slug}.md", "--auto-pick", "--name", name]
    if mode == "short":
        args.insert(2, "--short")
    return args, name


def validate(a):
    """None when the action is well formed, else why not."""
    if not isinstance(a, dict):
        return "not an object"
    if not isinstance(a.get("id"), str) or not ID_RX.match(a["id"]):
        return "bad id"
    k = a.get("kind")
    if k not in KINDS:
        return "unknown kind"
    if k in APPROVAL_KINDS or k == "rerender":
        v = a.get("video")
        if not isinstance(v, str) or not NAME_RX.match(v) or not os.path.isfile(os.path.join(_out_dir(v), "plan.json")):
            return "unknown video"
    if k == "queue_story":
        s = a.get("story")
        if not isinstance(s, str) or not NAME_RX.match(s) or not os.path.isfile(_story_file(s)):
            return "unknown story"
        if a.get("mode") not in ("short", "full"):
            return "mode must be short or full"
    for f in ("note", "url"):
        if a.get(f) is not None and not isinstance(a.get(f), str):
            return f"{f} must be text"
    return None


def load_queue():
    q = read_json(QUEUE, None)
    return q if isinstance(q, dict) and isinstance(q.get("jobs"), list) else {"jobs": []}


def runner_pids():
    try:
        import psutil
    except ImportError:
        return []
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower().startswith("python") and any(c.endswith("run_queue.py") for c in (p.info["cmdline"] or [])):
                out.append(p.info["pid"])
        except Exception:
            pass
    return out


def start_runner_if_needed(log=None):
    """Start tools/run_queue.py (hidden, BelowNormal) only if jobs are pending and no runner is alive."""
    if runner_pids():
        return None
    if not any(j.get("status") == "pending" for j in load_queue()["jobs"]):
        return None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) \
        | getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    p = subprocess.Popen([PYTHON, "-u", RUNNER], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    if log:
        log(f"started queue runner pid {p.pid}")
    return p.pid


def _approve(a, now):
    with FileLock(APPROVALS):
        ap = read_json(APPROVALS, {}) or {}
        ap[a["video"]] = {"status": APPROVAL_KINDS[a["kind"]], "note": (a.get("note") or "")[:2000],
                          "url": (a.get("url") or "")[:500] or None, "at": a.get("at") or iso(now), "id": a["id"],
                          "applied_at": iso(now)}
        write_json(APPROVALS, ap, indent=1)
    return f"{a['video']} -> {APPROVAL_KINDS[a['kind']]}"


def _enqueue(a, now):
    if a["kind"] == "queue_story":
        args, name = queue_story_args(a["story"], a["mode"])
        label = f"Queue {a['mode']} video of {a['story']} ({name})"
    else:
        args = make_args_for(a["video"])
        if not args:
            raise ValueError("cannot rebuild the make command for this video")
        label = f"Re-render {a['video']}"
    with FileLock(QUEUE):
        q = load_queue()
        if any(j.get("id") == a["id"] for j in q["jobs"]):
            return "already queued"
        q["jobs"].append({"id": a["id"], "label": label, "args": args, "status": "pending", "added_at": iso(now),
                          "source": "dashboard", "note": (a.get("note") or "")[:500]})
        write_json(QUEUE, q, indent=1)
    return label


def apply(actions, source, log=None, now=None):
    """Apply new actions (unknown ids only). Returns the list of result rows written to actions_applied.json."""
    now = time.time() if now is None else now
    os.makedirs(DATA, exist_ok=True)
    results = []
    with FileLock(APPLIED):
        done = read_json(APPLIED, []) or []
        seen = {r.get("id") for r in done}
        for a in actions:
            aid = a.get("id") if isinstance(a, dict) else None
            if not aid or aid in seen:
                continue
            why = validate(a)
            row = {"id": aid, "kind": a.get("kind"), "video": a.get("video"), "story": a.get("story"),
                   "at": a.get("at"), "applied_at": iso(now), "via": source}
            if why:
                row.update(result="rejected", msg=why)
            else:
                try:
                    row.update(result="ok", msg=_approve(a, now) if a["kind"] in APPROVAL_KINDS else _enqueue(a, now))
                except Exception as e:
                    row.update(result="error", msg=f"{type(e).__name__}: {e}"[:300])
            seen.add(aid)
            done.append(row)
            results.append(row)
            if log:
                log(f"action {aid} {row['kind']} via {source}: {row['result']} {row.get('msg', '')}")
        if results:
            write_json(APPLIED, done[-MAX_APPLIED:], indent=1)
    if any(r["result"] == "ok" and r["kind"] in ("queue_story", "rerender") for r in results):
        try:
            start_runner_if_needed(log)
        except Exception as e:
            if log:
                log(f"queue runner start failed: {e}")
    return results
