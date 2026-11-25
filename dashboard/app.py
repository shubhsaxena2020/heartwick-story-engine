"""Heartwick dashboard: story queue, per-video status + previews, Kaggle jobs/GPU quota, errors, approve / needs-changes.

Run:   python dashboard/app.py                 -> http://127.0.0.1:8766  (localhost only)
       HEARTWICK_DASH_PASSWORD=... python dashboard/app.py --host 0.0.0.0   (LAN; password required, Basic auth)
Never exposes anything without a password when bound beyond localhost. Approvals are stored locally in
dashboard/data/approvals.json. Everything shown is read from the engine's real files (stories/, out/, cache/, kaggle/usage.jsonl).
"""
import argparse, base64, datetime, glob, html, json, os, secrets, socket, subprocess, sys, time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "dashboard", "data")
APPROVALS = os.path.join(DATA, "approvals.json")
GPU_WEEKLY_QUOTA_H = 30.0
STAGES = ["script", "narration", "images", "render", "review", "approved"]
PASSWORD = os.environ.get("HEARTWICK_DASH_PASSWORD", "")
sys.path.insert(0, ROOT)

app = FastAPI(title="Heartwick dashboard", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if PASSWORD:
        h = request.headers.get("authorization", "")
        ok = False
        if h.startswith("Basic "):
            try:
                _, pw = base64.b64decode(h[6:]).decode().split(":", 1)
                ok = secrets.compare_digest(pw, PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return PlainTextResponse("auth required", 401, headers={"WWW-Authenticate": 'Basic realm="heartwick"'})
    return await call_next(request)


# ------------------------------------------------------------------ state
def _json(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default


def approvals():
    return _json(APPROVALS, {}) or {}


def _video_state(name):
    out = os.path.join(ROOT, "out", name)
    plan = _json(os.path.join(out, "plan.json"))
    v = {"name": name, "stage": "script", "files": {}, "errors": [], "qa": None, "story": None}
    if not plan:
        return None
    slug = plan.get("slug")
    v["story"] = slug
    v["title"] = plan.get("title")
    assets = os.path.join(ROOT, "cache", slug, "assets")
    have = {os.path.splitext(f)[0] for f in os.listdir(assets)} if os.path.isdir(assets) else set()
    utts = [u["id"] for s in plan["sections"] for sh in s["shots"] for u in sh["utts"]]
    locs = {sh["bg"] for s in plan["sections"] for sh in s["shots"]}
    poses = {f"pose_{c['char']}_{c['pose']}_cut" for s in plan["sections"] for sh in s["shots"] for c in sh["chars"]}
    need_img = {f"bg_{l}" for l in locs} | {f"bg_{l}_depth" for l in locs} | poses
    v["progress"] = {"narration": f"{sum(u in have for u in utts)}/{len(utts)}", "images": f"{len(need_img & have)}/{len(need_img)}"}
    narr = all(u in have for u in utts)
    imgs = need_img <= have
    mp4 = os.path.join(out, f"{name}.mp4")
    qa = _json(os.path.join(out, "qa_report.json"))
    v["qa"] = qa
    stage = "script"
    if narr:
        stage = "narration"
    if narr and imgs:
        stage = "images"
    if os.path.exists(mp4):
        stage = "render"
    if os.path.exists(mp4) and qa:
        stage = "review"
    ap = approvals().get(name)
    v["approval"] = ap
    if ap and ap.get("status") == "approved" and stage == "review":
        stage = "approved"
    v["stage"] = stage
    for f in ("thumbnail.jpg", "contact_sheet.jpg", "vs_reference.jpg", f"{name}.mp4", f"{name}.srt", "SEO_PACK.md", "qa_report.json",
              "master_candidates.jpg"):
        if os.path.exists(os.path.join(out, f)):
            v["files"][f] = f"/files/{name}/{f}"
    samples = sorted(glob.glob(os.path.join(out, "samples", "*.wav")))
    v["samples"] = [f"/files/{name}/samples/{os.path.basename(p)}" for p in samples]
    wl = _json(os.path.join(out, "worker_log.json"))
    if wl:
        v["errors"] += [f"Kaggle worker: {e}" for e in wl.get("errors", [])]
    if qa:
        a = qa.get("audio") or {}
        if a.get("integrated_lufs") is not None and not (-16.5 <= a["integrated_lufs"] <= -13.5):
            v["errors"].append(f"Loudness {a['integrated_lufs']} LUFS is outside -16..-14")
        if qa.get("thumbnail_error"):
            v["errors"].append("Thumbnail: " + qa["thumbnail_error"])
    v["mtime"] = os.path.getmtime(mp4) if os.path.exists(mp4) else os.path.getmtime(os.path.join(out, "plan.json"))
    return v


def _stories():
    from engine import story as S
    res = []
    for p in sorted(glob.glob(os.path.join(ROOT, "stories", "*.md"))):
        try:
            meta, secs = S.parse_story(p)
            words = sum(len(u["text"].split()) for s in secs for sh in s["shots"] for u in sh["utts"])
            res.append({"slug": meta.get("slug"), "title": meta.get("title"), "file": os.path.relpath(p, ROOT),
                        "chapters": len(secs), "shots": sum(len(s["shots"]) for s in secs), "words": words})
        except Exception as e:
            res.append({"slug": os.path.basename(p), "title": "(parse error)", "error": repr(e)})
    return res


_kstatus = {"t": 0, "data": {}}


def _kaggle_live():
    if time.time() - _kstatus["t"] < 90:
        return _kstatus["data"]
    exe = os.environ.get("KAGGLE_EXE", r"C:\Users\shubh\.kaggle-venv\Scripts\kaggle.exe")
    data = {}
    slugs = []
    for j in _kaggle_usage()["jobs"]:
        sl = j["slug"].split("/")[-1]
        if sl not in slugs:
            slugs.append(sl)
    for slug in slugs[:4]:
        try:
            r = subprocess.run([exe, "kernels", "status", f"shubhsaxena2008/{slug}"], capture_output=True, text=True, timeout=30)
            txt = (r.stdout + r.stderr).strip().splitlines()
            data[slug] = txt[-1][-120:] if txt else "?"
        except Exception as e:
            data[slug] = f"unavailable ({type(e).__name__})"
    _kstatus.update(t=time.time(), data=data)
    return data


def _kaggle_usage():
    p = os.path.join(ROOT, "kaggle", "usage.jsonl")
    runs = []
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try:
                runs.append(json.loads(l))
            except Exception:
                pass
    by = {}
    for r in runs:  # one row per push: the final record replaces the "pushed" record
        if "state" in r:
            by[(r["slug"].split("/")[-1], r["pushed"][:16])] = r
    jobs = list(by.values())
    for j in jobs:
        if j["state"] == "pushed":
            j["state"] = "queued/running"
    now = datetime.datetime.now()
    wk = now - datetime.timedelta(days=7)
    gpu_week = sum(r.get("run_min_est", 0) for r in jobs if r.get("gpu") and datetime.datetime.fromisoformat(r["pushed"]) >= wk)
    gpu_all = sum(r.get("run_min_est", 0) for r in jobs if r.get("gpu"))
    errs = [f"Kaggle job {r['slug']} pushed {r['pushed']} ended '{r['state']}'" for r in jobs if r.get("state") not in ("complete", "queued/running") and not r["state"].startswith("cancelled")]
    ql = os.path.join(ROOT, "kaggle", "queue_runner.log")
    queue_log = open(ql, encoding="utf-8").read().splitlines()[-12:] if os.path.exists(ql) else []
    return {"queue_log": queue_log, "jobs": list(reversed(jobs))[:25], "gpu_minutes_7d": round(gpu_week, 1), "gpu_minutes_total": round(gpu_all, 1),
            "quota_minutes": GPU_WEEKLY_QUOTA_H * 60, "errors": errs}


def state(live=False):
    vids = []
    outd = os.path.join(ROOT, "out")
    for n in sorted(os.listdir(outd)) if os.path.isdir(outd) else []:
        v = _video_state(n)
        if v:
            vids.append(v)
    vids.sort(key=lambda v: -v["mtime"])
    ku = _kaggle_usage()
    errors = [{"video": v["name"], "msg": e} for v in vids for e in v["errors"]] + [{"video": "-", "msg": e} for e in ku["errors"]]
    return {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "stages": STAGES, "stories": _stories(),
            "videos": vids, "kaggle": ku, "kaggle_live": _kaggle_live() if live else None, "errors": errors}


# ------------------------------------------------------------------ routes
@app.get("/api/state")
def api_state(live: int = 0):
    return JSONResponse(state(bool(live)))


@app.post("/api/approve")
async def api_approve(request: Request):
    body = await request.json()
    name, status = body.get("name"), body.get("status")
    if status not in ("approved", "needs_changes", "pending") or not name or not os.path.isdir(os.path.join(ROOT, "out", name)):
        raise HTTPException(400, "bad request")
    os.makedirs(DATA, exist_ok=True)
    ap = approvals()
    ap[name] = {"status": status, "note": str(body.get("note", ""))[:2000], "at": datetime.datetime.now().isoformat(timespec="seconds")}
    tmp = APPROVALS + ".tmp"
    json.dump(ap, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, APPROVALS)
    return {"ok": True, "approval": ap[name]}


@app.get("/files/{name}/{path:path}")
def files(name: str, path: str):
    base = os.path.realpath(os.path.join(ROOT, "out", name))
    p = os.path.realpath(os.path.join(base, path))
    if not p.startswith(base + os.sep) or not os.path.isfile(p):
        raise HTTPException(404)
    if p.endswith(".md") or p.endswith(".srt"):
        return PlainTextResponse(open(p, encoding="utf-8").read())
    return FileResponse(p)


@app.get("/fonts/{fname}")
def fonts(fname: str):
    if fname not in ("Nunito-VF.ttf", "Fraunces-VF.ttf"):
        raise HTTPException(404)
    return FileResponse(os.path.join(ROOT, "brand", "fonts", fname))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(open(os.path.join(ROOT, "dashboard", "index.html"), encoding="utf-8").read())


def port_free(host, port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) != 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    if a.host not in ("127.0.0.1", "localhost") and not PASSWORD:
        sys.exit("Refusing to listen beyond localhost without HEARTWICK_DASH_PASSWORD set.")
    if not port_free(a.host, a.port):
        sys.exit(f"Port {a.port} is already in use; pick another with --port.")
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
