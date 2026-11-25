"""Engine v4 laptop side: render "living still" shots on parallel Kaggle CPU kernels (no GPU quota).

    from engine4.la import render_la
    res = render_la(jobs, out_dir, n_kernels=3, tag="main")
    # -> {shot_id: {"mp4": path or None, "record": {...}}}  plus res["_kernels"] = per-kernel timings/state

jobs: list of dicts in the schema documented at the top of engine4/kaggle/la_render.py, with LOCAL paths in "image"
and "tail_of". Each file is staged under a content-hash name, the job list is split across n kernels (balanced by frame
count + per-key prep cost), each kernel gets the renderer source + its jobs, runs, and its output is downloaded to
out_dir/<slug>/out/. Slugs are heartwick-v4-la-<tag>-<i>.

Input transport (transport="auto"): if every kernel's script with its files embedded as base64 stays < 950 KB, files are
embedded; otherwise (the normal case: a 1088x1920 FLUX PNG is ~3.4 MB) ALL files go into one private dataset
<USER>/heartwick-v4-la-<tag>-in (created or versioned via the kaggle CLI, waited until ready) mounted by every kernel;
the renderer finds files by name under /kaggle/input/**. Use a distinct tag for batches that run at the same time.

Rules kept: CPU only (enable_gpu False, usage logged with gpu False so GPU-slot accounting ignores these); a slug that
is still queued/running is never re-pushed (we wait for it first); kernels are never deleted; only heartwick-v4-la-*
slugs are touched."""
import base64, hashlib, json, os, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine3 import kaggle3 as K  # noqa: E402

PREFIX = "heartwick-v4-la-"
RENDER_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle", "la_render.py")
BUILD = os.path.join(ROOT, "kaggle", "build")
MAX_SCRIPT = 950_000
PREP_FRAMES = 150   # per unique key/crop: ~20 s of prep ~ 150 frames of render

KERNEL_MAIN = r'''
import base64, json, os, subprocess, sys, time
from pathlib import Path
T0 = time.time()
W = Path("/kaggle/working"); OUT = W / "out"; IN = Path("/tmp/la_in"); OUT.mkdir(parents=True, exist_ok=True); IN.mkdir(parents=True, exist_ok=True)
for name, b in EMBED.items():
    (IN / name).write_bytes(base64.b64decode(b))
Path("/tmp/la_render.py").write_text(RENDER_SRC)
Path("/tmp/jobs.json").write_text(json.dumps(JOBS))
r = subprocess.run("pip install -q 'rembg[cpu]' opencv-python-headless 2>&1 | tail -2; nproc; free -g | head -2; ls -R /kaggle/input | head -40",
                   shell=True, capture_output=True, text=True)
print(r.stdout[-3000:], flush=True)
setup_s = round(time.time() - T0, 1)
os.environ["LA_SEARCH"] = "/tmp/la_in:/kaggle/input"
t = time.time()
p = subprocess.run([sys.executable, "/tmp/la_render.py", "/tmp/jobs.json", str(OUT)], capture_output=True, text=True)
print(p.stdout[-20000:], p.stderr[-6000:], flush=True)
(OUT / "render.log").write_text(p.stdout + "\n----- stderr -----\n" + p.stderr)
json.dump({"setup_s": setup_s, "render_wall_s": round(time.time() - t, 1), "kernel_total_s": round(time.time() - T0, 1),
           "returncode": p.returncode, "n_jobs": len(JOBS)}, open(OUT / "status.json", "w"), indent=1)
'''


def _sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:10]


def stage(jobs):
    """-> (kernel-side jobs, {staged name: local path}). Same content => same staged name."""
    files, kj = {}, []
    for j in jobs:
        j2 = dict(j)
        for key in ("image", "tail_of"):
            p = j.get(key)
            if p:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"{j['id']}: {key} {p}")
                name = f"{_sha(p)}_{os.path.basename(p)}"
                files[name] = p; j2[key] = name
        kj.append(j2)
    return kj, files


def split(kjobs, n):
    """Greedy longest-first split by frames (+ prep per unique key) into <= n non-empty bins."""
    n = max(1, min(n, len(kjobs)))
    cost = lambda j: round(j["dur"] * 24) + PREP_FRAMES
    bins = [[] for _ in range(n)]; load = [0] * n
    for j in sorted(kjobs, key=cost, reverse=True):
        i = load.index(min(load)); bins[i].append(j); load[i] += cost(j)
    return [b for b in bins if b]


def _script(kjobs, embed):
    src = open(RENDER_SRC, encoding="utf-8").read()
    return (f"import json\nEMBED = json.loads({json.dumps(embed)!r})\nJOBS = json.loads({json.dumps(kjobs)!r})\n"
            f"RENDER_SRC = {src!r}\n" + KERNEL_MAIN)


def _push(slug, script, dataset_sources, note):
    assert slug.startswith(PREFIX), slug
    if len(script) > MAX_SCRIPT:
        raise RuntimeError(f"{slug}: script {len(script)} bytes > {MAX_SCRIPT}")
    st = K.status(slug)
    while st in ("queued", "running"):           # never re-push a queued/running kernel: wait it out
        print(K.now(), slug, "is", st, "- waiting before re-push", flush=True)
        time.sleep(60); st = K.status(slug)
    d = os.path.join(BUILD, slug)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    open(os.path.join(d, "main.py"), "w", encoding="utf-8").write(script)
    meta = {"id": f"{K.USER}/{slug}", "title": slug.replace("-", " ").title(), "code_file": "main.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": list(dataset_sources), "competition_sources": [], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    for attempt in range(60):
        out = K._k("kernels", "push", "-p", d, check=False, timeout=600).strip()
        low = out.lower()
        if "successfully pushed" in low or "kernel version" in low:
            print(K.now(), slug, out[-200:], flush=True)
            break
        if any(w in low for w in ("session", "maximum", "limit", "429", "too many")):
            print(K.now(), slug, "push refused, retrying:", out[-200:], flush=True)
            time.sleep(60); continue
        raise RuntimeError(f"push {slug} failed: {out[-800:]}")
    else:
        raise RuntimeError(f"push {slug}: no slot after 60 tries")
    rec = {"slug": f"{K.USER}/{slug}", "gpu": False, "pushed": K.now(), "renders": [note], "engine": "v4", "state": "pushed",
           "by": "engine-v4-la"}
    K.log_usage(rec)
    return rec


def render_la(jobs, out_dir, n_kernels=3, tag="main", transport="auto", poll=30, max_wait_h=3.0):
    """Render living-still shots on up to n_kernels parallel Kaggle CPU kernels.
    Returns {shot_id: {"mp4": local path or None, "record": dict}, "_kernels": {slug: timing/state}}."""
    ids = [j["id"] for j in jobs]
    assert len(set(ids)) == len(ids), "duplicate shot ids"
    kjobs, files = stage(jobs)
    bins = split(kjobs, n_kernels)
    os.makedirs(out_dir, exist_ok=True)
    b64 = {}
    if transport in ("auto", "embed"):
        b64 = {n: base64.b64encode(open(p, "rb").read()).decode() for n, p in files.items()}
        need = lambda b: {j[k] for j in b for k in ("image", "tail_of") if j.get(k)}
        fits = all(len(_script(b, {n: b64[n] for n in need(b)})) < MAX_SCRIPT for b in bins)
        if transport == "embed" and not fits:
            raise RuntimeError("embedded inputs exceed the 950 KB kernel script limit; use transport='dataset'")
        transport = "embed" if fits else "dataset"
    ds = []
    if transport == "dataset":
        name = f"{PREFIX}{tag}-in"
        t = time.time()
        man = os.path.join(BUILD, f"ds_{name}.manifest")     # staged names are content hashes: same set => same data
        same = os.path.exists(man) and open(man).read() == "\n".join(sorted(files)) and \
            "ready" in K._k("datasets", "status", f"{K.USER}/{name}", check=False).lower()
        ref = f"{K.USER}/{name}" if same else K.dataset(name, files, title=f"Heartwick v4 LA {tag} inputs")
        os.makedirs(BUILD, exist_ok=True); open(man, "w").write("\n".join(sorted(files)))
        print(K.now(), "dataset", ref, "ready in", round(time.time() - t), "s", flush=True)
        ds = [ref]
    kern = {}
    for i, b in enumerate(bins):
        slug = f"{PREFIX}{tag}-{i}"
        emb = {n: b64[n] for n in {j[k] for j in b for k in ("image", "tail_of") if j.get(k)}} if transport == "embed" else {}
        script = _script(b, emb)
        rec = _push(slug, script, ds, f"la {tag}: {len(b)} shots")
        kern[slug] = {"jobs": [j["id"] for j in b], "frames": sum(round(j["dur"] * 24) for j in b), "t_push": time.time(),
                      "t_run": None, "t_done": None, "state": "pushed", "script_bytes": len(script), "transport": transport, "rec": rec}
    # poll all kernels together
    t0 = time.time()
    while time.time() - t0 < max_wait_h * 3600:
        live = [s for s, k in kern.items() if k["t_done"] is None]
        if not live:
            break
        for s in live:
            try:
                st = K.status(s)
            except Exception as e:
                st = "unknown:" + repr(e)[:60]
            k = kern[s]
            if st != k["state"]:
                print(K.now(), s, st, flush=True); k["state"] = st
            if st in ("queued", "running"):
                k["seen_active"] = True
            if st == "running" and k["t_run"] is None:
                k["t_run"] = time.time()
            # right after a push the status can still be the PREVIOUS version's terminal state: ignore it until the
            # new version has been seen queued/running (or 5 min passed)
            if st in ("complete", "error", "cancel") and (k.get("seen_active") or time.time() - k["t_push"] > 300):
                k["t_done"] = time.time()
                K.log_usage(dict(k["rec"], state=st, wall_min=round((k["t_done"] - k["t_push"]) / 60, 1),
                                 run_min_est=round((k["t_done"] - (k["t_run"] or k["t_push"])) / 60, 1)))
        time.sleep(poll)
    res = {}
    for s, k in kern.items():
        dest = os.path.join(out_dir, s)
        if k["state"] in ("complete", "error"):
            try:
                K.download(s, dest)
            except Exception as e:
                print(s, "download failed", repr(e)[:200], flush=True)
            if k["state"] == "error":
                open(os.path.join(dest, "kernel_log.txt"), "w", encoding="utf-8").write(K.logs(s, tail=200))
        o = os.path.join(dest, "out")
        st_path = os.path.join(o, "status.json")
        k["status"] = json.load(open(st_path)) if os.path.exists(st_path) else None
        k["wall_s"] = round((k["t_done"] or time.time()) - k["t_push"], 1)
        k["run_s_est"] = round((k["t_done"] or time.time()) - (k["t_run"] or k["t_push"]), 1)
        for sid in k["jobs"]:
            rp, mp = os.path.join(o, f"{sid}.json"), os.path.join(o, f"{sid}.mp4")
            rec = json.load(open(rp)) if os.path.exists(rp) else {"id": sid, "ok": False, "error": f"no record ({k['state']})"}
            rec["kernel"] = s
            res[sid] = {"mp4": mp if os.path.exists(mp) and rec.get("ok") else None, "record": rec}
    res["_kernels"] = {s: {kk: v for kk, v in k.items() if kk not in ("rec", "t_push", "t_run", "t_done")} for s, k in kern.items()}
    json.dump(res, open(os.path.join(out_dir, "render_la.json"), "w"), indent=1, default=str)
    return res
