"""Push a job to Kaggle as a script kernel, poll it, download outputs, log GPU time.
Pattern copied from the sewacircle kernels: kernel-metadata.json + single script, `kaggle kernels push/status/output`."""
import base64, io, json, os, shutil, subprocess, sys, time, zipfile, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.environ.get("KAGGLE_EXE", r"C:\Users\shubh\.kaggle-venv\Scripts\kaggle.exe")
USER = os.environ.get("KAGGLE_USER", "shubhsaxena2008")
USAGE = os.path.join(ROOT, "kaggle", "usage.jsonl")


def _k(*args, check=True, timeout=900):
    # explicit utf-8: Windows' default cp1252 cannot decode some Kaggle CLI output (stdout came back None -> TypeError)
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    out, err = r.stdout or "", r.stderr or ""
    if check and r.returncode:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {out[-800:]} {err[-800:]}")
    return out + err


def build_zip(job, refs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("job.json", json.dumps(job))
        for f in os.listdir(os.path.join(ROOT, "engine")):
            if f.endswith(".py"):
                z.write(os.path.join(ROOT, "engine", f), f"engine/{f}")
        # fonts (640 KB) are fetched by the worker from github.com/google/fonts: the script must stay < ~1 MB
        for name, path in refs.items():
            if name.lower().endswith(".png"):  # compact JPEG refs keep the kernel script small (IP-Adapter sees 224px)
                from PIL import Image
                im = Image.open(path).convert("RGB"); im.thumbnail((512, 512))
                b = io.BytesIO(); im.save(b, "JPEG", quality=88)
                z.writestr(f"refs/{name[:-4]}.jpg", b.getvalue())
            else:
                z.write(path, f"refs/{name}")
    return base64.b64encode(buf.getvalue()).decode()


def push(job, refs, slug="heartwick-gpu", gpu=True, use_cache_dataset=None, title=None):
    d = os.path.join(ROOT, "kaggle", "build", slug)
    os.makedirs(d, exist_ok=True)
    src = open(os.path.join(ROOT, "kaggle", "worker.py"), encoding="utf-8").read()
    script = f'JOB_ZIP_B64 = "{build_zip(job, refs)}"\n' + src
    if len(script) > 950_000:
        raise RuntimeError(f"kernel script is {len(script)} bytes; Kaggle rejects ~1 MB (HTTP 400). Move refs into the cache dataset.")
    open(os.path.join(d, "heartwick_worker.py"), "w", encoding="utf-8").write(script)
    meta = {"id": f"{USER}/{slug}", "title": title or slug.replace("-", " ").title(), "code_file": "heartwick_worker.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [use_cache_dataset] if use_cache_dataset else [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    args = ["kernels", "push", "-p", d]
    if gpu:
        args += ["--accelerator", "NvidiaTeslaT4"]
    for attempt in range(240):  # Kaggle allows 2 batch GPU sessions (queued ones count): wait for a free slot
        out = _k(*args, check=False).strip()
        if "successfully pushed" in out.lower():
            print(out[-300:])
            break
        if "session count" in out.lower() or "maximum" in out.lower():
            if attempt == 0:
                print("Kaggle GPU session limit reached; waiting for a free slot...", flush=True)
            time.sleep(120)
            continue
        raise RuntimeError(f"kaggle push failed: {out[-600:]}")
    else:
        raise RuntimeError("no free Kaggle session slot after 8 h")
    rec = {"slug": slug, "gpu": gpu, "pushed": datetime.datetime.now().isoformat(timespec="seconds"),
           "images": len(job.get("images", [])), "utts": len((job.get("tts") or {}).get("utts", [])), "render": bool(job.get("render"))}
    with open(USAGE, "a") as f:  # so the dashboard can show queued/running jobs
        f.write(json.dumps(dict(rec, slug=f"{USER}/{slug}", state="pushed")) + "\n")
    rec["slug"] = f"{USER}/{slug}"
    return f"{USER}/{slug}", rec, time.time()


def wait(ref, rec, t_push, poll=60, max_h=6):
    last = None
    t_run = None
    while time.time() - t_push < max_h * 3600:
        try:
            st = _k("kernels", "status", ref, check=False, timeout=120)
        except subprocess.TimeoutExpired:
            st = "timeout"
        s = st.lower()
        state = ("complete" if "complete" in s else "error" if "error" in s or "fail" in s else
                 "cancel" if "cancel" in s else "running" if "running" in s else "queued" if "queued" in s else s.strip()[-60:])
        if state != last:
            print(time.strftime("%H:%M:%S"), ref, state, flush=True)
            last = state
        if state == "running" and t_run is None:
            t_run = time.time()
        if state in ("complete", "error", "cancel"):
            rec.update(state=state, wall_min=round((time.time() - t_push) / 60, 1),
                       run_min_est=round((time.time() - (t_run or t_push)) / 60, 1))
            with open(USAGE, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return state
        time.sleep(poll)
    return "timeout"


def download(ref, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    print(_k("kernels", "output", ref, "-p", dest, "-o", timeout=3600).strip()[-300:])
    lg = os.path.join(dest, "worker_log.json")
    if os.path.exists(lg):
        L = json.load(open(lg))
        with open(USAGE, "a") as f:
            f.write(json.dumps({"slug": ref, "worker_log": L, "at": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
        print("worker log:", json.dumps(L)[:1500])
    return dest


def upload_cache(asset_dir, files, story="pip"):
    """Version the private heartwick-cache-<story> dataset with the given asset files. Returns its ref."""
    CACHE_DATASET = f"{USER}/heartwick-cache-{story.replace('_', '-')}"
    d = os.path.join(ROOT, "kaggle", "build", "cache_" + story)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in files:
        shutil.copy(os.path.join(asset_dir, f), os.path.join(d, f))
    json.dump({"title": CACHE_DATASET.split("/")[1], "id": CACHE_DATASET, "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(d, "dataset-metadata.json"), "w"))
    exists = CACHE_DATASET in _k("datasets", "list", "--mine", "-s", CACHE_DATASET.split("/")[1], check=False)
    if exists:
        print(_k("datasets", "version", "-p", d, "-m", time.strftime("%Y-%m-%d %H:%M"), "-r", "skip", timeout=3600)[-300:])
    else:
        print(_k("datasets", "create", "-p", d, "-r", "skip", timeout=3600)[-300:])
    for _ in range(60):  # wait until processed
        s = _k("datasets", "status", CACHE_DATASET, check=False)
        if "ready" in s.lower():
            return CACHE_DATASET
        time.sleep(15)
    return None


def usage_summary():
    if not os.path.exists(USAGE):
        return {}
    runs = [json.loads(l) for l in open(USAGE) if l.strip()]
    gpu = [r for r in runs if r.get("gpu") and "run_min_est" in r and r.get("state") != "pushed"]
    return {"gpu_runs": len(gpu), "gpu_minutes_est": round(sum(r["run_min_est"] for r in gpu), 1),
            "cpu_runs": len([r for r in runs if r.get("gpu") is False and "run_min_est" in r])}
