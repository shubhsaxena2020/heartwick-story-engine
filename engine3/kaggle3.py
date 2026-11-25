"""Kaggle plumbing for engine v3: one-GPU-session discipline, push, poll, fetch, usage log.

Kaggle allows 2 batch GPU sessions per account (queued ones count) and another agent shares them, so engine v3 holds
at most ONE GPU session at a time: before every GPU push it checks `kernels list --mine` + `kernels status` and waits
while any other heartwick-v3 GPU kernel is running/queued or both slots are taken. Every push and every final state
goes to kaggle/usage.jsonl (the Heartwick Studio dashboard reads it)."""
import datetime, json, os, re, shutil, time

from engine.kaggle_run import KAGGLE, USER, USAGE, _k  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "kaggle", "build")
PREFIX = "heartwick-v3-"


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log_usage(rec):
    with open(USAGE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def status(slug):
    s = _k("kernels", "status", f"{USER}/{slug}", check=False, timeout=120).lower()
    for w in ("complete", "error", "cancel", "running", "queued"):
        if w in s:
            return w
    return "unknown:" + s.strip()[-80:]


def recent_kernels(hours=14):
    out = _k("kernels", "list", "--mine", "--page-size", "60", "--sort-by", "dateRun", check=False, timeout=180)
    res = []
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
    for ln in out.splitlines():
        m = re.match(rf"^{USER}/(\S+)\s+.*?(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})", ln)
        if m:
            t = datetime.datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S")
            if t >= cutoff:
                res.append(m.group(1))
    return res


def _gpu_flags():
    flags = {}
    if os.path.exists(USAGE):
        for ln in open(USAGE, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if "gpu" in r:
                flags[r["slug"].split("/")[-1]] = bool(r["gpu"])
    return flags


def gpu_busy():
    """(mine, total): running/queued GPU kernels of engine v3, and of the whole account (unknown slugs count as GPU)."""
    flags = _gpu_flags()
    mine = total = 0
    active = []
    for slug in recent_kernels():
        st = status(slug)
        if st in ("running", "queued"):
            g = flags.get(slug, True)
            active.append((slug, st, g))
            if g:
                total += 1
                if slug.startswith(PREFIX):
                    mine += 1
    return mine, total, active


def push(slug, script_src, gpu, dataset_sources=(), kernel_sources=(), note="", wait_slot=True):
    assert slug.startswith("heartwick-"), slug      # v3 (heartwick-v3-) and v4 (heartwick-v4-) kernels
    if len(script_src) > 950_000:
        raise RuntimeError(f"kernel script is {len(script_src)} bytes (> ~1 MB is rejected); move data into a dataset")
    d = os.path.join(BUILD, slug)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    open(os.path.join(d, "main.py"), "w", encoding="utf-8").write(script_src)
    meta = {"id": f"{USER}/{slug}", "title": slug.replace("-", " ").title(), "code_file": "main.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": list(dataset_sources), "competition_sources": [], "kernel_sources": list(kernel_sources),
            "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    args = ["kernels", "push", "-p", d] + (["--accelerator", "NvidiaTeslaT4"] if gpu else [])
    for attempt in range(400):
        if gpu and wait_slot:
            mine, total, active = gpu_busy()
            if mine or total >= 2:
                if attempt % 10 == 0:
                    print(now(), "waiting for a GPU slot:", active, flush=True)
                time.sleep(60)
                continue
        out = _k(*args, check=False, timeout=600).strip()
        if "successfully pushed" in out.lower() or "kernel version" in out.lower():
            print(out[-300:], flush=True)
            break
        if "session" in out.lower() or "maximum" in out.lower() or "limit" in out.lower():
            print(now(), "push refused (slots):", out[-200:], flush=True)
            time.sleep(90)
            continue
        raise RuntimeError(f"kaggle push failed: {out[-800:]}")
    else:
        raise RuntimeError("no free Kaggle slot")
    rec = {"slug": f"{USER}/{slug}", "gpu": gpu, "pushed": now(), "renders": [note or slug], "engine": "v3",
           "state": "pushed", "by": "engine-v3"}
    log_usage(rec)
    return rec


def wait(rec, poll=60, max_h=8, on_poll=None):
    slug = rec["slug"].split("/")[-1]
    t_push = time.time(); t_run = None; last = None
    while time.time() - t_push < max_h * 3600:
        try:
            st = status(slug)
        except Exception as e:  # network hiccup
            st = "unknown:" + repr(e)[:60]
        if st != last:
            print(now(), slug, st, flush=True); last = st
        if st == "running" and t_run is None:
            t_run = time.time()
        if on_poll:
            on_poll(st)
        if st in ("complete", "error", "cancel"):
            fin = dict(rec, state=st, wall_min=round((time.time() - t_push) / 60, 1),
                       run_min_est=round((time.time() - (t_run or t_push)) / 60, 1))
            log_usage(fin)
            return fin
        time.sleep(poll)
    return dict(rec, state="timeout")


def logs(slug, tail=40):
    return "\n".join(_k("kernels", "logs", f"{USER}/{slug}", check=False, timeout=180).splitlines()[-tail:])


def download(slug, dest):
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    print(_k("kernels", "output", f"{USER}/{slug}", "-p", dest, "-o", timeout=3600).strip()[-300:], flush=True)
    return dest


def dataset(name, files, title=None):
    """Create or version the private dataset USER/name from {relative path: local path}. Returns its ref when ready."""
    ref = f"{USER}/{name}"
    d = os.path.join(BUILD, "ds_" + name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for rel, src in files.items():
        p = os.path.join(d, rel.replace("/", "__"))
        shutil.copy(src, p)
    json.dump({"title": title or name, "id": ref, "licenses": [{"name": "CC0-1.0"}]}, open(os.path.join(d, "dataset-metadata.json"), "w"))
    exists = "ready" in _k("datasets", "status", ref, check=False).lower()
    if exists:
        print(_k("datasets", "version", "-p", d, "-m", now(), "-r", "skip", timeout=3600)[-200:], flush=True)
    else:
        print(_k("datasets", "create", "-p", d, "-r", "skip", timeout=3600)[-200:], flush=True)
    time.sleep(10)
    for _ in range(80):
        if "ready" in _k("datasets", "status", ref, check=False).lower():
            return ref
        time.sleep(15)
    raise RuntimeError(f"dataset {ref} not ready")
