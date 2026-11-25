"""Push engine v2 renders to a Kaggle CPU kernel (no GPU queue, no GPU quota), wait, download.
Small narration sets ride inside the script as FLAC; big ones go through the private cache dataset (engine.kaggle_run)."""
import base64, datetime, io, json, os, time, zipfile
import soundfile as sf
from engine import kaggle_run as K

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 950_000


def build_zip(job, stories, tts_dir=None, utt_ids=(), extra=()):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("job.json", json.dumps(job))
        for pkg in ("engine", "engine2"):
            for f in os.listdir(os.path.join(ROOT, pkg)):
                if f.endswith(".py"):
                    z.write(os.path.join(ROOT, pkg, f), f"{pkg}/{f}")
        z.write(os.path.join(ROOT, "research", "tools", "pb_analyze.py"), "research/tools/pb_analyze.py")
        for s in stories:
            z.write(os.path.join(ROOT, s), s.replace("\\", "/"))
        for arc, src in (extra.items() if isinstance(extra, dict) else ((e, os.path.join(ROOT, e)) for e in extra)):
            z.write(src, arc)
        for u in utt_ids:  # narration as FLAC (half the size of wav)
            x, sr = sf.read(os.path.join(tts_dir, u + ".wav"))
            b = io.BytesIO()
            sf.write(b, x, sr, format="FLAC")
            z.writestr(f"tts_in/{u}.flac", b.getvalue())
            z.write(os.path.join(tts_dir, u + ".json"), f"tts_in/{u}.json")
    return base64.b64encode(buf.getvalue()).decode()


def push(job, stories, slug, tts_dir=None, utt_ids=(), dataset=None, extra=()):
    slug = slug.lower()  # Kaggle kernel slugs are lowercase
    d = os.path.join(ROOT, "kaggle", "build", slug)
    os.makedirs(d, exist_ok=True)
    src = open(os.path.join(ROOT, "kaggle", "worker_v2.py"), encoding="utf-8").read()
    script = f'JOB_ZIP_B64 = "{build_zip(job, stories, tts_dir, utt_ids, extra)}"\n' + src
    if len(script) > LIMIT:
        raise RuntimeError(f"kernel script is {len(script)} bytes (> {LIMIT}); send narration through the cache dataset")
    open(os.path.join(d, "worker_v2.py"), "w", encoding="utf-8").write(script)
    meta = {"id": f"{K.USER}/{slug}", "title": slug.replace("-", " ").title(), "code_file": "worker_v2.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": (dataset if isinstance(dataset, list) else [dataset]) if dataset else [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    out = K._k("kernels", "push", "-p", d, check=False)
    if "successfully pushed" not in out.lower():
        raise RuntimeError(f"kaggle push failed: {out[-600:]}")
    print(out.strip()[-200:])
    rec = {"slug": f"{K.USER}/{slug}", "gpu": False, "pushed": datetime.datetime.now().isoformat(timespec="seconds"),
           "renders": [r["out"] for r in job["renders"]], "engine": "v2", "state": "pushed"}
    with open(K.USAGE, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return f"{K.USER}/{slug}", dict(rec), time.time()


def run(job, stories, slug, dest, **kw):
    ref, rec, t0 = push(job, stories, slug, **kw)
    state = K.wait(ref, rec, t0, poll=45)
    K.download(ref, dest)
    return state
