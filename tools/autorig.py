"""Turn rig-ready character images into raster puppets on a Kaggle CPU kernel (no GPU queue).

    python tools/autorig.py research/art/rigB_nell_s11.png [more.png ...] --name nell
    -> characters/<name>/rig/ (rig.json + part PNGs) + characters/<name>/rig_preview.png

When characters/<name>/rig/rig.json exists, engine v2 uses that raster rig instead of the vector puppet
(engine2/direct.py rig_for); the pose library, scheduler and camera are unchanged.
"""
import argparse, base64, json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import kaggle_run as K  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--name", help="character name (one image) to install under characters/<name>/rig")
    a = ap.parse_args()
    from PIL import Image
    import io
    imgs = {}
    for p in a.images:
        im = Image.open(p).convert("RGB")
        im.thumbnail((900, 1300))
        b = io.BytesIO()
        im.save(b, "PNG", optimize=True)
        imgs[os.path.splitext(os.path.basename(p))[0]] = base64.b64encode(b.getvalue()).decode()
    files = {f: open(os.path.join(ROOT, "engine2", f), encoding="utf-8").read() for f in ("__init__.py", "autorig.py", "rig.py", "poses.py", "style.py", "mathx.py")}
    src = open(os.path.join(ROOT, "kaggle", "autorig_job.py"), encoding="utf-8").read()
    d = os.path.join(ROOT, "kaggle", "build", "autorig")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "autorig_job.py"), "w", encoding="utf-8").write(f"JOB = {json.dumps({'images': imgs})!r}\nJOB = __import__('json').loads(JOB)\nENGINE_FILES = {files!r}\n" + src)
    slug = "heartwick-autorig"
    json.dump({"id": f"{K.USER}/{slug}", "title": "Heartwick Autorig", "code_file": "autorig_job.py", "language": "python", "kernel_type": "script",
               "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [],
               "competition_sources": [], "kernel_sources": [], "model_sources": []}, open(os.path.join(d, "kernel-metadata.json"), "w"))
    print(K._k("kernels", "push", "-p", d)[-200:])
    ref = f"{K.USER}/{slug}"
    state = K.wait(ref, {"slug": ref, "gpu": False, "engine": "v2-autorig"}, time.time(), poll=30)
    dl = os.path.join(ROOT, "out", "autorig")
    K.download(ref, dl)
    print("kaggle:", state, "->", dl)
    if a.name and len(imgs) == 1:
        nm = next(iter(imgs))
        dst = os.path.join(ROOT, "characters", a.name)
        if os.path.isdir(os.path.join(dl, nm, "rig")):
            shutil.copytree(os.path.join(dl, nm, "rig"), os.path.join(dst, "rig"), dirs_exist_ok=True)
            shutil.copy(os.path.join(dl, nm, "preview.png"), os.path.join(dst, "rig_preview.png"))
            print("installed raster rig:", dst)


if __name__ == "__main__":
    main()
