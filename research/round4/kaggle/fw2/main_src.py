# Heartwick round 4, run 2 (Kaggle T4 x2): GPU0 = FastWan2.2-TI2V-5B clips (weights from the CPU weight-cache kernel, umt5
# embeddings from heartwick-r4-bench); GPU1 = FLUX.2-klein-4B keyframes / blink + mouth variants. IMGS, JOBS, FW, GEN_SRC,
# FW_SRC are prepended by build.py.
import base64, glob, json, os, subprocess, sys, threading, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs"; LOG.mkdir(exist_ok=True); IN = W / "in"; IN.mkdir(exist_ok=True)
for k, b in IMGS.items():
    (IN / k).write_bytes(base64.b64decode(b))
(W / "gen.py").write_text(GEN_SRC); (W / "fw_gen2.py").write_text(FW_SRC)
status = {"t_start": time.time()}


def save():
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def sh(cmd, log, env=None, timeout=None):
    t = time.time()
    with open(LOG / log, "a") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
            if "it/s]" not in line and "s/it]" not in line:
                print(f"[{log}] {line}", end="", flush=True)
        rc = p.wait(timeout=timeout)
    print(f"[{log}] rc={rc} {round(time.time() - t, 1)}s", flush=True)
    return {"rc": rc, "s": round(time.time() - t, 1)}


status["mounted"] = sh("ls /kaggle/input; find /kaggle/input -maxdepth 6 -name 'emb_*.pt' -o -maxdepth 6 -name 'fastwan5b' | head", "setup.log")
status["setup"] = sh("pip install -q -U 'diffusers>=0.37' 'transformers>=4.57' accelerate imageio imageio-ffmpeg ftfy 2>&1 | tail -2; "
                     "python -c 'import diffusers,transformers,torch;print(diffusers.__version__,transformers.__version__,torch.__version__)'",
                     "setup.log", timeout=1500)
save()


def env(g):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def emb(pid):
    r = sorted(glob.glob(f"/kaggle/input/**/emb_{pid}.pt", recursive=True)); return r[0] if r else None


CL = OUT / "clips"; CL.mkdir(exist_ok=True)
plan = {"name": "fw", "clips": [dict(j, image=str(IN / j["image"]), emb=emb(j["prompt"])) for j in FW]}
(W / "fw_plan.json").write_text(json.dumps(plan))
KO = OUT / "keys"; KO.mkdir(exist_ok=True)
(W / "flux_jobs.json").write_text(json.dumps([dict(j, refs=[str(IN / r) for r in j["refs"]]) for j in JOBS]))


def gpu0():
    status["fw"] = sh(f"python {W/'fw_gen2.py'} {W/'fw_plan.json'} {CL}", "fw.log", env(0), 5400)


def gpu1():
    t = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download("black-forest-labs/FLUX.2-klein-4B", allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*",
                                                                          "tokenizer/*", "transformer/*", "vae/*"])
    status["flux_download_s"] = round(time.time() - t, 1)
    time.sleep(20)   # let GPU0 take the load lock first (host RAM)
    status["flux"] = sh(f"python {W/'gen.py'} {W/'flux_jobs.json'} {KO} g1", "flux.log", env(1), 5400)


ts = [threading.Thread(target=gpu0), threading.Thread(target=gpu1)]
[t.start() for t in ts]
while any(t.is_alive() for t in ts):
    time.sleep(60); save()
SHEET = r'''
import glob, os, imageio.v3 as iio
from PIL import Image, ImageDraw
for p in sorted(glob.glob("/kaggle/working/out/clips/*.mp4")):
    fr = list(iio.imiter(p)); n = len(fr); idx = [round(i * (n - 1) / 7) for i in range(8)]
    ims = [Image.fromarray(fr[i]) for i in idx]; w = 240; h = round(ims[0].height * w / ims[0].width)
    S = Image.new("RGB", (w * 8, h + 18), "white"); d = ImageDraw.Draw(S); d.text((4, 2), os.path.basename(p), fill="black")
    for i, im in enumerate(ims): S.paste(im.resize((w, h)), (i * w, 18))
    S.save(p[:-4] + "_sheet.jpg", quality=88)
'''
(W / "sheet.py").write_text(SHEET); sh(f"python {W/'sheet.py'}", "sheet.log", timeout=900)
status["total_s"] = round(time.time() - status["t_start"], 1); save()
print(json.dumps(status, indent=1))
