# Heartwick round 4 bench kernel (Kaggle T4 x2). IMGS (base64 keys), SRC (scripts), PROMPTS, JOBS are prepended by build.py.
# GPU1: umt5 embeddings -> limited-animation renders (la.py) -> FastWan jobs "B".  GPU0: FastWan jobs "A" once embeddings exist.
import base64, glob, json, os, subprocess, sys, threading, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs"; LOG.mkdir(exist_ok=True); IN = W / "in"; IN.mkdir(exist_ok=True); EMB = W / "emb"; EMB.mkdir(exist_ok=True)
for k, b in IMGS.items():
    (IN / k).write_bytes(base64.b64decode(b))
for k, s in SRC.items():
    (W / k).write_text(s)
status = {"t_start": time.time()}


def save():
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def sh(cmd, log, env=None, timeout=None):
    t = time.time()
    with open(LOG / log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
            if "it/s]" not in line and "s/it]" not in line and not line.startswith(("Loading", "Fetching", "\r")):
                print(f"[{log}] {line}", end="", flush=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); rc = -999
    print(f"[{log}] rc={rc} {round(time.time() - t, 1)}s", flush=True)
    return rc, round(time.time() - t, 1)


status["setup"] = sh("pip install -q -U 'diffusers>=0.36' 'transformers>=4.51' accelerate ftfy imageio imageio-ffmpeg 'rembg[cpu]' "
                     "opencv-python-headless 2>&1 | tail -3; nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2; "
                     "df -h /root /kaggle/working | tail -2; python -c 'import diffusers,transformers,torch;print(diffusers.__version__,transformers.__version__,torch.__version__)'",
                     "setup.log", timeout=1500)
save()


def env(g):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1",
                HF_HUB_ENABLE_HF_TRANSFER="0")


# weights download in the background (timed) while the text encoder loads
def dl():
    t = time.time()
    from huggingface_hub import snapshot_download
    snapshot_download("FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", allow_patterns=["transformer/*", "vae/*", "model_index.json"])
    status["dl_dit_vae_s"] = round(time.time() - t, 1)


th = threading.Thread(target=dl); th.start()
(W / "prompts.json").write_text(json.dumps(PROMPTS))
status["te"] = sh(f"python {W/'te.py'} {W/'prompts.json'} {EMB}", "te.log", env(1), 2400); save()
sh("rm -rf ~/.cache/huggingface/hub/models--FastVideo--FastWan2.2-TI2V-5B-FullAttn-Diffusers/blobs/*.incomplete; du -sh ~/.cache/huggingface/hub/* | tail -5", "te.log")
th.join(); save()

CL = OUT / "clips"; CL.mkdir(exist_ok=True)


def fix(j):
    j = dict(j); j["image"] = str(IN / j["image"]); j["emb"] = str(EMB / f"emb_{j['prompt']}.pt"); return j


plans = {}
for g in (0, 1):
    pf = W / f"plan_{g}.json"; pf.write_text(json.dumps({"name": f"g{g}", "clips": [fix(j) for j in JOBS[str(g)]]})); plans[g] = pf


def gpu0():
    status["fw_g0"] = sh(f"python {W/'fw_gen.py'} {plans[0]} {CL}", "fw_g0.log", env(0), 4800)


def gpu1():
    (W / "la_shots.json").write_text(json.dumps([dict(s, image=str(IN / s["image"])) for s in LA]))
    status["la"] = sh(f"python {W/'la.py'} {W/'la_shots.json'} {OUT/'la'}", "la.log", env(1), 2400)
    status["fw_g1"] = sh(f"python {W/'fw_gen.py'} {plans[1]} {CL}", "fw_g1.log", env(1), 4800)


ts = [threading.Thread(target=gpu0), threading.Thread(target=gpu1)]
[t.start() for t in ts]
while any(t.is_alive() for t in ts):
    time.sleep(60); save()
    print(time.strftime("%H:%M:%S"), "clips so far:", sorted(os.path.basename(p) for p in glob.glob(str(CL / "*.mp4"))), flush=True)

# frame sheets (8 frames per clip) for review
SHEET = r'''
import glob, os, imageio.v3 as iio
from PIL import Image, ImageDraw
for p in sorted(glob.glob("/kaggle/working/out/clips/*.mp4")) + sorted(glob.glob("/kaggle/working/out/la/*.mp4")):
    fr = list(iio.imiter(p)); n = len(fr); idx = [round(i * (n - 1) / 7) for i in range(8)]
    ims = [Image.fromarray(fr[i]) for i in idx]; w = 240; h = round(ims[0].height * w / ims[0].width)
    S = Image.new("RGB", (w * 8, h + 18), "white"); d = ImageDraw.Draw(S); d.text((4, 2), os.path.basename(p), fill="black")
    for i, im in enumerate(ims): S.paste(im.resize((w, h)), (i * w, 18))
    S.save(p[:-4] + "_sheet.jpg", quality=88)
'''
(W / "sheet.py").write_text(SHEET)
sh(f"python {W/'sheet.py'}", "sheet.log", timeout=900)
status["total_s"] = round(time.time() - status["t_start"], 1); save()
sh(f"ls -la {OUT} {CL} {OUT/'la'}; cat {CL}/rec_g0.json {CL}/rec_g1.json | grep -v '\"cfg\"' | head -120", "final.log")
print(json.dumps(status, indent=1))
