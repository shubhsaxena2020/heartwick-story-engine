# Heartwick engine v3 kernel 2 (Kaggle T4 x2): umt5 prompt embeddings -> Wan2.2-I2V-A14B lightx2v 4-step distill GGUF,
# first-last-frame for action shots and plain I2V for holds, one process per T4 (expert-phase split, gen_flf.py) ->
# QA gate on every clip (best of 2 seeds for hero shots) -> finish on the GPUs: RIFE (MIT) 4x, retimed to the shot's
# narration length at 30 fps, Real-ESRGAN animevideov3 x4 (BSD-3) -> 1080x1920 with a gentle always-moving camera.
# GEN_SRC, QA_SRC, PLAN (clips + finish targets) are prepended by engine3/stages.py. Keys come from kernel sources.
import glob, json, os, subprocess, sys, textwrap, threading, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = W / "logs"; LOG.mkdir(exist_ok=True)
for name, s in (("gen_flf.py", GEN_SRC), ("qa.py", QA_SRC)):
    (W / name).write_text(s)
P = json.loads(PLAN)
status = {"t_start": time.time()}


def save():
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def sh(cmd, log, timeout=None, env=None, tee=True):
    t = time.time()
    with open(LOG / log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
            if tee and "it/s]" not in line and not line.startswith(("Loading", "Fetching")):
                print(f"[{log}] {line}", end="", flush=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); rc = -999
    print(f"[{log}] rc={rc} {round(time.time() - t, 1)}s", flush=True)
    return rc, round(time.time() - t, 1)


def find(name):
    r = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return r[0] if r else None


rc, s = sh("pip install -q -U 'diffusers>=0.36' 'transformers>=4.51' accelerate gguf ftfy imageio imageio-ffmpeg safetensors 2>&1 | tail -3; "
           "nvidia-smi --query-gpu=name,memory.total --format=csv; ls -R /kaggle/input | head -60", "setup.log", 1200)
status["setup"] = {"rc": rc, "s": s}
keys = {}
for c in P["clips"]:
    for f in ("start_file", "end_file"):
        if c.get(f):
            keys[c[f]] = find(c[f])
status["keys_found"] = keys; save()
missing = [k for k, v in keys.items() if not v]
if missing:
    print("MISSING KEYS", missing, flush=True)

# ---- downloads in the background while the text encoder runs ------------------------------------------------------
REPO = "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF"; Q = P.get("quant", "Q5_K_M"); GG = {}
def dl(w):
    from huggingface_hub import hf_hub_download
    GG[w] = hf_hub_download(REPO, f"{w}_noise_260412/wan2.2_i2v_A14b_{w}_noise_lightx2v_4step_720p_260412-{Q}.gguf")
ths = [threading.Thread(target=dl, args=(w,)) for w in ("high", "low")]; [t.start() for t in ths]

prompts = {c["shot"]: c["prompt"] for c in P["clips"]}
T0 = textwrap.dedent(f'''
import torch, json, gc
from transformers import AutoTokenizer, UMT5EncoderModel
repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"; PROMPTS = {json.dumps(prompts)}
tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer"); r = {{}}
def load(dev, dt): return UMT5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dt).to(dev)
te, dev = load("cuda", torch.float16), "cuda"
for sid, P in PROMPTS.items():
    t = tok([P], padding="max_length", max_length=512, truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    n = int(t.attention_mask.sum())
    with torch.no_grad():
        e = te(t.input_ids.to(dev), t.attention_mask.to(dev)).last_hidden_state[:, :n].float().cpu()
    if not torch.isfinite(e).all() and dev == "cuda":
        del te; gc.collect(); torch.cuda.empty_cache(); te, dev = load("cpu", torch.bfloat16)
        with torch.no_grad():
            e = te(t.input_ids, t.attention_mask).last_hidden_state[:, :n].float()
    r[sid] = {{"tokens": n, "finite": bool(torch.isfinite(e).all()), "dev": dev}}
    torch.save({{"pos": torch.cat([e, e.new_zeros(1, 512 - n, e.shape[-1])], 1)}}, f"/kaggle/working/emb_{{sid}}.pt")
print(json.dumps(r)); open("/kaggle/working/out/emb.json", "w").write(json.dumps(r))
''')
(W / "t0.py").write_text(T0)
status["emb"] = dict(zip(("rc", "s"), sh(f"CUDA_VISIBLE_DEVICES=0 python {W/'t0.py'}", "t0.log", 2400))); save()
sh("rm -rf ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-I2V-A14B-Diffusers/blobs/*text_encoder* 2>/dev/null; true", "t0.log")
t = time.time(); [x.join() for x in ths]; status["wan_dl_wait_s"] = round(time.time() - t, 1); status["quant"] = Q; save()

# ---- clips: one gen_flf.py per T4, balanced by count ----------------------------------------------------------------
CL = OUT / "clips"; CL.mkdir(exist_ok=True)
jobs = []
for c in P["clips"]:
    if not keys.get(c["start_file"]) or (c.get("end_file") and not keys.get(c["end_file"])):
        continue
    for seed in c["seeds"]:
        j = {"id": f"{c['shot']}_s{seed}", "image": keys[c["start_file"]], "w": P.get("w", 480), "h": P.get("h", 832),
             "frames": c.get("frames", P.get("frames", 81)), "seed": seed, "emb": f"/kaggle/working/emb_{c['shot']}.pt"}
        if c.get("end_file"):
            j["last"] = keys[c["end_file"]]
        jobs.append(j)
plans = {0: [], 1: []}
for i, j in enumerate(jobs):
    plans[i % 2].append(j)
procs = {}
for gpu, clips in plans.items():
    if not clips:
        continue
    pf = W / f"plan_g{gpu}.json"
    pf.write_text(json.dumps({"name": f"g{gpu}", "high": GG["high"], "low": GG["low"], "quant": Q, "clips": clips}))
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")
    procs[gpu] = (subprocess.Popen([sys.executable, str(W / "gen_flf.py"), "--plan", str(pf), "--out", str(CL)],
                                   stdout=open(LOG / f"gen_g{gpu}.log", "w"), stderr=subprocess.STDOUT, env=e), time.time())
status["clips"] = {"n": len(jobs), "per_gpu": {g: len(v) for g, v in plans.items()}}; save()
last = 0
while any(p.poll() is None for p, _ in procs.values()):
    time.sleep(30)
    if time.time() - last > 300:   # progress line for `kaggle kernels logs`
        last = time.time()
        for g in procs:
            r = glob.glob(str(CL / f"rec_g{g}.json"))
            if r:
                try:
                    rec = json.load(open(r[0])); n = sum(1 for c in rec["clips"].values() if "low_step_s" in c)
                    d = sum(1 for c in rec["clips"].values() if "vae_decode_s" in c)
                    print(time.strftime("%H:%M:%S"), f"gpu{g}: low-noise done {n}, decoded {d}/{len(plans[g])}", flush=True)
                except Exception:
                    pass
for g, (p, t) in procs.items():
    status["clips"][f"g{g}"] = {"rc": p.returncode, "wall_s": round(time.time() - t, 1)}
save()
sh("tail -5 /kaggle/working/logs/gen_g0.log /kaggle/working/logs/gen_g1.log", "gen.log")
sh("rm -rf ~/.cache/huggingface/hub/models--jayn7*; df -h /root | tail -1", "gen.log")

# ---- QA gate on every clip, pick the best seed per shot --------------------------------------------------------------
QA = textwrap.dedent(f'''
import os, sys, json, glob, math
os.environ["QA_DEVICE"] = "cuda"; sys.path.insert(0, "/kaggle/working"); import qa
bn = qa.ref_bank({json.dumps([find(x) for x in P["qa_refs_nell"] if find(x)])}, "nell")
bp = qa.ref_bank({json.dumps([find(x) for x in P["qa_refs_pip"] if find(x)])}, "pip")
res, allr = {{}}, []
for p in sorted(glob.glob("/kaggle/working/out/clips/*.mp4")):
    rows = qa.analyse(p, bn, bp, 9); allr.append((p, rows))
    res[os.path.basename(p)[:-4]] = {{"summary": qa.summary(rows), "frames": [{{k: v for k, v in r.items() if not k.startswith("_")}} for r in rows]}}
json.dump(res, open("/kaggle/working/out/clips_qa.json", "w"), indent=1)
qa.sheet(allr, "/kaggle/working/out/clips_qa_sheet.jpg")
print(json.dumps({{k: v["summary"] for k, v in res.items()}}))
''')
(W / "qa_clips.py").write_text(QA)
status["qa"] = dict(zip(("rc", "s"), sh(f"python {W/'qa_clips.py'}", "qa_clips.log", 2400, tee=False))); save()
qa = json.load(open(OUT / "clips_qa.json")) if (OUT / "clips_qa.json").exists() else {}


def clip_score(name, want_nell, want_pip):
    s = (qa.get(name) or {}).get("summary") or {}
    import math
    sc = 0.0
    if want_nell:
        sc += s.get("body_sim", {}).get("min", 0) + s.get("head_sim", {}).get("median", 0) - 0.15 * s.get("frames_cut", 0)
    if want_pip:
        sc -= math.log(s.get("pip_scale", {}).get("max_over_min", 1.0))
    return round(sc, 3)


best = {}
for c in P["clips"]:
    cands = [(clip_score(f"{c['shot']}_s{sd}", c["nell"], c["pip"]), f"{c['shot']}_s{sd}") for sd in c["seeds"]
             if (CL / f"{c['shot']}_s{sd}.mp4").exists()]
    if cands:
        cands.sort(reverse=True); best[c["shot"]] = {"take": cands[0][1], "cands": cands}
status["best"] = best; save()

# ---- finish on both GPUs: RIFE 4x -> retime to the shot length at 60 fps -> Real-ESRGAN x4 -> 1080x1920 ---------------
FIN = textwrap.dedent(r'''
import os, sys, json, subprocess, numpy as np, torch, imageio.v3 as iio
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
A = json.load(open(sys.argv[1])); dev = "cuda"
sys.path.insert(0, os.path.dirname(hf_hub_download("TensorForger/RIFE-safetensors", "interpolation_model.py")))
from interpolation_model import IFNet
rife = IFNet(); rife.load_state_dict(load_file(hf_hub_download("TensorForger/RIFE-safetensors", "flownet.safetensors"))); rife = rife.to(dev).half().eval()
class SRVGGNetCompact(nn.Module):   # Real-ESRGAN realesr-animevideov3 (BSD-3)
    def __init__(s, c=3, f=64, n=16, up=4):
        super().__init__(); s.up = up
        L = [nn.Conv2d(c, f, 3, 1, 1), nn.PReLU(f)]
        for _ in range(n): L += [nn.Conv2d(f, f, 3, 1, 1), nn.PReLU(f)]
        L += [nn.Conv2d(f, c * up * up, 3, 1, 1)]; s.body = nn.ModuleList(L); s.ps = nn.PixelShuffle(up)
    def forward(s, x):
        o = x
        for l in s.body: o = l(o)
        return s.ps(o) + F.interpolate(x, scale_factor=s.up, mode="nearest")
w = "/tmp/realesr-animevideov3.pth"
# both GPU finishers fetched the weights into the same file at once; GPU 0 read it half-written (EOFError, 24 Nov fix
# run: S08/S10 never finished). Each fetches to its own temp file and renames atomically; a bad read retries.
import time as _time
for _try in range(6):
    if not os.path.exists(w):
        _tmp = f"{w}.{os.getpid()}"
        subprocess.run(f"wget -q -O {_tmp} https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", shell=True)
        if os.path.exists(_tmp) and os.path.getsize(_tmp) > 500_000:
            os.replace(_tmp, w)
    try:
        sd = torch.load(w, map_location="cpu"); break
    except Exception as _e:
        print("ESRGAN weights not readable yet:", repr(_e)[:120], flush=True); _time.sleep(5)
        if _try >= 2 and os.path.exists(w):
            os.remove(w)
esr = SRVGGNetCompact(); esr.load_state_dict(sd.get("params", sd)); esr = esr.to(dev).half().eval()
def t(im): return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev).half()
def mid(a, b):
    with torch.no_grad():
        m = rife(torch.cat([t(a), t(b)], 1)).clamp(0, 1)
    return (m[0].float().permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
for job in A["jobs"]:
    fr = list(iio.imiter(job["src"]))
    for _ in range(2):   # 4x: 81 -> 321 frames
        seq = []
        for i in range(len(fr)):
            seq.append(fr[i])
            if i + 1 < len(fr): seq.append(mid(fr[i], fr[i + 1]))
        fr = seq
    FPS_O = int(job.get("fps", 60))                # v3: 60 fps (round-3 winner); v4: 24 fps at natural length
    n_out = max(2, int(round(job["dur"] * FPS_O)))
    idx = np.linspace(0, len(fr) - 1, n_out).round().astype(int)   # retime: the clip's motion spread over the shot length
    cam = job.get("cam", "push")
    out = []
    with torch.no_grad():
        for k, i in enumerate(idx):
            y = esr(t(fr[i])).clamp(0, 1)[0].float().permute(1, 2, 0).cpu().numpy()
            y = Image.fromarray((y * 255).round().astype(np.uint8))
            u = k / max(1, n_out - 1); u = u * u * (3 - 2 * u)
            z = 1.0 + (0.045 * u if cam == "push" else 0.045 * (1 - u) if cam == "pull" else 0.02)
            dx = (0.012 * (u - 0.5) if cam == "drift" else 0.0)
            Wd, Hd = 1080 * z, 1920 * z
            s = max(Wd / y.width, Hd / y.height); y = y.resize((round(y.width * s), round(y.height * s)), Image.LANCZOS)
            x0 = (y.width - 1080) / 2 + dx * y.width; y0 = (y.height - 1920) / 2
            out.append(np.asarray(y.crop((round(x0), round(y0), round(x0) + 1080, round(y0) + 1920))))
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920", "-r", str(FPS_O), "-i", "-",
                            "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", job["out"]], stdin=subprocess.PIPE)
    for fr_ in out:
        enc.stdin.write(np.ascontiguousarray(fr_, dtype=np.uint8).tobytes())
    enc.stdin.close(); enc.wait()
    print(os.path.basename(job["out"]), "frames", len(fr), "->", n_out, flush=True)
''')
(W / "finish.py").write_text(FIN)
FD = OUT / "finished"; FD.mkdir(exist_ok=True)
fjobs = [{"src": str(CL / f"{b['take']}.mp4"), "out": str(FD / f"{sid}.mp4"), "dur": P["durations"].get(sid, 5.0), "fps": P.get("fps", 60),
          "cam": P["cams"].get(sid, "push")} for sid, b in best.items()]
halves = {0: fjobs[0::2], 1: fjobs[1::2]}
fp = {}
for g, js in halves.items():
    if js:
        (W / f"fin_g{g}.json").write_text(json.dumps({"jobs": js}))
        e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        fp[g] = subprocess.Popen(f"python {W/'finish.py'} {W/f'fin_g{g}.json'} > {LOG}/finish_g{g}.log 2>&1", shell=True, env=e)
t = time.time()
for g, p in fp.items():
    p.wait()
status["finish"] = {"s": round(time.time() - t, 1), "rc": {g: p.returncode for g, p in fp.items()},
                    "done": sorted(os.path.basename(x) for x in glob.glob(str(FD / "*.mp4")))}; save()
sh(f"tail -3 {LOG}/finish_g0.log {LOG}/finish_g1.log", "finish.log")

# contact sheets: 5 frames of every raw take
SHEET = textwrap.dedent('''
import imageio.v3 as iio, glob, os
from PIL import Image, ImageDraw
rows = []
for p in sorted(glob.glob("/kaggle/working/out/clips/*.mp4")):
    fr = list(iio.imiter(p)); n = len(fr); idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    ims = [Image.fromarray(fr[i]).resize((192, 333)) for i in idx]
    r = Image.new("RGB", (192 * 5 + 150, 333), "white"); d = ImageDraw.Draw(r); d.text((4, 4), os.path.basename(p)[:-4], fill="black")
    for i, im in enumerate(ims): r.paste(im, (150 + i * 192, 0))
    rows.append(r)
for k in range(0, len(rows), 8):
    S = Image.new("RGB", (rows[0].width, 333 * len(rows[k:k + 8])), "white")
    for i, r in enumerate(rows[k:k + 8]): S.paste(r, (0, i * 333))
    S.save(f"/kaggle/working/out/clips_sheet_{k // 8}.jpg", quality=85)
''')
(W / "sheet.py").write_text(SHEET)
sh(f"python {W/'sheet.py'}", "sheet.log", 900)
status["total_s"] = round(time.time() - status["t_start"], 1); save()
sh(f"cp {LOG}/*.log {OUT}/ ; ls -la {OUT} {OUT}/finished", "final.log")
print(json.dumps(status, indent=1))
