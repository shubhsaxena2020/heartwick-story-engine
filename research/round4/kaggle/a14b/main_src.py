# Heartwick round 4, run 3 (Kaggle T4 x2): Wan2.2-I2V-A14B lightx2v 4-step GGUF Q5 (the v3 production model) on key K2 with
# the "mouth closed / silent" prompt: GPU0 = 81 frames (+ fp16-VAE decode test), GPU1 = 49 frames (a shorter action clip).
# umt5 embeddings come from heartwick-r4-bench (kernel source). IMGS, GEN_SRC are prepended by build.py.
import base64, glob, json, os, subprocess, sys, threading, time
from pathlib import Path
W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True); LOG = OUT / "logs"; LOG.mkdir(exist_ok=True)
IN = W / "in"; IN.mkdir(exist_ok=True)
for k, b in IMGS.items():
    (IN / k).write_bytes(base64.b64decode(b))
(W / "gen_flf_r4.py").write_text(GEN_SRC)
status = {"t_start": time.time()}
def save(): (OUT / "status.json").write_text(json.dumps(status, indent=1))
def sh(cmd, log, env=None, timeout=None):
    t = time.time()
    with open(LOG / log, "a") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
            if "it/s]" not in line and "s/it]" not in line: print(f"[{log}] {line}", end="", flush=True)
        rc = p.wait(timeout=timeout)
    print(f"[{log}] rc={rc} {round(time.time() - t, 1)}s", flush=True); return {"rc": rc, "s": round(time.time() - t, 1)}
status["setup"] = sh("pip install -q -U 'diffusers>=0.36' 'transformers>=4.51' accelerate gguf ftfy imageio imageio-ffmpeg 2>&1 | tail -2", "setup.log", timeout=1500)
t = time.time()
from huggingface_hub import hf_hub_download, snapshot_download
REPO = "jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF"; Q = "Q5_K_M"
GG = {w: hf_hub_download(REPO, f"{w}_noise_260412/wan2.2_i2v_A14b_{w}_noise_lightx2v_4step_720p_260412-{Q}.gguf") for w in ("high", "low")}
snapshot_download("Wan-AI/Wan2.2-I2V-A14B-Diffusers", allow_patterns=["model_index.json", "vae/*", "transformer/config.json", "transformer_2/config.json", "scheduler/*"])
status["download_s"] = round(time.time() - t, 1); save()
emb = sorted(glob.glob("/kaggle/input/**/emb_k2_mouth.pt", recursive=True))[0]
CL = OUT / "clips"; CL.mkdir(exist_ok=True)
plans = {0: [{"id": "a14b_k2_mouth_81f", "image": str(IN / "K2.jpg"), "w": 480, "h": 832, "frames": 81, "seed": 7, "emb": emb}],
         1: [{"id": "a14b_k2_mouth_49f", "image": str(IN / "K2.jpg"), "w": 480, "h": 832, "frames": 49, "seed": 7, "emb": emb}]}
procs = {}
for g, clips in plans.items():
    pf = W / f"plan_g{g}.json"
    pf.write_text(json.dumps({"name": f"g{g}", "high": GG["high"], "low": GG["low"], "quant": Q, "clips": clips, "fp16_decode_test": g == 0}))
    e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")
    procs[g] = subprocess.Popen([sys.executable, str(W / "gen_flf_r4.py"), "--plan", str(pf), "--out", str(CL)],
                                stdout=open(LOG / f"gen_g{g}.log", "w"), stderr=subprocess.STDOUT, env=e)
for g, p in procs.items():
    p.wait(); status[f"g{g}_rc"] = p.returncode
status["total_s"] = round(time.time() - status["t_start"], 1); save()
sh(f"tail -n 30 {LOG}/gen_g0.log {LOG}/gen_g1.log", "final.log")
