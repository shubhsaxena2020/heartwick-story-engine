# Heartwick round 4 (run 2): FastWan2.2-TI2V-5B DMD 3-step I2V on ONE T4, phase-split so the 10 GB fp16 transformer and the
# fp32 VAE decode are never resident together (run 1 OOMed in the decode: 14.3 GB allocated).
# A: VAE-encode every first frame.  B: load the transformer (from the weight-cache kernel when mounted), run 3 DMD steps per
# clip, latents to CPU.  C: free the transformer, VAE-decode every clip (untiled, tiled fallback).
# usage: python fw_gen2.py plan.json outdir
import gc, glob, json, os, sys, time, fcntl
import numpy as np, torch
from PIL import Image

plan = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
REPO = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
CACHE = sorted(glob.glob("/kaggle/input/**/fastwan5b/transformer", recursive=True))
SRC = os.path.dirname(CACHE[0]) if CACHE else REPO
dev = "cuda"
rec = {"name": plan["name"], "weights_from": SRC, "clips": {}, "phases": {}}


def save():
    json.dump(rec, open(os.path.join(OUT, f"rec_{plan['name']}.json"), "w"), indent=1)


def say(*a):
    print(time.strftime("%H:%M:%S"), plan["name"], *a, flush=True)


def sync():
    torch.cuda.synchronize()


from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

t = time.time()
vae = AutoencoderKLWan.from_pretrained(SRC, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
rec["phases"]["vae_load_s"] = round(time.time() - t, 1)
vp = VideoProcessor(vae_scale_factor=16)
Z = vae.config.z_dim
mean = torch.tensor(vae.config.latents_mean).view(1, Z, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, Z, 1, 1, 1)
SIG = [s / 1000.0 for s in plan.get("dmd", [1000, 757, 522])]

# ---- A: encode ------------------------------------------------------------------------------------------------------------
S = {}
for c in plan["clips"]:
    r = rec["clips"].setdefault(c["id"], {"cfg": c})
    w, h, F = c["w"], c["h"], c["frames"]
    sync(); t = time.time()
    img = Image.open(c["image"]).convert("RGB").resize((w, h), Image.LANCZOS)
    x = vp.preprocess(img, height=h, width=w).to(dev, torch.float32).unsqueeze(2)
    with torch.no_grad():
        cond = vae.encode(x).latent_dist.mode().cpu()
    sync(); r["encode_s"] = round(time.time() - t, 1)
    S[c["id"]] = (cond - mean) / std
save()
vae.to("cpu"); torch.cuda.empty_cache()

# ---- B: denoise -------------------------------------------------------------------------------------------------------------
lk = open("/kaggle/working/load.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
t = time.time()
tr = WanTransformer3DModel.from_pretrained(SRC, subfolder="transformer", torch_dtype=torch.float16).to(dev).eval()
sync(); rec["phases"]["dit_load_s"] = round(time.time() - t, 1); lk.close(); save(); say("dit loaded", rec["phases"]["dit_load_s"], "from", SRC)
LAT = {}
for c in plan["clips"]:
    r = rec["clips"][c["id"]]
    try:
        w, h, F = c["w"], c["h"], c["frames"]
        T, lh, lw = (F - 1) // 4 + 1, h // 16, w // 16
        cond = S[c["id"]].to(dev)
        g = torch.Generator("cpu").manual_seed(c["seed"])
        lat = torch.randn((1, Z, T, lh, lw), generator=g).to(dev)
        mask = torch.ones(1, 1, T, lh, lw, device=dev); mask[:, :, 0] = 0
        emb = torch.load(c["emb"])["pos"].to(dev, torch.float16)
        r["tokens"] = int(T * lh * lw / 4); per = []
        for i, s in enumerate(SIG):
            sync(); t = time.time()
            xt = (1 - mask) * cond + mask * lat
            ts = (mask[0][0][:, ::2, ::2] * (s * 1000.0)).flatten().unsqueeze(0)
            with torch.no_grad():
                v = tr(hidden_states=xt.half(), timestep=ts, encoder_hidden_states=emb, return_dict=False)[0].float()
            x0 = xt - s * v
            lat = (1 - SIG[i + 1]) * x0 + SIG[i + 1] * torch.randn(lat.shape, generator=g).to(dev) if i + 1 < len(SIG) else x0
            del v, x0, xt
            sync(); per.append(round(time.time() - t, 1))
        lat = (1 - mask) * cond + mask * lat
        r["step_s"] = per; r["finite"] = bool(torch.isfinite(lat).all())
        r["peak_vram_gb_dit"] = round(torch.cuda.max_memory_allocated() / 1e9, 2); torch.cuda.reset_peak_memory_stats()
        LAT[c["id"]] = lat.cpu(); del lat, cond, emb, mask
        say(c["id"], "steps", per, "tokens", r["tokens"], "finite", r["finite"])
    except Exception:
        import traceback; r["error_dit"] = traceback.format_exc()[-1200:]; say(c["id"], "DIT ERROR", r["error_dit"][-300:])
    gc.collect(); torch.cuda.empty_cache(); save()
del tr; gc.collect(); torch.cuda.empty_cache()

# ---- C: decode --------------------------------------------------------------------------------------------------------------
vae.to(dev)
for cid, lat in LAT.items():
    r = rec["clips"][cid]; c = r["cfg"]
    try:
        sync(); t = time.time()
        z = (lat * std + mean).to(dev, torch.float32)
        with torch.no_grad():
            try:
                vid = vae.decode(z, return_dict=False)[0]; r["tiled"] = False
            except torch.OutOfMemoryError:
                vid = None; gc.collect(); torch.cuda.empty_cache(); vae.enable_tiling(); r["tiled"] = True
                vid = vae.decode(z, return_dict=False)[0]
        fr = vp.postprocess_video(vid, output_type="np")[0]
        sync(); r["decode_s"] = round(time.time() - t, 1)
        r["peak_vram_gb_decode"] = round(torch.cuda.max_memory_allocated() / 1e9, 2); torch.cuda.reset_peak_memory_stats()
        del vid, z
        export_to_video(list(fr), os.path.join(OUT, f"{cid}.mp4"), fps=c.get("fps", 24))
        r["gpu_s_total"] = round(r["encode_s"] + sum(r["step_s"]) + r["decode_s"], 1)
        say(cid, "decoded", r["decode_s"], "tiled", r["tiled"], "total", r["gpu_s_total"])
    except Exception:
        import traceback; r["error_decode"] = traceback.format_exc()[-1200:]; say(cid, "DECODE ERROR", r["error_decode"][-300:])
    gc.collect(); torch.cuda.empty_cache(); save()
