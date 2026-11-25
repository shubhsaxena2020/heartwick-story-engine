# Heartwick round 4: FastWan2.2-TI2V-5B (FastVideo DMD 3-step, Apache-2.0) image-to-video on ONE T4, fp16.
# TI2V-5B I2V = "expand_timesteps": the first latent frame is the clean image latent with timestep 0, every other token
# gets the step's timestep (same as diffusers WanImageToVideoPipeline with expand_timesteps=True).
# DMD sampling (FastVideo DmdDenoisingStage): x0 = x_t - sigma_t * v; re-noise x0 to the next sigma with fresh noise;
# sigmas = t/1000 for t in [1000, 757, 522]; the last x0 is the output. No CFG, so negative prompts do nothing.
# usage: python fw_gen.py plan.json outdir
import gc, json, os, sys, time, fcntl
import numpy as np, torch
from PIL import Image

plan = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
REPO = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
dev = "cuda"
rec = {"name": plan["name"], "clips": {}, "phases": {}}


def save():
    json.dump(rec, open(os.path.join(OUT, f"rec_{plan['name']}.json"), "w"), indent=1)


def say(*a):
    print(time.strftime("%H:%M:%S"), plan["name"], *a, flush=True)


from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

t = time.time()
lk = open("/kaggle/working/load.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)   # one 10 GB load at a time (host RAM)
vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
tr = WanTransformer3DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.float16).to(dev).eval()
lk.close()
rec["phases"]["load_s"] = round(time.time() - t, 1); save(); say("loaded", rec["phases"]["load_s"])
vp = VideoProcessor(vae_scale_factor=16)
Z = vae.config.z_dim
mean = torch.tensor(vae.config.latents_mean).view(1, Z, 1, 1, 1).to(dev)
std = torch.tensor(vae.config.latents_std).view(1, Z, 1, 1, 1).to(dev)
SIG = [s / 1000.0 for s in plan.get("dmd", [1000, 757, 522])]

for c in plan["clips"]:
    r = {"cfg": c}; rec["clips"][c["id"]] = r
    try:
        w, h, F = c["w"], c["h"], c["frames"]
        torch.cuda.synchronize(); t = time.time()
        img = Image.open(c["image"]).convert("RGB").resize((w, h), Image.LANCZOS)
        x = vp.preprocess(img, height=h, width=w).to(dev, torch.float32).unsqueeze(2)
        with torch.no_grad():
            cond = vae.encode(x).latent_dist.mode()
        cond = (cond - mean) / std
        torch.cuda.synchronize(); r["encode_s"] = round(time.time() - t, 1)
        T, lh, lw = (F - 1) // 4 + 1, h // 16, w // 16
        g = torch.Generator("cpu").manual_seed(c["seed"])
        lat = torch.randn((1, Z, T, lh, lw), generator=g).to(dev)
        mask = torch.ones(1, 1, T, lh, lw, device=dev); mask[:, :, 0] = 0
        emb = torch.load(c["emb"])["pos"].to(dev, torch.float16)
        r["tokens"] = int(T * lh * lw / 4)
        per = []
        for i, s in enumerate(SIG):
            torch.cuda.synchronize(); t = time.time()
            xt = (1 - mask) * cond + mask * lat
            ts = (mask[0][0][:, ::2, ::2] * (s * 1000.0)).flatten().unsqueeze(0)
            with torch.no_grad():
                v = tr(hidden_states=xt.half(), timestep=ts, encoder_hidden_states=emb, return_dict=False)[0].float()
            x0 = xt - s * v
            if i + 1 < len(SIG):
                n = torch.randn(lat.shape, generator=g).to(dev)
                lat = (1 - SIG[i + 1]) * x0 + SIG[i + 1] * n
            else:
                lat = x0
            torch.cuda.synchronize(); per.append(round(time.time() - t, 1))
        r["step_s"] = per; r["finite"] = bool(torch.isfinite(lat).all())
        lat = (1 - mask) * cond + mask * lat
        say(c["id"], "steps", per, "finite", r["finite"], "tokens", r["tokens"])
        torch.cuda.synchronize(); t = time.time()
        z = (lat * std + mean).float()
        with torch.no_grad():
            try:
                vid = vae.decode(z, return_dict=False)[0]; r["tiled"] = False
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); vae.enable_tiling(); r["tiled"] = True
                vid = vae.decode(z, return_dict=False)[0]
        fr = vp.postprocess_video(vid, output_type="np")[0]
        torch.cuda.synchronize(); r["decode_s"] = round(time.time() - t, 1)
        del vid, z; torch.cuda.empty_cache()
        p = os.path.join(OUT, f"{c['id']}.mp4")
        export_to_video(list(fr), p, fps=c.get("fps", 24))
        r["gpu_s_total"] = round(r["encode_s"] + sum(per) + r["decode_s"], 1)
        r["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2); torch.cuda.reset_peak_memory_stats()
        say(c["id"], "done", r["gpu_s_total"], "s; decode", r["decode_s"], "tiled", r["tiled"])
    except Exception as e:
        import traceback; r["error"] = traceback.format_exc()[-1500:]; say(c["id"], "ERROR", repr(e)[:300])
        torch.cuda.empty_cache()
    save()
del tr; gc.collect(); torch.cuda.empty_cache()
