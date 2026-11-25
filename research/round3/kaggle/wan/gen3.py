# Heartwick round 3: Wan2.2-I2V-A14B (lightx2v 4-step distill 260412, GGUF), phase-split by expert.
# Based on research/round2/compute/kaggle_t4/gen.py (+ round2 quality first-last-frame and 2-GPU shard).
# New in round 3:
#   * per-clip step count (4 / 6 / 8): sigmas from the lightx2v WanStepDistillScheduler (shift 5, linear timesteps
#     from 1000), high-noise expert while sigma >= 0.9 (the Wan2.2 I2V boundary), low-noise expert after;
#   * per-clip last frame (FLF2V);
#   * plan["split"]: ONE expert across BOTH T4s (first half of the blocks on cuda:0, second half on cuda:1), for Q8_0
#     (15.4 GB per expert) and 720x1280 activations;
#   * VAE tiling for big frames, high-quality mp4 (crf 12) so the finishing pass starts from clean frames;
#   * peak VRAM per GPU and per-clip GPU seconds in rec_<plan>.json.
import argparse, gc, json, os, subprocess, time
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
plan = json.load(open(a.plan))
OUT = a.out; os.makedirs(OUT, exist_ok=True)
dev = "cuda"
DT = torch.float16
BASE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
SHIFT = 5.0
rec = {"plan": plan, "phases": {}, "clips": {}}
SPLIT = bool(plan.get("split")) and torch.cuda.device_count() > 1


def log():
    json.dump(rec, open(os.path.join(OUT, f"rec_{plan['name']}.json"), "w"), indent=1)


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def sigmas(n):
    steps = [int(round(1000 - i * 1000 / n)) for i in range(n)]          # 4 -> [1000, 750, 500, 250]
    s = torch.linspace(1.0, 0.0, 1001)[:-1]
    s = SHIFT * s / (1 + (SHIFT - 1) * s)
    return [float(s[1000 - x]) for x in steps] + [0.0]


from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel

t0 = time.time()
vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
if any(c["w"] * c["h"] > 560 * 1000 for c in plan["clips"]):
    vae.enable_tiling()
    rec["phases"]["vae_tiling"] = True
pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=None, image_processor=None,
                               image_encoder=None, transformer=None, transformer_2=None, boundary_ratio=0.9)
rec["phases"]["vae_load_s"] = round(time.time() - t0, 1)


def mem(tag):
    for i in range(torch.cuda.device_count()):
        rec["phases"][f"peak_vram_gb_{tag}_gpu{i}"] = round(torch.cuda.max_memory_allocated(i) / 1e9, 2)
        torch.cuda.reset_peak_memory_stats(i)


# ---- phase A: encode conditions -------------------------------------------------------------------------------------
state = {}
for c in plan["clips"]:
    t = time.time()
    img = Image.open(c["image"]).convert("RGB").resize((c["w"], c["h"]), Image.LANCZOS)
    x = pipe.video_processor.preprocess(img, height=c["h"], width=c["w"]).to(dev, dtype=torch.float32)
    g = torch.Generator("cpu").manual_seed(c["seed"])
    y = None
    if c.get("last"):
        im2 = Image.open(c["last"]).convert("RGB").resize((c["w"], c["h"]), Image.LANCZOS)
        y = pipe.video_processor.preprocess(im2, height=c["h"], width=c["w"]).to(dev, dtype=torch.float32)
    with torch.no_grad():
        lat, cond = pipe.prepare_latents(x, 1, 16, c["h"], c["w"], c["frames"], torch.float32, "cpu", g, last_image=y)
    sync()
    emb = torch.load(c["emb"])["pos"]
    sig = sigmas(c.get("steps", 4))
    hi = [i for i in range(len(sig) - 1) if sig[i] >= 0.9]
    lo = [i for i in range(len(sig) - 1) if sig[i] < 0.9]
    state[c["id"]] = {"lat": lat.cpu(), "cond": cond.cpu(), "emb": emb, "sig": sig, "high": hi, "low": lo}
    rec["clips"][c["id"]] = {"cfg": c, "vae_encode_s": round(time.time() - t, 1), "sigmas": [round(v, 4) for v in sig],
                             "tokens": int(lat.shape[2] * lat.shape[3] * lat.shape[4] / 4)}
    log()
mem("encode")
gc.collect(); torch.cuda.empty_cache()


def _mv(x, d):
    if torch.is_tensor(x):
        return x.to(d)
    if isinstance(x, (list, tuple)):
        return type(x)(_mv(y, d) for y in x)
    if isinstance(x, dict):
        return {k: _mv(v, d) for k, v in x.items()}
    return x


def shard(m):
    d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
    n = len(m.blocks); cut = n // 2
    for name, ch in m.named_children():
        if name != "blocks":
            ch.to(d0)
    for p_name, p in m.named_parameters(recurse=False):
        p.data = p.data.to(d0)
    for p_name, b in m.named_buffers(recurse=False):
        b.data = b.data.to(d0)
    for i, b in enumerate(m.blocks):
        d = d0 if i < cut else d1
        b.to(d)
        b.register_forward_pre_hook(lambda mod, args, kw, d=d: (_mv(args, d), _mv(kw, d)), with_kwargs=True)
    for name in ("norm_out", "proj_out"):
        if hasattr(m, name):
            getattr(m, name).register_forward_pre_hook(lambda mod, args, kw: (_mv(args, d0), _mv(kw, d0)), with_kwargs=True)
    return m


def load_expert(which):
    lk = None
    if not SPLIT:  # serialize the host-RAM peak of GGUF loading between the per-GPU processes
        import fcntl
        lk = open("/kaggle/working/load.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
    t = time.time()
    from diffusers import GGUFQuantizationConfig
    m = WanTransformer3DModel.from_single_file(
        plan[which], quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16), config=BASE,
        subfolder="transformer" if which == "high" else "transformer_2", torch_dtype=torch.float16)
    m = shard(m.eval()) if SPLIT else m.to(dev).eval()
    sync(); gc.collect()
    if lk:
        lk.close()
    return m, round(time.time() - t, 1)


def run_steps(model, tag):
    for cid, s in state.items():
        idx = s[tag]
        if not idx:
            continue
        lat = s["lat"].to(dev); cond = s["cond"].to(dev); emb = s["emb"].to(dev, DT); SIG = s["sig"]
        per = []
        for i in idx:
            sync(); t = time.time()
            with torch.no_grad():
                inp = torch.cat([lat, cond], 1).to(DT)
                ts = torch.tensor([SIG[i] * 1000.0], device=dev)
                v = model(hidden_states=inp, timestep=ts, encoder_hidden_states=emb, return_dict=False)[0].float().to(dev)
            lat = lat + (SIG[i + 1] - SIG[i]) * v
            sync(); per.append(round(time.time() - t, 1))
        s["lat"] = lat.cpu()
        rec["clips"][cid][f"{tag}_step_s"] = per
        rec["clips"][cid][f"{tag}_finite"] = bool(torch.isfinite(lat).all())
        log()


for which in ("high", "low"):
    m, ls = load_expert(which)
    rec["phases"][f"{which}_load_s"] = ls
    run_steps(m, which)
    mem(which)
    del m; gc.collect(); torch.cuda.empty_cache()


def write_mp4(frames, path, fps=16):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()


# ---- phase D: decode --------------------------------------------------------------------------------------------------
mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
for cid, s in state.items():
    t = time.time()
    lat = (s["lat"] * std + mean).to(dev, torch.float32)
    with torch.no_grad():
        try:
            vid = vae.decode(lat, return_dict=False)[0]
            tiled = bool(rec["phases"].get("vae_tiling"))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); vae.enable_tiling(); tiled = True
            vid = vae.decode(lat, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(vid, output_type="np")[0]
    sync()
    fr8 = [(f * 255).clip(0, 255).astype(np.uint8) for f in frames]
    c = rec["clips"][cid]
    c["vae_decode_s"] = round(time.time() - t, 1); c["decode_tiled"] = tiled
    write_mp4(fr8, os.path.join(OUT, f"{cid}.mp4"))
    Image.fromarray(fr8[-1]).save(os.path.join(OUT, f"{cid}_last.png"))
    c["gpu_s_total"] = round(c["vae_encode_s"] + sum(c.get("high_step_s", [])) + sum(c.get("low_step_s", [])) + c["vae_decode_s"], 1)
    c["gpus_used"] = 2 if SPLIT else 1
    del vid, lat; torch.cuda.empty_cache()
    log()
mem("decode")
rec["phases"]["wall_s"] = round(time.time() - t0, 1)
log()
print(json.dumps({k: v.get("gpu_s_total") for k, v in rec["clips"].items()}, indent=1))
