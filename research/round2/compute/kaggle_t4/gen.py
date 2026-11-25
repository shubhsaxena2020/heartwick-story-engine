# Heartwick round 2: Wan2.2-I2V-A14B (lightx2v 4-step distill, GGUF) on ONE GPU, phase-split by expert.
# Phase A: VAE-encode every clip's first frame (+ noise).  Phase B: load the high-noise expert once, run steps 0-1
# for every clip.  Phase C: free it, load the low-noise expert once, run steps 2-3 for every clip.  Phase D: VAE decode.
# Only one 14B expert is ever resident, so a 15 GB T4 holds it in VRAM and the process needs little host RAM; two
# copies of this script (one per T4) run side by side without exhausting Kaggle's ~31 GB RAM.
# Sampler = lightx2v WanStepDistillScheduler: shift 5, denoising_step_list [1000,750,500,250], Euler, CFG off.
import argparse, gc, json, os, time
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--test", action="store_true", help="tiny random models on CPU: checks the plumbing only")
ap.add_argument("--device", default="cuda", help="cuda (Kaggle T4) or xpu (laptop Arc iGPU)")
a = ap.parse_args()
plan = json.load(open(a.plan))
OUT = a.out; os.makedirs(OUT, exist_ok=True)
dev = "cpu" if a.test else a.device
DT = torch.float32 if a.test else torch.float16
BASE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
SHIFT, STEP_LIST, BOUNDARY = 5.0, [1000, 750, 500, 250], 2
rec = {"plan": plan, "phases": {}, "clips": {}}


def log():
    json.dump(rec, open(os.path.join(OUT, f"rec_{plan['name']}.json"), "w"), indent=1)


def sync():
    if dev != "cpu":
        getattr(torch, dev).synchronize()


def sigmas():
    s = torch.linspace(1.0, 0.0, 1001)[:-1]
    s = SHIFT * s / (1 + (SHIFT - 1) * s)
    return [float(s[1000 - x]) for x in STEP_LIST] + [0.0]


SIG = sigmas()

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video

t0 = time.time()
if a.test:
    vae = AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1, temperal_downsample=[False, True, True])
else:
    vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32)
vae = vae.to(dev).eval()
pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=None, image_processor=None,
                               image_encoder=None, transformer=None, transformer_2=None, boundary_ratio=0.9)
rec["phases"]["vae_load_s"] = round(time.time() - t0, 1)


def mem(tag):
    if dev != "cpu":
        acc = getattr(torch, dev)
        rec["phases"][f"peak_vram_gb_{tag}"] = round(acc.max_memory_allocated() / 1e9, 2)
        acc.reset_peak_memory_stats()


# ---- phase A: encode conditions -------------------------------------------------------------------------------------
state = {}
for c in plan["clips"]:
    t = time.time()
    img = Image.open(c["image"]).convert("RGB").resize((c["w"], c["h"]), Image.LANCZOS)
    x = pipe.video_processor.preprocess(img, height=c["h"], width=c["w"]).to(dev, dtype=torch.float32)
    g = torch.Generator("cpu").manual_seed(c["seed"])
    with torch.no_grad():
        lat, cond = pipe.prepare_latents(x, 1, 16, c["h"], c["w"], c["frames"], torch.float32, "cpu", g)
    sync()
    emb = torch.load(c["emb"])["pos"] if not a.test else torch.randn(1, 512, 4096)
    state[c["id"]] = {"lat": lat.cpu(), "cond": cond.cpu(), "emb": emb}
    rec["clips"][c["id"]] = {"cfg": c, "vae_encode_s": round(time.time() - t, 1),
                             "tokens": int(lat.shape[2] * lat.shape[3] * lat.shape[4] / 4)}
mem("encode")
log()
if dev != "cpu":  # hand the VAE-encode workspace back before a 10 GB expert arrives (matters on the shared-RAM iGPU)
    gc.collect(); getattr(torch, dev).empty_cache()


def load_expert(which):
    lk = None
    if dev == "cuda":  # serialize the host-RAM peak of GGUF loading between the two per-GPU processes
        import fcntl
        lk = open("/kaggle/working/load.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
    t = time.time()
    if a.test:
        m = WanTransformer3DModel(num_attention_heads=2, attention_head_dim=16, in_channels=36, out_channels=16,
                                  text_dim=4096, ffn_dim=64, num_layers=2)
    else:
        from diffusers import GGUFQuantizationConfig
        extra = {}
        if dev == "xpu":  # iGPU memory IS system RAM: stream each GGUF tensor straight to the device, never a full CPU copy
            import diffusers.models.model_loading_utils as mlu
            from diffusers.quantizers.gguf.utils import GGUFParameter
            import gguf

            def _load(path):
                out = {}
                for t in gguf.GGUFReader(path).tensors:
                    w = torch.from_numpy(t.data.copy()).to(dev)
                    q = t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
                    out[t.name] = GGUFParameter(w, quant_type=t.tensor_type) if q else w
                return out
            mlu.load_gguf_checkpoint = _load
            extra["device"] = dev
        m = WanTransformer3DModel.from_single_file(
            plan[which], quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16), config=BASE,
            subfolder="transformer" if which == "high" else "transformer_2", torch_dtype=torch.float16, **extra)
    m = m.to(dev).eval()
    sync()
    gc.collect()
    if lk:
        lk.close()
    return m, round(time.time() - t, 1)


def run_steps(model, steps, tag):
    for cid, s in state.items():
        lat = s["lat"].to(dev)
        cond = s["cond"].to(dev)
        emb = s["emb"].to(dev, DT)
        per = []
        for i in steps:
            sync(); t = time.time()
            with torch.no_grad():
                inp = torch.cat([lat, cond], 1).to(DT)
                ts = torch.tensor([SIG[i] * 1000.0], device=dev)
                v = model(hidden_states=inp, timestep=ts, encoder_hidden_states=emb, return_dict=False)[0].float()
            lat = lat + (SIG[i + 1] - SIG[i]) * v
            sync(); per.append(round(time.time() - t, 1))
        s["lat"] = lat.cpu()
        rec["clips"][cid][f"{tag}_step_s"] = per
        rec["clips"][cid][f"{tag}_finite"] = bool(torch.isfinite(lat).all())
        log()


for which, steps in (("high", range(0, BOUNDARY)), ("low", range(BOUNDARY, len(STEP_LIST)))):
    m, ls = load_expert(which)
    rec["phases"][f"{which}_load_s"] = ls
    run_steps(m, steps, which)
    mem(which)
    del m; gc.collect()
    if dev != "cpu":
        getattr(torch, dev).empty_cache()

# ---- phase D: decode --------------------------------------------------------------------------------------------------
mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
for cid, s in state.items():
    t = time.time()
    lat = (s["lat"] * std + mean).to(dev, torch.float32)
    with torch.no_grad():
        try:
            vid = vae.decode(lat, return_dict=False)[0]
            tiled = False
        except torch.OutOfMemoryError:
            getattr(torch, dev).empty_cache(); vae.enable_tiling(); tiled = True
            vid = vae.decode(lat, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(vid, output_type="np")[0]
    sync()
    rec["clips"][cid]["vae_decode_s"] = round(time.time() - t, 1)
    rec["clips"][cid]["decode_tiled"] = tiled
    p = os.path.join(OUT, f"{cid}.mp4")
    export_to_video(list(frames), p, fps=16)
    Image.fromarray((frames[-1] * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(OUT, f"{cid}_last.png"))
    c = rec["clips"][cid]
    c["gpu_s_total"] = round(c["vae_encode_s"] + sum(c["high_step_s"]) + sum(c["low_step_s"]) + c["vae_decode_s"], 1)
    log()
mem("decode")
rec["phases"]["wall_s"] = round(time.time() - t0, 1)
log()
print(json.dumps(rec["clips"], indent=1))
