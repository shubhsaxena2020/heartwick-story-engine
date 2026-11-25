# Heartwick engine v3, keys stage (runs on Kaggle T4 x2). Qwen-Image-Edit-2511 (Apache-2.0) + Lightning 4-step LoRA
# (Apache-2.0): GPU0 = GGUF transformer + VAE, GPU1 = Qwen2.5-VL text/vision encoder in NF4 (fp32 compute; bnb int8 is
# not supported on a T4) + the QA models. Character sheets first, then every key as an edit of its source key (same
# scene) or of the sheets (new scene), with a pose skeleton as an extra image. N seeds per key; the QA gate (OWLv2 +
# DINOv2) picks the best: identity vs the master, Pip's size vs the source key (+-15 %), framing (cut-offs, headroom).
# usage: python keys_gen.py jobs.json outdir
import gc, inspect, json, math, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # bf16 Q4 OOMed by ~0.6 GB from fragmentation
import numpy as np, torch
from PIL import Image

jobs = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
REC = {"jobs": {}, "phases": {}, "chosen": {}}
DONE = set(jobs.get("skip", []))


def save():
    json.dump(REC, open(os.path.join(OUT, "keys_rec.json"), "w"), indent=1)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as QE
if hasattr(QE, "VAE_IMAGE_SIZE"):
    QE.VAE_IMAGE_SIZE = 768 * 768      # reference latents at ~0.6 MP instead of 1 MP: fewer tokens, fits a T4
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

REPO = "Qwen/Qwen-Image-Edit-2511"
GG = {"Q4_K_M": "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/blob/main/qwen-image-edit-2511-Q4_K_M.gguf",
      "Q5_K_M": "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/blob/main/qwen-image-edit-2511-Q5_K_M.gguf",
      "Q3_K_M": "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/blob/main/qwen-image-edit-2511-Q3_K_M.gguf"}
QUANT = jobs.get("quant", "Q4_K_M")
D0, D1 = torch.device("cuda:0"), torch.device("cuda:1")
CDT = torch.float16
t0 = time.time()

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32,
                       llm_int8_skip_modules=["visual", "lm_head"])
te = Qwen2_5_VLForConditionalGeneration.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.float32,
                                                        quantization_config=q, device_map={"": 1}).eval()
REC["phases"]["te_load_s"] = round(time.time() - t0, 1); save(); say("text encoder on GPU1", REC["phases"]["te_load_s"])

sched = FlowMatchEulerDiscreteScheduler.from_config({
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192, "max_shift": math.log(3),
    "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False, "time_shift_type": "exponential",
    "use_beta_sigmas": False, "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False})

QwenImageEditPlusPipeline._execution_device = property(lambda self: D0)
_orig = QwenImageEditPlusPipeline._get_qwen_prompt_embeds
_sig = inspect.signature(_orig)


def _patched(self, *a, **k):   # text/vision encoding on GPU1, the denoiser on GPU0
    b = _sig.bind(self, *a, **k); b.arguments["device"] = D1; b.arguments["dtype"] = torch.float32
    e, m = _orig(*b.args, **b.kwargs)
    return e.to(D0, CDT), m.to(D0)


QwenImageEditPlusPipeline._get_qwen_prompt_embeds = _patched

# The VAE stays float32 (fp16 VAE overflows), but prepare_latents casts the reference images to the
# denoiser dtype before encoding: "Input type (Half) and bias type (float)" on every smoke test
# (23 Nov run 1). Encode in the VAE's own dtype, hand the latents back in the denoiser dtype.
_orig_enc = QwenImageEditPlusPipeline._encode_vae_image


def _enc_patched(self, image, *a, **k):
    out = _orig_enc(self, image.to(self.vae.dtype), *a, **k)
    return out.to(CDT) if torch.is_tensor(out) else out


QwenImageEditPlusPipeline._encode_vae_image = _enc_patched

# Memory (24 Nov run 3): bf16 OOMed on the T4 even with Q3, short by one 2 GiB block. EditPlus resizes every
# reference image to ~1 MP before VAE-encoding it, so the reference tokens dwarf a 480x832 output. Encode the
# references at the output area instead (keys are 480x832 now, the Wan clip size).
import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as _qep
if hasattr(_qep, "VAE_IMAGE_SIZE"):
    _qep.VAE_IMAGE_SIZE = 480 * 832
REC["phases"]["vae_image_size"] = getattr(_qep, "VAE_IMAGE_SIZE", "n/a")


def build(cdt, quant):
    global CDT
    CDT = cdt
    t = time.time()
    tr = QwenImageTransformer2DModel.from_single_file(GG[quant], quantization_config=GGUFQuantizationConfig(compute_dtype=cdt),
                                                      config=REPO, subfolder="transformer", torch_dtype=cdt).to(D0)
    from diffusers import AutoencoderKLQwenImage
    vae = AutoencoderKLQwenImage.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).to(D0)
    vae.enable_tiling()
    p = QwenImageEditPlusPipeline.from_pretrained(REPO, transformer=tr, text_encoder=te, vae=vae, scheduler=sched, torch_dtype=cdt)
    p.set_progress_bar_config(disable=True)
    p.load_lora_weights("lightx2v/Qwen-Image-Edit-2511-Lightning", weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
    REC["phases"][f"build_{quant}_{str(cdt)[6:]}_s"] = round(time.time() - t, 1); save()
    return p


def run(p, refs, prompt, W, H, seed, steps=4):
    imgs = [Image.open(r).convert("RGB") for r in refs]
    # 24 Nov run 4: 1 reference fit (14.0 GB peak), but keys with 2-4 references (sheets + previous key + pose
    # skeleton) OOMed by ~2.6 GiB. Keep the TOTAL reference area at the size that fit: each reference gets
    # (480*832)/n pixels, floored at 192*336 so faces stay readable.
    # 24 Nov run 4b: every 3-reference key still OOMed by ~0.4 GiB (each image also adds vision tokens to the
    # prompt), so 3+ references start at 70 % of that area, and an OOM retries at 60 % / 40 % of it.
    base = (480 * 832) // max(1, len(imgs)) * (0.7 if len(imgs) >= 3 else 1.0)
    for f in (1.0, 0.6, 0.4):
        if hasattr(_qep, "VAE_IMAGE_SIZE"):
            _qep.VAE_IMAGE_SIZE = int(max(160 * 288, base * f))
        gc.collect(); torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                return p(image=imgs, prompt=prompt, height=H, width=W, num_inference_steps=steps, true_cfg_scale=1.0,
                         generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        except torch.cuda.OutOfMemoryError:
            if f == 0.4:
                raise
            say("OOM at ref area", getattr(_qep, "VAE_IMAGE_SIZE", "?"), "- retrying smaller")


def ok(im):
    a = np.asarray(im).astype(np.float32) / 255
    return bool(np.isfinite(a).all() and a.std() > 0.04 and a.mean() > 0.03)


# smoke test at the real size. 23 Nov runs: fp16 gives an all-black image (Qwen-Image overflows to NaN in fp16 even
# with diffusers' clipping, Q4 and Q3 alike), so bf16 (emulated on a T4, slower) goes first; bf16+Q4 OOMed by ~0.6 GB,
# so the smaller Q3 quant leads, then Q4 with expandable segments; fp16 only as a last resort.
pipe = None
for cdt, quant in ((torch.bfloat16, "Q3_K_M"), (torch.bfloat16, QUANT), (torch.float16, QUANT)):
    if pipe is not None:
        del pipe; gc.collect(); torch.cuda.empty_cache()
    tag = f"{quant}_{str(cdt)[6:]}"; t = time.time(); good = False
    try:
        pipe = build(cdt, quant)
        sm = run(pipe, [jobs["smoke_ref"]], "The same woman, now smiling, everything else identical.", 480, 832, 1, steps=2)
        sm.save(os.path.join(OUT, f"smoke_{tag}.png")); good = ok(sm)
    except Exception as e:
        REC["phases"][f"smoke_{tag}_err"] = repr(e)[:600]
    REC["phases"][f"smoke_{tag}"] = {"ok": good, "s": round(time.time() - t, 1), "peak_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2)}
    save(); say("smoke", tag, REC["phases"][f"smoke_{tag}"], REC["phases"].get(f"smoke_{tag}_err", ""))
    torch.cuda.reset_peak_memory_stats(0)
    if good:
        REC["phases"]["config"] = tag; break
else:
    save(); raise SystemExit("no Qwen-Image-Edit config ran on this T4")

os.environ["QA_DEVICE"] = "cuda:1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa


def score(path, want_nell, want_pip, pip_ref, bank_n, bank_p):
    r = qa.analyse(path, bank_n, bank_p, 1)[0]
    d = {k: v for k, v in r.items() if not k.startswith("_")}
    s = 0.0
    if want_nell:
        s += d.get("body_sim", 0) + d.get("head_sim", 0)
        if "body_sim" not in d: s -= 1.0                  # Nell not found at all
        if d.get("cut"): s -= 0.3 * len([c for c in d["cut"] if c != "bottom"]) + (0.15 if "bottom" in d["cut"] else 0)
        hr = d.get("headroom")
        if hr is not None and hr < jobs.get("min_headroom", 0.03): s -= 0.5
    if want_pip:
        if "pip_sim" in d: s += 0.5 * d["pip_sim"]
        else: s -= 0.4
        if pip_ref and "pip_scale" in d:
            dev = abs(math.log(d["pip_scale"] / pip_ref)); d["pip_scale_dev"] = round(dev, 3)
            if dev > math.log(1.15): s -= 2.0 * dev        # outside the +-15 % band
    d["score"] = round(s, 3); return d


def contact(paths, out):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB"); im.thumbnail((288, 512)); ims.append((os.path.basename(p), im))
    if not ims: return
    from PIL import ImageDraw
    cols = min(8, len(ims)); rows = (len(ims) + cols - 1) // cols
    S = Image.new("RGB", (cols * 292, rows * 540), "white"); d = ImageDraw.Draw(S)
    for i, (n, im) in enumerate(ims):
        x, y = (i % cols) * 292, (i // cols) * 540
        S.paste(im, (x + 2, y + 24)); d.text((x + 4, y + 4), n[:40], fill="black")
    S.save(out, quality=88)


chosen = {k: v for k, v in jobs.get("given", {}).items()}   # e.g. {"K0": "/kaggle/working/in/K0.png"}
bank_n = bank_p = None
for j in jobs["keys"]:
    if j["id"] in DONE:
        continue
    if bank_n is None and not j["id"].startswith("sheet"):  # QA banks: the master + the sheets once they exist
        bank_n = qa.ref_bank([chosen["K0"]] + ([chosen["sheet_nell"]] if "sheet_nell" in chosen else []), "nell")
        bank_p = qa.ref_bank([chosen["sheet_pip"]] if "sheet_pip" in chosen else [chosen["K0"]], "pip")
    refs = [chosen.get(r[1:], r) if r.startswith("@") else r for r in j["refs"]]   # "@K3" -> the chosen file of key K3
    if any(r.startswith("@") or not os.path.exists(r) for r in refs):
        REC["jobs"][j["id"]] = {"error": f"missing ref {refs}"}; save(); continue
    pip_ref = None
    if j.get("pip_from") and j["pip_from"] in REC["jobs"]:
        c = REC["jobs"][j["pip_from"]].get("best") or {}
        pip_ref = c.get("pip_scale")
    pip_ref = pip_ref or j.get("pip_target")
    cands = []
    for seed in j["seeds"]:
        t = time.time()
        try:
            im = run(pipe, refs, j["prompt"], j["W"], j["H"], seed)
        except Exception as e:
            REC["jobs"].setdefault(j["id"], {}).setdefault("errors", []).append(repr(e)[:400]); save(); continue
        p = os.path.join(OUT, f"{j['id']}_s{seed}.png"); im.save(p)
        gs = round(time.time() - t, 1)
        if j["id"].startswith("sheet") or bank_n is None:
            d = {"score": 0.0 if ok(im) else -9}
        else:
            d = score(p, "nell" in j["chars"], bool({"pip", "pip_baby"} & set(j["chars"])), pip_ref, bank_n, bank_p)
        d.update(gen_s=gs, seed=seed, file=os.path.basename(p))
        cands.append((d["score"], p, d))
        REC["jobs"].setdefault(j["id"], {}).setdefault("cands", []).append(d); save()
        say(j["id"], "seed", seed, gs, "s score", d.get("score"), {k: d.get(k) for k in ("body_sim", "head_sim", "pip_scale", "cut", "headroom")})
    if cands:
        cands.sort(key=lambda x: -x[0]); chosen[j["id"]] = cands[0][1]
        REC["jobs"][j["id"]].update(chosen=os.path.basename(cands[0][1]), best=cands[0][2], pip_ref=pip_ref)
        REC["chosen"][j["id"]] = os.path.basename(cands[0][1]); save()
REC["phases"]["total_s"] = round(time.time() - t0, 1); save()
import glob
contact(sorted(glob.glob(os.path.join(OUT, "*_s*.png"))), os.path.join(OUT, "keys_candidates.jpg"))
contact([os.path.join(OUT, f) for f in REC["chosen"].values()], os.path.join(OUT, "keys_chosen.jpg"))
say("done", json.dumps(REC["chosen"]))
