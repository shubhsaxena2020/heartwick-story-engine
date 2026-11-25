# Heartwick round 3, keyframe axis (Kaggle T4 x2, this process owns BOTH GPUs for ~30 min).
# Builds the END key of the beat (Nell kneeling on ONE knee, Pip held at her chest, smiling) three ways, plus the
# jewel-tone style pair, so the Wan FLF2V clips can be compared on the keyframe method alone:
#   ZI  Z-Image-Turbo alone (text only, Apache-2.0). T4 has no bf16: DiT in fp16 with an overflow-safe FFN
#       (the FFN output feeds an RMSNorm, so it is computed scaled down by 1/32: same result, no inf), text encoder fp32
#       on the CPU; fallback = fp32 DiT sharded over both T4s.
#   QE  Qwen-Image-Edit-2511 (Apache-2.0) + Lightning 4-step LoRA (Apache-2.0), CHAINED edits K0 -> kneel-reach -> hold
#   QP  the same chain with a hand-blocked OpenPose skeleton as image 2 (kneel_reach, then kneel_hold)
#   QD  one direct edit K0 -> hold with the kneel_hold skeleton (chain vs single edit)
#   JT  jewel-tone restyle of K0 and of the chosen QP end key (same restyle prompt)
# Qwen setup on T4 (round 2 died on cuBLASLt): DISABLE_ADDMM_CUDA_LT=1; text encoder NF4 (fp32 compute, fp32 norms and
# embeddings, vision tower fp16, lm_head replaced by Identity) on cuda:1; GGUF Q4_K_M transformer sharded 36/24 blocks
# over cuda:0/cuda:1; VAE on cuda:0; prompt embeddings computed separately and passed in.
# QA (OWLv2 + DINOv2) picks the best seed per key. usage: python keys3.py jobs.json outdir
import os
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
import gc, json, math, sys, time, traceback
import numpy as np, torch
from PIL import Image
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

J = json.load(open(sys.argv[1])); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
REC = {"jobs": {}, "phases": {}, "errors": []}
D0, D1 = torch.device("cuda:0"), torch.device("cuda:1")


def save():
    json.dump(REC, open(os.path.join(OUT, "keys_rec.json"), "w"), indent=1)


def err(tag, e):
    REC["errors"].append({"tag": tag, "err": repr(e)[:600], "tb": traceback.format_exc()[-1500:]}); save()
    print("ERROR", tag, repr(e)[:300], flush=True)


def peak():
    return {f"gpu{i}": round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())}


def _mv(x, d):
    if torch.is_tensor(x):
        return x.to(d)
    if isinstance(x, (list, tuple)):
        return type(x)(_mv(y, d) for y in x)
    if isinstance(x, dict):
        return {k: _mv(v, d) for k, v in x.items()}
    return x


def shard(m, attr, n0):
    """blocks[:n0] on cuda:0, the rest on cuda:1, everything else on cuda:0; hooks move activations."""
    blocks = getattr(m, attr)
    for name, ch in m.named_children():
        if name != attr:
            ch.to(D0)
    for _, p in m.named_parameters(recurse=False):
        p.data = p.data.to(D0)
    for i, b in enumerate(blocks):
        d = D0 if i < n0 else D1
        b.to(d)
        b.register_forward_pre_hook(lambda mod, args, kw, d=d: (_mv(args, d), _mv(kw, d)), with_kwargs=True)
    for name, ch in m.named_children():
        if name != attr:
            ch.register_forward_pre_hook(lambda mod, args, kw: (_mv(args, D0), _mv(kw, D0)), with_kwargs=True)
    return m


def ok_img(im):
    a = np.asarray(im).astype(np.float32) / 255
    return bool(np.isfinite(a).all() and a.std() > 0.04)


# ---- QA (CPU while models are on the GPUs) ----------------------------------------------------------------------------
os.environ.setdefault("QA_DEVICE", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_consistency as qa
bank_n = qa.ref_bank([J["k0"]], "nell"); bank_p = qa.ref_bank([J["k0"]], "pip")
k0_rows = qa.analyse(J["k0"], bank_n, bank_p, 1)
PIP0 = k0_rows[0].get("pip_scale") or 1.28
REC["phases"]["k0_pip_scale"] = PIP0


def score(path, want_pip=True):
    r = qa.analyse(path, bank_n, bank_p, 1)[0]
    d = {k: v for k, v in r.items() if not k.startswith("_")}
    s = d.get("body_sim", 0) + d.get("head_sim", 0)
    if d.get("cut"): s -= 0.3 * len(d["cut"])
    hr = d.get("headroom")
    if hr is not None and hr < 0.04: s -= 0.5
    if want_pip:
        if "pip_scale" in d: s -= 2.0 * abs(math.log(d["pip_scale"] / PIP0))
        else: s -= 0.4
    d["score"] = round(s, 3); return d


def pick(jid, cands):
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    REC["jobs"][jid]["chosen"] = os.path.basename(cands[0][1]); save()
    return cands[0][1]


CHOSEN = {}

# ==== 1. Z-Image-Turbo ===================================================================================================
def zimage():
    from diffusers import ZImagePipeline
    import diffusers.models.transformers.transformer_z_image as ZT
    from transformers import AutoModel
    repo = J.get("zimage_repo") or "Tongyi-MAI/Z-Image-Turbo"
    t = time.time()
    # text encoder alone first (fp32 on the CPU, 16 GB host RAM), freed before the DiT loads: never both in RAM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer")
    te = AutoModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.float32).eval()
    embs = {}
    with torch.no_grad():
        for j in J["zimage"]:   # = ZImagePipeline._encode_prompt
            txt = tok.apply_chat_template([{"role": "user", "content": j["prompt"]}], tokenize=False, add_generation_prompt=True,
                                          enable_thinking=True)
            ti = tok([txt], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
            m = ti.attention_mask.bool()
            hs = te(input_ids=ti.input_ids, attention_mask=m, output_hidden_states=True).hidden_states[-2]
            e = [hs[0][m[0]].float()]
            embs[j["id"]] = e
            REC["phases"].setdefault("zi_emb_finite", []).append(bool(torch.isfinite(e[0]).all()))
    del te; gc.collect()
    pipe = ZImagePipeline.from_pretrained(repo, text_encoder=None, torch_dtype=torch.float16)
    ZImagePipeline._execution_device = property(lambda self: D0)
    REC["phases"]["zi_te_s"] = round(time.time() - t, 1); save()

    # overflow-safe fp16 FFN (scale-invariant: the output goes through ffn_norm2, an RMSNorm)
    def ffn_fwd(self, x):
        h = (torch.nn.functional.silu(self.w1(x).float()) * self.w3(x).float() / 32.0).clamp(-6e4, 6e4).to(x.dtype)
        return self.w2(h)
    ZT.FeedForward.forward = ffn_fwd
    pipe.vae.to(D0, torch.float32)
    modes = [("fp16", None), ("fp32_split", None)]
    for mode, _ in modes:
        try:
            if mode == "fp16":
                pipe.transformer.to(D0, torch.float16)
            else:
                pipe.transformer.to(torch.float32)
                shard(pipe.transformer, "layers", len(pipe.transformer.layers) // 2)
            dt = torch.float16 if mode == "fp16" else torch.float32
            good_all = True
            for j in J["zimage"]:
                cands = []
                REC["jobs"].setdefault(j["id"], {"method": "zimage", "mode": mode, "cands": []})
                for seed in j["seeds"]:
                    t = time.time()
                    with torch.no_grad():
                        im = pipe(prompt_embeds=[x.to(D0, dt) for x in embs[j["id"]]], height=j["H"], width=j["W"],
                                  num_inference_steps=9, guidance_scale=0.0, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
                    if not ok_img(im):
                        good_all = False; REC["jobs"][j["id"]]["cands"].append({"seed": seed, "bad": True, "mode": mode}); save()
                        break
                    p = os.path.join(OUT, f"{j['id']}_s{seed}.png"); im.save(p)
                    d = score(p); d.update(seed=seed, gen_s=round(time.time() - t, 1), mode=mode)
                    REC["jobs"][j["id"]]["cands"].append(d); cands.append((d["score"], p, d)); save()
                if not good_all:
                    break
                CHOSEN[j["id"]] = pick(j["id"], cands)
            REC["phases"]["zi_mode"] = mode; REC["phases"]["zi_peak"] = peak(); save()
            if good_all:
                break
        except Exception as e:
            err(f"zimage_{mode}", e)
    del pipe; gc.collect(); torch.cuda.empty_cache()


# ==== 2. Qwen-Image-Edit-2511 =============================================================================================
def qwen():
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as QE
    QE.VAE_IMAGE_SIZE = 768 * 768      # reference latents at ~0.6 MP: fewer tokens on a T4
    from diffusers import (QwenImageEditPlusPipeline, QwenImageTransformer2DModel, GGUFQuantizationConfig,
                           FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage)
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    REPO = J.get("qwen_repo") or "Qwen/Qwen-Image-Edit-2511"   # local snapshot without the 40 GB bf16 transformer
    t = time.time()
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32,
                           llm_int8_skip_modules=["visual", "lm_head"])
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=torch.float32,
                                                            quantization_config=q, device_map={"": 1})
    te.lm_head = torch.nn.Identity()
    vis = te.model.visual if hasattr(te.model, "visual") else te.visual
    vis.half()
    for n, b in vis.named_buffers():   # keep the vision RoPE frequencies in fp32
        if "inv_freq" in n:
            b.data = b.data.float()
    gc.collect(); torch.cuda.empty_cache()
    REC["phases"]["qe_te_load_s"] = round(time.time() - t, 1); REC["phases"]["qe_te_peak"] = peak(); save()

    sched = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192, "max_shift": math.log(3),
        "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False, "time_shift_type": "exponential",
        "use_beta_sigmas": False, "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False})
    t = time.time()
    tr = QwenImageTransformer2DModel.from_single_file(J["qwen_gguf"], quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
                                                      config=REPO, subfolder="transformer", torch_dtype=torch.float16)
    vae = AutoencoderKLQwenImage.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).to(D0)
    vae.enable_tiling()
    pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, transformer=tr, text_encoder=te, vae=vae, scheduler=sched, torch_dtype=torch.float16)
    lora = "none"
    try:
        pipe.load_lora_weights("lightx2v/Qwen-Image-Edit-2511-Lightning", weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
        lora = "lightning-4step"
    except Exception as e:
        err("lora", e)
    shard(pipe.transformer, "transformer_blocks", J.get("qwen_split", 36))
    gc.collect(); torch.cuda.empty_cache()
    QwenImageEditPlusPipeline._execution_device = property(lambda self: D0)
    REC["phases"].update(qe_build_s=round(time.time() - t, 1), qe_lora=lora, qe_build_peak=peak()); save()

    def cond_images(paths):
        ims = [Image.open(p).convert("RGB") for p in paths]
        out = []
        for im in ims:   # exactly what __call__ does before encode_prompt
            w, h = QE.calculate_dimensions(QE.CONDITION_IMAGE_SIZE, im.width / im.height)
            out.append(pipe.image_processor.resize(im, h, w))
        return ims, out

    def edit(refs, prompt, W, H, seed):
        ims, conds = cond_images(refs)
        with torch.no_grad():
            pe, pm = pipe.encode_prompt(prompt=prompt, image=conds, device=D1)
        fin = bool(torch.isfinite(pe).all())
        pe = pe.to(D0, torch.float16); pm = pm.to(D0) if pm is not None else None
        steps, cfg = (4, 1.0) if lora != "none" else (20, 4.0)
        kw = dict(image=ims, prompt_embeds=pe, prompt_embeds_mask=pm, height=H, width=W, num_inference_steps=steps,
                  true_cfg_scale=cfg, generator=torch.Generator("cpu").manual_seed(seed))
        if cfg > 1:
            npe, npm = pipe.encode_prompt(prompt=" ", image=conds, device=D1)
            kw.update(negative_prompt_embeds=npe.to(D0, torch.float16), negative_prompt_embeds_mask=npm.to(D0) if npm is not None else None)
        with torch.no_grad():
            return pipe(**kw).images[0], fin

    def run_layer(jobs):
        for j in jobs:
            refs = [CHOSEN.get(r[1:], None) if r.startswith("@") else r for r in j["refs"]]
            if any(r is None for r in refs):
                REC["jobs"][j["id"]] = {"skipped": "missing ref"}; save(); continue
            REC["jobs"][j["id"]] = {"method": j["method"], "refs": [os.path.basename(r) for r in refs], "cands": []}
            cands = []
            for seed in j["seeds"]:
                t = time.time()
                try:
                    im, fin = edit(refs, j["prompt"], j["W"], j["H"], seed)
                except Exception as e:
                    err(f"{j['id']}_s{seed}", e); torch.cuda.empty_cache(); continue
                p = os.path.join(OUT, f"{j['id']}_s{seed}.png"); im.save(p)
                d = score(p, j.get("pip", True)) if ok_img(im) else {"score": -99, "bad": True}
                d.update(seed=seed, gen_s=round(time.time() - t, 1), emb_finite=fin)
                REC["jobs"][j["id"]]["cands"].append(d); cands.append((d["score"], p, d)); save()
                print(j["id"], seed, d.get("score"), d["gen_s"], flush=True)
            CHOSEN[j["id"]] = pick(j["id"], [c for c in cands if not c[2].get("bad")])
        REC["phases"].setdefault("qe_peaks", []).append(peak()); save()

    for layer in J["qwen_layers"]:
        run_layer(layer)


t0 = time.time()
try:
    zimage()
except Exception as e:
    err("zimage", e)
REC["phases"]["zimage_wall_s"] = round(time.time() - t0, 1); save()
gc.collect(); torch.cuda.empty_cache()
t1 = time.time()
try:
    qwen()
except Exception as e:
    err("qwen", e)
REC["phases"]["qwen_wall_s"] = round(time.time() - t1, 1)
REC["chosen"] = {k: (os.path.basename(v) if v else None) for k, v in CHOSEN.items()}
REC["phases"]["total_s"] = round(time.time() - t0, 1); save()
json.dump({k: v for k, v in CHOSEN.items() if v}, open(os.path.join(OUT, "chosen.json"), "w"), indent=1)
print(json.dumps(REC["chosen"], indent=1))
