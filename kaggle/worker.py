# Heartwick GPU worker (runs on Kaggle, T4 x2). Generic: everything it does is described by job.json.
# Stages: (1) Kokoro TTS with word timestamps, (2) SDXL images (backgrounds, character masters, IP-Adapter poses)
# on both GPUs in parallel, (3) BiRefNet cut-outs, (4) Depth-Anything-V2 depth maps, (5) optional render via engine.
# The launcher prepends:  JOB_ZIP_B64 = "<base64 zip of job.json + engine/ + brand/ + refs/>"
import base64, io, json, os, shutil, subprocess, sys, time, traceback, zipfile

T0 = time.time()
W = "/kaggle/working"
WORK = "/kaggle/tmp_hw"
os.makedirs(WORK, exist_ok=True)
LOG = {"stages": {}, "errors": []}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def stage(name):
    class S:
        def __enter__(s):
            s.t = time.time(); log("== stage", name)
        def __exit__(s, et, ev, tb):
            LOG["stages"][name] = round(time.time() - s.t, 1)
            if et:
                LOG["errors"].append(f"{name}: {et.__name__}: {ev}")
                traceback.print_exception(et, ev, tb)
            json.dump(LOG, open(f"{W}/worker_log.json", "w"), indent=1)
            return True  # never kill the whole run
    return S()


def sh(cmd):
    log("$", cmd[:160])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        log("rc", r.returncode, r.stdout[-1500:], r.stderr[-2500:])
    return r.returncode


# ---------------------------------------------------------------- unpack job
zipfile.ZipFile(io.BytesIO(base64.b64decode(JOB_ZIP_B64))).extractall(WORK)  # noqa: F821
sys.path.insert(0, WORK)
JOB = json.load(open(f"{WORK}/job.json"))
os.makedirs(f"{WORK}/brand/fonts", exist_ok=True)
for name, url in (("Nunito-VF.ttf", "ofl/nunito/Nunito%5Bwght%5D.ttf"), ("Fraunces-VF.ttf", "ofl/fraunces/Fraunces%5BSOFT,WONK,opsz,wght%5D.ttf")):
    if not os.path.exists(f"{WORK}/brand/fonts/{name}"):  # OFL fonts, fetched instead of shipped (script size limit)
        subprocess.run(["curl", "-sSL", "-o", f"{WORK}/brand/fonts/{name}", f"https://github.com/google/fonts/raw/main/{url}"])
OUT = f"{W}/assets"
os.makedirs(OUT, exist_ok=True)
for root, _, files in os.walk("/kaggle/input"):  # cached assets from the heartwick-cache dataset
    for f in files:
        if f.startswith(("bg_", "master_", "pose_", "u_")) and not os.path.exists(f"{OUT}/{f}"):
            shutil.copy(os.path.join(root, f), f"{OUT}/{f}")
log("cached assets available:", len(os.listdir(OUT)))
for f in os.listdir(f"{WORK}/refs") if os.path.isdir(f"{WORK}/refs") else []:
    shutil.copy(f"{WORK}/refs/{f}", f"{OUT}/{f}")
log("job:", {k: (len(v) if isinstance(v, list) else v) for k, v in JOB.items() if k != "tts"})

# ---------------------------------------------------------------- installs
with stage("install"):
  if JOB.get("images") or (JOB.get("tts") or {}).get("utts") or (JOB.get("tts") or {}).get("samples"):
    sh("apt-get -qq -y install espeak-ng > /dev/null 2>&1 || true")
    # Kokoro/misaki wheels pin python<3.13; Kaggle is on 3.13 now, the code itself runs fine
    sh(f"{sys.executable} -m pip install -q --ignore-requires-python 'kokoro>=0.9.4' 'misaki[en]>=0.9.4' soundfile einops kornia timm")
    if sh(f"{sys.executable} -c 'import kokoro, misaki'"):
        sh(f"{sys.executable} -m pip install -q --ignore-requires-python --no-deps kokoro misaki")
        sh(f"{sys.executable} -m pip install -q loguru num2words spacy phonemizer-fork espeakng-loader addict regex")
    sh(f"{sys.executable} -c 'import kokoro; print(kokoro.__file__)'")
    sh(f"{sys.executable} -m pip install -q -U diffusers accelerate peft")
    import torch
    log("torch", torch.__version__, "gpus", torch.cuda.device_count(),
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# ---------------------------------------------------------------- TTS
if JOB.get("tts") and (JOB["tts"].get("utts") or JOB["tts"].get("samples")):
    with stage("tts"):
        import numpy as np, soundfile as sf, torch
        from kokoro import KPipeline
        tts = JOB["tts"]
        pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device="cuda")
        os.makedirs(f"{W}/samples", exist_ok=True)

        def say(text, voice, speed):
            chunks, words, off = [], [], 0.0
            for r in pipe(text, voice=voice, speed=speed, split_pattern=r"\n+"):
                a = r.audio.cpu().numpy() if hasattr(r.audio, "cpu") else np.asarray(r.audio)
                for t in (r.tokens or []):
                    if t.start_ts is None:
                        continue
                    words.append({"w": t.text, "ws": t.whitespace, "s": round(off + t.start_ts, 3),
                                  "e": round(off + (t.end_ts or t.start_ts), 3)})
                chunks.append(a)
                off += len(a) / 24000
            return (np.concatenate(chunks) if chunks else np.zeros(1, np.float32)), words

        for smp in tts.get("samples", []):
            a, _ = say(smp["text"], smp["voice"], smp.get("speed", 1.0))
            sf.write(f"{W}/samples/sample_{smp['id']}.wav", a, 24000)
            log("sample", smp["id"], round(len(a) / 24000, 1), "s")
        tot = 0.0
        for u in tts.get("utts", []):
            a, words = say(u["text"], u.get("voice", tts["voice"]), u.get("speed", tts.get("speed", 1.0)))
            sf.write(f"{OUT}/{u['id']}.wav", a, 24000)
            json.dump({"id": u["id"], "text": u["text"], "dur": round(len(a) / 24000, 3), "words": words}, open(f"{OUT}/{u['id']}.json", "w"))
            tot += len(a) / 24000
        log("tts utts", len(tts.get("utts", [])), "total", round(tot, 1), "s")
        del pipe; torch.cuda.empty_cache()

# ---------------------------------------------------------------- images (one process per GPU)
GEN_SRC = r'''
import json, sys, os, time, torch
from PIL import Image
gpu, items_path, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
items = json.load(open(items_path))
if not items: sys.exit(0)
dev = f"cuda:{gpu}"
from diffusers import StableDiffusionXLPipeline, AutoPipelineForImage2Image, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
need_ip = any(i.get("ip_ref") for i in items)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
kw = {}
if need_ip:
    kw["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16,
                                                 variant="fp16", use_safetensors=True, **kw).to(dev)
if need_ip:
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
pipe.vae.enable_tiling()
pipe.set_progress_bar_config(disable=True)
i2i = AutoPipelineForImage2Image.from_pipe(pipe)
blank = Image.new("RGB", (224, 224), (128, 128, 128))
for it in items:
    t = time.time()
    try:
        for k in range(it.get("n", 1)):
            seed = it["seed"] + k * 1000
            g = torch.Generator(dev).manual_seed(seed)
            extra = {}
            if need_ip:
                ref = it.get("ip_ref")
                sc = it.get("ip_scale", 0.6) if ref else 0.0
                if ref and it.get("ip_mode") == "style":  # InstantStyle: style blocks only, layout from the prompt
                    sc = {"up": {"block_0": [0.0, sc, 0.0]}}
                pipe.set_ip_adapter_scale(sc)
                extra["ip_adapter_image"] = Image.open(next(f"{out}/{ref}{e}" for e in (".png", ".jpg") if os.path.exists(f"{out}/{ref}{e}"))).convert("RGB") if ref else blank
            img = pipe(prompt=it["prompt"], negative_prompt=it["neg"], width=it["w"], height=it["h"],
                       num_inference_steps=it.get("steps", 30), guidance_scale=it.get("cfg", 6.5), generator=g, **extra).images[0]
            hf = it.get("hires", 0)
            if hf:
                big = img.resize((int(it["w"] * hf) // 8 * 8, int(it["h"] * hf) // 8 * 8), Image.LANCZOS)
                if need_ip:
                    extra["ip_adapter_image"] = extra["ip_adapter_image"]
                img = i2i(prompt=it["prompt"], negative_prompt=it["neg"], image=big, strength=it.get("hires_strength", 0.35),
                          num_inference_steps=24, guidance_scale=it.get("cfg", 6.5), generator=torch.Generator(dev).manual_seed(seed), **extra).images[0]
            name = it["id"] if it.get("n", 1) == 1 else f"{it['id']}_c{k}"
            img.save(f"{out}/{name}.png")
        print("GEN", gpu, it["id"], round(time.time() - t, 1), "s", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("FAIL", gpu, it["id"], repr(e)[:300], flush=True)
'''

def run_gen(items, tag):
    if not items:
        return
    open(f"{WORK}/gen.py", "w").write(GEN_SRC)
    ngpu = max(1, __import__("torch").cuda.device_count())
    # balance by pixel*steps cost
    shares = [[] for _ in range(ngpu)]
    load = [0.0] * ngpu
    for it in sorted(items, key=lambda i: -i["w"] * i["h"] * (1 + 2 * bool(i.get("hires"))) * i.get("n", 1)):
        j = load.index(min(load)); shares[j].append(it); load[j] += it["w"] * it["h"] * (1 + 2 * bool(it.get("hires"))) * it.get("n", 1)
    procs = []
    for g, sh_items in enumerate(shares):
        p = f"{WORK}/items_{tag}_{g}.json"; json.dump(sh_items, open(p, "w"))
        procs.append(subprocess.Popen([sys.executable, f"{WORK}/gen.py", str(g), p, OUT], stdout=open(f"{W}/gen_{tag}_{g}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    for g in range(ngpu):
        txt = open(f"{W}/gen_{tag}_{g}.log").read()
        log(f"gen {tag} gpu{g}:", " | ".join(l for l in txt.splitlines() if l.startswith(("GEN", "FAIL")))[-1500:])
        if "FAIL" in txt or "Traceback" in txt:
            log(txt[-3000:])


imgs = JOB.get("images", [])
with stage("images_phase1"):
    run_gen([i for i in imgs if i["kind"] != "pose"], "p1")   # plates + masters (style-anchored)
with stage("images_phase2"):
    run_gen([i for i in imgs if i["kind"] == "pose"], "p2")   # poses need the masters

# ---------------------------------------------------------------- cut-outs
cut = [i for i in imgs if i.get("cutout")]
if cut:
    with stage("cutout"):
        import torch, numpy as np
        from PIL import Image
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        seg = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to("cuda").eval().half()
        tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        for it in cut:
            names = [it["id"]] if it.get("n", 1) == 1 else [f"{it['id']}_c{k}" for k in range(it["n"])]
            for nm in names:
                p = f"{OUT}/{nm}.png"
                if not os.path.exists(p):
                    continue
                im = Image.open(p).convert("RGB")
                with torch.no_grad():
                    pred = seg(tf(im).unsqueeze(0).to("cuda").half())[-1].sigmoid().float().cpu()[0, 0]
                m = Image.fromarray((pred.numpy() * 255).astype(np.uint8)).resize(im.size, Image.LANCZOS)
                rgba = im.copy(); rgba.putalpha(m)
                bb = m.point(lambda v: 255 if v > 20 else 0).getbbox()
                if bb:
                    pad = 6
                    bb = (max(0, bb[0] - pad), max(0, bb[1] - pad), min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
                    rgba = rgba.crop(bb)
                rgba.save(f"{OUT}/{nm}_cut.png")
        log("cutouts", len(cut))
        del seg; torch.cuda.empty_cache()

# ---------------------------------------------------------------- depth
dep = [i for i in imgs if i.get("depth")]
if dep:
    with stage("depth"):
        import numpy as np
        from PIL import Image
        from transformers import pipeline
        de = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
        for it in dep:
            p = f"{OUT}/{it['id']}.png"
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            de(im)["depth"].resize(im.size, Image.BICUBIC).save(f"{OUT}/{it['id']}_depth.png")  # bright = near
        log("depth maps", len(dep))

# ---------------------------------------------------------------- render (engine code shipped in the zip)
if JOB.get("render"):
    with stage("render"):
        os.chdir(WORK)
        from engine import timeline, render
        tl = timeline.build(JOB["plan"], OUT, JOB["render"])
        json.dump(tl, open(f"{W}/timeline.json", "w"), indent=0)
        render.render_video(tl, assets_dir=OUT, tts_dir=OUT, out_path=f"{W}/{JOB['render']['name']}.mp4",
                            cfg=JOB["render"], workers=JOB["render"].get("workers", 4))

# ---------------------------------------------------------------- verify: Whisper listens to the final mix
if JOB.get("render") and os.path.exists(f"{W}/{JOB['render']['name']}_mix.wav"):
    with stage("verify_asr"):
        import torch
        from transformers import pipeline
        gpu_ok = torch.cuda.is_available()
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0 if gpu_ok else -1,
                       torch_dtype=torch.float16 if gpu_ok else torch.float32)
        r = asr(f"{W}/{JOB['render']['name']}_mix.wav", return_timestamps="word", chunk_length_s=30, batch_size=8)
        json.dump({"text": r["text"], "chunks": [{"w": c["text"], "t": c["timestamp"]} for c in r.get("chunks", [])]},
                  open(f"{W}/asr.json", "w"), indent=0)
        log("asr words", len(r.get("chunks", [])))

LOG["total_s"] = round(time.time() - T0, 1)
json.dump(LOG, open(f"{W}/worker_log.json", "w"), indent=1)
log("DONE", LOG)
