# Heartwick v2 GPU job (Kaggle 2x T4): music library + anime hero shots. Free, open models only:
#   ACE-Step 1.5 (MIT)            4 instrumental moods x 2 takes -> /kaggle/working/music/<mood>_<k>.wav
#   Animagine XL 4.0 (OpenRAIL++) anime keyframes for the story's 2-3 most emotional beats -> keyframes/<beat>_s<seed>.png
#   LTX-Video 2B 0.9.5 (LTXV licence, free under $10M revenue) and Wan2.2-TI2V-5B (Apache-2.0) image-to-video
#   RIFE v4 (MIT) 2x interpolation, resampled to 30 fps -> hero/<beat>_<model>.mp4
# Prepended by tools/hero_music.py: JOB = {"moods": {...}, "beats": {...}, "models": [...]}
# Each stage runs in its own venv/subprocess with a timeout; caches are deleted after each stage
# (an earlier research run filled the container until it went read-only).
import json, os, shutil, subprocess, sys, textwrap, threading, time

W = "/kaggle/working"
for d in ("music", "keyframes", "hero", "logs", "raw"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
STATUS, LOCK, BIG = {}, threading.Lock(), threading.Lock()
KEYS_READY = threading.Event()
HF = "/tmp/hf"
os.makedirs(HF, exist_ok=True)


def sh(cmd, log, timeout=3600, env=None):
    t = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        try:
            rc = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            f.write("TIMEOUT\n"); rc = -9
    return rc, round(time.time() - t, 1)


def stage(name, gpu, setup, code, timeout=3600, big=False, venv=None, cleanup=()):
    lock = BIG if big else threading.Lock()
    with lock:
        log = f"{W}/logs/{name}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=HF, PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0")
        rec = {"gpu": gpu}
        t0 = time.time()
        if setup:
            rc, dt = sh(setup, log, timeout=2400, env=env)
            rec["setup_s"] = dt
            if rc:
                rec.update(ok=False, err=f"setup rc={rc}")
                with LOCK: STATUS[name] = rec
                json.dump(STATUS, open(f"{W}/logs/status.json", "w"), indent=1)
                return False
        src = f"/tmp/stage_{name}.py"
        open(src, "w").write("JOB = " + repr(JOB) + "\nW = " + repr(W) + "\n" + textwrap.dedent(code))  # noqa: F821
        rc, dt = sh(f"cd /tmp && {venv}/bin/python {src}", log, timeout=timeout, env=env)
        rec.update(ok=rc == 0, run_s=dt, total_s=round(time.time() - t0, 1))
        if rc:
            rec["err"] = f"run rc={rc}"
        for c in cleanup:
            sh(f"rm -rf {c}", log, timeout=600)
        sh("uv cache clean -q || true; rm -rf ~/.cache/pip; df -h / /tmp | tail -2; free -g | head -2", log, timeout=600)
        with LOCK: STATUS[name] = rec
        json.dump(STATUS, open(f"{W}/logs/status.json", "w"), indent=1)
        return rc == 0


DIFF = "/tmp/v_diff"
SETUP_DIFF = (f"[ -x {DIFF}/bin/python ] || (pip install -q uv && uv venv -q -p 3.11 {DIFF} && uv pip install -q -p {DIFF}/bin/python "
              "torch torchvision diffusers transformers accelerate sentencepiece imageio imageio-ffmpeg ftfy pillow safetensors protobuf)")

# ---------------------------------------------------------------- music (ACE-Step 1.5)
ACE = "/tmp/ace"
SETUP_ACE = (f"pip install -q uv && rm -rf {ACE} && git clone -q --depth 1 https://github.com/ace-step/ACE-Step-1.5 {ACE} && cd {ACE} && "
             "uv sync -q --python 3.11")
MUSIC = r'''
import os, sys, time, json, glob, shutil
os.chdir("/tmp/ace"); sys.path.insert(0, "/tmp/ace")
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
dit = AceStepHandler()
dit.initialize_service(project_root="/tmp/ace", config_path="acestep-v15-turbo", device="cuda")
llm = LLMHandler()
timings = {}
for mood, spec in JOB["moods"].items():
    t = time.time()
    p = GenerationParams(caption=spec["caption"], lyrics="[Instrumental]", instrumental=True, bpm=spec.get("bpm"),
                         duration=spec.get("duration", 90), inference_steps=8, thinking=False)
    c = GenerationConfig(batch_size=2, audio_format="wav")
    out = f"/tmp/music_{mood}"
    os.makedirs(out, exist_ok=True)
    try:
        r = generate_music(dit, llm, p, c, save_dir=out)
        ok = r.success
        err = None if ok else str(r.error)[:300]
    except Exception as e:
        ok, err = False, repr(e)[:300]
    if not ok and not getattr(llm, "llm_initialized", False):  # retry with the 1.7B planner LM if DiT-only is refused
        try:
            llm.initialize(checkpoint_dir="/tmp/ace/checkpoints", lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda")
            r = generate_music(dit, llm, p, c, save_dir=out)
            ok, err = r.success, None if r.success else str(r.error)[:300]
        except Exception as e:
            err = repr(e)[:300]
    files = sorted(glob.glob(out + "/**/*.wav", recursive=True) + glob.glob(out + "/**/*.flac", recursive=True))
    for k, f in enumerate(files[:2]):
        shutil.copy(f, f"{W}/music/{mood}_{k + 1}{os.path.splitext(f)[1]}")
    timings[mood] = {"ok": ok, "files": len(files), "s": round(time.time() - t, 1), "err": err}
    print(mood, timings[mood], flush=True)
json.dump(timings, open(f"{W}/music/timings.json", "w"), indent=1)
'''

# ---------------------------------------------------------------- keyframes (Animagine XL 4.0)
KEYS = r'''
import torch, time, json
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
pipe = StableDiffusionXLPipeline.from_pretrained("cagliostrolab/animagine-xl-4.0", torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda"); pipe.vae.enable_tiling()
Q = ", masterpiece, high score, great score, absurdres"
NEG = ("lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, "
       "low quality, low score, bad score, average score, signature, watermark, username, blurry, nsfw, candle, fire, flame")
tim = {}
for beat, b in JOB["beats"].items():
    for seed in b.get("seeds", [11, 23]):
        t = time.time()
        img = pipe(b["prompt"] + Q, negative_prompt=NEG, width=832, height=1216, num_inference_steps=28, guidance_scale=5.0,
                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        img.save(f"{W}/keyframes/{beat}_s{seed}.png")
        tim[f"{beat}_s{seed}"] = round(time.time() - t, 1)
        print(beat, seed, tim[f"{beat}_s{seed}"], flush=True)
json.dump(tim, open(f"{W}/keyframes/timings.json", "w"), indent=1)
'''

# ---------------------------------------------------------------- image to video
LTX = r'''
import torch, time, json, os
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling()
neg = "worst quality, inconsistent motion, blurry, jittery, distorted, morphing, extra limbs, deformed face, text, watermark"
tim = {}
for beat, b in JOB["beats"].items():
    img = load_image(f"{W}/keyframes/{beat}_s{b.get('seeds', [11])[0]}.png").resize((512, 768))
    t = time.time()
    try:
        fr = pipe(image=img, prompt=b["motion"], negative_prompt=neg, width=512, height=768, num_frames=73, num_inference_steps=30,
                  guidance_scale=3.0, generator=torch.Generator("cpu").manual_seed(7)).frames[0]
        export_to_video(fr, f"{W}/raw/{beat}_ltx.mp4", fps=24)
        tim[beat] = {"s": round(time.time() - t, 1), "frames": len(fr), "res": "512x768"}
    except Exception as e:
        tim[beat] = {"err": repr(e)[:300]}
    print("ltx", beat, tim[beat], flush=True)
json.dump(tim, open(f"{W}/raw/ltx_timings.json", "w"), indent=1)
'''

WAN = r'''
import torch, time, json
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
mid = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
neg = "photorealistic, live action, morphing, distorted face, extra limbs, flicker, text, watermark, static image, blurry"
tim = {}
for beat, b in JOB["beats"].items():
    img = load_image(f"{W}/keyframes/{beat}_s{b.get('seeds', [11])[0]}.png").resize((480, 704))
    t = time.time()
    try:
        fr = pipe(image=img, prompt=b["motion"], negative_prompt=neg, height=704, width=480, num_frames=49, num_inference_steps=20,
                  guidance_scale=5.0, generator=torch.Generator("cpu").manual_seed(7)).frames[0]
        export_to_video(fr, f"{W}/raw/{beat}_wan.mp4", fps=24)
        tim[beat] = {"s": round(time.time() - t, 1), "frames": len(fr), "res": "480x704"}
    except Exception as e:
        tim[beat] = {"err": repr(e)[:300]}
    print("wan", beat, tim[beat], flush=True)
json.dump(tim, open(f"{W}/raw/wan_timings.json", "w"), indent=1)
'''

RIFE = r'''
import glob, json, os, subprocess, sys, time
import numpy as np, torch, imageio
from huggingface_hub import snapshot_download
d = snapshot_download("TensorForger/RIFE-safetensors")
sys.path.insert(0, d)
from interpolation_model import IFNet
from safetensors.torch import load_file
m = IFNet(); m.load_state_dict(load_file(os.path.join(d, "flownet.safetensors"))); m.to("cuda").eval()
tim = {}
for f in sorted(glob.glob(f"{W}/raw/*.mp4")):
    t = time.time()
    r = imageio.get_reader(f); fps = r.get_meta_data().get("fps", 24)
    fr = [np.asarray(x) for x in r]
    H, Wd = fr[0].shape[:2]; ph, pw = (32 - H % 32) % 32, (32 - Wd % 32) % 32
    def T(a): return torch.from_numpy(np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="edge")).permute(2, 0, 1).float().div(255)[None].cuda()
    dbl = []
    with torch.no_grad():
        for a, b in zip(fr, fr[1:]):
            mid = m(torch.cat([T(a), T(b)], 1))[0, :, :H, :Wd].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            dbl += [a, mid]
    dbl.append(fr[-1])
    src_fps = fps * 2
    n_out = int(len(dbl) / src_fps * 30)
    out = [dbl[min(len(dbl) - 1, round(i * src_fps / 30))] for i in range(n_out)]
    name = os.path.basename(f)
    imageio.mimwrite(f"{W}/hero/{name}", out, fps=30, quality=9, macro_block_size=16)
    tim[name] = {"in_frames": len(fr), "in_fps": fps, "out_frames": len(out), "s": round(time.time() - t, 1)}
    print(name, tim[name], flush=True)
json.dump(tim, open(f"{W}/hero/rife_timings.json", "w"), indent=1)
'''


# ---------------------------------------------------------------- V1-style rig-ready painted characters
V1CHARS = r'''
import base64, io, json, os, sys, time
import numpy as np, torch
from PIL import Image
os.makedirs("/tmp/e2/engine2", exist_ok=True)
for name, src in JOB["engine_files"].items():
    open(f"/tmp/e2/engine2/{name}", "w").write(src)
sys.path.insert(0, "/tmp/e2")
from engine2 import autorig as AR
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
OUT = f"{W}/v1chars"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, image_encoder=enc,
                                                 torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
pipe.vae.enable_tiling(); pipe.set_progress_bar_config(disable=True)
S = JOB["v1"]
tim = {}
for name, c in S["chars"].items():
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    ref = Image.open(io.BytesIO(base64.b64decode(c["ref_b64"]))).convert("RGB")
    pipe.set_ip_adapter_scale(c.get("ip_scale", 0.55))
    for k, seed in enumerate(c["seeds"]):
        t = time.time()
        img = pipe(prompt=c["prompt"], negative_prompt=S["neg"], width=c.get("w", 832), height=c.get("h", 1216), num_inference_steps=30,
                   guidance_scale=6.5, ip_adapter_image=ref, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        img.save(f"{OUT}/{name}/cand_{k}.png")
        tim[f"{name}_{k}"] = round(time.time() - t, 1)
        print(name, k, tim[f"{name}_{k}"], flush=True)
del pipe; torch.cuda.empty_cache()
# cut-outs
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
seg = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to("cuda").eval().half()
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def cut(p):
    im = Image.open(p).convert("RGB")
    with torch.no_grad():
        m = seg(tf(im).unsqueeze(0).to("cuda").half())[-1].sigmoid().float().cpu()[0, 0].numpy()
    a = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(im.size, Image.LANCZOS))
    return np.dstack([np.asarray(im), a])
cuts = {}
for name in S["chars"]:
    for f in sorted(os.listdir(f"{OUT}/{name}")):
        if f.startswith("cand_"):
            rgba = cut(f"{OUT}/{name}/{f}")
            Image.fromarray(rgba, "RGBA").save(f"{OUT}/{name}/{f[:-4]}_cut.png")
            cuts[(name, f)] = rgba
del seg; torch.cuda.empty_cache()
# joints + pick the best candidate per human (confident joints, wrists clear of the hips) + auto-rig
from rtmlib import Body
body = Body(mode="balanced", backend="onnxruntime", device="cpu")
try:
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()
    def inpaint(rgb, hole):
        o = lama(Image.fromarray(rgb), Image.fromarray((hole * 255).astype(np.uint8)))
        return np.asarray(o.convert("RGB").resize((rgb.shape[1], rgb.shape[0])))
except Exception as e:
    print("LaMa unavailable:", repr(e)[:200], flush=True); inpaint = None
report = {}
for name, c in S["chars"].items():
    if c.get("kind") != "humanoid":
        continue
    best = None
    for (n, f), rgba in cuts.items():
        if n != name:
            continue
        rgb = np.asarray(Image.open(f"{OUT}/{name}/{f}").convert("RGB"))
        kps, sc = body(rgb[..., ::-1])
        if len(kps) == 0:
            continue
        i = int(np.argmax(sc.mean(1))); k, s = kps[i], sc[i]
        kp = {nm: (float(k[j][0]), float(k[j][1])) for j, nm in enumerate(AR.COCO)}
        clear = min(abs(kp["l_wrist"][0] - kp["l_hip"][0]), abs(kp["r_wrist"][0] - kp["r_hip"][0])) / max(1, rgb.shape[1])
        score = float(s.mean()) + 2.0 * min(clear, 0.12)
        if best is None or score > best[0]:
            best = (score, f, kp, rgba)
    if best is None:
        report[name] = "no person found"; continue
    score, f, kp, rgba = best
    json.dump(kp, open(f"{OUT}/{name}/keypoints.json", "w"))
    Image.fromarray(rgba, "RGBA").save(f"{OUT}/{name}/{name}.png")
    J = AR.autorig(rgba, kp, f"{OUT}/{name}/rig", name=name, inpaint=inpaint)
    report[name] = {"picked": f, "score": round(score, 3), "parts": len(J["parts"])}
    print(name, report[name], flush=True)
json.dump({"gen_s": tim, "rig": report}, open(f"{OUT}/report.json", "w"), indent=1)
'''
SETUP_V1 = SETUP_DIFF + f" && uv pip install -q -p {DIFF}/bin/python timm kornia einops rtmlib onnxruntime simple-lama-inpainting"


def track0():
    if JOB["moods"]:  # noqa: F821
        stage("music_acestep", 0, SETUP_ACE, MUSIC, timeout=3600, venv=f"{ACE}/.venv", cleanup=(ACE, f"{HF}/hub"))
    if not JOB["beats"]:  # noqa: F821  (music-only run)
        return
    KEYS_READY.wait(timeout=7200)
    if "ltx" in JOB["models"]:  # noqa: F821
        stage("i2v_ltx", 0, SETUP_DIFF, LTX, timeout=4800, big=True, venv=DIFF, cleanup=(f"{HF}/hub/models--Lightricks--LTX-Video-0.9.5",))


def track1():
    if JOB.get("v1"):  # noqa: F821
        stage("v1_characters", 1, SETUP_V1, V1CHARS, timeout=5400, venv=DIFF,
              cleanup=(f"{HF}/hub/models--stabilityai--stable-diffusion-xl-base-1.0",))
    if not JOB["beats"]:  # noqa: F821
        return
    stage("keyframes_animagine", 1, SETUP_DIFF, KEYS, timeout=3600, venv=DIFF, cleanup=(f"{HF}/hub/models--cagliostrolab--animagine-xl-4.0",))
    KEYS_READY.set()
    if "wan" in JOB["models"]:  # noqa: F821
        stage("i2v_wan", 1, SETUP_DIFF, WAN, timeout=7200, big=True, venv=DIFF, cleanup=(f"{HF}/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers",))


if __name__ == "__main__":
    t0 = time.time()
    sh("nvidia-smi; df -h / /tmp /kaggle/working; free -g; apt-get -qq update > /dev/null; apt-get -qq install -y git-lfs > /dev/null",
       f"{W}/logs/setup.log", timeout=900)
    a, b = threading.Thread(target=track0), threading.Thread(target=track1)
    a.start(); b.start(); a.join(); b.join()
    if JOB["beats"]:  # noqa: F821
        stage("rife", 0, SETUP_DIFF, RIFE, timeout=1800, venv=DIFF)
    STATUS["_wall_s"] = round(time.time() - t0, 1)
    json.dump(STATUS, open(f"{W}/logs/status.json", "w"), indent=1)
    print(json.dumps(STATUS, indent=1))
