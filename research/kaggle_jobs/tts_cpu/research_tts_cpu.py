# CPU-ONLY variant of the GPU job (generated from gpu/research_gpu.py): TTS bake-off + Whisper on a Kaggle CPU kernel.
# Heartwick research GPU job (Kaggle, T4 x2). Two tracks in parallel:
#   track A (GPU0): TTS bake-off (one uv venv per model) + Whisper WER
#   track B (GPU1): art-direction stills (Z-Image-Turbo) + image-to-video feasibility (Wan2.2-TI2V-5B, LTX-Video 2B)
# Every stage runs in its own venv/subprocess with a timeout; a failing stage is logged and skipped.
import json, os, subprocess, sys, textwrap, threading, time

W = "/kaggle/working"
for d in ("voices", "art", "video", "logs"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
STATUS = {}
LOCK = threading.Lock()
BIG = threading.Lock()  # stages that need >8 GB host RAM never overlap (Kaggle has ~29 GB)

T = ("Every autumn, the old maple behind Nell's house turned the color of embers. "
     "That morning, tangled in its roots, she found something small and shivering, wrapped in a blanket of wet red leaves. "
     "\"Well, hello there, little one,\" she whispered, and for the first time in three years, the kitchen didn't feel so quiet.")
REF_TEXT = ("The wind came down from the hills that night, and the lanterns along the road flickered one by one. "
            "Nobody in the village could remember a storm quite like it.")
DESC = {
    "female": "A warm, gentle American woman in her late thirties telling a cozy bedtime story. Soft, clear and close to the microphone, unhurried, with tender emotion and a smile in her voice.",
    "male": "A warm, deep American man in his late forties telling a fireside story. Calm, rich and clear, close to the microphone, unhurried and kind, slightly gravelly.",
}
CONST = dict(T=T, REF_TEXT=REF_TEXT, DESC=DESC, W=W)


def sh(cmd, log, timeout=3600, env=None):
    t = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        try:
            r = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            f.write("TIMEOUT\n"); rc = -9
    return rc, round(time.time() - t, 1)


def stage(name, gpu, py, pip, code, timeout=1800, pre=None, workdir=None, runner=None, big=False, cleanup=()):
    if big:
        with BIG:
            return _stage(name, gpu, py, pip, code, timeout, pre, workdir, runner, cleanup)
    return _stage(name, gpu, py, pip, code, timeout, pre, workdir, runner, cleanup)


def _stage(name, gpu, py, pip, code, timeout, pre, workdir, runner, cleanup):
    log = f"{W}/logs/{name}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_ENABLE_HF_TRANSFER="0", PYTHONUNBUFFERED="1")
    rec = {"gpu": gpu}
    t0 = time.time()
    if pre:
        rc, dt = sh(pre, log, timeout=2400, env=env); rec["pre_s"] = dt
        if rc:
            rec.update(ok=False, err=f"pre rc={rc}")
            with LOCK: STATUS[name] = rec
            return False
    if runner is None:
        vpy = f"/tmp/v_{name}/bin/python"
        rc, dt = sh(f"uv venv -q -p {py} /tmp/v_{name} && uv pip install -q -p {vpy} {pip}", log, timeout=2400, env=env)
        rec["install_s"] = dt
        if rc:
            rec.update(ok=False, err=f"install rc={rc}")
            with LOCK: STATUS[name] = rec
            return False
        runner = vpy
    src = f"/tmp/stage_{name}.py"
    open(src, "w").write("CONST = " + repr(CONST) + "\nglobals().update(CONST)\n" + textwrap.dedent(code))
    rc, dt = sh(f"cd {workdir or '/tmp'} && {runner} {src}", log, timeout=timeout, env=env)
    rec.update(ok=(rc == 0), run_s=dt, total_s=round(time.time() - t0, 1))
    if rc:
        rec["err"] = f"run rc={rc}"
    for rid in cleanup:  # keep disk use bounded
        sh(f"rm -rf ~/.cache/huggingface/hub/models--{rid.replace('/', '--')}", log, timeout=300)
    sh("df -h /tmp | tail -1; free -g | head -2", log, timeout=60)
    with LOCK: STATUS[name] = rec
    json.dump(STATUS, open(f"{W}/logs/status.json", "w"), indent=1)
    return rc == 0


# ---------------------------------------------------------------- TTS stage code
COMMON_TTS = r'''
import time, json, os, numpy as np, soundfile as sf
OUT = W + "/voices"
def save(tag, wav, sr, t0, extra=None):
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    sf.write(f"{OUT}/{tag}.wav", wav, int(sr))
    gen = time.time() - t0
    rec = dict(tag=tag, sr=int(sr), dur=round(len(wav) / sr, 2), gen_s=round(gen, 1), rtf=round(gen / max(1e-3, len(wav) / sr), 3))
    if extra: rec.update(extra)
    open(f"{OUT}/timings.jsonl", "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
def ok(wav):
    w = np.asarray(wav, dtype=np.float32).squeeze()
    return w.size > 1000 and np.isfinite(w).all() and float(np.sqrt((w ** 2).mean())) > 1e-3
'''

KOKORO = COMMON_TTS + r'''
from kokoro import KPipeline
p = KPipeline(lang_code="a")
for tag, v in [("female", "af_heart"), ("male", "am_michael")]:
    t0 = time.time()
    wav = np.concatenate([np.asarray(a) for _, _, a in p(T, voice=v, speed=1.0)])
    save(f"kokoro_{tag}", wav, 24000, t0, {"voice": v})
    # reference clip for cloning models if Qwen voice design fails
    wav = np.concatenate([np.asarray(a) for _, _, a in p(REF_TEXT, voice=v, speed=1.0)])
    sf.write(f"{OUT}/kokororef_{tag}.wav", wav, 24000)
'''

QWEN = COMMON_TTS + r'''
import torch
from qwen_tts import Qwen3TTSModel
def load(dtype):
    return Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cpu", dtype=dtype, attn_implementation="sdpa")
dt = torch.float32
m = load(dt)
for tag in ("female", "male"):
    for what, text in (("narr", T), ("ref", REF_TEXT)):
        for attempt in range(2):
            t0 = time.time()
            wavs, sr = m.generate_voice_design(text=text, language="English", instruct=DESC[tag])
            if ok(wavs[0]): break
            print("bad output in", dt, "-> float32", flush=True); dt = torch.float32; del m; torch.cuda.empty_cache(); m = load(dt)
        if what == "narr":
            save(f"qwen3tts_{tag}", wavs[0], sr, t0, {"dtype": str(dt)})
        else:
            sf.write(f"{OUT}/ref_{tag}.wav", np.asarray(wavs[0], dtype=np.float32), sr)
'''

VOXCPM = COMMON_TTS + r'''
import torch, glob, shutil
from voxcpm import VoxCPM
def load():
    try:
        m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cpu")
        return m, "bf16-default"
    except Exception as e:
        print("default load failed:", repr(e)[:300], flush=True)
    from huggingface_hub import snapshot_download
    d = snapshot_download("openbmb/VoxCPM2", local_dir="/tmp/voxcpm2")
    c = json.load(open(d + "/config.json")); c["dtype"] = "float32"; json.dump(c, open(d + "/config.json", "w"))
    return VoxCPM.from_pretrained(d, load_denoiser=False, optimize=False, device="cpu"), "float32"
m, mode = load()
sr = m.tts_model.sample_rate
for tag in ("female", "male"):
    t0 = time.time()
    wav = m.generate(text=f"({DESC[tag]}){T}", cfg_value=2.0, inference_timesteps=10)
    if not ok(wav) and mode != "float32":
        print("bad output, retry float32", flush=True)
        del m; torch.cuda.empty_cache()
        from huggingface_hub import snapshot_download
        d = snapshot_download("openbmb/VoxCPM2", local_dir="/tmp/voxcpm2")
        c = json.load(open(d + "/config.json")); c["dtype"] = "float32"; json.dump(c, open(d + "/config.json", "w"))
        m = VoxCPM.from_pretrained(d, load_denoiser=False, optimize=False, device="cpu"); mode = "float32"; sr = m.tts_model.sample_rate
        t0 = time.time(); wav = m.generate(text=f"({DESC[tag]}){T}", cfg_value=2.0, inference_timesteps=10)
    save(f"voxcpm2_design_{tag}", wav, sr, t0, {"mode": mode})
    ref = f"{OUT}/ref_{tag}.wav" if os.path.exists(f"{OUT}/ref_{tag}.wav") else f"{OUT}/kokororef_{tag}.wav"
    t0 = time.time()
    wav = m.generate(text=T, prompt_wav_path=ref, prompt_text=REF_TEXT, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=10)
    save(f"voxcpm2_clone_{tag}", wav, sr, t0, {"mode": mode, "ref": os.path.basename(ref)})
'''

CHATTERBOX = COMMON_TTS + r'''
import torch
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device="cpu")
for tag in ("female", "male"):
    ref = f"{OUT}/ref_{tag}.wav" if os.path.exists(f"{OUT}/ref_{tag}.wav") else f"{OUT}/kokororef_{tag}.wav"
    t0 = time.time()
    wav = m.generate(T, audio_prompt_path=ref, exaggeration=0.5, cfg_weight=0.4)
    save(f"chatterbox_{tag}", wav.cpu().numpy(), m.sr, t0, {"ref": os.path.basename(ref), "exaggeration": 0.5, "cfg_weight": 0.4})
del m; torch.cuda.empty_cache()
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    m = ChatterboxTurboTTS.from_pretrained(device="cpu")
    for tag in ("female", "male"):
        ref = f"{OUT}/ref_{tag}.wav" if os.path.exists(f"{OUT}/ref_{tag}.wav") else f"{OUT}/kokororef_{tag}.wav"
        t0 = time.time()
        wav = m.generate(T, audio_prompt_path=ref)
        save(f"chatterboxturbo_{tag}", wav.cpu().numpy(), m.sr, t0, {"ref": os.path.basename(ref)})
except Exception as e:
    import traceback; traceback.print_exc()
'''

INDEX = COMMON_TTS + r'''
import sys
sys.path.insert(0, "/tmp/index-tts")
from indextts.infer_v2 import IndexTTS2
import inspect
kw = dict(cfg_path="/tmp/index-tts/checkpoints/config.yaml", model_dir="/tmp/index-tts/checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
if "device" in inspect.signature(IndexTTS2.__init__).parameters: kw["device"] = "cpu"
tts = IndexTTS2(**kw)
for tag in ("female", "male"):
    ref = f"{OUT}/ref_{tag}.wav" if os.path.exists(f"{OUT}/ref_{tag}.wav") else f"{OUT}/kokororef_{tag}.wav"
    t0 = time.time()
    tts.infer(spk_audio_prompt=ref, text=T, output_path=f"/tmp/idx_{tag}.wav", verbose=False)
    wav, sr = sf.read(f"/tmp/idx_{tag}.wav")
    save(f"indextts2_{tag}", wav if wav.ndim == 1 else wav.mean(1), sr, t0, {"ref": os.path.basename(ref)})
'''

HIGGS = COMMON_TTS + r'''
import inspect, torch, sys
sys.path.insert(0, "/tmp/higgs-audio")
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
kw = dict(device="cuda")
sig = inspect.signature(HiggsAudioServeEngine.__init__).parameters
if "torch_dtype" in sig: kw["torch_dtype"] = torch.float16
if "kv_cache_lengths" in sig: kw["kv_cache_lengths"] = [1024, 2048]
print("engine kwargs", kw, flush=True)
eng = HiggsAudioServeEngine("bosonai/higgs-audio-v2-generation-3B-base", "bosonai/higgs-audio-v2-tokenizer", **kw)
system = "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
for tag in ("female", "male"):
    ref = f"{OUT}/ref_{tag}.wav" if os.path.exists(f"{OUT}/ref_{tag}.wav") else f"{OUT}/kokororef_{tag}.wav"
    msgs = [Message(role="system", content=system), Message(role="user", content=REF_TEXT),
            Message(role="assistant", content=AudioContent(audio_url=ref)), Message(role="user", content=T)]
    t0 = time.time()
    out = eng.generate(chat_ml_sample=ChatMLSample(messages=msgs), max_new_tokens=1536, temperature=0.3, top_p=0.95, top_k=50,
                       stop_strings=["<|end_of_text|>", "<|eot_id|>"])
    save(f"higgs2_{tag}", out.audio, out.sampling_rate, t0, {"ref": os.path.basename(ref)})
'''

ASR = r'''
import glob, json, os, re, soundfile as sf, numpy as np
from faster_whisper import WhisperModel
import ctypes, site
for sp in site.getsitepackages():
    for pat in ("nvidia/cublas/lib/libcublas*.so*", "nvidia/cudnn/lib/libcudnn*.so*"):
        for lib in sorted(glob.glob(os.path.join(sp, pat))):
            try: ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError: pass
try:
    raise RuntimeError("cpu kernel")
    list(m.transcribe(np.zeros(16000, np.float32))[0])
    print("whisper on cuda", flush=True)
except Exception as e:
    print("cuda whisper failed, using cpu:", repr(e)[:200], flush=True)
    m = WhisperModel("large-v3", device="cpu", compute_type="int8")
def norm(s):
    s = s.lower().replace("’", "'"); s = re.sub(r"[^a-z' ]+", " ", s); return s.split()
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
res = {}
for f in sorted(glob.glob(W + "/voices/*.wav")):
    tag = os.path.basename(f)[:-4]
    if tag.startswith("ref_") or tag.startswith("kokororef_"): continue
    x, sr = sf.read(f, dtype="float32")
    x = x if x.ndim == 1 else x.mean(1)
    if sr != 16000:
        import scipy.signal as ss
        x = ss.resample_poly(x, 16000, sr).astype(np.float32)
    segs, _ = m.transcribe(x, language="en", beam_size=5, word_timestamps=True)
    segs = list(segs)
    hyp = " ".join(s.text for s in segs).strip()
    words = [w for s in segs for w in (s.words or [])]
    span = (words[-1].end - words[0].start) if words else 0
    gaps = [b.start - a.end for a, b in zip(words, words[1:])]
    res[tag] = dict(hyp=hyp, wer=round(wer(T, hyp), 4), words=len(words), wps=round(len(words) / max(span, 1e-3), 2),
                    pauses_ge_0_25=[round(g, 2) for g in gaps if g >= 0.25])
    print(tag, res[tag]["wer"], hyp, flush=True)
json.dump(res, open(W + "/voices/asr.json", "w"), indent=1)
'''

# ---------------------------------------------------------------- art + video stage code
ART = r'''
import torch, time, json
from diffusers import ZImagePipeline
import numpy as np
def load(dt):
    p = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=dt); p.enable_model_cpu_offload(); return p
dt = torch.bfloat16; pipe = load(dt)
P = json.load(open("/tmp/art_prompts.json"))
def gen(prompt, seed):
    return pipe(prompt=prompt, height=1536, width=864, num_inference_steps=9, guidance_scale=0.0,
                generator=torch.Generator("cpu").manual_seed(seed)).images[0]
for name, prompt in P.items():
    for seed in (11, 23):
        t0 = time.time()
        try:
            img = gen(prompt, seed); bad = float(np.asarray(img).mean()) < 2
        except Exception as e:
            print("gen failed", dt, repr(e)[:300], flush=True); bad = True
        if bad and dt == torch.bfloat16:
            del pipe; torch.cuda.empty_cache(); dt = torch.float16; pipe = load(dt); t0 = time.time(); img = gen(prompt, seed)
        img.save(f"{W}/art/{name}_s{seed}.png")
        open(f"{W}/art/timings.jsonl", "a").write(json.dumps(dict(name=name, seed=seed, s=round(time.time() - t0, 1))) + "\n")
        print(name, seed, round(time.time() - t0, 1), flush=True)
'''

WAN = r'''
import torch, time, json, glob, os
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
mid = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
def load(dt):
    p = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=dt); p.enable_model_cpu_offload(); return p
DT = [torch.bfloat16, torch.float16]
pipe = load(DT[0])
src = sorted(glob.glob(W + "/art/B_*_s11.png")) or sorted(glob.glob(W + "/art/*.png"))
img = load_image(src[0]).resize((480, 832))
prompt = ("2D storybook animation. The old woman slowly raises her right hand to her chest and turns her head toward the small fox kit, "
          "she smiles and blinks; autumn leaves drift down; gentle slow camera push-in. Clean animated illustration, consistent character.")
neg = "photorealistic, live action, morphing, distorted face, extra limbs, flicker, text, watermark, static image"
for steps in (20,):
    t0 = time.time()
    call = lambda: pipe(image=img, prompt=prompt, negative_prompt=neg, height=832, width=480, num_frames=49,
                        num_inference_steps=steps, guidance_scale=5.0, generator=torch.Generator("cpu").manual_seed(7)).frames[0]
    try:
        frames = call()
    except Exception as e:
        print("bf16 failed:", repr(e)[:300], "-> fp16", flush=True)
        del pipe; torch.cuda.empty_cache(); pipe = load(DT[1]); t0 = time.time(); frames = call()
    export_to_video(frames, f"{W}/video/wan22_ti2v5b_{steps}steps.mp4", fps=24)
    open(f"{W}/video/timings.jsonl", "a").write(json.dumps(dict(model="wan2.2-ti2v-5b", steps=steps, frames=49, res="480x832", src=os.path.basename(src[0]), s=round(time.time() - t0, 1))) + "\n")
    print("wan", steps, round(time.time() - t0, 1), flush=True)
'''

LTX = r'''
import torch, time, json, glob, os
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
def load(dt):
    p = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", torch_dtype=dt); p.enable_model_cpu_offload(); return p
pipe = load(torch.bfloat16)
src = sorted(glob.glob(W + "/art/B_*_s11.png")) or sorted(glob.glob(W + "/art/*.png"))
img = load_image(src[0]).resize((480, 832))
prompt = ("2D storybook animation. The old woman slowly raises her right hand to her chest and turns her head toward the small fox kit, "
          "she smiles and blinks; autumn leaves drift down; gentle slow camera push-in. Clean animated illustration.")
neg = "worst quality, inconsistent motion, blurry, jittery, distorted, photorealistic, morphing"
t0 = time.time()
call = lambda: pipe(image=img, prompt=prompt, negative_prompt=neg, width=480, height=832, num_frames=97, num_inference_steps=40,
                    generator=torch.Generator("cpu").manual_seed(7)).frames[0]
try:
    frames = call()
except Exception as e:
    print("bf16 failed:", repr(e)[:300], "-> fp16", flush=True)
    del pipe; torch.cuda.empty_cache(); pipe = load(torch.float16); t0 = time.time(); frames = call()
export_to_video(frames, f"{W}/video/ltxv095_2b.mp4", fps=24)
open(f"{W}/video/timings.jsonl", "a").write(json.dumps(dict(model="ltx-video-0.9.5-2b", steps=40, frames=97, res="480x832", src=os.path.basename(src[0]), s=round(time.time() - t0, 1))) + "\n")
print("ltx", round(time.time() - t0, 1), flush=True)
'''

ART_PROMPTS = {
    # A: gilded folk tapestry (illuminated manuscript / lacquer-box miniature, Western folk tale)
    "A_gilded_folk": ("Vertical storybook illustration in the style of a gilded folk-art lacquer box miniature and illuminated manuscript. "
                      "An elderly New England grandmother in an embroidered emerald-green shawl and garnet-red skirt with gold filigree trim kneels "
                      "beneath a giant maple tree with flame-red and gold leaves, cradling a tiny red fox kit. Ornate gold leaf border patterns, "
                      "deep sapphire night sky with gold stars, a white farmhouse with glowing amber windows. Saturated jewel tones, gold accents, "
                      "flat decorative shapes, fine patterned fabrics, clean edges, no text."),
    # B: storybook vector cel (clean shapes, soft gradients, puppet-rig friendly)
    "B_vector_cel": ("Vertical 2D animated film still, clean vector cut-out animation style with smooth gradient shading and crisp edges. "
                     "A kind elderly woman with silver hair in a bun, round glasses, a richly patterned plum cardigan with gold buttons and an "
                     "embroidered teal apron, standing in full body side three-quarter view, arms slightly away from her body, looking down at a "
                     "small red fox kit sitting in fallen maple leaves. Behind her an ornate autumn orchard with golden lanterns, carved wooden porch "
                     "with gilded trim, saturated jewel tones: ruby, emerald, sapphire, marigold, gold accents. Simple readable shapes, no text."),
    # C: paper theatre / jointed paper puppets (rig joints become the style)
    "C_paper_theatre": ("Vertical photograph of a handmade Victorian toy paper theatre diorama, layered cut-paper scenery with gold leaf edges, "
                        "lit from behind like stained glass in jewel colours: emerald, ruby, sapphire and amber. On the little stage, a jointed paper "
                        "puppet of an old woman with brass split-pin fasteners at her shoulders, elbows and knees, wearing a patterned paper dress, "
                        "reaching toward a tiny jointed paper fox among cut-paper maple leaves. Ornate gilded proscenium arch, depth layers, "
                        "warm lantern glow, crisp paper edges, no text."),
}

PRE_INDEX = ("rm -rf /tmp/index-tts && GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/index-tts/index-tts /tmp/index-tts && "
             "cd /tmp/index-tts && uv sync -q && uv pip install -q huggingface_hub soundfile && "
             ".venv/bin/python -c \"from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2', local_dir='/tmp/index-tts/checkpoints')\"")
PRE_HIGGS = ("rm -rf /tmp/higgs-audio && git clone -q --depth 1 https://github.com/boson-ai/higgs-audio /tmp/higgs-audio && "
             "uv venv -q -p 3.11 /tmp/v_higgs2 && cd /tmp/higgs-audio && uv pip install -q -p /tmp/v_higgs2/bin/python torch torchaudio soundfile "
             "-r requirements.txt && uv pip install -q -p /tmp/v_higgs2/bin/python -e .")


def track_a():
    stage("kokoro", 0, "3.11", "pip kokoro soundfile numpy", KOKORO, timeout=1800)
    stage("qwen3tts", 0, "3.11", "qwen-tts soundfile", QWEN, timeout=5400)
    stage("voxcpm2", 0, "3.11", "voxcpm soundfile huggingface_hub", VOXCPM, timeout=5400, cleanup=("openbmb/VoxCPM2",))
    stage("chatterbox", 0, "3.11", "chatterbox-tts soundfile", CHATTERBOX, timeout=3600)
    stage("indextts2", 0, None, None, INDEX, timeout=5400, pre=PRE_INDEX, runner="/tmp/index-tts/.venv/bin/python", workdir="/tmp/index-tts", big=True)
    stage("asr", 0, "3.11", "faster-whisper soundfile scipy", ASR, timeout=1800)


def track_b():
    json.dump(ART_PROMPTS, open("/tmp/art_prompts.json", "w"))
    pip = "torch diffusers transformers accelerate sentencepiece imageio imageio-ffmpeg ftfy pillow"
    stage("art_zimage", 1, "3.11", pip, ART, timeout=3000, big=True, cleanup=("Tongyi-MAI/Z-Image-Turbo",))
    # the video tests reuse the same venv (same libs)
    stage("video_wan22", 1, None, None, WAN, timeout=3300, runner="/tmp/v_art_zimage/bin/python", big=True,
          cleanup=("Wan-AI/Wan2.2-TI2V-5B-Diffusers",))
    stage("video_ltx", 1, None, None, LTX, timeout=2400, runner="/tmp/v_art_zimage/bin/python", big=True,
          cleanup=("Lightricks/LTX-Video-0.9.5",))


if __name__ == "__main__":
    t0 = time.time()
    sh("pip install -q uv && nproc && df -h /tmp /kaggle/working && free -g", f"{W}/logs/setup.log", timeout=600)
    sh("apt-get -qq update && apt-get -qq install -y espeak-ng git-lfs > /dev/null", f"{W}/logs/setup.log", timeout=900)
    track_a()
    STATUS["_wall_s"] = round(time.time() - t0, 1)
    json.dump(STATUS, open(f"{W}/logs/status.json", "w"), indent=1)
    print(json.dumps(STATUS, indent=1))
