# Heartwick voice library (Kaggle, T4 x2). 14 original voices, no real person's likeness:
#   1. DESIGN  (GPU0) Qwen3-TTS-12Hz-1.7B-VoiceDesign turns a text description into a ~12 s reference clip per voice
#                     (up to 3 tries, the one whose median pitch sits in the voice's target range wins).
#   2. CLONE   every candidate cloning model renders the same passage + one in-character dialogue line from that
#              reference, 2 seeds each:  GPU0 Qwen3-TTS-1.7B-Base clone | GPU1 VoxCPM2 clone, then Chatterbox (MIT).
#   3. SCORE   (GPU0) faster-whisper large-v3 WER + word timings (pace, pauses), ECAPA speaker similarity to the
#              designed reference, pitch, clipping; best WER-0 pair per voice wins.
#   4. MASTER  70 Hz high-pass, ffmpeg loudnorm two-pass to -16 LUFS / -1.5 dBTP, re-measured; mp3 + voices.json.
# Licences: Qwen3-TTS (Apache-2.0), VoxCPM2 (Apache-2.0), Chatterbox (MIT), Whisper (MIT), ECAPA (Apache-2.0).
import json, os, shutil, subprocess, textwrap, threading, time

W = "/kaggle/working"
for d in ("refs", "takes", "library", "library/refs", "logs"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
STATUS, LOCK = {}, threading.Lock()

T = ("Every autumn, the old maple behind Nell's house turned the color of embers. "
     "That morning, tangled in its roots, she found something small and shivering, wrapped in a blanket of wet red leaves. "
     "\"Well, hello there, little one,\" she whispered, and for the first time in three years, the kitchen didn't feel so quiet.")

# id: (description for voice design, reference text read by the designer, dialogue line, target median F0 Hz, best use)
VOICES = {
    "narrator_baritone": (
        "A warm, deep American male baritone narrator in his fifties. Rich, resonant chest voice, close to the microphone in a "
        "quiet studio, calm and measured, gentle gravitas, crisp clear diction, kind and reassuring, like a classic storybook narrator.",
        "Long ago, in a valley where the rivers ran clear and cold, there lived a woodcutter who never told a lie. "
        "People came from miles around just to hear him speak.",
        "And that, my friends, is how a single maple leaf changed the whole village.",
        (85, 135), "Main narrator for Shorts and long-form (the reference channel uses a male baritone)."),
    "narrator_warm_f": (
        "A warm, gentle American woman in her late thirties narrating a cozy story. Soft, clear mid-range voice, close to the "
        "microphone, unhurried, tender, with a smile in her voice, like a beloved audiobook narrator.",
        "There is a little house at the end of Willow Lane, with a blue door and a garden full of sunflowers. "
        "Every morning, the woman who lives there sets out a bowl of seeds for the birds.",
        "So she wrapped him in her softest scarf, and carried him inside.",
        (160, 235), "Main female narrator; cozy Shorts, Nell stories."),
    "grandmother": (
        "An elderly American grandmother in her late seventies. Soft, slightly thin and airy voice with a gentle age tremble, "
        "affectionate and warm, speaking slowly and lovingly, close to the microphone.",
        "When I was a little girl, we didn't have much, but we had each other, and we had the old apple tree. "
        "Oh, how we loved that tree.",
        "Oh, you poor little thing. Let's get you warm by the stove, shall we?",
        (150, 260), "Nell and other grandmother characters."),
    "grandfather": (
        "An elderly American grandfather in his late seventies. Low, gravelly, slightly husky voice, slow and thoughtful, "
        "warm with gentle humor, a rural New England storyteller, close to the microphone.",
        "Now, I've lived on this farm going on sixty years, and I'll tell you something. "
        "The land always remembers who took good care of it.",
        "Back in my day, son, we fixed that fence with nothing but rope and patience.",
        (80, 140), "Grandfathers, wise old farmers, the wise elder."),
    "young_woman": (
        "A bright, friendly American young woman in her early twenties. Clear, lively, expressive mid-high voice with natural "
        "energy and curiosity, close to the microphone.",
        "I had never seen the ocean before that summer. The first time the waves touched my feet, "
        "I laughed so hard I nearly fell over.",
        "Wait, did you hear that? Something's moving under the porch!",
        (175, 270), "Dee, heroines, young mothers, sisters."),
    "young_man": (
        "A friendly American young man in his mid twenties. Clear, warm tenor voice, relaxed and sincere, natural and "
        "conversational, close to the microphone.",
        "My dad always said a good fence makes good neighbors. I never understood it until the day the cows got out.",
        "Don't worry, I've got the flashlight. Stay right behind me.",
        (95, 155), "Young heroes, fathers, brothers, farmhands."),
    "child_like": (
        "An adult female voice actress performing a cheerful eight-year-old boy character for an animated film. High, light, "
        "playful and bouncy, curious and excited, cartoon-like but clear, close to the microphone.",
        "I found a frog in the creek today and I named him Captain Biscuit! "
        "He can jump super far, even farther than my sister.",
        "Grandma, look! He's holding onto my finger!",
        (220, 400), "Eli and child characters (a performed adult voice, not a real child)."),
    "british_narrator": (
        "A warm, refined British man in his forties with a received pronunciation accent, narrating a classic fairy tale. "
        "Rich, articulate and elegant, gently theatrical, close to the microphone.",
        "Once upon a time, in a kingdom of mist and moss, there stood a castle with a hundred chimneys. "
        "And in the smallest room of all lived a very curious mouse.",
        "Far beyond the hills, in a kingdom of mist and moss, there lived a very peculiar king.",
        (85, 145), "Classic fairy tales, folk-tale cast stories, long-form."),
    "villain_king": (
        "A deep, commanding older man playing a menacing king in an animated fairy tale. Very low, dark and resonant voice, "
        "slow, cold and theatrical with a slight rasp, powerful, close to the microphone.",
        "You dare to stand before my throne and ask for mercy? "
        "Then kneel, little farmer, and pray that I am in a generous mood.",
        "Bring me the golden acorn by dawn, or this village will never see spring again.",
        (65, 120), "Villains, kings, giants, storm spirits."),
    "sleep_storyteller": (
        "A soft, gentle American woman in her forties telling a bedtime sleep story. Hushed, slow, soothing and calm, low and "
        "breathy, very relaxed, intimate and close to the microphone, long gentle pauses.",
        "The snow is falling softly over the sleeping village. One by one, the windows go dark, "
        "and the whole world grows quiet and still.",
        "Close your eyes now. The snow is falling softly, and everyone you love is safe and warm.",
        (140, 215), "Sleep stories and long cozy compilations."),
    "hook_reader": (
        "An energetic, punchy American man in his thirties reading a dramatic hook for a short video. Upbeat, confident, "
        "fast and exciting, strong emphasis, clear and bright, close to the microphone.",
        "Nobody in town believed her. Not the sheriff, not the mayor, not even her own family. "
        "But what she found in that barn changed everything.",
        "She found him after the storm, and nobody believed what happened next!",
        (105, 190), "First 2 seconds of a Short (the paradox hook), trailers, CTAs."),
    "sidekick_critter": (
        "An adult male voice actor performing a quirky squirrel sidekick in a cartoon. High-pitched, nasal, bouncy and fast "
        "talking, comedic and excitable, very expressive, clear, close to the microphone.",
        "Okay, okay, okay, listen. I buried exactly one hundred and twelve acorns this year, "
        "and I can only remember where three of them are!",
        "Acorns? Did somebody say acorns? Because I have been waiting all day for this!",
        (180, 400), "Pip (when he 'talks'), animal sidekicks, comic relief."),
    "british_female": (
        "A graceful British woman in her thirties with a gentle received pronunciation accent, narrating a fairy tale. Clear, "
        "warm and elegant, softly expressive, close to the microphone.",
        "In a cottage at the edge of the old forest, there lived a weaver whose threads were said to shimmer like moonlight. "
        "Travellers would knock at her door just to see them.",
        "The queen smiled, folded the letter twice, and tucked it into her sleeve.",
        (160, 245), "Queens, fairy godmothers, classic tales, British-flavoured series."),
    "sleep_storyteller_m": (
        "A calm, deep American man in his fifties telling a slow bedtime sleep story. Soft, low and soothing, hushed and "
        "relaxed, very slow with long gentle pauses, intimate and close to the microphone.",
        "The rain is tapping gently on the roof of the old farmhouse. Down the hall, the clock is ticking slowly, "
        "and there is nowhere at all that you need to be.",
        "Breathe in slowly. Outside, the wind settles, and the old house hums you to sleep.",
        (75, 130), "Male sleep stories and long-form compilations."),
}
CONST = dict(T=T, VOICES=VOICES, W=W)


def sh(cmd, log, timeout=3600, env=None):
    t = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        try:
            rc = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            f.write("TIMEOUT\n"); rc = -9
    return rc, round(time.time() - t, 1)


def stage(name, gpu, py, pip, code, timeout=3600, runner=None, keep_venv=False, hf_cleanup=()):
    log = f"{W}/logs/{name}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", HF_HUB_ENABLE_HF_TRANSFER="0",
               UV_CACHE_DIR=f"/tmp/uvcache_{name}")
    rec, t0 = {"gpu": gpu}, time.time()
    if runner is None:
        vpy = f"/tmp/v_{name}/bin/python"
        rc, dt = sh(f"uv venv -q -p {py} /tmp/v_{name} && uv pip install -q -p {vpy} {pip}", log, timeout=2400, env=env)
        rec["install_s"] = dt
        if rc:
            rec.update(ok=False, err=f"install rc={rc}")
            with LOCK: STATUS[name] = rec; json.dump(STATUS, open(f"{W}/logs/status.json", "w"), indent=1)
            return False
        runner = vpy
    src = f"/tmp/stage_{name}.py"
    open(src, "w").write("CONST = " + repr(CONST) + "\nglobals().update(CONST)\n" + textwrap.dedent(COMMON) + textwrap.dedent(code))
    rc, dt = sh(f"cd /tmp && {runner} {src}", log, timeout=timeout, env=env)
    rec.update(ok=(rc == 0), run_s=dt, total_s=round(time.time() - t0, 1))
    if rc:
        rec["err"] = f"run rc={rc}"
    for rid in hf_cleanup:  # keep the container's writable layer small (run 1 went read-only when it filled up)
        sh(f"rm -rf ~/.cache/huggingface/hub/models--{rid.replace('/', '--')}", log, timeout=300)
    if not keep_venv:
        sh(f"rm -rf /tmp/v_{name} /tmp/uvcache_{name} ~/.cache/pip", log, timeout=600)
    sh("df -h / /tmp | tail -2; du -sh ~/.cache 2>/dev/null; free -g | head -2", log, timeout=60)
    with LOCK:
        STATUS[name] = rec
        json.dump(STATUS, open(f"{W}/logs/status.json", "w"), indent=1)
    return rc == 0


COMMON = r'''
import time, json, os, random, numpy as np, soundfile as sf
def seed_all(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def ok(wav):
    w = np.asarray(wav, dtype=np.float32).squeeze()
    return w.size > 2000 and np.isfinite(w).all() and float(np.sqrt((w ** 2).mean())) > 1e-3
def save(path, wav, sr):
    sf.write(path, np.asarray(wav, dtype=np.float32).squeeze(), int(sr))
def log_take(rec):
    open(W + "/takes/timings.jsonl", "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
def f0_median(wav, sr):
    """Median F0 of voiced 40 ms frames (normalised autocorrelation), numpy only."""
    x = np.asarray(wav, np.float32).squeeze()
    if sr != 16000:
        from scipy.signal import resample_poly
        x = resample_poly(x, 16000, sr).astype(np.float32); sr = 16000
    n, hop = 640, 160
    lo, hi = int(sr / 450), int(sr / 60)
    rms_all = np.sqrt((x ** 2).mean()) + 1e-9
    f0s = []
    for i in range(0, len(x) - n, hop):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt((fr ** 2).mean()) < 0.5 * rms_all: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        ac = ac / (ac[0] + 1e-9)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5: f0s.append(sr / k)
    return float(np.median(f0s)) if f0s else 0.0
def parts():
    for vid, (desc, ref_text, line, f0r, use) in VOICES.items():
        yield vid, desc, ref_text, line, f0r
def wait_refs(timeout=5400):
    t = time.time()
    while not os.path.exists(W + "/refs/DONE"):
        if time.time() - t > timeout: raise SystemExit("refs never arrived")
        time.sleep(10)
'''

# ---------------------------------------------------------------- GPU0: Qwen3 design (refs) then Qwen3 Base clone
QWEN = r'''
import torch, gc
from qwen_tts import Qwen3TTSModel
DT = torch.float16
def load(mid):
    return Qwen3TTSModel.from_pretrained(mid, device_map="cuda:0", dtype=DT, attn_implementation="sdpa")
m = load("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
for vid, desc, ref_text, line, (flo, fhi) in parts():
    best = None
    for tri in range(3):
        seed_all(1000 + tri * 17)
        t0 = time.time()
        wavs, sr = m.generate_voice_design(text=ref_text, language="English", instruct=desc)
        w = np.asarray(wavs[0], np.float32).squeeze()
        if not ok(w): continue
        f0 = f0_median(w, sr)
        dist = 0 if flo <= f0 <= fhi else min(abs(f0 - flo), abs(f0 - fhi))
        print(vid, "design try", tri, "f0", round(f0, 1), "dist", round(dist, 1), round(time.time() - t0, 1), "s", flush=True)
        if best is None or dist < best[0]: best = (dist, w, sr, f0, tri)
        if dist == 0: break
    dist, w, sr, f0, tri = best
    save(f"{W}/refs/{vid}.wav", w, sr)
    open(f"{W}/refs/{vid}.txt", "w").write(ref_text)
    json.dump({"f0": round(f0, 1), "in_range": dist == 0, "try": tri, "sr": int(sr), "dur": round(len(w) / sr, 2)},
              open(f"{W}/refs/{vid}.json", "w"))
open(W + "/refs/DONE", "w").write("ok")
del m; gc.collect(); torch.cuda.empty_cache()
m = load("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
for vid, desc, ref_text, line, f0r in parts():
    ref = f"{W}/refs/{vid}.wav"
    try:
        prompt = m.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text)
    except Exception as e:
        print("no reusable clone prompt:", repr(e)[:200], flush=True); prompt = None
    for s in (1, 2):
        for part, text in (("passage", T), ("line", line)):
            seed_all(s * 101)
            t0 = time.time()
            try:
                if prompt is None: raise TypeError("no prompt")
                wavs, sr = m.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt)
            except TypeError:
                wavs, sr = m.generate_voice_clone(text=text, language="English", ref_audio=ref, ref_text=ref_text)
            w = np.asarray(wavs[0], np.float32).squeeze()
            if not ok(w): print("bad", vid, part, s, flush=True); continue
            save(f"{W}/takes/{vid}__qwen3clone__{part}__s{s}.wav", w, sr)
            log_take(dict(voice=vid, model="qwen3clone", part=part, seed=s, sr=int(sr), dur=round(len(w) / sr, 2), gen_s=round(time.time() - t0, 1)))
'''

# ---------------------------------------------------------------- GPU1: VoxCPM2 clone, then Chatterbox
VOXCPM = r'''
import torch
from voxcpm import VoxCPM
m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)
sr = m.tts_model.sample_rate
wait_refs()
for vid, desc, ref_text, line, f0r in parts():
    ref = f"{W}/refs/{vid}.wav"
    for s in (1, 2):
        for part, text in (("passage", T), ("line", line)):
            seed_all(s * 101)
            t0 = time.time()
            w = m.generate(text=text, prompt_wav_path=ref, prompt_text=ref_text, reference_wav_path=ref, cfg_value=2.0, inference_timesteps=12)
            if not ok(w): print("bad", vid, part, s, flush=True); continue
            save(f"{W}/takes/{vid}__voxcpm2__{part}__s{s}.wav", w, sr)
            log_take(dict(voice=vid, model="voxcpm2", part=part, seed=s, sr=int(sr), dur=round(len(np.asarray(w).squeeze()) / sr, 2), gen_s=round(time.time() - t0, 1)))
'''

EXAG = {"sidekick_critter": 0.75, "villain_king": 0.6, "hook_reader": 0.7, "child_like": 0.65, "young_woman": 0.55,
        "sleep_storyteller": 0.3, "sleep_storyteller_m": 0.3}
CONST["EXAG"] = EXAG
CHATTER = r'''
import torch
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device="cuda")
wait_refs()
for vid, desc, ref_text, line, f0r in parts():
    ref = f"{W}/refs/{vid}.wav"
    ex = EXAG.get(vid, 0.5)
    for s in (1, 2):
        for part, text in (("passage", T), ("line", line)):
            seed_all(s * 101)
            t0 = time.time()
            w = m.generate(text, audio_prompt_path=ref, exaggeration=ex, cfg_weight=0.3 if ex >= 0.6 else 0.5).cpu().numpy()
            if not ok(w): print("bad", vid, part, s, flush=True); continue
            save(f"{W}/takes/{vid}__chatterbox__{part}__s{s}.wav", w, m.sr)
            log_take(dict(voice=vid, model="chatterbox", part=part, seed=s, sr=int(m.sr), exaggeration=ex,
                          dur=round(len(np.asarray(w).squeeze()) / m.sr, 2), gen_s=round(time.time() - t0, 1)))
'''

# ---------------------------------------------------------------- GPU0 at the end: score, pick, master
SCORE = r'''
import glob, re, ctypes, site, subprocess, math, shutil
for sp in site.getsitepackages():
    for pat in ("nvidia/cublas/lib/libcublas*.so*", "nvidia/cudnn/lib/libcudnn*.so*"):
        for lib in sorted(glob.glob(os.path.join(sp, pat))):
            try: ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError: pass
from faster_whisper import WhisperModel
from scipy.signal import resample_poly, butter, sosfilt
import pyloudnorm as pyln
try:
    wm = WhisperModel("large-v3", device="cuda", compute_type="float16"); list(wm.transcribe(np.zeros(16000, np.float32))[0]); DEV = "cuda"
except Exception as e:
    print("cuda whisper failed -> cpu", repr(e)[:200], flush=True)
    wm = WhisperModel("large-v3", device="cpu", compute_type="int8"); DEV = "cpu"
try:
    import torch, torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/ecapa",
                                         run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})
except Exception as e:
    print("speaker model unavailable:", repr(e)[:300], flush=True); spk = None

def load16(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
    return (resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x), x, sr
def norm(s):
    s = s.lower().replace("â€™", "'").replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    nums = {"3": "three", "112": "one hundred and twelve", "8": "eight", "60": "sixty", "100": "one hundred"}
    return " ".join(nums.get(w, w) for w in s.split()).split()
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
def emb(x16):
    if spk is None: return None
    import torch
    with torch.no_grad():
        e = spk.encode_batch(torch.tensor(x16)[None]).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)
def asr(x16, ref_text):
    segs, _ = wm.transcribe(x16, language="en", beam_size=5, word_timestamps=True, condition_on_previous_text=False)
    segs = list(segs)
    hyp = " ".join(s.text for s in segs).strip()
    words = [w for s in segs for w in (s.words or [])]
    span = (words[-1].end - words[0].start) if words else 0.0
    gaps = [round(b.start - a.end, 2) for a, b in zip(words, words[1:])]
    lp = float(np.mean([s.avg_logprob for s in segs])) if segs else -9
    return dict(hyp=hyp, wer=round(wer(ref_text, hyp), 4), words=len(words), wps=round(len(words) / max(span, 1e-3), 2),
                pauses=[g for g in gaps if g >= 0.25], max_pause=max(gaps) if gaps else 0.0, logprob=round(lp, 3),
                lead=round(words[0].start, 2) if words else 0, tail=round(len(x16) / 16000 - words[-1].end, 2) if words else 0)

res, refemb = {}, {}
for vid, desc, ref_text, line, f0r in parts():
    p = f"{W}/refs/{vid}.wav"
    if not os.path.exists(p): continue
    x16, x, sr = load16(p)
    a = asr(x16, ref_text); a["f0"] = round(f0_median(x, sr), 1)
    refemb[vid] = emb(x16)
    res[f"{vid}__REF"] = a
    print(vid, "REF", a["wer"], a["f0"], a["hyp"], flush=True)
for f in sorted(glob.glob(W + "/takes/*.wav")):
    vid, model, part, s = os.path.basename(f)[:-4].split("__")
    line = VOICES[vid][2]
    x16, x, sr = load16(f)
    a = asr(x16, T if part == "passage" else line)
    a.update(f0=round(f0_median(x, sr), 1), peak_dbfs=round(20 * math.log10(float(np.abs(x).max()) + 1e-9), 2),
             clip_frac=round(float((np.abs(x) > 0.995).mean()), 6), raw_lufs=round(pyln.Meter(sr).integrated_loudness(x), 1))
    e = emb(x16)
    a["spk_sim"] = round(float(np.dot(e, refemb[vid])), 4) if e is not None and refemb.get(vid) is not None else None
    res[os.path.basename(f)[:-4]] = a
    print(os.path.basename(f), a["wer"], a["wps"], a["spk_sim"], a["hyp"][:80], flush=True)
json.dump(res, open(W + "/takes/scores.json", "w"), indent=1)

PRIOR = {"voxcpm2": 0.03, "qwen3clone": 0.02, "chatterbox": 0.0}  # run 1: voxcpm2_clone and qwen3tts were the best
def take_score(a):
    if a is None: return -99
    s = -5.0 * a["wer"] + (a["spk_sim"] or 0.0) + 0.15 * a["logprob"]
    s -= 0.5 * max(0.0, a["max_pause"] - 1.4) + 50 * a["clip_frac"]
    return s
choice = {}
for vid in VOICES:
    best = None
    for model in ("voxcpm2", "qwen3clone", "chatterbox"):
        P = [(k, res[k]) for k in res if k.startswith(f"{vid}__{model}__passage__")]
        L = [(k, res[k]) for k in res if k.startswith(f"{vid}__{model}__line__")]
        if not P or not L: continue
        kp, ap = max(P, key=lambda kv: take_score(kv[1]))
        kl, al = max(L, key=lambda kv: take_score(kv[1]))
        clean = ap["wer"] == 0 and al["wer"] == 0
        sc = take_score(ap) + 0.5 * take_score(al) + PRIOR[model] + (10 if clean else 0)
        if best is None or sc > best[0]: best = (sc, model, kp, kl, clean)
    if best: choice[vid] = best
    print("CHOICE", vid, best, flush=True)

SR = 48000
def master(vid, kp, kl):
    xs = []
    for k, gap in ((kp, 0.75), (kl, 0.0)):
        x, sr = sf.read(f"{W}/takes/{k}.wav", dtype="float32", always_2d=True); x = x.mean(1)
        if sr != SR:
            g = math.gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        a = res[k]
        i0 = max(0, int((a["lead"] - 0.2) * SR)); i1 = len(x) - max(0, int((a["tail"] - 0.35) * SR))
        x = x[i0:i1]
        n = int(0.03 * SR); x[:n] *= np.linspace(0, 1, n); x[-n:] *= np.linspace(1, 0, n)
        xs += [x, np.zeros(int(gap * SR), np.float32)]
    x = np.concatenate(xs)
    x = sosfilt(butter(2, 70, "highpass", fs=SR, output="sos"), x).astype(np.float32)
    raw = f"/tmp/{vid}_raw.wav"; sf.write(raw, x, SR)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", raw, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    js = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
    af = ("loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:"
          "measured_thresh={input_thresh}:offset={target_offset}").format(**js)
    wav = f"{W}/library/voice_{vid}.wav"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", raw, "-af", af, "-ar", str(SR), "-ac", "1", wav], check=True)
    y, _ = sf.read(wav, dtype="float32")
    lufs = pyln.Meter(SR).integrated_loudness(y)
    if abs(lufs + 16) > 0.3:  # linear mode can fall back to dynamic: correct the residual with a plain gain
        y = y * 10 ** ((-16 - lufs) / 20)
        pk = np.abs(y).max()
        if pk > 10 ** (-1.5 / 20): y = y * (10 ** (-1.5 / 20) / pk)
        sf.write(wav, y, SR); lufs = pyln.Meter(SR).integrated_loudness(y)
    mp3 = f"{W}/library/voice_{vid}.mp3"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav, "-c:a", "libmp3lame", "-b:a", "192k", mp3], check=True)
    x16 = resample_poly(y, 1, 3).astype(np.float32)
    a = asr(x16, T + " " + VOICES[vid][2])
    return dict(lufs=round(lufs, 2), peak_dbfs=round(20 * math.log10(float(np.abs(y).max()) + 1e-9), 2), dur_s=round(len(y) / SR, 2), **{f"final_{k}": v for k, v in a.items() if k in ("wer", "wps", "max_pause", "hyp")}, pauses_ge_0_25=len(a["pauses"]), mean_pause=round(float(np.mean(a["pauses"])) if a["pauses"] else 0.0, 2))

LIC = {"voxcpm2": "VoxCPM2: Apache-2.0 (commercial use OK)", "qwen3clone": "Qwen3-TTS-1.7B-Base: Apache-2.0 (commercial use OK)",
       "chatterbox": "Chatterbox: MIT (commercial use OK; output carries Resemble's imperceptible Perth watermark)"}
lib = []
for vid, (sc, model, kp, kl, clean) in choice.items():
    m = master(vid, kp, kl)
    shutil.copy(f"{W}/refs/{vid}.wav", f"{W}/library/refs/voice_{vid}_ref.wav")
    refinfo = json.load(open(f"{W}/refs/{vid}.json"))
    desc, ref_text, line, f0r, use = VOICES[vid]
    lib.append(dict(id=vid, description=desc, dialogue_line=line, best_use=use,
                    design_model="Qwen3-TTS-12Hz-1.7B-VoiceDesign (Apache-2.0)", model=model, licence=LIC[model] + "; designed reference: Qwen3-TTS VoiceDesign, Apache-2.0",
                    reference_file=f"refs/voice_{vid}_ref.wav", reference_text=ref_text, reference_f0_hz=refinfo["f0"], target_f0_hz=list(f0r),
                    sample_file=f"voice_{vid}.mp3", passage_take=kp, line_take=kl, all_takes_wer0=clean,
                    measured=dict(lufs=m["lufs"], peak_dbfs=m["peak_dbfs"], wer=m["final_wer"], words_per_s=m["final_wps"],
                                  pauses_ge_0_25=m["pauses_ge_0_25"], mean_pause_s=m["mean_pause"], max_pause_s=m["final_max_pause"],
                                  speaker_similarity_to_ref=res[kp]["spk_sim"], median_f0_hz=res[kp]["f0"], clip_frac=res[kp]["clip_frac"],
                                  passage_wer=res[kp]["wer"], line_wer=res[kl]["wer"], dur_s=m["dur_s"]),
                    transcript=m["final_hyp"]))
    print("LIB", vid, model, m, flush=True)
json.dump(dict(passage=T, target_lufs=-16, true_peak_dbtp=-1.5, voices=lib), open(W + "/library/voices.json", "w"), indent=1)
# shrink the takes for download: mp3 at 128k
for f in glob.glob(W + "/takes/*.wav"):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f, "-c:a", "libmp3lame", "-b:a", "128k", f[:-4] + ".mp3"])
    if os.path.exists(f[:-4] + ".mp3"): os.remove(f)
'''


def gpu0():
    stage("qwen", 0, "3.11", "qwen-tts soundfile scipy", QWEN, timeout=7200, hf_cleanup=("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))


def gpu1():
    stage("voxcpm2", 1, "3.11", "voxcpm soundfile scipy huggingface_hub", VOXCPM, timeout=7200, hf_cleanup=("openbmb/VoxCPM2",))
    stage("chatterbox", 1, "3.11", "chatterbox-tts soundfile scipy", CHATTER, timeout=3600, hf_cleanup=("ResembleAI/chatterbox",))


if __name__ == "__main__":
    t0 = time.time()
    sh("pip install -q uv && nvidia-smi && df -h / /tmp /kaggle/working && free -g", f"{W}/logs/setup.log", timeout=600)
    if not os.path.exists(f"{W}/refs/DONE"):
        open(f"{W}/logs/start", "w").write(str(t0))
    a, b = threading.Thread(target=gpu0), threading.Thread(target=gpu1)
    a.start(); b.start(); a.join()
    if not os.path.exists(f"{W}/refs/DONE"):  # the designer failed: nothing to clone from
        open(f"{W}/refs/DONE", "w").write("failed")
    b.join()
    ok = stage("score", 0, "3.11", "faster-whisper soundfile scipy pyloudnorm numpy speechbrain torch torchaudio", SCORE, timeout=3600)
    if not ok:  # scoring must not be lost to a dependency clash: retry without the speaker model
        stage("score_nospk", 0, "3.11", "faster-whisper soundfile scipy pyloudnorm numpy nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*",
              SCORE, timeout=3600)
    STATUS["_wall_s"] = round(time.time() - t0, 1)
    json.dump(STATUS, open(f"{W}/logs/status.json", "w"), indent=1)
    print(json.dumps(STATUS, indent=1))
