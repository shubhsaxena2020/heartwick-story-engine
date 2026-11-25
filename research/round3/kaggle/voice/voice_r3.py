# Heartwick round 3 voice test on a Kaggle CPU kernel (no GPU session used).
# A. The top 3 library narrators read the SAME ~20 s line (the round-3 beat), sentence by sentence with designed pauses,
#    each in its library model, 2 seeds: narrator_baritone (Qwen3-TTS-1.7B-Base clone), narrator_warm_f (VoxCPM2
#    clone), british_narrator (Qwen3 clone). All Apache-2.0.
# B. Known gaps: (1) a TRUE-bass villain: Qwen3 VoiceDesign re-design aiming at 70-95 Hz, and a DSP pitch-down of the
#    library villain; (2) slower sleep voices: long designed pauses + 0.85x pitch-preserving time-stretch.
# C. Scores: faster-whisper large-v3 WER (clarity), UTMOS-22 strong predicted MOS (naturalness, MIT), pitch range /
#    s.d. in semitones (expressiveness), HuBERT-large SUPERB emotion probabilities (Apache-2.0), pace and pauses.
import glob, json, os, subprocess, textwrap, time
W = "/kaggle/working"; OUT = f"{W}/out"; os.makedirs(f"{OUT}/takes", exist_ok=True); os.makedirs(f"{W}/logs", exist_ok=True)
IN = os.path.dirname(sorted(glob.glob("/kaggle/input/**/voices.json", recursive=True))[0])
LIB = {v["id"]: v for v in json.load(open(f"{IN}/voices.json"))["voices"]}
REFS = {}
for vid in ("narrator_baritone", "narrator_warm_f", "british_narrator", "villain_king", "sleep_storyteller", "sleep_storyteller_m"):
    subprocess.run(f"ffmpeg -v error -y -i {IN}/voice_{vid}_ref.wav -ar 24000 -ac 1 {W}/{vid}_ref.wav", shell=True)
    REFS[vid] = {"wav": f"{W}/{vid}_ref.wav", "text": LIB[vid]["reference_text"]}
LINE = [
    {"text": "Nell knelt down in the cold, wet leaves and held out her hand.", "pause": 0.6},
    {"text": "For a long moment, the little grey squirrel only trembled.", "pause": 0.65},
    {"text": "Then, slowly, he crept into her palm, as light as a falling leaf.", "pause": 0.75},
    {"text": "\"There you are,\" she whispered.", "pause": 0.7},
    {"text": "And for the first time in three long years, Nell smiled.", "pause": 0.3},
]
VILLAIN = [{"text": "You may keep your little kingdom of leaves, old woman.", "pause": 0.5},
           {"text": "By morning, the whole forest belongs to me.", "pause": 0.3}]
SLEEP = [{"text": "Breathe in slowly.", "pause": 1.3}, {"text": "The rain is soft on the roof, and the fire is warm.", "pause": 1.3},
         {"text": "And the old house hums you gently to sleep.", "pause": 0.5}]
CFG = {"REFS": REFS, "LINE": LINE, "VILLAIN": VILLAIN, "SLEEP": SLEEP}
json.dump(CFG, open(f"{W}/cfg.json", "w"))
STATUS = {}


def sh(cmd, log, timeout=3600):
    t = time.time()
    with open(f"{W}/logs/{log}", "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        try:
            rc = subprocess.run(["bash", "-lc", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, round(time.time() - t, 1)


COMMON = r'''
import json, time, numpy as np, soundfile as sf, torch, random
torch.set_num_threads(4)
W = "/kaggle/working"; C = json.load(open(W + "/cfg.json")); REFS = C["REFS"]
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
def join(parts, sr, name, gen_s, extra=None):
    out = []
    for (w, pause) in parts:
        w = np.asarray(w, np.float32).squeeze()
        nz = np.where(np.abs(w) > 0.01 * (np.abs(w).max() + 1e-9))[0]
        w = w[max(0, nz[0] - int(0.03 * sr)): nz[-1] + int(0.08 * sr)] if len(nz) else w
        out += [w, np.zeros(int(pause * sr), np.float32)]
    sf.write(f"{W}/out/takes/{name}.wav", np.concatenate(out), int(sr))
    d = {"gen_s": round(gen_s, 1), "sr": int(sr)}; d.update(extra or {})
    json.dump(d, open(f"{W}/out/takes/{name}.json", "w")); print(name, d, flush=True)
'''

QWEN = r'''
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cpu", dtype=torch.float32)
def say(vid, sents, name, seed, extra=None):
    seed_all(seed); t = time.time(); parts = []
    for s in sents:
        wavs, sr = m.generate_voice_clone(text=s["text"], language="English", ref_audio=REFS[vid]["wav"], ref_text=REFS[vid]["text"])
        parts.append((wavs[0], s["pause"]))
    join(parts, sr, name, time.time() - t, extra)
for seed in (7, 11):
    say("narrator_baritone", C["LINE"], f"line__narrator_baritone__qwen3__s{seed}", seed)
    say("british_narrator", C["LINE"], f"line__british_narrator__qwen3__s{seed}", seed)
say("villain_king", C["VILLAIN"], "villain__library", 7)
say("sleep_storyteller", C["SLEEP"], "sleep__f_library", 7)
say("sleep_storyteller_m", C["SLEEP"], "sleep__m_library", 7)
# true-bass villain: re-design the voice (Qwen3 VoiceDesign), keep the lowest-pitched of 3 tries, then clone the line
import gc
del m; gc.collect()
d = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", device_map="cpu", dtype=torch.float32)
DESC = ("A very deep, dark bass male voice of a menacing old villain king, cavernous and resonant, extremely low pitch, "
        "slow and deliberate, gravelly chest voice, close to the microphone. Basso profundo.")
REFTXT = "Long ago, when the old forest still belonged to me, every creature bowed when I walked beneath the trees."
def f0med(w, sr):
    from scipy.signal import resample_poly
    x = resample_poly(np.asarray(w, np.float32).squeeze(), 16000, sr); n, hop, lo, hi = 640, 160, int(16000 / 400), int(16000 / 50)
    rms = np.sqrt((x ** 2).mean()) + 1e-9; out = []
    for i in range(0, len(x) - n, hop):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt((fr ** 2).mean()) < 0.5 * rms: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]; ac = ac / (ac[0] + 1e-9); k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5: out.append(16000 / k)
    return float(np.median(out)) if out else 0.0
best = None
for tri in range(3):
    seed_all(500 + tri); wavs, sr = d.generate_voice_design(text=REFTXT, language="English", instruct=DESC)
    f0 = f0med(wavs[0], sr); print("villain design try", tri, "f0", round(f0, 1), flush=True)
    if f0 > 40 and (best is None or f0 < best[0]): best = (f0, wavs[0], sr)
sf.write(W + "/villain_bass_ref.wav", np.asarray(best[1], np.float32).squeeze(), best[2])
json.dump({"ref_f0": round(best[0], 1)}, open(W + "/out/takes/villain_bass_ref.json", "w"))
del d; gc.collect()
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cpu", dtype=torch.float32)
REFS["villain_bass"] = {"wav": W + "/villain_bass_ref.wav", "text": REFTXT}
say("villain_bass", C["VILLAIN"], "villain__bass_redesign", 7, {"ref_f0": round(best[0], 1)})
'''

VOX = r'''
from voxcpm import VoxCPM
m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False); sr = m.tts_model.sample_rate
r = REFS["narrator_warm_f"]
for seed in (7, 11):
    seed_all(seed); t = time.time(); parts = []
    for s in C["LINE"]:   # the library's way: "ultimate cloning" with the reference text (WER 0 in the library run)
        parts.append((m.generate(text=s["text"], prompt_wav_path=r["wav"], prompt_text=r["text"], reference_wav_path=r["wav"],
                                 cfg_value=2.0, inference_timesteps=10), s["pause"]))
    join(parts, sr, f"line__narrator_warm_f__voxcpm2__s{seed}", time.time() - t)
'''

sh("pip install -q uv; nproc; free -g; ffmpeg -hide_banner -filters 2>/dev/null | grep -i rubberband", "setup.log", 900)
for name, pip, code in (("qwen3", "qwen-tts soundfile scipy", QWEN), ("voxcpm", "voxcpm soundfile", VOX)):
    rc, s = sh(f"uv venv -q -p 3.11 /tmp/v_{name} && uv pip install -q -p /tmp/v_{name}/bin/python {pip}", f"{name}.log", 2400)
    STATUS[name] = {"install_rc": rc, "install_s": s}
    if rc == 0:
        open(f"/tmp/st_{name}.py", "w").write(COMMON + textwrap.dedent(code))
        rc, s = sh(f"cd /tmp && /tmp/v_{name}/bin/python /tmp/st_{name}.py", f"{name}.log", 4 * 3600)
        STATUS[name].update(run_rc=rc, run_s=s)
    sh(f"rm -rf /tmp/v_{name} ~/.cache/uv", f"{name}.log", 600)
    json.dump(STATUS, open(f"{OUT}/status.json", "w"), indent=1)

# ---- DSP fixes (ffmpeg only) ----------------------------------------------------------------------------------------------
T = f"{OUT}/takes"
rb = subprocess.run("ffmpeg -hide_banner -filters 2>/dev/null | grep -c rubberband", shell=True, capture_output=True, text=True).stdout.strip()
if os.path.exists(f"{T}/villain__library.wav"):
    if rb and rb != "0":   # pitch down 3.5 semitones, formants kept (a bigger voice, not a chipmunk-in-reverse)
        af = "rubberband=pitch=0.817:formant=preserved"
    else:                  # resample trick: lowers pitch AND formants (a 'giant' timbre), tempo restored with atempo
        af = "asetrate=24000*0.84,aresample=24000,atempo=1.19"
    subprocess.run(f"ffmpeg -v error -y -i {T}/villain__library.wav -af '{af}' {T}/villain__library_pitchdown.wav", shell=True)
    json.dump({"gen_s": 0, "sr": 24000, "dsp": af}, open(f"{T}/villain__library_pitchdown.json", "w"))
for g in ("f", "m"):
    p = f"{T}/sleep__{g}_library.wav"
    if os.path.exists(p):   # 0.85x pitch-preserving stretch (WSOLA) on top of the long designed pauses
        subprocess.run(f"ffmpeg -v error -y -i {p} -af atempo=0.85 {T}/sleep__{g}_slow085.wav", shell=True)
        json.dump({"gen_s": 0, "sr": 24000, "dsp": "atempo=0.85"}, open(f"{T}/sleep__{g}_slow085.json", "w"))

SCORE = r'''
import glob, json, os, re, numpy as np, soundfile as sf, torch, subprocess
from faster_whisper import WhisperModel
import pyloudnorm as pyln
from scipy.signal import resample_poly
torch.set_num_threads(4)
W = "/kaggle/working"; C = json.load(open(W + "/cfg.json"))
TXT = {k: " ".join(s["text"] for s in C[k.upper()]) for k in ("line", "villain", "sleep")}
wm = WhisperModel("large-v3", device="cpu", compute_type="int8")
try:
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
except Exception as e:
    print("utmos unavailable", repr(e)[:200], flush=True); mos = None
try:
    from transformers import pipeline
    ser = pipeline("audio-classification", model="superb/hubert-large-superb-er", device=-1)
except Exception as e:
    print("ser unavailable", repr(e)[:200], flush=True); ser = None
def norm(t): return re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " ")).split()
def wer(r, h):
    r, h = norm(r), norm(h); d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p = d.copy(); d[0] = i
        for j in range(1, len(h) + 1): d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
def f0s(x, sr=16000):
    n, hop, lo, hi = 640, 160, int(sr / 400), int(sr / 50); rms = np.sqrt((x ** 2).mean()) + 1e-9; out = []
    for i in range(0, len(x) - n, hop):
        fr = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt((fr ** 2).mean()) < 0.5 * rms: continue
        ac = np.correlate(fr, fr, "full")[n - 1:]; ac = ac / (ac[0] + 1e-9); k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.5: out.append(sr / k)
    return np.array(out)
res = {}
for p in sorted(glob.glob(W + "/out/takes/*.wav")):
    name = os.path.basename(p)[:-4]; kind = name.split("__")[0]
    y, sr = sf.read(p, dtype="float32"); y = y.mean(1) if y.ndim > 1 else y
    x = resample_poly(y, 16000, sr).astype(np.float32) if sr != 16000 else y
    seg, _ = wm.transcribe(x, language="en", beam_size=5, word_timestamps=True); seg = list(seg)
    words = [w for s in seg for w in s.words]; hyp = " ".join(s.text for s in seg)
    gaps = [b.start - a.end for a, b in zip(words, words[1:]) if b.start - a.end >= 0.25]
    span = (words[-1].end - words[0].start) if words else 1
    f = f0s(x); st = 12 * np.log2(f / np.median(f)) if len(f) else np.zeros(1)
    r = {"wer": round(wer(TXT.get(kind, ""), hyp), 3), "dur_s": round(len(x) / 16000, 2), "words_per_s": round(len(words) / max(span, 1e-3), 2),
         "pauses_ge_0_25": len(gaps), "mean_pause_s": round(float(np.mean(gaps)), 2) if gaps else 0,
         "median_f0_hz": round(float(np.median(f)), 1) if len(f) else 0, "p10_f0_hz": round(float(np.percentile(f, 10)), 1) if len(f) else 0,
         "pitch_range_st_p5_p95": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2), "pitch_sd_st": round(float(st.std()), 2),
         "transcript": hyp.strip()}
    if mos is not None:
        with torch.no_grad():
            r["utmos"] = round(float(mos(torch.from_numpy(x)[None], 16000)), 3)
    if ser is not None:
        r["emotion"] = {d["label"]: round(d["score"], 3) for d in ser({"raw": x, "sampling_rate": 16000}, top_k=4)}
    r.update(json.load(open(p[:-4] + ".json")))
    r["raw_lufs"] = round(float(pyln.Meter(sr).integrated_loudness(y)), 2)
    subprocess.run(f"ffmpeg -v error -y -i {p} -af highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11 -ar 48000 -b:a 192k {W}/out/{name}.mp3", shell=True)
    res[name] = r; print(name, {k: v for k, v in r.items() if k != "transcript"}, flush=True)
json.dump(res, open(W + "/out/voice_scores.json", "w"), indent=1)
'''
rc, s = sh("uv venv -q -p 3.11 /tmp/v_score && uv pip install -q -p /tmp/v_score/bin/python faster-whisper pyloudnorm soundfile scipy torch torchaudio transformers --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu", "score.log", 2400)
open("/tmp/score.py", "w").write(SCORE)
rc, s = sh("cd /tmp && /tmp/v_score/bin/python /tmp/score.py", "score.log", 5400)
STATUS["score"] = {"rc": rc, "s": s}
json.dump(STATUS, open(f"{OUT}/status.json", "w"), indent=1)
sh(f"cp {W}/logs/*.log {OUT}/; ls -la {OUT} {OUT}/takes", "final.log")
print(json.dumps(STATUS, indent=1))
