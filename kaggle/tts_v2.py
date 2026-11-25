# Heartwick v2 narration worker (Kaggle). Prepended by engine2/narrate.py:  JOB = {...}
#   JOB = {"voice": "kokoro:<voice>" | "voxcpm2:female|male" | "chatterbox:female|male"   (add a GEN entry for more),
#          "speed": 1.0, "utts": [{"id", "text"}]}
# Writes /kaggle/working/tts/<id>.wav + <id>.json ({id, text, dur, words: [{w, ws, s, e}]}).
# Kokoro gives word timestamps itself (runs on CPU: no GPU queue). Cloning voices use the designed house reference
# voices (research: Qwen3-TTS VoiceDesign, no real person cloned) from the heartwick-research-voices1 dataset, and
# get word timings from faster-whisper, aligned back onto the script's own words.
import difflib, json, os, re, subprocess, sys, textwrap, time

W = "/kaggle/working"
OUT = f"{W}/tts"
os.makedirs(OUT, exist_ok=True)
REFS = "/kaggle/input/heartwick-research-voices1"
REF_TEXT = ("The wind came down from the hills that night, and the lanterns along the road flickered one by one. "
            "Nobody in the village could remember a storm quite like it.")
T0 = time.time()
engine, spk = JOB["voice"].split(":", 1)  # noqa: F821
UTTS = JOB["utts"]  # noqa: F821
json.dump(UTTS, open("/tmp/utts.json", "w"))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sh(cmd):
    log("$", cmd[:180])
    return subprocess.run(["bash", "-lc", cmd]).returncode


GEN = {
    "kokoro": ("pip kokoro 'misaki[en]' soundfile numpy torch", r'''
import torch
from kokoro import KPipeline
pipe = KPipeline(lang_code="a" if SPK.startswith("a") else "b", repo_id="hexgrad/Kokoro-82M", device="cuda" if torch.cuda.is_available() else "cpu")
for u in UTTS:
    chunks, words, off = [], [], 0.0
    for r in pipe(u["text"], voice=SPK, speed=SPEED, split_pattern=r"\n+"):
        a = r.audio.cpu().numpy() if hasattr(r.audio, "cpu") else np.asarray(r.audio)
        for t in (r.tokens or []):
            if t.start_ts is not None:
                words.append({"w": t.text, "ws": t.whitespace, "s": round(off + t.start_ts, 3), "e": round(off + (t.end_ts or t.start_ts), 3)})
        chunks.append(a)
        off += len(a) / 24000
    a = np.concatenate(chunks)
    sf.write(f"{OUT}/{u['id']}.wav", a, 24000)
    json.dump({"id": u["id"], "text": u["text"], "dur": round(len(a) / 24000, 3), "words": words}, open(f"{OUT}/{u['id']}.json", "w"))
    print("kokoro", u["id"], round(len(a) / 24000, 2), flush=True)
'''),
    "voxcpm2": ("voxcpm soundfile huggingface_hub", r'''
from voxcpm import VoxCPM
m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)
sr = m.tts_model.sample_rate
for u in UTTS:
    wav = m.generate(text=u["text"], prompt_wav_path=REF, prompt_text=REF_TEXT, reference_wav_path=REF, cfg_value=2.0, inference_timesteps=10)
    sf.write(f"{OUT}/{u['id']}.wav", np.asarray(wav, np.float32).squeeze(), sr)
'''),
    "chatterbox": ("chatterbox-tts soundfile", r'''
from chatterbox.tts import ChatterboxTTS
m = ChatterboxTTS.from_pretrained(device="cuda")
for u in UTTS:
    wav = m.generate(u["text"], audio_prompt_path=REF, exaggeration=0.5, cfg_weight=0.4)
    sf.write(f"{OUT}/{u['id']}.wav", wav.cpu().numpy().squeeze(), m.sr)
'''),
}


def run_gen():
    """Every engine runs in its own Python 3.11 venv (Kaggle's 3.13 image cannot build Kokoro's spaCy dependency)."""
    pip, code = GEN[engine]
    ref = f"{REFS}/ref_{spk}.wav"
    head = (f"import json, numpy as np, soundfile as sf\nUTTS = json.load(open('/tmp/utts.json'))\nOUT = {OUT!r}\nREF = {ref!r}\n"
            f"REF_TEXT = {REF_TEXT!r}\nSPK = {spk!r}\nSPEED = {JOB.get('speed', 1.0)!r}\n")  # noqa: F821
    open("/tmp/gen.py", "w").write(head + textwrap.dedent(code))
    if engine == "kokoro":
        sh("apt-get -qq update > /dev/null 2>&1; apt-get -qq -y install espeak-ng > /dev/null 2>&1")
    sh(f"pip install -q uv && uv venv -q -p 3.11 /tmp/v_gen && uv pip install -q -p /tmp/v_gen/bin/python {pip}")
    sh("/tmp/v_gen/bin/python /tmp/gen.py")
    if engine != "kokoro":  # Kokoro gives its own word timestamps
        align()


def _norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def align():
    """Whisper word times mapped onto the script's own tokens (so captions show the exact script text)."""
    sh(f"{sys.executable} -m pip install -q faster-whisper")
    import numpy as np, soundfile as sf
    from faster_whisper import WhisperModel
    import torch
    m = WhisperModel("small.en", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
    for u in UTTS:
        p = f"{OUT}/{u['id']}.wav"
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        from scipy.signal import resample_poly  # numpy in, not a path: Kaggle's old PyAV breaks faster-whisper's decoder
        mono = (x if x.ndim == 1 else x.mean(1)).astype(np.float32)
        segs, _ = m.transcribe(resample_poly(mono, 16000, sr).astype(np.float32) if sr != 16000 else mono, language="en", word_timestamps=True)
        hw = [w for s in segs for w in (s.words or [])]
        toks = re.findall(r"[\w'’]+|[^\w\s]", u["text"])
        spaces = []
        pos = 0
        for t in toks:  # whitespace after each token, from the text itself
            i = u["text"].find(t, pos)
            pos = i + len(t)
            spaces.append(" " if pos < len(u["text"]) and u["text"][pos] == " " else "")
        words = [(i, t) for i, t in enumerate(toks) if re.match(r"[\w'’]", t)]
        sm = difflib.SequenceMatcher(a=[_norm(t) for _, t in words], b=[_norm(w.word) for w in hw], autojunk=False)
        times = [None] * len(words)
        for a0, b0, n in sm.get_matching_blocks():
            for k in range(n):
                times[a0 + k] = (hw[b0 + k].start, hw[b0 + k].end)
        known = [k for k, v in enumerate(times) if v]
        for k in range(len(times)):  # interpolate unmatched words between known neighbours
            if times[k] is None:
                lo = max([j for j in known if j < k], default=None)
                hi = min([j for j in known if j > k], default=None)
                s0 = times[lo][1] if lo is not None else 0.0
                s1 = times[hi][0] if hi is not None else len(x) / sr
                n_ = (hi if hi is not None else len(times)) - (lo if lo is not None else -1)
                step = (s1 - s0) / max(1, n_)
                off = k - (lo if lo is not None else -1)
                times[k] = (s0 + step * (off - 1), s0 + step * off)
        out = []
        wi = 0
        for i, t in enumerate(toks):
            if re.match(r"[\w'’]", t):
                s, e = times[wi]
                wi += 1
            else:
                s = e = out[-1]["e"] if out else 0.0
            out.append({"w": t, "ws": spaces[i], "s": round(float(s), 3), "e": round(float(e), 3)})
        json.dump({"id": u["id"], "text": u["text"], "dur": round(len(x) / sr, 3), "words": out}, open(f"{OUT}/{u['id']}.json", "w"))


run_gen()
done = [u["id"] for u in UTTS if os.path.exists(f"{OUT}/{u['id']}.json")]
json.dump({"voice": JOB["voice"], "done": len(done), "asked": len(UTTS), "s": round(time.time() - T0, 1)}, open(f"{W}/tts_log.json", "w"))  # noqa: F821
log("DONE", len(done), "/", len(UTTS))
