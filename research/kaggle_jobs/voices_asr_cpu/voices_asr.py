# Whisper WER / pace for the voice comparison (CPU kernel: no GPU queue). Reads the dataset of synthesised voices.
import glob, json, os, re, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper"], check=True)
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel
T = ("Every autumn, the old maple behind Nell's house turned the color of embers. "
     "That morning, tangled in its roots, she found something small and shivering, wrapped in a blanket of wet red leaves. "
     "\"Well, hello there, little one,\" she whispered, and for the first time in three years, the kitchen didn't feel so quiet.")
W = "/kaggle/working"
os.makedirs(f"{W}/voices", exist_ok=True)
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)
def norm(s):
    s = s.lower().replace("\u2019", "'"); s = re.sub(r"[^a-z' ]+", " ", s); return s.split()
def wer(ref, hyp):
    r, h = norm(ref), norm(hyp); d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
res = {}
for f in sorted(glob.glob("/kaggle/input/**/*.wav", recursive=True)):
    tag = os.path.basename(f)[:-4]
    if tag.startswith(("ref_", "kokororef_")):
        continue
    x, sr = sf.read(f, dtype="float32")  # numpy in: the image's old PyAV breaks faster-whisper's file decoder
    x = x if x.ndim == 1 else x.mean(1)
    if sr != 16000:
        from scipy.signal import resample_poly
        x = resample_poly(x, 16000, sr).astype(np.float32)
    segs, _ = m.transcribe(x, language="en", beam_size=5, word_timestamps=True)
    segs = list(segs)
    hyp = " ".join(s.text for s in segs).strip()
    words = [w for s in segs for w in (s.words or [])]
    span = (words[-1].end - words[0].start) if words else 0
    gaps = [b.start - a.end for a, b in zip(words, words[1:])]
    res[tag] = dict(hyp=hyp, wer=round(wer(T, hyp), 4), words=len(words), wps=round(len(words) / max(span, 1e-3), 2),
                    pauses_ge_0_25=[round(g, 2) for g in gaps if g >= 0.25])
    print(tag, res[tag]["wer"], res[tag]["wps"], hyp, flush=True)
json.dump(res, open(f"{W}/voices/asr.json", "w"), indent=1)