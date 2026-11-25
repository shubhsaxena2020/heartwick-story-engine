# Heartwick research: transcribe the Parabrahma World reel audio (Hindi) with word timestamps.
# Runs as a Kaggle CPU kernel (no GPU quota). Output: /kaggle/working/asr/<id>.json
import glob, json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faster-whisper", "soundfile"], check=True)
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

src = sorted(glob.glob("/kaggle/input/**/*.flac", recursive=True))
print("files", src, flush=True)
os.makedirs("/kaggle/working/asr", exist_ok=True)
t0 = time.time()
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
print("model loaded", round(time.time() - t0), "s", flush=True)
for f in src:
    sid = os.path.basename(f)[:-5]
    t = time.time()
    x, sr = sf.read(f, dtype="float32")
    assert sr == 16000, sr
    x = x if x.ndim == 1 else x.mean(1)
    segs, info = m.transcribe(x, language="hi", word_timestamps=True, vad_filter=False, beam_size=5,
                              condition_on_previous_text=True)
    out = []
    for s in segs:
        out.append(dict(start=s.start, end=s.end, text=s.text,
                        words=[dict(w=w.word, s=w.start, e=w.end, p=round(w.probability, 3)) for w in (s.words or [])]))
    # English gist (translate task) for the beat sheet
    tsegs, _ = m.transcribe(x, task="translate", beam_size=5)
    eng = [dict(start=s.start, end=s.end, text=s.text) for s in tsegs]
    json.dump(dict(id=sid, duration=info.duration, segments=out, english=eng), open(f"/kaggle/working/asr/{sid}.json", "w"),
              ensure_ascii=False, indent=1)
    print(sid, "done in", round(time.time() - t), "s", flush=True)
print("ALL DONE", round(time.time() - t0), "s")
