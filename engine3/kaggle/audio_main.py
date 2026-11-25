# Heartwick engine v3 kernel (Kaggle CPU, no GPU session): narration with a library voice cloned from its stored
# reference (Qwen3-TTS-12Hz-1.7B-Base, Apache-2.0), one take per line, retried with a new seed if Whisper hears it
# wrong; then faster-whisper (MIT) word timestamps over the joined narration (captions + cuts on pauses).
# REF_B64, REF_TEXT, LINES (list of {id, text, pause}), VOICE_ID are prepended by engine3/stages.py.
import base64, json, os, subprocess, time

W = "/kaggle/working"; OUT = f"{W}/out"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{W}/logs", exist_ok=True)
open(f"{W}/ref_in", "wb").write(base64.b64decode(REF_B64))
subprocess.run(f"ffmpeg -v error -y -i {W}/ref_in -ar 24000 -ac 1 {W}/ref.wav", shell=True)
json.dump({"lines": LINES, "ref_text": REF_TEXT}, open(f"{W}/lines.json", "w"))
STATUS = {"t0": time.time(), "voice": VOICE_ID}


def sh(cmd, log, timeout=3600):
    t = time.time()
    with open(f"{W}/logs/{log}", "a") as f:
        f.write(f"\n$ {cmd}\n"); f.flush()
        p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
            if "it/s]" not in line and "pad_token_id" not in line:
                print(line, end="", flush=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); rc = -9
    print(f"[{log}] rc={rc} {round(time.time() - t, 1)}s", flush=True)
    return rc, round(time.time() - t, 1)


TTS = r'''
import json, time, re, numpy as np, soundfile as sf, torch
torch.set_num_threads(4)
from qwen_tts import Qwen3TTSModel
from faster_whisper import WhisperModel
W = "/kaggle/working"; L = json.load(open(W + "/lines.json"))
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cpu", dtype=torch.float32)
wm = WhisperModel("small.en", device="cpu", compute_type="int8")
norm = lambda t: re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " ")).split()
def wer(r, h):
    r, h = norm(r), norm(h); d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        p = d.copy(); d[0] = i
        for j in range(1, len(h) + 1): d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
def trim(w, sr):
    w = np.asarray(w, np.float32).squeeze(); nz = np.where(np.abs(w) > 0.01 * (np.abs(w).max() + 1e-9))[0]
    return w[max(0, nz[0] - int(0.03 * sr)): nz[-1] + int(0.1 * sr)] if len(nz) else w
rep = []
for ln in L["lines"]:
    best = None
    for seed in (7, 11, 23):
        torch.manual_seed(seed); t = time.time()
        wavs, sr = m.generate_voice_clone(text=ln["text"], language="English", ref_audio=W + "/ref.wav", ref_text=L["ref_text"])
        w = trim(wavs[0], sr); sf.write(f"{W}/out/line_{ln['id']}.wav", w, sr)
        seg, _ = wm.transcribe(f"{W}/out/line_{ln['id']}.wav", beam_size=5); hyp = " ".join(s.text for s in seg)
        e = wer(ln["text"], hyp); r = {"id": ln["id"], "seed": seed, "wer": round(e, 3), "dur": round(len(w) / sr, 2), "gen_s": round(time.time() - t, 1), "hyp": hyp.strip()}
        print(json.dumps(r), flush=True)
        if best is None or e < best[0]:
            best = (e, w, r)
        if e <= 0.05:
            break
    sf.write(f"{W}/out/line_{ln['id']}.wav", best[1], sr); rep.append(best[2])
json.dump({"sr": sr, "lines": rep}, open(W + "/out/tts.json", "w"), indent=1)
'''

JOIN = r'''
import json, re, numpy as np, soundfile as sf
from faster_whisper import WhisperModel
W = "/kaggle/working"; L = json.load(open(W + "/lines.json"))["lines"]; T = json.load(open(W + "/out/tts.json"))
sr = T["sr"]; parts, t, lines = [], 0.25, []
parts.append(np.zeros(int(0.25 * sr), np.float32))
for ln in L:
    w, _ = sf.read(f"{W}/out/line_{ln['id']}.wav", dtype="float32")
    lines.append({"id": ln["id"], "text": ln["text"], "start": round(t, 3), "end": round(t + len(w) / sr, 3), "pause_after": ln["pause"]})
    parts += [w, np.zeros(int(ln["pause"] * sr), np.float32)]; t += len(w) / sr + ln["pause"]
x = np.concatenate(parts); sf.write(W + "/out/narration.wav", x, sr)
wm = WhisperModel("small.en", device="cpu", compute_type="int8")
segs, _ = wm.transcribe(W + "/out/narration.wav", word_timestamps=True, beam_size=5)
words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)} for s in segs for w in s.words]
for li in lines:   # words inside each line's span
    li["words"] = [w for w in words if w["s"] >= li["start"] - 0.15 and w["e"] <= li["end"] + 0.25]
json.dump({"sr": sr, "duration": round(len(x) / sr, 3), "lines": lines, "words": words}, open(W + "/out/narration.json", "w"), indent=1)
print("narration", round(len(x) / sr, 2), "s", len(words), "words")
'''
open("/tmp/tts.py", "w").write(TTS); open("/tmp/join.py", "w").write(JOIN)
rc, s = sh("pip install -q uv && uv venv -q -p 3.11 /tmp/v && uv pip install -q -p /tmp/v/bin/python qwen-tts soundfile faster-whisper "
           "torch --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu", "setup.log", 2400)
STATUS["install"] = {"rc": rc, "s": s}
rc, s = sh("cd /tmp && /tmp/v/bin/python /tmp/tts.py", "tts.log", 4 * 3600)
STATUS["tts"] = {"rc": rc, "s": s}
rc, s = sh("cd /tmp && /tmp/v/bin/python /tmp/join.py", "join.log", 1800)
STATUS["join"] = {"rc": rc, "s": s}
STATUS["total_s"] = round(time.time() - STATUS["t0"], 1)
json.dump(STATUS, open(f"{OUT}/status.json", "w"), indent=1)
sh(f"cp {W}/logs/*.log {OUT}/; ls -la {OUT}", "final.log")
