# Heartwick engine v4, GPU session A (Kaggle T4 x2): narration + master keys.
#   GPU1: Qwen3-TTS-12Hz-1.7B-Base voice clone (Apache-2.0), one take per line, retried with a new seed when Whisper
#         hears it wrong; faster-whisper word timestamps over the joined narration. Then it joins the key queue.
#   GPU0: FLUX.2-klein-4B (Apache-2.0, 4-step, multi-reference) master keys at 1088x1920 from the moment it is up.
# Masters may reference an earlier master ("@M03"): a shared file queue hands a job to whichever GPU is free once its
# references exist. Prepended by engine4/stages4.py: IMGS {name: b64}, JOBS [{id, prompt, refs, w, h, seed}],
# LINES [{id, text, pause}], REF_B64, REF_TEXT, VOICE_ID.
import base64, json, os, subprocess, time
from pathlib import Path

W = Path("/kaggle/working"); OUT = W / "out"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "logs"; LOG.mkdir(exist_ok=True); IN = W / "in"; IN.mkdir(exist_ok=True)
KO = OUT / "keys"; KO.mkdir(exist_ok=True); Q = W / "queue"; Q.mkdir(exist_ok=True)
for k, b in IMGS.items():
    (IN / k).write_bytes(base64.b64decode(b))
    (KO / k).write_bytes(base64.b64decode(b))      # the clips kernel's identity QA mounts this output for the sheets
(W / "ref_in").write_bytes(base64.b64decode(REF_B64))
subprocess.run(f"ffmpeg -v error -y -i {W}/ref_in -ar 24000 -ac 1 {W}/ref.wav", shell=True)
json.dump({"lines": LINES, "ref_text": REF_TEXT}, open(W / "lines.json", "w"))
json.dump(JOBS, open(W / "jobs.json", "w"))
status = {"t_start": time.time(), "voice": VOICE_ID, "n_jobs": len(JOBS), "n_lines": len(LINES)}


def save():
    (OUT / "status.json").write_text(json.dumps(status, indent=1))


def sh(cmd, log, env=None, timeout=None):
    t = time.time()
    with open(LOG / log, "a") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in p.stdout:
            f.write(line); f.flush()
            if "it/s]" not in line and "s/it]" not in line and "pad_token_id" not in line:
                print(f"[{log}] {line}", end="", flush=True)
        rc = p.wait(timeout=timeout)
    return {"rc": rc, "s": round(time.time() - t, 1)}


KEYS = r'''
import json, os, sys, time, torch, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import Flux2KleinPipeline
W = Path("/kaggle/working"); IN, KO, Q = W / "in", W / "out" / "keys", W / "queue"; g = sys.argv[1]
jobs = json.load(open(W / "jobs.json")); rec = {}
t = time.time()
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
rec["load_s"] = round(time.time() - t, 1)
def ref_path(r):
    return KO / (r[1:] + ".png") if r.startswith("@") else IN / r
while True:
    todo = [j for j in jobs if not (KO / (j["id"] + ".png")).exists() and not (Q / (j["id"] + ".claim")).exists()]
    if not todo:
        break
    for j in todo:      # a reference that failed for good: this job cannot be made either (no endless wait)
        if any(r.startswith("@") and (Q / (r[1:] + ".err")).exists() for r in j["refs"]):
            (Q / (j["id"] + ".claim")).write_text(g); (Q / (j["id"] + ".err")).write_text("ref failed")
            rec[j["id"]] = {"error": "reference master failed", "gpu": g}
    ready = [j for j in todo if all(ref_path(r).exists() for r in j["refs"])]
    if not ready:
        if all((Q / (j["id"] + ".claim")).exists() or (KO / (j["id"] + ".png")).exists() for j in jobs):
            break
        time.sleep(5); continue
    j = ready[0]
    try:
        fd = os.open(Q / (j["id"] + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, g.encode()); os.close(fd)
    except FileExistsError:
        continue
    refs = [Image.open(ref_path(r)).convert("RGB") for r in j["refs"]]
    torch.cuda.synchronize(); t = time.time()
    try:
        im = pipe(prompt=j["prompt"], image=refs or None, height=j["h"], width=j["w"], guidance_scale=1.0,
                  num_inference_steps=4, generator=torch.Generator("cuda").manual_seed(j.get("seed", 7))).images[0]
        a = np.asarray(im)
        im.save(KO / (j["id"] + ".tmp.png")); os.replace(KO / (j["id"] + ".tmp.png"), KO / (j["id"] + ".png"))
        rec[j["id"]] = {"s": round(time.time() - t, 1), "gpu": g, "n_refs": len(refs), "std": round(float(a.std()), 1)}
    except Exception as e:
        rec[j["id"]] = {"error": repr(e)[-600:], "gpu": g}
        (Q / (j["id"] + ".err")).write_text(rec[j["id"]]["error"][:300])
    print(j["id"], rec[j["id"]], flush=True)
    json.dump(rec, open(W / "out" / f"keys_rec_g{g}.json", "w"), indent=1)
'''

TTS = r'''
import json, time, re, numpy as np, soundfile as sf, torch
from qwen_tts import Qwen3TTSModel
from faster_whisper import WhisperModel
W = "/kaggle/working"; L = json.load(open(W + "/lines.json"))
m = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0", dtype=torch.float32)
wm = WhisperModel("small.en", device="cuda", compute_type="float16")
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
        e = wer(ln["text"], hyp); r = {"id": ln["id"], "seed": seed, "wer": round(e, 3), "dur": round(len(w) / sr, 2), "gen_s": round(time.time() - t, 1)}
        print(json.dumps(r), flush=True)
        if best is None or e < best[0]:
            best = (e, w, r)
        if e <= 0.05:
            break
    sf.write(f"{W}/out/line_{ln['id']}.wav", best[1], sr); rep.append(best[2])
json.dump({"sr": sr, "lines": rep}, open(W + "/out/tts.json", "w"), indent=1)
parts, t, lines = [np.zeros(int(0.25 * sr), np.float32)], 0.25, []
for ln in L["lines"]:
    w, _ = sf.read(f"{W}/out/line_{ln['id']}.wav", dtype="float32")
    lines.append({"id": ln["id"], "text": ln["text"], "start": round(t, 3), "end": round(t + len(w) / sr, 3), "pause_after": ln["pause"]})
    parts += [w, np.zeros(int(ln["pause"] * sr), np.float32)]; t += len(w) / sr + ln["pause"]
x = np.concatenate(parts); sf.write(W + "/out/narration.wav", x, sr)
segs, _ = wm.transcribe(W + "/out/narration.wav", word_timestamps=True, beam_size=5)
words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)} for s in segs for w in s.words]
for li in lines:
    li["words"] = [w for w in words if w["s"] >= li["start"] - 0.15 and w["e"] <= li["end"] + 0.25]
json.dump({"sr": sr, "duration": round(len(x) / sr, 3), "lines": lines, "words": words}, open(W + "/out/narration.json", "w"), indent=1)
print("narration", round(len(x) / sr, 2), "s", len(words), "words", flush=True)
'''
(W / "keys.py").write_text(KEYS); (W / "tts.py").write_text(TTS)

# installs in parallel: FLUX stack in the main env, Qwen3-TTS in its own venv (its pins must not touch diffusers)
p_tts_env = subprocess.Popen("pip install -q uv && uv venv -q -p 3.11 /tmp/v && uv pip install -q -p /tmp/v/bin/python "
                             "qwen-tts soundfile faster-whisper torch > " + str(LOG / "setup_tts.log") + " 2>&1", shell=True)
status["setup_keys"] = sh("pip install -q -U 'diffusers>=0.37' 'transformers>=4.57' accelerate 2>&1 | tail -2", "setup_keys.log", timeout=1500)
t = time.time()
from huggingface_hub import snapshot_download
snapshot_download("black-forest-labs/FLUX.2-klein-4B", allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*",
                                                                      "tokenizer/*", "transformer/*", "vae/*"])
status["flux_download_s"] = round(time.time() - t, 1); save()
env = lambda g: dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
k0 = subprocess.Popen(f"python {W/'keys.py'} 0 > {LOG}/keys_g0.log 2>&1", shell=True, env=env(0))
t_tts = time.time()
p_tts_env.wait(); status["setup_tts_rc"] = p_tts_env.returncode; save()
status["tts"] = sh(f"cd /tmp && /tmp/v/bin/python {W/'tts.py'}", "tts.log", env=env(1), timeout=4 * 3600); save()
k1 = subprocess.Popen(f"python {W/'keys.py'} 1 > {LOG}/keys_g1.log 2>&1", shell=True, env=env(1))
k0.wait(); k1.wait()
status["keys_rc"] = [k0.returncode, k1.returncode]
status["keys_done"] = sorted(p.stem for p in KO.glob("M*.png"))
status["total_s"] = round(time.time() - status["t_start"], 1); save()
sh(f"tail -n 5 {LOG}/keys_g0.log {LOG}/keys_g1.log {LOG}/tts.log", "final.log")
print(json.dumps(status, indent=1))
