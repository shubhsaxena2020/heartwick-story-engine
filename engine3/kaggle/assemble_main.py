# Heartwick engine v3 kernel 3 (Kaggle CPU): assembly. Finished 1080x1920 shots (cut on narration pauses, match cuts on
# shared keys) + caption chunks (2-4 words, one highlighted word, ~71 % height) + the Heartwick series tag + the end
# card; narration + ACE-Step pad 18-22 dB under the voice + sparse procedural SFX, mastered to -14 LUFS; .srt, SEO pack,
# thumbnail, contact sheet, QA report. ENGINE (brand/style/audio sources), PLAN are prepended by engine3/stages.py.
import glob, json, math, os, re, subprocess, sys, time
W = "/kaggle/working"; OUT = f"{W}/out"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{W}/engine", exist_ok=True)
os.makedirs(f"{W}/brand/fonts", exist_ok=True)
for name, s in ENGINE.items():
    open(f"{W}/engine/{name}", "w").write(s)
open(f"{W}/engine/__init__.py", "w").write("")
for name, url in (("Nunito-VF.ttf", "ofl/nunito/Nunito%5Bwght%5D.ttf"), ("Fraunces-VF.ttf", "ofl/fraunces/Fraunces%5BSOFT,WONK,opsz,wght%5D.ttf")):
    subprocess.run(["curl", "-sSL", "-o", f"{W}/brand/fonts/{name}", f"https://github.com/google/fonts/raw/main/{url}"])
subprocess.run("pip install -q pyloudnorm soundfile 2>&1 | tail -1", shell=True)
sys.path.insert(0, W)
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from engine import brand
from engine import audio2 as A
P = json.loads(PLAN); T0 = time.time(); rep = {"stages": {}}
FPS, WD, HT = int(P.get("fps", 60)), 1080, 1920   # v3: 60 (round-3 RIFE 60, 24 Nov); v4: 24 (limited animation, PLAN fps)


def find(name):
    r = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return r[0] if r else None


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        print("CMD FAILED", cmd[:300], r.stderr[-1500:], flush=True)
    return r


# ---------------------------------------------------------------- picture: shots back to back (cuts in pauses)
parts = []
for s in P["shots"]:
    f = None
    if s.get("src"):   # the newest clips kernel that made this shot (re-runs of single shots live in their own kernel)
        r = sorted(glob.glob(f"/kaggle/input/{s['src']}/**/finished/{s['id']}.mp4", recursive=True)) + \
            sorted(glob.glob(f"/kaggle/input/**/{s['src']}/**/finished/{s['id']}.mp4", recursive=True))
        f = r[0] if r else None
    f = f or find(f"finished/{s['id']}.mp4") or find(f"{s['id']}.mp4")
    if not f:
        print("missing finished shot", s["id"], flush=True); continue
    parts.append((s, f))
open("/tmp/list.txt", "w").write("".join(f"file '{f}'\n" for _, f in parts))
run(f"ffmpeg -v error -y -f concat -safe 0 -i /tmp/list.txt -c:v libx264 -crf 16 -preset medium -pix_fmt yuv420p -r {FPS} /tmp/body.mp4")
body_s = float(run("ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/body.mp4").stdout or 0)
rep["body_s"] = round(body_s, 3); rep["shots_used"] = [s["id"] for s, _ in parts]

# end card (animated, drawn in code)
EC_S = P.get("end_card_s", 3.0)
ec = brand.EndCard(WD, HT, P.get("cta", "Follow for more Heartwick stories"))
os.makedirs("/tmp/ec", exist_ok=True)
for i in range(int(EC_S * FPS)):
    Image.fromarray((ec.frame(i / FPS) * 255).astype(np.uint8)).save(f"/tmp/ec/{i:05d}.png")
run(f"ffmpeg -v error -y -framerate {FPS} -i /tmp/ec/%05d.png -c:v libx264 -crf 16 -pix_fmt yuv420p /tmp/endcard.mp4")
# a 0.5 s dip from the last shot into the end card
XF = 0.5
run(f"ffmpeg -v error -y -i /tmp/body.mp4 -i /tmp/endcard.mp4 -filter_complex "
    f"\"[0:v]settb=AVTB,fps={FPS},format=yuv420p[a];[1:v]settb=AVTB,fps={FPS},format=yuv420p[b];"
    f"[a][b]xfade=transition=fade:duration={XF}:offset={body_s - XF:.3f},format=yuv420p[v]\" -map [v] -c:v libx264 -crf 16 -preset medium /tmp/picture.mp4")
total = body_s + EC_S - XF
rep["total_s"] = round(total, 3)

# ---------------------------------------------------------------- captions from the narration words
N = json.load(open(find("narration.json")))
norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())
import difflib
script_words = []
for ln in N["lines"]:
    for w in ln["text"].split():
        script_words.append({"w": w, "line": ln["id"]})
asr = N["words"]
sm = difflib.SequenceMatcher(a=[norm(w["w"]) for w in script_words], b=[norm(w["w"]) for w in asr], autojunk=False)
for a0, b0, n in sm.get_matching_blocks():
    for k in range(n):
        script_words[a0 + k]["s"], script_words[a0 + k]["e"] = asr[b0 + k]["s"], asr[b0 + k]["e"]
lines = {ln["id"]: ln for ln in N["lines"]}
for i, w in enumerate(script_words):   # unmatched words: interpolate inside their line's span
    if "s" not in w:
        L = lines[w["line"]]; same = [j for j, x in enumerate(script_words) if x["line"] == w["line"]]
        k = same.index(i); span = (L["end"] - L["start"]) / max(1, len(same))
        w["s"], w["e"] = L["start"] + k * span, L["start"] + (k + 1) * span
matched = sum(1 for w in script_words if "s" in w)
rep["caption_words"] = len(script_words); rep["caption_words_matched_asr"] = sum(n for _, _, n in sm.get_matching_blocks())

HL = {h.lower() for h in P.get("highlight_words", [])}
chunks, cur = [], []
for i, w in enumerate(script_words):
    cur.append(w)
    nxt = script_words[i + 1] if i + 1 < len(script_words) else None
    gap = (nxt["s"] - w["e"]) if nxt else 9
    end_punct = w["w"][-1] in ".,!?;:…\"" or w["w"].endswith("...")
    if len(cur) >= 4 or (len(cur) >= 2 and (end_punct or gap > 0.28)) or not nxt or (nxt and nxt["line"] != w["line"]):
        chunks.append(cur); cur = []
cap = []
for i, c in enumerate(chunks):
    words = [x["w"] for x in c]
    plain = [norm(x) for x in words]
    hl = next((k for k, p in enumerate(plain) if p in HL), None)
    if hl is None:
        cands = [(len(p), k) for k, p in enumerate(plain) if len(p) >= 5]
        hl = max(cands)[1] if cands else -1
    a = c[0]["s"]; b = chunks[i + 1][0]["s"] if i + 1 < len(chunks) else c[-1]["e"] + 0.4
    b = min(b, c[-1]["e"] + 0.9)
    cap.append({"words": words, "hl": hl, "a": round(a, 3), "b": round(b, 3)})
rep["caption_chunks"] = len(cap)

os.makedirs("/tmp/cap", exist_ok=True)
inputs, filt, last = [], [], "[0:v]"
Y = int(HT * P.get("caption_y", 0.71))
for i, c in enumerate(cap):
    im = brand.caption_sprite(c["words"], c["hl"], scale=1.0)
    if im.width > WD - 40:
        im = brand.caption_sprite(c["words"], c["hl"], scale=(WD - 40) / im.width)
    p = f"/tmp/cap/c{i:03d}.png"; im.save(p); inputs.append(p)
    x = (WD - im.width) // 2; y = Y - im.height // 2
    filt.append(f"{last}[{i + 1}:v]overlay={x}:{y}:enable='between(t,{c['a']:.3f},{c['b']:.3f})'[v{i}]"); last = f"[v{i}]"
tag = brand.series_tag(P.get("series_label", "Pip"), scale=0.9); tag.save("/tmp/tag.png")
k = len(cap) + 1
filt.append(f"[{k}:v]format=rgba,fade=t=in:st=0.4:d=0.5:alpha=1,fade=t=out:st={P.get('tag_s', 4.0)}:d=0.6:alpha=1[tg]")
filt.append(f"{last}[tg]overlay=56:110:enable='between(t,0,{P.get('tag_s', 4.0) + 0.7})'[vout]")
open("/tmp/filt.txt", "w").write(";\n".join(filt))
ins = " ".join(f"-i {p}" for p in inputs) + " -loop 1 -t 6 -i /tmp/tag.png"
run(f"ffmpeg -v error -y -i /tmp/picture.mp4 {ins} -filter_complex_script /tmp/filt.txt -map [vout] -c:v libx264 -crf 17 -preset medium -pix_fmt yuv420p -r {FPS} /tmp/captioned.mp4")
rep["stages"]["picture_s"] = round(time.time() - T0, 1)

# ---------------------------------------------------------------- sound: voice + music pad + sparse SFX, -14 LUFS
SR = A.SR; n = int(total * SR)
v = A.load_mono(find("narration.wav")); voice = np.zeros(n, np.float32); voice[: min(n, len(v))] = v[:n]
from scipy.signal import butter, sosfilt
voice = A.compress(sosfilt(butter(2, 70, "highpass", fs=SR, output="sos"), voice)).astype(np.float32)
music = None
mf = find(P.get("music_file", "music_1.wav"))
if mf:
    m = A.load_mono(mf)
    if len(m) < n:   # loop with a crossfade if the cue is short
        f = int(2 * SR); m2 = m.copy()
        while len(m2) < n:
            m2 = np.concatenate([m2[:-f], m2[-f:] * np.linspace(1, 0, f) + m[:f] * np.linspace(0, 1, f), m[f:]])
        m = m2
    m = m[:n].copy(); fi, fo = int(1.0 * SR), int(2.5 * SR)
    m[:fi] *= np.linspace(0, 1, fi); m[-fo:] *= np.linspace(1, 0, fo); music = m
rep["music_file"] = os.path.basename(mf) if mf else None


def rustle(d, seed=5):
    k = int(d * SR); r = np.random.default_rng(seed); x = np.zeros(k, np.float32)
    for p in r.integers(0, max(1, k - 4000), int(d * 14)):
        L = int(r.uniform(0.01, 0.05) * SR); x[p:p + L] += r.normal(0, 1, L) * np.hanning(L) * r.uniform(0.2, 1.0)
    return sosfilt(butter(2, [1500, 7000], "bandpass", fs=SR, output="sos"), x).astype(np.float32) * 0.15


sfx = np.zeros(n, np.float32); used = []
for s in P["shots"]:
    kw = (s.get("sfx") or "").lower(); a, b = int(s["t0"] * SR), int(s["t1"] * SR); d = (b - a) / SR
    if d <= 0: continue
    seg = None
    if "rain" in kw or "storm" in kw or "thunder" in kw:
        seg = A.sfx_rain(d, seed=2)[: b - a] + A.sfx_wind(d, seed=1)[: b - a]
    elif "wind" in kw or "breeze" in kw:
        seg = A.sfx_wind(d, seed=4)[: b - a]
    elif "leaf" in kw or "leaves" in kw or "rustl" in kw:
        seg = rustle(d)
    if seg is not None:
        seg = np.pad(seg, (0, max(0, (b - a) - len(seg))))[: b - a]
        e = np.minimum(1, np.minimum(np.arange(b - a) / (0.4 * SR), np.arange(b - a)[::-1] / (0.4 * SR))).astype(np.float32)
        sfx[a:b] += seg * e; used.append(s["id"])
rep["sfx_shots"] = used
mix, mrep = A.master(voice, music=music, sfx=sfx if used else None, target=-14.0, pad_gap_db=P.get("pad_gap_db", 20.0), sfx_gap_db=22.0)
rep["audio"] = mrep
sf.write("/tmp/mix.wav", np.stack([mix, mix], 1), SR)
name = P["name"]
final = f"{OUT}/{name}.mp4"
run(f"ffmpeg -v error -y -i /tmp/captioned.mp4 -i /tmp/mix.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -ar 48000 -shortest -movflags +faststart {final}")
# verify loudness of the delivered file
run(f"ffmpeg -v error -y -i {final} -vn -ac 2 -ar 48000 /tmp/check.wav")
import pyloudnorm as pyln
x, sr = sf.read("/tmp/check.wav", dtype="float64")
rep["audio"]["delivered_lufs"] = round(pyln.Meter(sr).integrated_loudness(x), 2)
rep["duration_s"] = round(float(run(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {final}").stdout or 0), 2)

# ---------------------------------------------------------------- srt, thumbnail, contact sheet, SEO pack, QA report
def ts(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"
with open(f"{OUT}/{name}.srt", "w") as f:
    for i, c in enumerate(cap):
        f.write(f"{i + 1}\n{ts(c['a'])} --> {ts(c['b'])}\n{' '.join(c['words'])}\n\n")
rep["srt_cues"] = len(cap)

def frame_at(t, p=final):
    run(f"ffmpeg -v error -y -ss {t:.2f} -i {p} -frames:v 1 /tmp/f.png"); return Image.open("/tmp/f.png").convert("RGB")
th_shot = next((s for s in P["shots"] if s["id"] == P.get("thumb_shot")), P["shots"][0])
run(f"ffmpeg -v error -y -ss {(th_shot['t0'] + th_shot['t1']) / 2:.2f} -i /tmp/picture.mp4 -frames:v 1 /tmp/th.png")
th = Image.open("/tmp/th.png").convert("RGBA")
hk = brand.hook_sprite("A HEARTWICK STORY", P.get("hook_lines", ["She saved his life.", "Then she let him go."]), P.get("hook_highlight", "go."))
th.alpha_composite(hk, (0, int(HT * 0.10)))
th.convert("RGB").save(f"{OUT}/thumbnail.jpg", quality=92)
tiles = []
for s in P["shots"]:
    im = frame_at((s["t0"] + s["t1"]) / 2); im.thumbnail((270, 480)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 270, 22], fill=(0, 0, 0)); d.text((4, 4), f"{s['id']} {s['t0']:.1f}-{s['t1']:.1f}s", fill="white"); tiles.append(im)
cols = 7; rows = (len(tiles) + cols - 1) // cols
S = Image.new("RGB", (cols * 274, rows * 484), "white")
for i, t in enumerate(tiles): S.paste(t, ((i % cols) * 274 + 2, (i // cols) * 484 + 2))
S.save(f"{OUT}/contact_sheet.jpg", quality=88)
json.dump(P.get("seo", {}), open(f"{OUT}/SEO_PACK.json", "w"), indent=1)
rep.update(shots=len(parts), shots_per_min=round(len(parts) / (total / 60), 1), name=name, captions=cap)
rep["stages"]["total_s"] = round(time.time() - T0, 1)
json.dump(rep, open(f"{OUT}/qa_report.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "captions"}, indent=1))
