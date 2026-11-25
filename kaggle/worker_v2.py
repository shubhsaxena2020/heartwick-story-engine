# Heartwick engine v2 render worker (Kaggle CPU kernel: no GPU queue, no GPU quota).
# The launcher (engine2/kaggle2.py) prepends:  JOB_ZIP_B64 = "<base64 zip: job.json, engine/, engine2/, research/tools/, stories/, tts/>"
# job.json: {"renders": [{"story", "out", "cfg"}], "tts_dirs": [...]}. Writes mp4s, reports, contact sheets to /kaggle/working.
import base64, glob, io, json, os, shutil, subprocess, sys, time, traceback, zipfile

T0 = time.time()
W = "/kaggle/working"
WORK = "/kaggle/tmp_hw2"
os.makedirs(WORK, exist_ok=True)
LOG = {"stages": {}, "errors": []}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sh(cmd):
    log("$", cmd[:200])
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        log("rc", r.returncode, r.stdout[-1500:], r.stderr[-2500:])
    return r.returncode


zipfile.ZipFile(io.BytesIO(base64.b64decode(JOB_ZIP_B64))).extractall(WORK)  # noqa: F821
os.chdir(WORK)
sys.path.insert(0, WORK)
JOB = json.load(open("job.json"))
t = time.time()
sh(f"{sys.executable} -m pip install -q skia-python==87.7 pyloudnorm")
if sh(f"{sys.executable} -c 'import skia'"):
    sh("apt-get -qq update && apt-get -qq install -y libgl1 libegl1 > /dev/null 2>&1")
os.makedirs("brand/fonts", exist_ok=True)
for name, url in (("Nunito-VF.ttf", "ofl/nunito/Nunito%5Bwght%5D.ttf"), ("Fraunces-VF.ttf", "ofl/fraunces/Fraunces%5BSOFT,WONK,opsz,wght%5D.ttf")):
    if not os.path.exists(f"brand/fonts/{name}"):
        subprocess.run(["curl", "-sSL", "-o", f"brand/fonts/{name}", f"https://github.com/google/fonts/raw/main/{url}"])
LOG["stages"]["install"] = round(time.time() - t, 1)

# narration: FLAC in the zip (size limit) or files from the attached dataset -> wav + json in tts/
import soundfile as sf
os.makedirs("tts", exist_ok=True)
for root, _, files in list(os.walk("/kaggle/input")) + list(os.walk("tts_in")):
    for f in files:
        p = os.path.join(root, f)
        if f.startswith("A__"):  # rigs, painted plates, hero clips, music: flattened paths from the asset dataset
            dst = os.path.join(WORK, f[3:].replace("-D-", "/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(p, dst)
        elif f.startswith("u_") and f.endswith(".flac"):
            x, sr = sf.read(p)
            sf.write(os.path.join("tts", f[:-5] + ".wav"), x, sr)
        elif f.startswith("u_") and f.endswith((".json", ".wav")):
            shutil.copy(p, os.path.join("tts", f))
log("tts files", len(os.listdir("tts")))

from engine2 import render as R, qa  # noqa: E402

for job in JOB["renders"]:
    name = job["out"]
    t = time.time()
    try:
        out = f"{W}/{name}.mp4"
        fr = job.get("frames")
        if fr:  # one part of a render split across parallel kernels: video only (+ the full mix on part 0)
            rep = R.render(job["story"], "tts", out, job["cfg"], workers=job.get("workers", 4), frame_range=tuple(fr), with_audio=False)
            if job.get("part", 0) == 0:
                rep.update(R.mix_audio(job["story"], "tts", f"{W}/{name}", job["cfg"]))
            t_off = fr[0] / job["cfg"].get("fps", 30)
        else:
            rep = R.render(job["story"], "tts", out, job["cfg"], workers=job.get("workers", 4))
            rep["loudness_ffmpeg"] = qa.loudness(out)
            t_off = 0.0
        md = qa.motion_density(out, t_end=rep.get("story_s"))
        rep["motion"] = {k: v for k, v in md.items() if k != "per_second"}
        rep["motion_per_second"] = [dict(p, t=p["t"] + t_off) for p in md["per_second"]]
        qa.contact_sheet(out, f"{W}/{name}_sheet.jpg", every=job.get("sheet_every", 0.5), t_end=None, t0=t_off)
        json.dump(rep, open(f"{W}/{name}_report.json", "w"), indent=1)
        if not fr:
            for f in glob.glob(f"{W}/{name}_mix.wav"):
                os.remove(f)
        log(name, "done", {k: v for k, v in rep.items() if k in ("render_s", "s_per_frame", "shots_per_min", "character_motion_seconds_pct")}, rep["motion"]["motion_seconds_pct"])
    except Exception as e:
        traceback.print_exc()
        LOG["errors"].append(f"{name}: {type(e).__name__}: {e}")
    LOG["stages"][name] = round(time.time() - t, 1)
    json.dump(LOG, open(f"{W}/worker_log.json", "w"), indent=1)

LOG["total_s"] = round(time.time() - T0, 1)
json.dump(LOG, open(f"{W}/worker_log.json", "w"), indent=1)
log("DONE", LOG)
