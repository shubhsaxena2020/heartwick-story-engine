"""Re-finish raw shot takes (RIFE + Real-ESRGAN, both T4s) in a small GPU kernel, from the raw takes that clips kernels
already made: no Wan generation. Used 24 Nov to move the finish from 30 to 60 fps and to finish S08/S10 (the fix
kernel's GPU-0 finisher hit the ESRGAN weight race).
  python -m engine3.refinish out/pip_short_v3 S01=heartwick-v3-clips-pip-short:S01_s7,...   (default: from state)
Writes refinish.json {slug, made}; then python -m engine3.refinish out/pip_short_v3 --merge (pipeline stopped)."""
import json, os, re, sys, time
from . import kaggle3 as K, stages as ST
from .pipeline import State, _od

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle", "clips_main.py")


def fin_src():
    s = open(SRC, encoding="utf-8").read()
    return re.search(r"FIN = textwrap\.dedent\(r'''(.*?)'''\)", s, re.S).group(1)


def main():
    out = sys.argv[1]; st = State(out)
    ks = os.path.basename(out).rsplit("_v3", 1)[0].replace("_", "-")
    side = os.path.join(out, "refinish.json")
    if "--merge" in sys.argv:
        r = json.load(open(side))
        cs = dict(st.get("clips", "sources") or {}); cs.update({sid: r["slug"] for sid in r["made"]})
        st.set("clips", sources=cs, refinished=r["made"], fps=60); st.set("assemble", status="pending")
        print("merged", r["made"]); return
    shots = json.load(open(os.path.join(out, "shots.json"), encoding="utf-8"))
    tl = {t["id"]: round(t["t1"] - t["t0"], 3) for t in json.load(open(os.path.join(out, "timeline.json")))}
    takes = dict(x.split("=") for x in sys.argv[2].split(","))            # S01=<kernel slug>:<take name>
    plan, _ = ST.build_clips(shots, json.load(open(os.path.join(out, "timeline.json"))), st.get("keys", "chosen"))
    jobs = [{"take": v.split(":")[1], "kernel": v.split(":")[0], "out": f"/kaggle/working/out/finished/{sid}.mp4",
             "dur": tl[sid], "cam": plan["cams"].get(sid, "push")} for sid, v in sorted(takes.items())]
    script = (f"JOBS = {json.dumps(jobs)!r}\nFIN = {fin_src()!r}\n" + r'''
import json, os, glob, subprocess, time
os.makedirs("/kaggle/working/out/finished", exist_ok=True); os.makedirs("/kaggle/working/logs", exist_ok=True)
subprocess.run("pip install -q -U imageio imageio-ffmpeg safetensors 2>&1 | tail -2", shell=True)
jobs = json.loads(JOBS)
for j in jobs:
    hit = sorted(glob.glob(f"/kaggle/input/**/{j['kernel']}/**/clips/{j['take']}.mp4", recursive=True)) or \
          sorted(glob.glob(f"/kaggle/input/**/{j['take']}.mp4", recursive=True))
    j["src"] = hit[0] if hit else None
    print(j["take"], "->", j["src"], flush=True)
open("/kaggle/working/finish.py", "w").write(FIN)
# fetch the ESRGAN weights ONCE before both finishers start (the race that lost S08/S10)
subprocess.run("wget -q -O /tmp/realesr-animevideov3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", shell=True)
ok = [j for j in jobs if j["src"]]
ps = []
for g, js in {0: ok[0::2], 1: ok[1::2]}.items():
    if js:
        json.dump({"jobs": js}, open(f"/kaggle/working/fin_g{g}.json", "w"))
        ps.append(subprocess.Popen(f"python /kaggle/working/finish.py /kaggle/working/fin_g{g}.json > /kaggle/working/logs/finish_g{g}.log 2>&1",
                                   shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
for p in ps: p.wait()
done = sorted(os.path.basename(x) for x in glob.glob("/kaggle/working/out/finished/*.mp4"))
json.dump({"rc": [p.returncode for p in ps], "done": done, "missing_src": [j["take"] for j in jobs if not j["src"]]},
          open("/kaggle/working/out/status.json", "w"))
print("done", done, flush=True)
''')
    slug = f"{K.PREFIX}refinish-{ks}-{int(time.time()) % 100000}"
    ksrc = sorted({v.split(":")[0] for v in takes.values()})
    while K.gpu_busy()[1] >= 2:
        print(K.now(), "account at 2 GPU kernels; waiting", flush=True); time.sleep(120)
    print("pushing", slug, len(jobs), "shots from", ksrc, len(script), "bytes", flush=True)
    rec = K.push(slug, script, True, kernel_sources=[f"{K.USER}/{s}" for s in ksrc], note="re-finish 60 fps", wait_slot=False)
    fin = K.wait(rec)
    dl = _od(K.download(slug, os.path.join(out, "kaggle", slug.replace(K.PREFIX, ""))))
    stt = json.load(open(os.path.join(dl, "status.json"))) if os.path.exists(os.path.join(dl, "status.json")) else {}
    made = [f[:-4] for f in stt.get("done", [])]
    json.dump({"slug": slug, "made": made, "status": stt, "kaggle_state": fin.get("state")}, open(side, "w"), indent=1)
    print("refinish done:", made, stt.get("rc"), stt.get("missing_src"))


if __name__ == "__main__":
    main()
