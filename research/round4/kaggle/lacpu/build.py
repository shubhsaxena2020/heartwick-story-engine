"""CPU-only Kaggle kernel heartwick-r4-lacpu: the limited-animation renderer (la.py) on Kaggle's 4-vCPU CPU session."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, "..", "bench")
sys.path.insert(0, B)
src = open(os.path.join(B, "build.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(B, "build.py")}
exec(src.split("src = {n:")[0], ns)                        # KEYS, b64, LA from the bench build
la = open(os.path.join(B, "la.py"), encoding="utf-8").read()
body = '''
import base64, json, os, subprocess, time
from pathlib import Path
W = Path("/kaggle/working"); OUT = W / "out"; IN = W / "in"; OUT.mkdir(exist_ok=True); IN.mkdir(exist_ok=True)
for k, b in IMGS.items():
    (IN / k).write_bytes(base64.b64decode(b))
(W / "la.py").write_text(LA_SRC)
t = time.time()
print(subprocess.run("pip install -q 'rembg[cpu]' opencv-python-headless 2>&1 | tail -2; nproc; free -g | head -2", shell=True, capture_output=True, text=True).stdout)
setup_s = round(time.time() - t, 1)
(W / "shots.json").write_text(json.dumps([dict(s, image=str(IN / s["image"])) for s in LA]))
t = time.time()
r = subprocess.run(f"python {W/'la.py'} {W/'shots.json'} {OUT/'la'} 2>&1 | tail -40", shell=True, capture_output=True, text=True)
print(r.stdout)
json.dump({"setup_s": setup_s, "la_wall_s": round(time.time() - t, 1)}, open(OUT / "status.json", "w"))
subprocess.run(f"cd {OUT}/la && for f in *.mp4; do ffmpeg -v error -y -i $f -vf \\"select=not(mod(n\\,18)),scale=240:-2,tile=8x1\\" -frames:v 1 ${{f%.mp4}}_sheet.jpg; done", shell=True)
'''
os.makedirs(os.path.join(HERE, "push"), exist_ok=True)
open(os.path.join(HERE, "push", "main.py"), "w", encoding="utf-8").write(
    f"IMGS = {json.dumps({k: ns['b64'](v) for k, v in ns['KEYS'].items()})}\nLA = {json.dumps(ns['LA'])}\nLA_SRC = {la!r}\n" + body)
json.dump({"id": "shubhsaxena2008/heartwick-r4-lacpu", "title": "Heartwick R4 Lacpu", "code_file": "main.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(HERE, "push", "kernel-metadata.json"), "w"), indent=1)
print("ok")
