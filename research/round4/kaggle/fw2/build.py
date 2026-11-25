"""Build research/round4/kaggle/fw2/push/main.py (heartwick-r4-fw2): FastWan run 2 + FLUX.2-klein keys."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); KD = os.path.join(HERE, "..", "keys")
ns = {"__file__": os.path.join(KD, "build.py")}
exec(open(os.path.join(KD, "build.py"), encoding="utf-8").read().split("IMGS = {k: b64(v)")[0], ns)
IMGS = {k: ns["b64"](v) for k, v in ns["SRC"].items()}
ksrc = open(os.path.join(KD, "kmain_src.py"), encoding="utf-8").read()
GEN = re.search(r"GEN = r'''(.*?)'''", ksrc, re.S).group(1)
J = lambda i, img, p, w, h, f=121, seed=7: {"id": i, "image": img, "prompt": p, "w": w, "h": h, "frames": f, "seed": seed, "fps": 24}
FW = [J("fw480_k2_plain_s7", "K2.jpg", "k2_plain", 480, 832), J("fw480_k2_mouth_s7", "K2.jpg", "k2_mouth", 480, 832),
      J("fw480_k10_mouth_s7", "K10.jpg", "k10_mouth", 480, 832), J("fw480_k2_mouth_s11", "K2.jpg", "k2_mouth", 480, 832, seed=11),
      J("fw704_k2_mouth_s7", "K2.jpg", "k2_mouth", 704, 1280)]
body = open(os.path.join(HERE, "main_src.py"), encoding="utf-8").read()
fw = open(os.path.join(HERE, "fw_gen2.py"), encoding="utf-8").read()
os.makedirs(os.path.join(HERE, "push"), exist_ok=True)
open(os.path.join(HERE, "push", "main.py"), "w", encoding="utf-8").write(
    f"IMGS = {json.dumps(IMGS)}\nJOBS = {json.dumps(ns['JOBS'])}\nFW = {json.dumps(FW)}\nGEN_SRC = {GEN!r}\nFW_SRC = {fw!r}\n" + body)
json.dump({"id": "shubhsaxena2008/heartwick-r4-fw2", "title": "Heartwick R4 Fw2", "code_file": "main.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [],
           "kernel_sources": ["shubhsaxena2008/heartwick-r4-bench", "shubhsaxena2008/heartwick-r4-wcache"], "model_sources": []},
          open(os.path.join(HERE, "push", "kernel-metadata.json"), "w"), indent=1)
print("bytes", os.path.getsize(os.path.join(HERE, "push", "main.py")))
