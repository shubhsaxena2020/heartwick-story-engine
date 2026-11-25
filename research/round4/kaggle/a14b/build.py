"""Build research/round4/kaggle/a14b/push/main.py (heartwick-r4-a14b)."""
import base64, io, json, os
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
b = io.BytesIO(); Image.open(os.path.join(ROOT, "out/pip_short_v3/kaggle/keys/out/keys/K2_s21.png")).convert("RGB").save(b, "JPEG", quality=96)
IMGS = {"K2.jpg": base64.b64encode(b.getvalue()).decode()}
gen = open(os.path.join(HERE, "gen_flf_r4.py"), encoding="utf-8").read()
body = open(os.path.join(HERE, "main_src.py"), encoding="utf-8").read()
os.makedirs(os.path.join(HERE, "push"), exist_ok=True)
open(os.path.join(HERE, "push", "main.py"), "w", encoding="utf-8").write(f"IMGS = {json.dumps(IMGS)}\nGEN_SRC = {gen!r}\n" + body)
json.dump({"id": "shubhsaxena2008/heartwick-r4-a14b", "title": "Heartwick R4 A14b", "code_file": "main.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": ["shubhsaxena2008/heartwick-r4-bench"], "model_sources": []},
          open(os.path.join(HERE, "push", "kernel-metadata.json"), "w"), indent=1)
print("ok")
