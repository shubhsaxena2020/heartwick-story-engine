"""Build research/round4/kaggle/tpu/push/main.py for the TPU v5e-8 kernel heartwick-r4-tpu."""
import base64, io, json, os, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "bench"))
b = io.BytesIO(); Image.open(os.path.join(ROOT, "out/pip_short_v3/kaggle/keys/out/keys/K2_s21.png")).convert("RGB").save(b, "JPEG", quality=96)
IMGS = {"K2.jpg": base64.b64encode(b.getvalue()).decode()}
V = "2D hand-drawn anime animation, clean line art, warm watercolor background. "
PROMPT = (V + "Very slow push-in on Nell, who sits still with her hands around the mug and looks down. Steam curls up from the mug. "
          "Cold light drifts across the table. Snow falls softly outside the window. Her mouth stays closed the whole time, lips "
          "gently pressed together; she is silent and does not speak. She breathes slowly and blinks once.")
body = open(os.path.join(HERE, "main_src.py"), encoding="utf-8").read()
os.makedirs(os.path.join(HERE, "push"), exist_ok=True)
open(os.path.join(HERE, "push", "main.py"), "w", encoding="utf-8").write(f"IMGS = {json.dumps(IMGS)}\nPROMPT = {PROMPT!r}\n" + body)
json.dump({"id": "shubhsaxena2008/heartwick-r4-tpu", "title": "Heartwick R4 Tpu", "code_file": "main.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": True, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(HERE, "push", "kernel-metadata.json"), "w"), indent=1)
print("ok")
