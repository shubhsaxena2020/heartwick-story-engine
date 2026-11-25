"""Build research/round4/kaggle/bench/push/main.py (keys + scripts + jobs inlined)."""
import base64, io, json, os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
KEYS = {"K2.jpg": "out/pip_short_v3/kaggle/keys/out/keys/K2_s21.png",
        "K10.jpg": "out/pip_short_v3/kaggle/keys-pip-short-r3/out/keys/K10_s7.png"}


def b64(p):
    b = io.BytesIO(); Image.open(os.path.join(ROOT, p)).convert("RGB").save(b, "JPEG", quality=96)
    return base64.b64encode(b.getvalue()).decode()


V = "2D hand-drawn anime animation, clean line art, warm watercolor background. "
MOUTH = (" Her mouth stays closed the whole time, lips gently pressed together; she is silent and does not speak. "
         "She breathes slowly and blinks once.")
PROMPTS = {
    "k2_plain": V + "Very slow push-in on Nell, who sits still with her hands around the mug and looks down. Steam curls up "
                    "from the mug. Cold light drifts across the table. Snow falls softly outside the window.",
    "k2_mouth": V + "Very slow push-in on Nell, who sits still with her hands around the mug and looks down. Steam curls up "
                    "from the mug. Cold light drifts across the table. Snow falls softly outside the window." + MOUTH,
    "k10_mouth": V + "Pip the grey squirrel turns his head toward the open window, then back to Nell; his whiskers twitch and his "
                     "tail sways. Nell watches him with a gentle closed-mouth smile and slowly lowers her hands. Leaves sway "
                     "outside and sunlight flickers." + MOUTH,
}
J = lambda i, img, p, w, h, f=121, seed=7: {"id": i, "image": img, "prompt": p, "w": w, "h": h, "frames": f, "seed": seed, "fps": 24}
JOBS = {"0": [J("fw480_k2_plain_s7", "K2.jpg", "k2_plain", 480, 832),
              J("fw480_k2_mouth_s7", "K2.jpg", "k2_mouth", 480, 832),
              J("fw704_k2_mouth_s7", "K2.jpg", "k2_mouth", 704, 1280)],
        "1": [J("fw480_k10_mouth_s7", "K10.jpg", "k10_mouth", 480, 832),
              J("fw704_k10_mouth_s7", "K10.jpg", "k10_mouth", 704, 1280),
              J("fw480_k2_mouth_s11", "K2.jpg", "k2_mouth", 480, 832, seed=11)]}
# anchors in normalised key coordinates (K2 mug steam at ~(355,425)/480x832; window 0-190 x 0-370)
LA = [{"id": "la_k2", "image": "K2.jpg", "dur": 6.0, "hip_y": 0.60, "steam": [0.735, 0.505], "snow_box": [0.0, 0.0, 0.40, 0.44],
       "cam": {"zoom": 0.07, "pan": -50}, "breath_px": 6, "sway_px": 2.0},
      {"id": "la_k10", "image": "K10.jpg", "dur": 6.0, "hip_y": 0.62, "dust_box": [0.0, 0.62, 0.65, 0.98],
       "cam": {"zoom": 0.05, "pan": 60}, "breath_px": 6, "sway_px": 2.5}]

src = {n: open(os.path.join(HERE, n), encoding="utf-8").read() for n in ("fw_gen.py", "te.py", "la.py")}
body = open(os.path.join(HERE, "main_src.py"), encoding="utf-8").read()
script = "".join([f"IMGS = {json.dumps({k: b64(v) for k, v in KEYS.items()})}\n", f"SRC = {json.dumps(src)}\n",
                  f"PROMPTS = {json.dumps(PROMPTS)}\n", f"JOBS = {json.dumps(JOBS)}\n", f"LA = {json.dumps(LA)}\n", body])
os.makedirs(os.path.join(HERE, "push"), exist_ok=True)
open(os.path.join(HERE, "push", "main.py"), "w", encoding="utf-8").write(script)
json.dump({"id": "shubhsaxena2008/heartwick-r4-bench", "title": "Heartwick R4 Bench", "code_file": "main.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(HERE, "push", "kernel-metadata.json"), "w"), indent=1)
print("script bytes", len(script))
