"""Build research/round4/kaggle/keys/push/main.py (FLUX.2-klein keyframe benchmark, heartwick-r4-keys)."""
import base64, io, json, os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
K = "out/pip_short_v3/kaggle/keys/out/keys/"
SRC = {"sheet_nell.jpg": K + "sheet_nell_s11.png", "sheet_pip.jpg": K + "sheet_pip_s11.png", "K0.jpg": K + "K0.png",
       "K2.jpg": K + "K2_s21.png", "K10.jpg": "out/pip_short_v3/kaggle/keys-pip-short-r3/out/keys/K10_s7.png"}


def b64(p):
    b = io.BytesIO(); im = Image.open(os.path.join(ROOT, p)).convert("RGB"); im.thumbnail((1024, 1024)); im.save(b, "JPEG", quality=90)
    return base64.b64encode(b.getvalue()).decode()


STYLE = ("Cozy 2D hand-drawn anime storybook illustration, clean confident line art, soft watercolor background painting, "
         "warm gentle light, Ghibli-like.")
NELL = "Nell, the elderly woman from the reference: silver hair in a bun, round glasses, green cardigan, cream blouse, long rust-red skirt"
JOBS = [
    {"id": "new_garden_1088", "refs": ["sheet_nell.jpg", "K0.jpg"], "w": 1088, "h": 1920, "seeds": [3, 21],
     "prompt": f"{STYLE} Image 1 is the character sheet of {NELL}. Image 2 shows the art style and palette. Paint a new complete "
               "vertical scene: Nell stands in her autumn garden beside a big red maple tree at golden hour, holding a watering can, "
               "looking up at falling leaves with a calm closed-mouth smile. Full body, medium-wide shot."},
    {"id": "new_window_768", "refs": ["sheet_nell.jpg", "sheet_pip.jpg"], "w": 768, "h": 1344, "seeds": [3, 21],
     "prompt": f"{STYLE} Image 1 is the character sheet of {NELL}. Image 2 is the character sheet of Pip, a small grey squirrel. "
               "Paint a new complete vertical scene: in a cozy kitchen at night, Nell sits at a wooden table with a mug of tea while "
               "Pip sits on the table next to the mug, both looking at each other, warm lamplight, rain on the window."},
    {"id": "edit_blink_K2", "refs": ["K2.jpg"], "w": 768, "h": 1344, "seeds": [3],
     "prompt": "Keep this exact image identical in every detail, framing and style. Change only this: the woman's eyes are gently "
               "closed (a blink), eyelids down behind her glasses. Mouth stays closed."},
    {"id": "edit_mouth_K2", "refs": ["K2.jpg"], "w": 768, "h": 1344, "seeds": [3],
     "prompt": "Keep this exact image identical in every detail, framing and style. Change only this: the woman's mouth is slightly "
               "open as if softly saying a word. Everything else unchanged."},
    {"id": "edit_closeup_K10", "refs": ["K10.jpg"], "w": 1088, "h": 1920, "seeds": [3],
     "prompt": "Redraw this exact scene as a high-resolution, crisp, clean anime illustration with the same characters, composition, "
               "colors and lighting; finer line art and painted detail. Nobody speaks; mouths closed."},
]
IMGS = {k: b64(v) for k, v in SRC.items()}
body = open(os.path.join(HERE, "kmain_src.py"), encoding="utf-8").read()
os.makedirs(os.path.join(HERE, "push"), exist_ok=True)
open(os.path.join(HERE, "push", "main.py"), "w", encoding="utf-8").write(f"IMGS = {json.dumps(IMGS)}\nJOBS = {json.dumps(JOBS)}\n" + body)
json.dump({"id": "shubhsaxena2008/heartwick-r4-keys", "title": "Heartwick R4 Keys", "code_file": "main.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": ["shubhsaxena2008/heartwick-r4-wcache"], "model_sources": []},
          open(os.path.join(HERE, "push", "kernel-metadata.json"), "w"), indent=1)
print("bytes", os.path.getsize(os.path.join(HERE, "push", "main.py")))
