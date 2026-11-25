"""Prototype B: open-weight AI keyframe + open-weight image-to-video, run through public
Hugging Face ZeroGPU Spaces with NO login (anonymous quota, ~2 GPU-min/day per IP).

Models (both permissive licences, output usable commercially):
  - Z-Image-Turbo (Tongyi-MAI, Apache-2.0)      -> keyframe
  - Wan2.2-I2V-A14B + Lightning (Apache-2.0)    -> 5 s image-to-video

Usage: python -I zerogpu_test.py <outdir>
"""
import json, shutil, sys, time
from pathlib import Path
from gradio_client import Client, handle_file

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
log = {}

KEY_PROMPT = (
    "Hand-drawn 2D anime film still, Studio-Ghibli-like warm watercolor backgrounds, clean line art, "
    "vertical 9:16 frame. Autumn New England yard: a big red-orange maple tree on the right, golden leaves "
    "falling, white farmhouse with picket fence far left, soft late-afternoon sun rays. "
    "Nell, a kind older woman about 70, silver hair in a low bun, round glasses, dark green knitted cardigan "
    "over a cream blouse, long rust-brown skirt, brown shoes, walks slowly across the leaf-covered grass "
    "toward the maple, full body visible, mid-stride. At the foot of the maple trunk sits a small grey squirrel "
    "looking up at her. No text, no watermark."
)
VID_PROMPT = (
    "2D hand-drawn anime animation. The older woman in the green cardigan takes two slow steps to the maple tree, "
    "kneels down on the leaves, and gently lifts the small grey squirrel in both hands, smiling. "
    "Golden maple leaves keep falling. The camera slowly pushes in and drifts right. Smooth natural motion, "
    "consistent character, no morphing."
)

t0 = time.time()
zi = Client("Tongyi-MAI/Z-Image-Turbo", verbose=False)
res = zi.predict(KEY_PROMPT, "720x1280 ( 9:16 )", 20261121, 8, 3.0, False, [], api_name="/generate")
gal = res[0]
img_path = gal[0]["image"] if isinstance(gal[0]["image"], str) else gal[0]["image"]["path"]
shutil.copy(img_path, out / "keyframe.png")
log["keyframe_s"] = round(time.time() - t0, 1)
print("keyframe", log, flush=True)

t1 = time.time()
wan = Client("zerogpu-aoti/wan2-2-fp8da-aoti-faster", verbose=False)
vres = wan.predict(handle_file(str(out / "keyframe.png")), VID_PROMPT, 6,
                   "static, morphing, extra limbs, blurry, text, watermark", 5.0, 1, 1, 7, False,
                   api_name="/generate_video")
vpath = vres[0]["video"] if isinstance(vres[0], dict) else vres[0]
shutil.copy(vpath, out / "wan22_i2v.mp4")
log["video_s"] = round(time.time() - t1, 1)
log["prompts"] = {"keyframe": KEY_PROMPT, "video": VID_PROMPT}
(out / "run.json").write_text(json.dumps(log, indent=1))
print("done", log, flush=True)
