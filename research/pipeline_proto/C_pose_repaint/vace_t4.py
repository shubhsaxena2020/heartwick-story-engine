# Prototype C on Kaggle T4: deterministic 3D blocking (Blender + CC0 animation library) -> OpenPose control video
# -> Wan2.1-VACE-1.3B (Apache-2.0) repaints it in the house anime style, with Nell's keyframe as the reference image.
# Idea: Claude controls WHAT happens and WHERE (timeline JSON -> Blender), the video model controls HOW it looks.
import base64, io, json, subprocess, time, zipfile
from pathlib import Path
W = Path("/kaggle/working"); W.mkdir(parents=True, exist_ok=True)
POSE_ZIP = "__POSE_ZIP__"; REF_PNG = "__REF_PNG__"
zipfile.ZipFile(io.BytesIO(base64.b64decode(POSE_ZIP))).extractall(W / "pose")
(W / "ref.png").write_bytes(base64.b64decode(REF_PNG))
def sh(c, log, t=None):
    t0 = time.time()
    with open(W / log, "a") as f:
        try: rc = subprocess.run(c, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=t).returncode
        except subprocess.TimeoutExpired: rc = -999
    return rc, round(time.time() - t0, 1)
st = {"setup": sh("pip install -q -U 'diffusers>=0.35' 'transformers>=4.51' accelerate ftfy imageio imageio-ffmpeg", "setup.log", 900)}
CODE = r'''
import time, json, torch, glob
from PIL import Image
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
PROMPT_ = ("2D hand-drawn anime film, Studio-Ghibli-like watercolor backgrounds, clean line art. Autumn New England yard with a big red-orange "
          "maple tree, golden leaves falling, white farmhouse and picket fence in the distance. An older woman with silver hair in a bun, round "
          "glasses, dark green knitted cardigan, cream blouse and long rust-brown skirt walks across the grass, kneels down under the maple and "
          "gently lifts a small grey squirrel in both hands. Smooth natural motion.")
NEG_ = "static, blurry, extra limbs, deformed hands, text, watermark, low quality, photorealistic"
t0 = time.time()
mid = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(mid, vae=vae, torch_dtype=torch.float16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
# v2: encode text once on GPU, then drop the 11 GB umt5 so transformer + VAE fit the T4; tile the VAE
pipe.text_encoder.to("cuda")
with torch.no_grad():
    pe, ne = pipe.encode_prompt(prompt=PROMPT_, negative_prompt=NEG_, do_classifier_free_guidance=True, device="cuda")
pipe.text_encoder.to("cpu"); pipe.text_encoder = None; import gc; gc.collect(); torch.cuda.empty_cache()
pipe.to("cuda"); pipe.vae.enable_tiling()
load_s = time.time() - t0
video = [Image.open(p).convert("RGB") for p in sorted(glob.glob("/kaggle/working/pose/*.png"))]
mask = [Image.new("L", v.size, 255) for v in video]
ref = Image.open("/kaggle/working/ref.png").convert("RGB")
t1 = time.time()
out = pipe(video=video, mask=mask, reference_images=[ref], prompt_embeds=pe, negative_prompt_embeds=ne, height=832, width=480,
           num_frames=81, num_inference_steps=25, guidance_scale=5.0, conditioning_scale=1.0,
           generator=torch.Generator("cpu").manual_seed(3)).frames[0]
gen_s = time.time() - t1
export_to_video(out, "/kaggle/working/c_vace_pose_repaint.mp4", fps=16)
d = {"load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "steps": 25, "frames": 81, "res": "480x832",
     "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
open("/kaggle/working/c.json", "w").write(json.dumps(d)); print(d)
'''
(W / "c.py").write_text(CODE)
st["vace"] = sh(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python {W/'c.py'}", "c.log", 5400)
(W / "status.json").write_text(json.dumps(st, indent=1)); print(st)
