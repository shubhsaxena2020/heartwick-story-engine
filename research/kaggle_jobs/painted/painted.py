# Heartwick painted (V1 storybook) assets for engine v2, on Kaggle T4 x2.
# Same look as V1: SDXL base + sdxl-vae-fp16-fix + IP-Adapter plus (h94), V1 prompt prefix/negatives, hires pass.
#   GPU0: humans in a rig-ready pose (OpenPose ControlNet, xinsir, Apache-2.0), V1 masters as the IP-Adapter reference
#         (king and elder: InstantStyle from brand/style_ref.png). 4 seeds each.
#   GPU1: animals steered by an edge map of their vector rig (canny ControlNet, xinsir, Apache-2.0), then background
#         plates (plain SDXL + InstantStyle, like V1).
#   then: BiRefNet cut-outs, RTMPose keypoints + engine2.autorig with LaMa fill (humans), Depth-Anything-V2 depth and
#         a LaMa-filled layer split (plates), contact sheets.
import json, os, subprocess, sys, threading, time, textwrap

W = "/kaggle/working"
IN = "/kaggle/input/heartwick-painted-inputs"
for d in ("chars", "animals", "plates", "rigs", "layers", "logs", "sheets"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
T0 = time.time()

STYLE_PREFIX = "storybook illustration, flat 2D cartoon, gouache, soft shading"
NEG = ("photo, photograph, photorealistic, realistic, 3d render, cgi, anime, text, letters, watermark, signature, "
       "logo, blurry, lowres, deformed, bad anatomy, extra limbs, candle, flame, fire")
CHAR_SUFFIX = "full body, plain light gray background"
CHAR_NEG = NEG + ", scenery, landscape, room, multiple characters, cropped, close-up, arms crossed, hands in pockets, holding objects"
BG_SUFFIX = "empty scene, no people, vertical composition"
BG_NEG = NEG + ", people, person, human, animal, squirrel, character"
RIG = "standing in side three-quarter view facing right, arms held slightly away from the body, hands open, mid-stride, feet apart"

HUMANS = {  # desc from stories/pip.md (V1), plus the folk-tale cast; ref = V1 master (identity) or style ref
    "nell": ("elderly woman, silver hair in a low bun, round glasses, kind wrinkled face, green knitted cardigan, cream blouse, "
             "long gray skirt, white apron", "master_nell", 0.55, 1101),
    "dee": ("woman in her fifties, long brown braid, gray fleece vest over red plaid shirt, blue jeans, green rubber boots",
            "master_dee", 0.55, 4404),
    "eli": ("nine year old boy, messy sandy blond hair, freckles, gap-toothed grin, red zip jacket, blue jeans, white sneakers",
            "master_eli", 0.55, 5505),
    "king": ("kind old fairy-tale king, short gray beard, small gold crown with red jewels, deep blue tunic with gold embroidery, "
             "red velvet cloak with white fur trim, brown boots", "style_ref", 0.5, 6606),
    "elder": ("wise old man with a long white beard, deep green hooded robe with embroidered hem, rope belt, leather satchel, "
              "simple sandals", "style_ref", 0.5, 7707),
}
ANIMALS = {  # control = edge map of the vector rig in the same pose the painted rig will rest in
    "pip": ("a young eastern gray squirrel, side view facing right, crouching on all fours, big dark shiny eye with a white eye ring, "
            "fluffy silver gray fur with a soft brown wash on the back, cream white belly, long bushy gray tail with a silvery white fringe "
            "curving up behind", "master_pip", 0.45, 3303),
    "pip_kit": ("a tiny baby eastern gray squirrel, side view facing right, crouching, short soft gray fuzz, eyes almost closed, "
                "small pink nose and paws, short thin fluffy gray tail", "master_pip_kit", 0.45, 2202),
    "hare": ("a brown hare, side view facing right, crouching, long upright ears with black tips, warm brown fur, cream belly, "
             "big dark eye, small white tail", "style_ref", 0.45, 8808),
}
PLATES = {  # (prompt, w, h, style scale, seed)  tall plates for tilt reveals
    "village_autumn": ("a small Vermont village in autumn at golden hour, a white church with a tall steeple, white clapboard houses, "
                       "a red covered bridge, red and orange sugar maples, rolling hills and blue mountains, a winding road in front",
                       704, 1472, 0.6, 9101),
    "maple_yard": ("the backyard of a white New England farmhouse in autumn morning light, a huge old red sugar maple tree, "
                   "a white picket fence, piles of fallen red leaves on green grass, a red barn and hills behind", 832, 1216, 0.6, 9202),
    "kitchen": ("a cozy old farmhouse kitchen in warm morning light, a window with an autumn maple outside, red checked curtains, "
                "wooden shelves with blue and white plates, copper pots, a wooden table, geraniums on the sill, wooden floor",
                832, 1216, 0.6, 9303),
    "night_sky": ("a quiet Vermont village at night under a sky full of stars and a big full moon, the milky way, dark blue hills, "
                  "small white houses with warm lit windows, bare trees", 704, 1472, 0.3, 9404),
    "storm": ("a dark stormy autumn night, swirling storm clouds, lightning, rain, a huge old maple tree bending in the wind, "
              "a small white farmhouse with one warm lit window", 832, 1216, 0.25, 9505),
    "castle_court": ("a fairy-tale castle on a hill at golden hour, towers with blue and red pointed roofs and banners, a cobbled "
                     "town square with half-timbered houses, a stone fountain, colourful bunting", 832, 1216, 0.55, 9606),
}
FORE = {"fore_leaves": ("a row of fallen red and orange maple leaves and tall autumn grass tufts along the bottom, plain light gray background", 9701),
        "fore_grass_dark": ("a row of dark green grass tufts and fern leaves along the bottom, plain light gray background", 9702)}
CONST = dict(W=W, IN=IN, STYLE_PREFIX=STYLE_PREFIX, NEG=NEG, CHAR_SUFFIX=CHAR_SUFFIX, CHAR_NEG=CHAR_NEG, BG_SUFFIX=BG_SUFFIX,
             BG_NEG=BG_NEG, RIG=RIG, HUMANS=HUMANS, ANIMALS=ANIMALS, PLATES=PLATES, FORE=FORE)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def run(name, code, gpu, timeout=7200):
    src = f"/tmp/{name}.py"
    open(src, "w").write("CONST = " + repr(CONST) + "\nglobals().update(CONST)\n" + textwrap.dedent(code))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    t = time.time()
    with open(f"{W}/logs/{name}.log", "w") as f:
        try:
            rc = subprocess.run([sys.executable, src], stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    log(name, "rc", rc, round(time.time() - t), "s")
    return rc


COMMON = r'''
import json, os, time, torch, numpy as np
from PIL import Image, ImageDraw
from diffusers import (StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL,
                       AutoPipelineForImage2Image)
from transformers import CLIPVisionModelWithProjection
dev = "cuda"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16)
def make(cn_id):
    cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=torch.float16)
    p = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, vae=vae,
            image_encoder=enc, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(dev)
    p.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
    p.vae.enable_tiling(); p.set_progress_bar_config(disable=True)
    return p
def ip(p, ref, scale):
    if ref == "style_ref":
        p.set_ip_adapter_scale({"up": {"block_0": [0.0, scale, 0.0]}})  # InstantStyle (V1 plates)
    else:
        p.set_ip_adapter_scale(scale)
    return Image.open(f"{IN}/{ref}.png").convert("RGB")
def hires(p, img, prompt, neg, seed, ref, scale, f=1.5, strength=0.3):
    i2i = AutoPipelineForImage2Image.from_pipe(StableDiffusionXLPipeline.from_pipe(p))
    big = img.resize((int(img.width * f) // 8 * 8, int(img.height * f) // 8 * 8), Image.LANCZOS)
    ipimg = ip(i2i, ref, scale)
    return i2i(prompt=prompt, negative_prompt=neg, image=big, strength=strength, num_inference_steps=24, guidance_scale=6.5,
               ip_adapter_image=ipimg, generator=torch.Generator(dev).manual_seed(seed)).images[0]
'''

HUMAN_GEN = COMMON + r'''
COL = [[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],[0,255,0],[0,255,85],[0,255,170],[0,255,255],
       [0,170,255],[0,85,255],[0,0,255],[85,0,255],[170,0,255],[255,0,255],[255,0,170],[255,0,85]]
LIMBS = [[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[1,0],[0,14],[14,16],[0,15],[15,17]]
def pose_image(kp, w, h):
    """OpenPose-18 drawing. kp: list of 18 (x, y) or None, in OpenPose order."""
    import math
    im = Image.new("RGB", (w, h), (0, 0, 0)); d = ImageDraw.Draw(im, "RGBA")
    for i, (a, b) in enumerate(LIMBS):
        if kp[a] is None or kp[b] is None: continue
        (x1, y1), (x2, y2) = kp[a], kp[b]
        L = math.hypot(x2 - x1, y2 - y1); ang = math.atan2(y2 - y1, x2 - x1); sw = 7
        pts = []
        for t in range(0, 360, 20):
            r = math.radians(t)
            ex, ey = (L / 2) * math.cos(r), sw * math.sin(r)
            pts.append(((x1 + x2) / 2 + ex * math.cos(ang) - ey * math.sin(ang), (y1 + y2) / 2 + ex * math.sin(ang) + ey * math.cos(ang)))
        d.polygon(pts, fill=tuple(COL[i]) + (153,))
    for i, p in enumerate(kp):
        if p is not None: d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=tuple(COL[i]))
    return im
# facing right, 3/4: the near side is the character's LEFT. 832 x 1216, ~8 heads tall.
def rig_pose(child=False):
    s = 0.86 if child else 1.0
    def P(x, y): return (416 + (x - 416) * s, 1150 - (1150 - y) * s)
    kp = [P(472, 188), P(420, 262), P(392, 282), P(345, 420), P(300, 548), P(444, 286), P(500, 420), P(572, 540),
          P(400, 600), P(380, 830), P(350, 1062), P(432, 600), P(472, 830), P(506, 1062), None, P(456, 172), None, P(412, 182)]
    return kp
p = make("xinsir/controlnet-openpose-sdxl-1.0")
for name, (desc, ref, scale, seed) in HUMANS.items():
    ctrl = pose_image(rig_pose(name == "eli"), 832, 1216)
    ctrl.save(f"{W}/chars/{name}_pose.png")
    prompt = f"{STYLE_PREFIX}, {desc}, {RIG}, {CHAR_SUFFIX}"
    for k in range(4):
        t = time.time()
        img = p(prompt=prompt, negative_prompt=CHAR_NEG, image=ctrl, controlnet_conditioning_scale=0.85, width=832, height=1216,
                num_inference_steps=30, guidance_scale=6.5, ip_adapter_image=ip(p, ref, scale),
                generator=torch.Generator(dev).manual_seed(seed + k * 1000)).images[0]
        img = hires(p, img, prompt, CHAR_NEG, seed + k * 1000, ref, scale)
        img.save(f"{W}/chars/{name}_c{k}.png")
        print("GEN", name, k, round(time.time() - t, 1), flush=True)
'''

ANIMAL_PLATE_GEN = COMMON + r'''
p = make("xinsir/controlnet-canny-sdxl-1.0")
for name, (desc, ref, scale, seed) in ANIMALS.items():
    ctrl = Image.open(f"{IN}/ctrl_{name}.png").convert("RGB")
    prompt = f"{STYLE_PREFIX}, {desc}, full body, plain light gray background"
    for k in range(4):
        t = time.time()
        img = p(prompt=prompt, negative_prompt=CHAR_NEG, image=ctrl, controlnet_conditioning_scale=0.6, control_guidance_end=0.7,
                width=1024, height=1024, num_inference_steps=30, guidance_scale=6.5, ip_adapter_image=ip(p, ref, scale),
                generator=torch.Generator(dev).manual_seed(seed + k * 1000)).images[0]
        img = hires(p, img, prompt, CHAR_NEG, seed + k * 1000, ref, scale, f=1.25, strength=0.28)
        img.save(f"{W}/animals/{name}_c{k}.png")
        print("GEN", name, k, round(time.time() - t, 1), flush=True)
base = StableDiffusionXLPipeline.from_pipe(p)
del p; torch.cuda.empty_cache()
base = base.to(dev)
for name, (desc, w, h, scale, seed) in PLATES.items():
    prompt = f"{STYLE_PREFIX}, {desc}, rich warm light, detailed, {BG_SUFFIX}"
    for k in range(2):
        t = time.time()
        img = base(prompt=prompt, negative_prompt=BG_NEG, width=w, height=h, num_inference_steps=30, guidance_scale=6.5,
                   ip_adapter_image=ip(base, "style_ref", scale), generator=torch.Generator(dev).manual_seed(seed + k * 1000)).images[0]
        img = hires(base, img, prompt, BG_NEG, seed + k * 1000, "style_ref", scale, f=1.5, strength=0.33)
        img.save(f"{W}/plates/{name}_c{k}.png")
        print("GEN", name, k, round(time.time() - t, 1), flush=True)
for name, (desc, seed) in FORE.items():
    img = base(prompt=f"{STYLE_PREFIX}, {desc}", negative_prompt=BG_NEG, width=1216, height=832, num_inference_steps=30,
               guidance_scale=6.5, ip_adapter_image=ip(base, "style_ref", 0.6), generator=torch.Generator(dev).manual_seed(seed)).images[0]
    img.save(f"{W}/plates/{name}.png")
'''

POST = r'''
import glob, json, os, sys, io, numpy as np, torch
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation, pipeline
sys.path.insert(0, "/tmp/eng")
os.makedirs("/tmp/eng/engine2", exist_ok=True)
for f in glob.glob(IN + "/engine2__*.py"):
    open("/tmp/eng/engine2/" + os.path.basename(f).split("__", 1)[1], "w").write(open(f).read())
from engine2 import autorig as AR, rig as RG, poses as PO, style as ST
seg = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to("cuda").eval().half()
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def cut(im):
    with torch.no_grad():
        m = seg(tf(im).unsqueeze(0).to("cuda").half())[-1].sigmoid().float().cpu()[0, 0].numpy()
    a = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(im.size, Image.LANCZOS))
    return np.dstack([np.asarray(im.convert("RGB")), a])
import cv2
try:
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()
    def inpaint(rgb, hole):
        out = lama(Image.fromarray(rgb), Image.fromarray((hole * 255).astype(np.uint8)))
        return np.asarray(out.convert("RGB").resize((rgb.shape[1], rgb.shape[0])))
except Exception as e:
    print("LaMa unavailable -> Telea", repr(e)[:200], flush=True)
    def inpaint(rgb, hole):
        return cv2.inpaint(np.ascontiguousarray(rgb), (hole * 255).astype(np.uint8), 9, cv2.INPAINT_TELEA)
from rtmlib import Body
body = Body(mode="performance", backend="onnxruntime", device="cpu")
report = {}
# ---- humans: cut-out, keypoints, auto-rig, preview
for f in sorted(glob.glob(W + "/chars/*_c*.png")):
    tag = os.path.basename(f)[:-4]
    im = Image.open(f).convert("RGB")
    rgba = cut(im)
    Image.fromarray(rgba).save(W + f"/chars/{tag}_cut.png")
    kps, sc = body(np.asarray(im)[..., ::-1])
    if len(kps) == 0:
        report[tag] = "no person"; continue
    i = int(np.argmax(sc.mean(1)))
    kp = {n: [float(kps[i][j][0]), float(kps[i][j][1])] for j, n in enumerate(AR.COCO)}
    json.dump({"keypoints": kp, "scores": [float(s) for s in sc[i]]}, open(W + f"/chars/{tag}_kp.json", "w"))
    out = W + f"/rigs/{tag}"
    try:
        J = AR.autorig(rgba, {k: tuple(v) for k, v in kp.items()}, out, name=tag.split("_c")[0], inpaint=inpaint)
        r = RG.load_partsheet(out)
        import skia
        s = skia.Surface.MakeRasterN32Premul(1500, 760); c = s.getCanvas(); c.clear(skia.Color(225, 228, 232))
        for k2, pose in enumerate(["_image", "point", "kneel_reach", "sit_hold", "walk"]):
            root = skia.Matrix(); root.setTranslate(150 + k2 * 300, 720); root.preScale(0.62, 0.62)
            sem = PO.walk(0.25) if pose == "walk" else (PO.HUMAN.get(pose, {}))
            ang = RG.image_pose_angles(out) if pose == "_image" else PO.to_bones(sem)
            root.preTranslate(0, -r.meta.get("pelvis_h", 470))
            r.draw(c, ang, root, ST.VectorCel())
        s.makeImageSnapshot().save(W + f"/rigs/{tag}_preview.png")
        report[tag] = {"parts": len(J["parts"]), "kp_score": round(float(sc[i].mean()), 3)}
    except Exception as e:
        import traceback; traceback.print_exc(); report[tag] = repr(e)[:200]
    print("RIG", tag, report[tag], flush=True)
# ---- animals: cut-outs only (parts are cut locally with the vector rig's masks)
for f in sorted(glob.glob(W + "/animals/*_c*.png")):
    Image.fromarray(cut(Image.open(f).convert("RGB"))).save(f[:-4] + "_cut.png")
for f in glob.glob(W + "/plates/fore_*.png"):
    Image.fromarray(cut(Image.open(f).convert("RGB"))).save(f[:-4] + "_cut.png")
# ---- plates: depth + layer split (sky/far/mid/ground), LaMa fill behind nearer layers
de = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
import cv2
for f in sorted(glob.glob(W + "/plates/*_c*.png")):
    tag = os.path.basename(f)[:-4]
    im = Image.open(f).convert("RGB"); rgb = np.asarray(im)
    d = np.asarray(de(im)["depth"].resize(im.size, Image.BICUBIC)).astype(np.float32)
    d = cv2.GaussianBlur(d, (0, 0), 3); d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    Image.fromarray((d * 255).astype(np.uint8)).save(W + f"/layers/{tag}_depth.png")
    qs = np.quantile(d, [0.3, 0.55, 0.75])
    band = np.digitize(d, qs)  # 0 far .. 3 near
    od = W + f"/layers/{tag}"; os.makedirs(od, exist_ok=True)
    for k in range(4):
        nearer = band > k
        own = band == k
        if k == 0:  # back plate: opaque, nearer parts painted out
            hole = cv2.dilate(nearer.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            out = inpaint(rgb, hole); a = np.full(own.shape, 255, np.uint8)
        else:
            ring = (cv2.dilate(nearer.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & nearer
            hole = cv2.dilate(nearer.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            filled = inpaint(rgb, hole)
            out = np.where(own[..., None], rgb, filled)
            a = ((own | ring) * 255).astype(np.uint8)
            a = cv2.GaussianBlur(a, (0, 0), 1.5)
        Image.fromarray(np.dstack([out, a])).save(f"{od}/L{k}.png")
    print("LAYERS", tag, [round(float(q), 3) for q in qs], flush=True)
json.dump(report, open(W + "/rigs/report.json", "w"), indent=1)
# ---- contact sheets
def sheet(files, out, h=480):
    ims = [Image.open(f).convert("RGB") for f in files]
    ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    S = Image.new("RGB", (sum(i.width for i in ims) + 6 * len(ims), h), (30, 30, 30)); x = 0
    for i in ims: S.paste(i, (x, 0)); x += i.width + 6
    S.save(out, quality=85)
for n in HUMANS: sheet(sorted(glob.glob(W + f"/chars/{n}_c?.png")), W + f"/sheets/{n}.jpg")
for n in ANIMALS: sheet(sorted(glob.glob(W + f"/animals/{n}_c?.png")), W + f"/sheets/{n}.jpg")
sheet(sorted(glob.glob(W + "/plates/*_c?.png")), W + "/sheets/plates.jpg", h=420)
print("POST DONE", flush=True)
'''

if __name__ == "__main__":
    subprocess.run("pip install -q -U diffusers accelerate rtmlib onnxruntime fire && pip install -q --no-deps simple-lama-inpainting && pip install -q "
                   "skia-python==87.7 timm kornia einops > /tmp/pip.log 2>&1; tail -3 /tmp/pip.log", shell=True)
    log("installed")
    res = {}
    t1 = threading.Thread(target=lambda: res.__setitem__("humans", run("humans", HUMAN_GEN, 0)))
    t2 = threading.Thread(target=lambda: res.__setitem__("animals_plates", run("animals_plates", ANIMAL_PLATE_GEN, 1)))
    t1.start(); t2.start(); t1.join(); t2.join()
    res["post"] = run("post", POST, 0, timeout=5400)
    res["_wall_s"] = round(time.time() - T0)
    json.dump(res, open(f"{W}/logs/status.json", "w"), indent=1)
    log(res)
