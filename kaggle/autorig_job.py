# Raster puppet source for engine v2 (Kaggle CPU kernel): rig-ready AI character image -> part sheet.
# Prepended by tools/autorig.py:  JOB = {"images": {"<name>": "<base64 png>"}}  (or images in an attached dataset)
#   1. BiRefNet cut-out (MIT)            2. RTMPose COCO-17 keypoints via rtmlib/onnxruntime (Apache-2.0)
#   3. engine2.autorig: capsule parts with joint caps, torso filled behind the limbs with LaMa (Apache-2.0)
# Output: /kaggle/working/<name>/rig/{rig.json, *.png} + preview.png (rest pose and three library poses).
import base64, io, json, os, subprocess, sys, time

W = "/kaggle/working"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rtmlib", "onnxruntime", "simple-lama-inpainting", "skia-python==87.7",
                "timm", "kornia", "einops"], check=False)
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

sys.path.insert(0, "/kaggle/tmp_rig")
os.makedirs("/kaggle/tmp_rig/engine2", exist_ok=True)
for name, src in ENGINE_FILES.items():  # noqa: F821  (engine2 modules shipped inline)
    open(f"/kaggle/tmp_rig/engine2/{name}", "w").write(src)
from engine2 import autorig as AR, rig as RG, poses as PO, style as ST  # noqa: E402

seg = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).eval()
tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
from rtmlib import Body
body = Body(mode="balanced", backend="onnxruntime", device="cpu")
try:
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()

    def inpaint(rgb, hole):
        out = lama(Image.fromarray(rgb), Image.fromarray((hole * 255).astype(np.uint8)))
        return np.asarray(out.convert("RGB").resize((rgb.shape[1], rgb.shape[0])))
except Exception as e:  # noqa: BLE001
    log("LaMa unavailable, OpenCV Telea fill:", repr(e)[:200])
    inpaint = None

report = {}
for name, b64 in JOB["images"].items():  # noqa: F821
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    with torch.no_grad():
        m = seg(tf(im).unsqueeze(0))[-1].sigmoid()[0, 0].numpy()
    alpha = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(im.size, Image.LANCZOS))
    rgba = np.dstack([np.asarray(im), alpha])
    kps, scores = body(np.asarray(im)[..., ::-1])
    if len(kps) == 0:
        report[name] = "no person found"
        continue
    k = kps[int(np.argmax(scores.mean(1)))]
    kp = {n: (float(k[i][0]), float(k[i][1])) for i, n in enumerate(AR.COCO)}
    if kp["r_shoulder"][0] < kp["l_shoulder"][0] and kp["nose"][0] < np.mean([kp["l_ear"][0], kp["r_ear"][0]]):
        rgba = rgba[:, ::-1].copy()  # facing left -> mirror so the rig faces right
        kp = {n: (rgba.shape[1] - 1 - x, y) for n, (x, y) in kp.items()}
        kp = {(n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")): v for n, v in kp.items()}
    out = f"{W}/{name}/rig"
    J = AR.autorig(rgba, kp, out, name=name, inpaint=inpaint)
    r = RG.load_partsheet(out)
    import skia
    Wd, Hd = 1200, 700
    s = skia.Surface.MakeRasterN32Premul(Wd, Hd)
    c = s.getCanvas()
    c.clear(skia.Color(225, 228, 232))
    for i, pose in enumerate(["_image", "point", "kneel_reach", "heart"]):
        root = skia.Matrix()
        root.setTranslate(150 + i * 300, 420)
        root.preScale(0.5, 0.5)
        ang = RG.image_pose_angles(out) if pose == "_image" else PO.to_bones(PO.HUMAN[pose])
        r.draw(c, ang, root, ST.VectorCel())
    s.makeImageSnapshot().save(f"{W}/{name}/preview.png")
    report[name] = {"parts": len(J["parts"]), "near_side": J["near_side"], "units_per_px": round(J["units_per_px"], 4)}
    log(name, report[name])
json.dump(report, open(f"{W}/autorig_report.json", "w"), indent=1)
log("DONE")
