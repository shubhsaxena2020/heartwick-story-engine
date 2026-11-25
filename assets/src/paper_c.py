"""Style C (paper theatre) part sheets from the rich vector part sheets: every part gets cut-paper fibre grain, a thin
gold-leaf edge, and a brass split-pin where it hinges (its joint pivot), so the joints become part of the look.
engine v2's PaperTheatre painter adds the paper drop shadow between pieces when it draws raster parts.
python assets/src/paper_c.py <src characters dir> <dst characters dir> [name ...]"""
import json, os, shutil, sys
import cv2
import numpy as np
from PIL import Image

PIN_PARTS = ("uarm", "farm", "hand", "thigh", "shin", "foot", "head", "paw", "leg", "tail", "scarf", "cape", "hair", "skirt_n", "skirt_f",
             "hem")


def grain(h, w, seed=5):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, (h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=2.2, sigmaY=0.7)  # fibres run horizontally
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    return 0.86 + 0.14 * g


def pin(img, x, y, r):
    """Domed brass split-pin with a highlight and the slot."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - x, yy - y)
    m = np.clip(r + 0.8 - d, 0, 1)
    hx, hy = x - r * 0.35, y - r * 0.4
    t = np.clip(np.hypot(xx - hx, yy - hy) / (r * 1.3), 0, 1)[..., None]
    c0, c1, c2 = np.array([255, 241, 176]), np.array([217, 167, 58]), np.array([122, 82, 16])
    col = np.where(t < 0.45, c0 + (c1 - c0) * (t / 0.45), c1 + (c2 - c1) * ((t - 0.45) / 0.55))
    sh = np.clip(r * 1.15 + 1.5 - np.hypot(xx - x - r * 0.25, yy - y - r * 0.35), 0, 1)[..., None] * 0.35
    img[..., :3] = img[..., :3] * (1 - sh) + 30 * sh
    img[..., :3] = img[..., :3] * (1 - m[..., None]) + col * m[..., None]
    img[..., 3] = np.maximum(img[..., 3], m * 255)
    slot = (np.abs(yy - y) < max(1.0, r * 0.12)) & (np.abs(xx - x) < r * 0.55)
    img[slot, :3] = [90, 58, 12]


def paperize(src_png, dst_png, anchor, part_name, q_px):
    im = np.asarray(Image.open(src_png).convert("RGBA")).astype(np.float32)
    pad = int(q_px * 6) + 4
    im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)))
    ax, ay = anchor[0] + pad, anchor[1] + pad
    H, W = im.shape[:2]
    a = im[..., 3] / 255
    # flatten the airbrush slightly toward cut paper and add the fibre grain
    rgb = im[..., :3]
    blur = cv2.GaussianBlur(rgb, (0, 0), 3)
    rgb = rgb * 0.75 + blur * 0.25
    rgb *= grain(H, W)[..., None]
    # gold-leaf edge along the silhouette
    am = (a > 0.5).astype(np.uint8)
    w = max(2, int(round(q_px * 0.9)))
    edge = am - cv2.erode(am, np.ones((w, w), np.uint8))
    e = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 0.6)[..., None]
    gold = np.array([226, 182, 80], np.float32)
    rgb = rgb * (1 - e * 0.85) + gold * e * 0.85
    im[..., :3] = np.clip(rgb, 0, 255)
    if part_name.startswith(PIN_PARTS):
        pin(im, ax, ay, max(4.5, q_px * 5.0))
    Image.fromarray(im.astype(np.uint8), "RGBA").save(dst_png, optimize=True)
    return [anchor[0] + pad, anchor[1] + pad]


def convert(src, dst, name):
    s, d = os.path.join(src, name, "rig"), os.path.join(dst, name, "rig")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    J = json.load(open(os.path.join(s, "rig.json")))
    q = 1.0 / J["units_per_px"]
    for p in J["parts"]:
        p["anchor_px"] = paperize(os.path.join(s, p["image"]), os.path.join(d, p["image"]), p["anchor_px"], p["name"], q)
    for pn, vs in J.get("variants", {}).items():
        for vn, v in vs.items():
            v["anchor_px"] = paperize(os.path.join(s, v["image"]), os.path.join(d, v["image"]), v["anchor_px"], pn, q)
    J["source"] = J.get("source", "") + " + paper-theatre finish (assets/src/paper_c.py)"
    json.dump(J, open(os.path.join(d, "rig.json"), "w"), indent=1)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    for n in sys.argv[3:] or sorted(os.listdir(src)):
        if os.path.exists(os.path.join(src, n, "rig", "rig.json")):
            convert(src, dst, n)
            print("C:", n)
