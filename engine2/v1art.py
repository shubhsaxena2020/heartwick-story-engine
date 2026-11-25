"""Art source A: engine v2 motion on the V1 painted storybook art (SDXL + IP-Adapter, Shubh's preferred look).

Scenes: a V1 plate (bg_<loc>.png) + its Depth-Anything map (bg_<loc>_depth.png, bright = near) become 3 parallax
layers: an inpainted base (depth 0.3), a mid band (0.65) and a near band (0.95), feathered, plus engine v2's particles,
rays and mist. Characters: V1-style rig-ready painted figures auto-rigged on Kaggle (characters/<name>/v1rig/rig/),
squirrels as sprite rigs cut from the V1 masters (head, body, tail: they bound, twitch and flick).
"""
import json, math, os
import numpy as np
import skia
from . import fx as FX
from .scene import Scene, Layer, ACTORS
from .rig import Bone, Skeleton, RasterPart, Rig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATE_DIRS = [os.path.join(ROOT, "cache", "pip", "assets"), os.path.join(ROOT, "cache", "pip", "kaggle_out", "pip_full", "assets")]


def plate_paths(loc, dirs=None):
    for d in dirs or PLATE_DIRS:
        p = os.path.join(d, f"bg_{loc}.png")
        if os.path.exists(p):
            dp = os.path.join(d, f"bg_{loc}_depth.png")
            return p, dp if os.path.exists(dp) else None
    return None, None


def _to_skia(rgba):
    img = skia.Image.fromarray(np.ascontiguousarray(rgba), skia.kRGBA_8888_ColorType, skia.kUnpremul_AlphaType)
    return FX.native(img)


def _layers_from_depth(rgb, depth):
    """-> base (inpainted where near things stood), mid RGBA, near RGBA (feathered by depth)."""
    import cv2
    d = depth.astype(np.float32) / 255.0
    d = cv2.GaussianBlur(d, (0, 0), 3)
    a_mid = np.clip((d - 0.42) / 0.12, 0, 1)
    a_near = np.clip((d - 0.68) / 0.1, 0, 1)
    hole = (a_mid > 0.5).astype(np.uint8) * 255
    small = cv2.resize(rgb, (rgb.shape[1] // 3, rgb.shape[0] // 3), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    hs = cv2.dilate(hs, np.ones((9, 9), np.uint8))
    fill = cv2.inpaint(small, hs, 9, cv2.INPAINT_TELEA)
    fill = cv2.resize(fill, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur((hole > 0).astype(np.float32), (0, 0), 6)[..., None]
    base = (rgb * (1 - m) + fill * m).astype(np.uint8)
    mid = np.dstack([rgb, (a_mid * (1 - a_near * 0.0) * 255).astype(np.uint8)])
    near = np.dstack([rgb, (a_near * 255).astype(np.uint8)])
    return base, mid, near


def load_v1_scene(loc, painter, kind_kw, seed=1, fx_override=None, plate_dir=None):
    from PIL import Image
    p, dp = plate_paths(loc, [plate_dir] if plate_dir else None)
    if p is None:
        return None
    im = np.asarray(Image.open(p).convert("RGB"))
    depth = np.asarray(Image.open(dp).convert("L")) if dp else np.full(im.shape[:2], 80, np.uint8)
    if depth.shape != im.shape[:2]:
        depth = np.asarray(Image.fromarray(depth).resize((im.shape[1], im.shape[0])))
    base, mid, near = _layers_from_depth(im, depth)
    Ww = 1400.0
    k = Ww / im.shape[1]
    Wh = im.shape[0] * k
    gy = Wh * 0.9
    imgs = {"base": _to_skia(np.dstack([base, np.full(base.shape[:2], 255, np.uint8)])), "mid": _to_skia(mid), "near": _to_skia(near)}
    margin = 260

    def plate(name, over=0.0):
        img = imgs[name]

        def draw(c):
            c.save()
            c.translate(-over * Ww / 2, -over * Wh / 2)
            c.scale(k * (1 + over), k * (1 + over))
            p_ = skia.Paint(AntiAlias=True)
            p_.setFilterQuality(skia.FilterQuality.kHigh_FilterQuality)
            c.drawImage(img, 0, 0, p_)
            c.restore()
        return draw

    parts = list(fx_override or kind_kw.get("fx_list", ("motes",)))
    pfx = [FX.Particles(f, (-200, 0, Ww + 200, Wh), seed=seed + i) for i, f in enumerate(parts)
           if f in ("leaves", "motes", "snow", "rain", "fireflies", "petals")]
    tod = kind_kw.get("tod", "day")

    def light(c, t):
        if tod in ("day", "golden", "morning", "dawn"):
            FX.god_rays(c, t, (Ww * 0.25, Wh * 0.08), 62, Wh * 1.1, spread=30, n=6, alpha=0.1, seed=seed)
        if kind_kw.get("mist") or "mist" in parts:
            FX.mist(c, t, gy - 300, Ww, 220, alpha=0.28, seed=seed)
        if "rain" in parts:
            FX.splashes(c, t, -100, Ww + 100, gy - 10, n=60, seed=seed)
        for q in pfx:
            q.draw(c, t)

    layers = [Layer("base", 0.3, plate("base", 0.06)), Layer("mid", 0.65, plate("mid", 0.03)), Layer("near", 0.95, plate("near")),
              Layer(ACTORS, 1.0), Layer("light", 1.05, None, light)]
    spots = {"left": (Ww * 0.3, gy), "center": (Ww * 0.5, gy), "right": (Ww * 0.7, gy)}
    return Scene(loc, (Ww, Wh), layers, gy, painter, spots, margin=margin, tall=Wh > 2300)


# ---------------------------------------------------------------- characters
def find_v1_rig(name, root=ROOT):
    d = os.path.join(root, "characters", name, "v1rig", "rig")
    return d if os.path.exists(os.path.join(d, "rig.json")) else None


def squirrel_sprite_rig(name, root=ROOT, src=None):
    """A painted squirrel (V1 master cut-out) as head / body / tail parts on the squirrel skeleton."""
    from PIL import Image
    p = src or os.path.join(root, "characters", name, "master_cut.png")
    if not os.path.exists(p):
        return None
    a = np.asarray(Image.open(p).convert("RGBA"))
    ys, xs = np.where(a[..., 3] > 40)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    H, W = a.shape[:2]
    alpha = a[..., 3] > 40
    cols = alpha.sum(0)
    # face right: the head side is the narrower, higher end; V1 masters face left, so mirror when the head is left
    tail_left = alpha[: H // 2, : W // 3].sum() > alpha[: H // 2, 2 * W // 3:].sum()
    head_left = not tail_left
    if head_left:
        a = a[:, ::-1].copy()
        alpha = alpha[:, ::-1]
    yy, xx = np.mgrid[0:H, 0:W]
    tail = alpha & (xx < W * 0.42) & (yy < H * 0.85)
    head = alpha & (xx > W * 0.5) & (yy < H * 0.45) & ~tail
    if head.sum() < 0.03 * alpha.sum():  # odd silhouettes: take the upper front of whatever is not tail
        head = alpha & ~tail & (yy < np.percentile(yy[alpha], 40)) & (xx > np.percentile(xx[alpha], 55))
    body = alpha & ~tail & ~head

    def part(mask, pivot, cap=None):
        m = mask.copy()
        if not m.any():
            return None, None
        if cap is not None:  # overlap disc so rotation never opens a gap
            m |= alpha & (np.hypot(xx - cap[0], yy - cap[1]) < W * 0.08)
        y0, y1 = np.where(m.any(1))[0][[0, -1]]
        x0, x1 = np.where(m.any(0))[0][[0, -1]]
        rgba = a[y0:y1 + 1, x0:x1 + 1].copy()
        rgba[..., 3] = np.where(m[y0:y1 + 1, x0:x1 + 1], rgba[..., 3], 0)
        return _to_skia(rgba), (pivot[0] - x0, pivot[1] - y0)

    units = 260.0
    k = units / H
    haunch = (W * 0.42, H * 0.8)
    neck = (W * 0.6, H * 0.4)
    tail_base = (W * 0.38, H * 0.72)
    B = Bone
    sk = Skeleton([B("body", None), B("head", "body", ((neck[0] - haunch[0]) * k, (neck[1] - haunch[1]) * k)),
                   B("paw_f", "body", (0, 0)), B("paw_n", "body", (0, 0)), B("leg_f", "body", (0, 0)), B("leg_n", "body", (0, 0)),
                   B("tail1", "body", ((tail_base[0] - haunch[0]) * k, (tail_base[1] - haunch[1]) * k)),
                   B("tail2", "tail1", (-40, 0)), B("tail3", "tail2", (-36, 0))])
    parts = []
    for bone, z, mask, piv, cap in (("tail1", 3, tail, tail_base, tail_base), ("body", 10, body, haunch, None), ("head", 14, head, neck, neck)):
        img, anc = part(mask, piv, cap)
        if img is not None:  # drawn upright as painted (the neutral pose cancels the pose angles)
            parts.append(RasterPart(bone, z, img, anc, scale=k, rot=0))
    contacts = [("body", (0.0, (H - haunch[1]) * k)), ("body", ((W * 0.75 - haunch[0]) * k, (H - haunch[1]) * k))]
    from .poses import SQUIRREL
    neutral = dict(SQUIRREL["sit"] if name == "pip" else SQUIRREL["curl"])  # the sprite is already painted in this pose
    return Rig(f"{name}_v1", sk, parts, contacts, units=units, kind="squirrel",
               meta={"pelvis_h": (H - haunch[1]) * k, "source": "v1-sprite", "neutral": neutral})
