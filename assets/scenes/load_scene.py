"""Turn a layered scene folder (scene.json + layer images, made by assets/src/scenes.py) into an engine v2 Scene.

    from assets.scenes.load_scene import load_scene
    scene = load_scene("assets/scenes/maple_yard", painter)   # drop-in for engine2.scenery.build_location(...)

Static layers are the PNG/JPG plates (engine2.scene caches each once per render scale); the moving bits come from
engine v2's own fx (particles, god rays, mist, twinkling stars, lightning flash, birds), configured in scene.json.
This file only reads engine2; it does not change it.
"""
import json, math, os
import numpy as np
import skia
from engine2.scene import Scene, Layer, ACTORS
from engine2 import fx as FX


def _static(path, margin, q):
    img = skia.Image.open(path)

    def draw(c):
        c.save()
        c.translate(-margin, -margin)
        c.scale(1 / q, 1 / q)
        p = skia.Paint(AntiAlias=True)
        p.setFilterQuality(skia.FilterQuality.kHigh_FilterQuality)
        c.drawImage(img, 0, 0, p)
        c.restore()
    return draw


def load_scene(folder, painter, seed=1, fx_override=None):
    J = json.load(open(os.path.join(folder, "scene.json")))
    W, H = J["world"]
    m, q = J.get("margin", 260), J.get("q", 1.0)
    cfg = J.get("fx", {})
    parts = list(fx_override or cfg.get("particles", []))
    pfx = [FX.Particles(k, (-200, 0, W + 200, H), seed=seed + i) for i, k in enumerate(parts)
           if k in ("leaves", "motes", "snow", "rain", "fireflies", "petals")]
    rng = np.random.default_rng(seed)
    tw = cfg.get("twinkle")
    twinkle = [(rng.uniform(0, W), rng.uniform(0, tw.get("y_max", H * 0.4)), rng.uniform(1, 2.6), rng.uniform(0, 6.28))
               for _ in range(tw.get("n", 60))] if tw else []

    def sky_dyn(c, t):
        pa = skia.Paint(AntiAlias=True)
        for x, y, r, ph in twinkle:
            a = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t + ph))
            pa.setColor(skia.Color(255, 244, 214, int(255 * a)))
            c.drawCircle(x, y, r, pa)
        lt = cfg.get("lightning")
        if lt and (t % lt.get("period", 4.7)) < lt.get("flash", 0.12):
            pa.setColor(skia.Color(223, 232, 255, 90))
            c.drawRect(skia.Rect(-400, -400, W + 400, H), pa)
        if cfg.get("birds"):
            bp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=4, Color=skia.Color(58, 48, 70, 200))
            bp.setStrokeCap(skia.Paint.kRound_Cap)
            for k in range(5):
                x = (t * 70 + k * 60 + seed * 137) % (W + 1000) - 500
                y = J["ground"] - 1400 + 40 * math.sin(k * 1.7) + 12 * math.sin(0.5 * t + k)
                f = 14 * math.sin(2 * math.pi * 2.2 * t + k)
                p = skia.Path()
                p.moveTo(x - 20, y - f)
                p.lineTo(x, y)
                p.lineTo(x + 20, y - f)
                c.drawPath(p, bp)

    def light(c, t):
        r = cfg.get("rays")
        if r:
            FX.god_rays(c, t, tuple(r["origin"]), r.get("angle", 62), r.get("length", 2200), spread=34, n=7, alpha=r.get("alpha", 0.12),
                        seed=seed, **({"color": r["color"]} if "color" in r else {}))
        g = cfg.get("glow")
        if g:
            FX.glow(c, g["at"][0], g["at"][1], g["r"], g["color"], g["alpha"] + 0.02 * math.sin(t * 0.8))
        mi = cfg.get("mist")
        if mi:
            FX.mist(c, t, mi["y"], W, 200, alpha=mi.get("alpha", 0.3), seed=seed)
        for p in pfx:
            p.draw(c, t)

    layers = []
    for i, L in enumerate(J["layers"]):
        if L["name"] == "actors":
            layers.append(Layer(ACTORS, 1.0))
            continue
        dyn = sky_dyn if i == 0 else None
        layers.append(Layer(L["name"], L["depth"], _static(os.path.join(folder, L["image"]), m, q), dyn))
    layers.append(Layer("light", 1.05, None, light))
    spots = {k: tuple(v) for k, v in J.get("spots", {}).items()}
    return Scene(J["name"], (W, H), layers, J["ground"], painter, spots, margin=m, tall=J.get("tall", False))
