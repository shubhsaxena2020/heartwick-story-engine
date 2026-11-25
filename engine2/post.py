"""Lighting and finishing passes (engine v2 round 2: "stop looking cheap").

  mood grades      per-scene colour matrix + lift: storm dark and cold, home warm, dusk rose, night blue...
  key light        a soft warm (or cool) light pool from the scene's light source
  rim light        a light edge on the characters' lit side (actors are drawn into their own layer)
  bloom            bright parts blurred and screened back (lamps, windows, sun, glows)
  film grain       animated, subtle
  atmosphere       far layers hazed + softly blurred, foreground layer defocused (applied once, to the cached layers)
All passes are a few ms at 1080x1920; the cached-layer ones cost nothing per frame.
"""
import math
import numpy as np
import skia
from .fx import native

# mood -> (colour matrix gains rgb, lift rgb (0-255), saturation, key light colour, key strength, rim colour, rim strength, bloom)
MOODS = {
    "storm": ((0.78, 0.86, 1.04), (4, 8, 18), 0.82, "#9fb8ff", 0.10, "#c9d8ff", 0.55, 0.30),
    "night": ((0.86, 0.9, 1.06), (2, 4, 14), 0.9, "#ffc56b", 0.22, "#ffd9a0", 0.55, 0.55),
    "dawn": ((1.03, 0.98, 0.98), (8, 4, 6), 1.0, "#ffd0a0", 0.18, "#fff0d8", 0.6, 0.45),
    "home": ((1.07, 1.0, 0.9), (10, 5, 0), 1.06, "#ffb860", 0.22, "#ffe2b0", 0.6, 0.45),
    "day": ((1.0, 1.0, 1.0), (4, 4, 4), 1.05, "#fff1c9", 0.12, "#fff6e0", 0.5, 0.35),
    "golden": ((1.08, 0.99, 0.86), (12, 6, 0), 1.08, "#ffc070", 0.26, "#ffe0a0", 0.75, 0.55),
    "dusk": ((1.02, 0.97, 0.97), (6, 2, 6), 1.0, "#ffb488", 0.12, "#ffd2b0", 0.6, 0.45),
    "winter": ((0.95, 1.0, 1.06), (6, 8, 14), 0.92, "#e8f2ff", 0.10, "#ffffff", 0.45, 0.35),
    "morning": ((0.97, 1.0, 1.02), (6, 6, 8), 0.98, "#fff1d8", 0.12, "#fff6e8", 0.5, 0.35),
}


def mood_of(kind, kw):
    """Location builder parameters -> mood name."""
    tod, weather, season = kw.get("tod", "day"), kw.get("weather"), kw.get("season", "autumn")
    if tod == "storm" or weather == "storm":
        return "storm"
    if tod == "night":
        return "night"
    if kind == "interior" and tod in ("day", "morning", "dawn"):
        return "home"
    if tod in ("dusk", "golden", "dawn", "morning"):
        return tod
    if season == "winter":
        return "winter"
    return "day"


def _matrix(gains, lift, sat):
    r, g, b = gains
    lr, lg, lb = (x / 255.0 for x in lift)
    s = sat
    # saturation around luma, then gains, then lift (skia's 4x5 matrix, translate in 0..1)
    lw = (0.2126, 0.7152, 0.0722)
    m = []
    for i, gain in enumerate((r, g, b)):
        row = [(1 - s) * lw[j] + (s if i == j else 0.0) for j in range(3)]
        m += [gain * row[0], gain * row[1], gain * row[2], 0.0, (lr, lg, lb)[i]]
    m += [0, 0, 0, 1, 0]
    return m


class Finisher:
    """Per-film finishing state (grain tiles, scratch surfaces)."""

    def __init__(self, W, H, seed=0, grain=0.06, gains=None):
        self.W, self.H = W, H
        self.gains = {"key": 1.0, "rim": 1.0, "bloom": 1.0}
        self.gains.update(gains or {})
        rng = np.random.default_rng(seed)
        self.grain_alpha = grain
        self.grain = []
        n = 256
        for _ in range(4):
            g = rng.normal(0.5, 0.22, (n, n)).clip(0, 1)
            v = (g * 255).astype(np.uint8)
            img = skia.Image.fromarray(np.ascontiguousarray(np.dstack([v, v, v, np.full_like(v, 255)])), skia.kRGBA_8888_ColorType)
            self.grain.append(native(img).makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat))
        self.small = skia.Surface.MakeRasterN32Premul(max(1, W // 4), max(1, H // 4))
        self.actor_surf = skia.Surface.MakeRasterN32Premul(W, H)
        self._grades = {}

    def grade_paint(self, mood):
        if mood not in self._grades:
            gains, lift, sat = MOODS[mood][:3]
            p = skia.Paint()
            p.setColorFilter(skia.ColorFilters.Matrix(_matrix(gains, lift, sat)))
            self._grades[mood] = p
        return self._grades[mood]

    def key_light(self, c, mood, light_xy):
        col, k = MOODS[mood][3], MOODS[mood][4] * self.gains["key"]
        x, y = light_xy
        r = max(self.W, self.H) * 0.9
        h = col.lstrip("#")
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        p = skia.Paint(AntiAlias=True)
        p.setShader(skia.GradientShader.MakeRadial((x, y), r, [skia.Color(*rgb, int(255 * k)), skia.Color(*rgb, 0)]))
        p.setBlendMode(skia.BlendMode.kSoftLight)
        c.drawRect(skia.Rect(0, 0, self.W, self.H), p)

    def rim(self, c, img, mood, light_xy):
        """Lit edge on the actors: the actor layer tinted, minus itself shifted away from the light."""
        col, k = MOODS[mood][5], min(1.0, MOODS[mood][6] * self.gains["rim"])
        x, y = light_xy
        dx, dy = self.W / 2 - x, self.H * 0.45 - y
        L = math.hypot(dx, dy) or 1
        off = 5.0 * self.W / 1080
        ox, oy = dx / L * off, dy / L * off
        h = col.lstrip("#")
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        tint = skia.Paint()
        tint.setColorFilter(skia.ColorFilters.Matrix([0, 0, 0, 0, rgb[0], 0, 0, 0, 0, rgb[1], 0, 0, 0, 0, rgb[2], 0, 0, 0, k, 0]))
        cut = skia.Paint()
        cut.setBlendMode(skia.BlendMode.kDstOut)
        scr = skia.Paint()
        scr.setBlendMode(skia.BlendMode.kScreen)
        c.saveLayer(None, scr)
        c.drawImage(img, 0, 0, tint)
        c.drawImage(img, ox, oy, cut)
        c.restore()

    def bloom(self, c, frame_img, mood):
        strength = min(1.0, MOODS[mood][7] * self.gains["bloom"])
        sc = self.small.getCanvas()
        sc.clear(skia.ColorBLACK)
        p = skia.Paint()
        # keep only highlights: (x - 0.72) * 3.5 per channel
        p.setColorFilter(skia.ColorFilters.Matrix([3.5, 0, 0, 0, -2.5, 0, 3.5, 0, 0, -2.5, 0, 0, 3.5, 0, -2.5, 0, 0, 0, 1, 0]))
        p.setImageFilter(skia.ImageFilters.Blur(6, 6))
        p.setFilterQuality(skia.FilterQuality.kLow_FilterQuality)
        sc.save()
        sc.scale(0.25, 0.25)
        sc.drawImage(frame_img, 0, 0, p)
        sc.restore()
        small = self.small.makeImageSnapshot()
        q = skia.Paint()
        q.setBlendMode(skia.BlendMode.kScreen)
        q.setAlphaf(strength)
        q.setFilterQuality(skia.FilterQuality.kMedium_FilterQuality)
        c.save()
        c.scale(4, 4)
        c.drawImage(small, 0, 0, q)
        c.restore()

    def film_grain(self, c, frame_index):
        p = skia.Paint()
        sh = self.grain[frame_index % 4]
        m = skia.Matrix()
        m.setTranslate((frame_index * 37) % 256, (frame_index * 91) % 256)
        m.preScale(self.W / 1080 * 1.5, self.W / 1080 * 1.5)
        p.setShader(sh.makeWithLocalMatrix(m))
        p.setBlendMode(skia.BlendMode.kOverlay)
        p.setAlphaf(self.grain_alpha)
        c.drawRect(skia.Rect(0, 0, self.W, self.H), p)


def atmosphere(img, depth, haze_color, q):
    """Cached-layer treatment: far layers get haze + a soft blur, the foreground a depth-of-field blur."""
    if 0.36 <= depth <= 1.1 or img is None:
        return img
    s = skia.Surface.MakeRasterN32Premul(img.width(), img.height())
    c = s.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    p = skia.Paint()
    sigma = (2.4 if depth < 0.36 else 7.0) * q
    p.setImageFilter(skia.ImageFilters.Blur(sigma, sigma))
    c.drawImage(img, 0, 0, p)
    if 0.12 < depth < 0.36:  # haze only on land layers, not the sky gradient itself
        h = haze_color.lstrip("#")
        hp = skia.Paint(Color=skia.Color(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 70))
        hp.setBlendMode(skia.BlendMode.kSrcATop)
        c.drawRect(skia.Rect(0, 0, img.width(), img.height()), hp)
    return s.makeImageSnapshot()
