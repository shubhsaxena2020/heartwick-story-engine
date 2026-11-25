"""Particles, god-rays, glows, mist and the vignette. All deterministic in t (seeded), drawn in world px."""
import math
import numpy as np
import skia
from .style import rgb, PALETTE

LEAF_COLS = ["#c8332b", "#e7892e", "#f2c14e", "#9e1f2f", "#d9572a"]


def leaf_path(s=1.0):
    p = skia.Path()
    pts = [(0, -14), (4, -5), (13, -8), (8, 1), (12, 9), (3, 6), (0, 14), (-3, 6), (-12, 9), (-8, 1), (-13, -8), (-4, -5)]
    p.moveTo(pts[0][0] * s, pts[0][1] * s)
    for x, y in pts[1:]:
        p.lineTo(x * s, y * s)
    p.close()
    return p


_LEAF = leaf_path()


class Particles:
    """kind: leaves | motes | snow | rain | fireflies | petals. Area = world rect (x0, y0, x1, y1)."""

    def __init__(self, kind, area, n=None, seed=0, wind=1.0):
        self.kind, self.area, self.wind = kind, area, wind
        n = n or {"leaves": 16, "motes": 46, "snow": 90, "rain": 120, "fireflies": 26, "petals": 18}[kind]
        r = np.random.default_rng(seed)
        self.p = np.column_stack([r.random(n), r.random(n), r.uniform(0.6, 1.4, n), r.uniform(0, 6.283, n), r.integers(0, 5, n)])

    def draw(self, c, t, painter=None):
        x0, y0, x1, y1 = self.area
        w, h = x1 - x0, y1 - y0
        k = self.kind
        paint = skia.Paint(AntiAlias=True)
        for px, py, sc, ph, ci in self.p:
            if k == "leaves" or k == "petals":
                y = y0 + ((py * h + 70 * t * sc) % (h + 80)) - 40
                x = x0 + (px * w + self.wind * 22 * t * sc + 30 * math.sin(0.9 * t * sc + ph)) % w
                c.save()
                c.translate(x, y)
                c.rotate((t * 110 * sc + ph * 57) % 360)
                c.scale(1.4 * sc, 1.4 * sc * abs(math.cos(t * 2.1 * sc + ph)) + 0.25)
                paint.setColor(rgb(LEAF_COLS[int(ci)] if k == "leaves" else "#f6c9d0"))
                c.drawPath(_LEAF, paint)
                c.restore()
            elif k == "motes":
                y = y0 + ((py * h - 14 * t * sc) % h)
                x = x0 + px * w + 16 * math.sin(0.5 * t + ph)
                a = 0.35 + 0.45 * (0.5 + 0.5 * math.sin(1.7 * t * sc + ph))
                r = 2.2 * sc + 1.0
                g = skia.GradientShader.MakeRadial((x, y), r * 3, [rgb(PALETTE["honey"], int(255 * a)), rgb(PALETTE["honey"], 0)])
                gp = skia.Paint(AntiAlias=True)
                gp.setShader(g)
                c.drawCircle(x, y, r * 3, gp)
            elif k == "snow":
                y = y0 + ((py * h + 60 * t * sc) % h)
                x = x0 + (px * w + 18 * math.sin(0.8 * t * sc + ph)) % w
                paint.setColor(rgb("#ffffff", 210))
                c.drawCircle(x, y, 2.6 * sc + 1, paint)
            elif k == "rain":
                y = y0 + ((py * h + 1500 * t * sc) % h)
                x = x0 + (px * w - 240 * t) % w
                paint.setColor(rgb("#cfe0ff", 110))
                paint.setStrokeWidth(2.0)
                c.drawLine(x, y, x - 10, y + 44 * sc, paint)
            elif k == "fireflies":
                x = x0 + px * w + 40 * math.sin(0.4 * t * sc + ph)
                y = y0 + py * h + 30 * math.sin(0.55 * t * sc + ph * 2)
                a = max(0.0, math.sin(1.3 * t * sc + ph)) ** 2
                g = skia.GradientShader.MakeRadial((x, y), 16, [rgb("#fff3a0", int(240 * a)), rgb("#fff3a0", 0)])
                gp = skia.Paint(AntiAlias=True)
                gp.setShader(g)
                c.drawCircle(x, y, 16, gp)


def god_rays(c, t, origin, angle_deg, length, spread=40, n=6, color=PALETTE["honey"], alpha=0.16, seed=0):
    """Soft additive light shafts fanning out from `origin`, slowly breathing."""
    r = np.random.default_rng(seed)
    paint = skia.Paint(AntiAlias=True)
    paint.setBlendMode(skia.BlendMode.kScreen)
    ox, oy = origin
    for i in range(n):
        a = math.radians(angle_deg + (i - n / 2) * spread / n + 3 * math.sin(0.15 * t + i))
        wdt = r.uniform(30, 90) * (1 + 0.15 * math.sin(0.4 * t + i * 1.7))
        al = alpha * (0.55 + 0.45 * math.sin(0.35 * t + i * 2.1)) * r.uniform(0.6, 1.0)
        ex, ey = ox + math.cos(a) * length, oy + math.sin(a) * length
        nx, ny = -math.sin(a), math.cos(a)
        p = skia.Path()
        p.moveTo(ox + nx * 6, oy + ny * 6)
        p.lineTo(ex + nx * wdt, ey + ny * wdt)
        p.lineTo(ex - nx * wdt, ey - ny * wdt)
        p.lineTo(ox - nx * 6, oy - ny * 6)
        p.close()
        paint.setShader(skia.GradientShader.MakeLinear([(ox, oy), (ex, ey)], [rgb(color, int(255 * al)), rgb(color, 0)]))
        c.drawPath(p, paint)


def glow(c, x, y, r, color, alpha=0.5):
    p = skia.Paint(AntiAlias=True)
    p.setShader(skia.GradientShader.MakeRadial((x, y), r, [rgb(color, int(255 * alpha)), rgb(color, 0)]))
    p.setBlendMode(skia.BlendMode.kScreen)
    c.drawCircle(x, y, r, p)


def mist(c, t, y, w, h, color="#dfe6ee", alpha=0.35, speed=8.0, seed=0):
    r = np.random.default_rng(seed)
    p = skia.Paint(AntiAlias=True)
    for i in range(7):
        x = (r.random() * w + speed * t * (0.6 + 0.4 * i / 7)) % (w + 600) - 300
        yy = y + r.uniform(-h / 2, h / 2)
        rx, ry = r.uniform(220, 420), r.uniform(40, 80)
        sh = skia.GradientShader.MakeRadial((x, yy), rx, [rgb(color, int(255 * alpha)), rgb(color, 0)])
        p.setShader(sh)
        c.save()
        c.translate(x, yy)
        c.scale(1, ry / rx)
        c.translate(-x, -yy)
        c.drawCircle(x, yy, rx, p)
        c.restore()


def vignette_image(W, H, strength=0.55, color=(16, 12, 20)):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.47) / (H * 0.6)) ** 2)
    a = np.clip((d - 0.55) / 0.75, 0, 1) ** 1.6 * strength
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = (a * 255).astype(np.uint8)
    # premultiply
    for k in range(3):
        rgba[..., k] = (rgba[..., k].astype(np.float32) * a).astype(np.uint8)
    return native(skia.Image.fromarray(np.ascontiguousarray(rgba), skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType))


def native(img):
    """Re-encode an image in the surface's native pixel format (numpy RGBA images are converted on every draw)."""
    s = skia.Surface.MakeRasterN32Premul(img.width(), img.height())
    s.getCanvas().clear(skia.ColorTRANSPARENT)
    s.getCanvas().drawImage(img, 0, 0)
    return s.makeImageSnapshot()


def splashes(c, t, x0, x1, y, n=40, seed=0, depth_px=160):
    """Rain hitting the ground: small rings that expand and fade, plus a spray tick, each on its own 0.45 s cycle."""
    r = np.random.default_rng(seed)
    pts = np.column_stack([r.uniform(x0, x1, n), r.uniform(0, depth_px, n), r.uniform(0, 1, n), r.uniform(0.35, 0.6, n)])
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=2.0)
    for x, dy, ph, per in pts:
        u = ((t / per) + ph) % 1.0
        a = int(150 * (1 - u))
        p.setColor(rgb("#dfe8ff", a))
        rx = 4 + 16 * u
        c.drawOval(skia.Rect(x - rx, y + dy - rx * 0.28, x + rx, y + dy + rx * 0.28), p)
        if u < 0.3:
            c.drawLine(x, y + dy, x - 3, y + dy - 12 * (1 - u / 0.3), p)
