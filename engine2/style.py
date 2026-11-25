"""Art-direction painters. Character parts and scene props call these instead of painting directly, so the same
rig and the same scene description render as

  B  "vector cel":   clean cut-out shapes, soft gradient shading, glows (closest to the reference technique)
  C  "paper theatre": flat cut-paper with grain, gold-leaf edges, paper drop shadows between layers,
                      brass split-pins on every joint, a gilded proscenium in front, stained-glass backlight

Heartwick palette rules (STYLE_BIBLE.md): warm inside / cool outside, jewel accents (ruby, emerald, sapphire,
marigold, gold), never the reference's indigo-and-gold devotional look, never candles or flames.
"""
import math
import numpy as np
import skia


def rgb(h, a=255):
    h = h.lstrip("#")
    return skia.Color(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


def mix(c1, c2, u):
    """Mix two '#rrggbb' strings -> '#rrggbb'."""
    a = [int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(c2.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{int(round(x + (y - x) * u)):02x}" for x, y in zip(a, b))


def shade(c, k):
    """k < 0 darkens toward a warm shadow, k > 0 lightens toward cream."""
    return mix(c, "#2a1426", -k) if k < 0 else mix(c, "#fff4e2", k)


PALETTE = {
    "ruby": "#a3203a", "emerald": "#1f7a5c", "sapphire": "#2b4fa8", "marigold": "#e9a53a", "gold": "#e7b53c",
    "cream": "#fff4e2", "night_pine": "#10233a", "dusk_teal": "#1f4e5c", "maple": "#b8432f", "ember": "#ffa552",
    "honey": "#f3c77a", "sage": "#8da67e",
}


class Painter:
    name = "base"
    joint_style = "cap"

    def __init__(self, seed=0):
        self.seed = seed
        self._paint = skia.Paint(AntiAlias=True)

    # -- primitives every part uses
    def fill(self, c, path, color, shade_k=0.32, kind="cloth", alpha=255, light=(-0.5, -0.8)):
        raise NotImplementedError

    def stroke(self, c, path, color, width, alpha=255, cap=skia.Paint.kRound_Cap):
        p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=width, Color=rgb(color, alpha))
        p.setStrokeCap(cap)
        p.setStrokeJoin(skia.Paint.kRound_Join)
        c.drawPath(path, p)

    def flat(self, c, path, color, alpha=255):
        c.drawPath(path, skia.Paint(AntiAlias=True, Color=rgb(color, alpha)))

    def joint(self, c, x, y, r, color):
        """Round joint cap (tube-limb puppet look)."""
        self.fill(c, circle(x, y, r), color, 0.25)

    def raster_part(self, c, img, x, y, paint):
        c.drawImage(img, x, y, paint)

    def pattern_dots(self, c, path, color, step=16, r=2.6, alpha=200, offset=(0, 0)):
        """Clip to path and scatter a regular dot pattern (dotted fabrics in the reference style)."""
        b = path.computeTightBounds()
        c.save()
        c.clipPath(path, skia.ClipOp.kIntersect, True)
        p = skia.Paint(AntiAlias=True, Color=rgb(color, alpha))
        y = b.top() - step + offset[1] % step
        row = 0
        while y < b.bottom() + step:
            x = b.left() - step + (offset[0] + (step / 2 if row % 2 else 0)) % step
            while x < b.right() + step:
                c.drawCircle(x, y, r, p)
                x += step
            y += step
            row += 1
        c.restore()

    def glow(self, c, x, y, r, color, alpha=0.6):
        sh = skia.GradientShader.MakeRadial((x, y), r, [rgb(color, int(255 * alpha)), rgb(color, 0)])
        p = skia.Paint(AntiAlias=True)
        p.setShader(sh)
        p.setBlendMode(skia.BlendMode.kScreen)
        c.drawCircle(x, y, r, p)

    # scene-level hooks
    def layer_shadow(self):
        return None

    def grade(self, c, W, H, t):
        pass


def circle(x, y, r):
    p = skia.Path()
    p.addCircle(x, y, r)
    return p


def ellipse(x, y, rx, ry):
    p = skia.Path()
    p.addOval(skia.Rect(x - rx, y - ry, x + rx, y + ry))
    return p


def capsule(x0, y0, x1, y1, r0, r1=None):
    """Tapered tube between two points (limbs, sleeves)."""
    r1 = r0 if r1 is None else r1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1e-6
    nx, ny = -dy / L, dx / L
    p = skia.Path()
    p.moveTo(x0 + nx * r0, y0 + ny * r0)
    p.lineTo(x1 + nx * r1, y1 + ny * r1)
    p.arcTo(skia.Rect(x1 - r1, y1 - r1, x1 + r1, y1 + r1), math.degrees(math.atan2(ny, nx)), -180, False)
    p.lineTo(x0 - nx * r0, y0 - ny * r0)
    p.arcTo(skia.Rect(x0 - r0, y0 - r0, x0 + r0, y0 + r0), math.degrees(math.atan2(-ny, -nx)), -180, False)
    p.close()
    return p


def poly(pts, smooth=False, closed=True):
    p = skia.Path()
    if not smooth:
        p.moveTo(*pts[0])
        for q in pts[1:]:
            p.lineTo(*q)
    else:  # Catmull-Rom through the points
        n = len(pts)
        p.moveTo(*pts[0])
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
            if not closed:
                p0 = pts[max(0, i - 1)]
                p3 = pts[min(n - 1, i + 2)]
            c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
            c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
            p.cubicTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    if closed:
        p.close()
    return p


def path_op(a, b, kind="union"):
    """skia.Op that never raises: degenerate shapes (e.g. mid run-cycle) fall back to the first path."""
    k = {"union": skia.PathOp.kUnion_PathOp, "difference": skia.PathOp.kDifference_PathOp}[kind]
    try:
        return skia.Op(a, b, k) or a
    except RuntimeError:
        return a


def hull(pts):
    pts = sorted(set((round(x, 2), round(y, 2)) for x, y in pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


# ---------------------------------------------------------------- B: vector cel
class VectorCel(Painter):
    name = "B"

    def fill(self, c, path, color, shade_k=0.32, kind="cloth", alpha=255, light=(-0.5, -0.8)):
        b = path.computeTightBounds()
        cx, cy = b.centerX(), b.centerY()
        r = max(b.width(), b.height()) * 0.6 + 1
        p0 = (cx + light[0] * r, cy + light[1] * r)
        p1 = (cx - light[0] * r, cy - light[1] * r)
        cols = [rgb(shade(color, shade_k * 0.55), alpha), rgb(color, alpha), rgb(shade(color, -shade_k), alpha)]
        p = skia.Paint(AntiAlias=True)
        p.setShader(skia.GradientShader.MakeLinear([p0, p1], cols, [0.0, 0.45, 1.0]))
        c.drawPath(path, p)

    def joint(self, c, x, y, r, color):
        self.fill(c, circle(x, y, r), color, 0.3)
        c.drawCircle(x, y, r * 0.34, skia.Paint(AntiAlias=True, Color=rgb(PALETTE["gold"], 230)))


# ---------------------------------------------------------------- C: paper theatre
def _grain_image(seed=5, n=192):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, (n, n)).astype(np.float32)
    # soft fibres: blur along x a little
    k = np.array([0.15, 0.2, 0.3, 0.2, 0.15], np.float32)
    g = sum(np.roll(g, i - 2, axis=1) * k[i] for i in range(5))
    g = (g - g.min()) / (g.max() - g.min())
    v = (205 + 50 * g).astype(np.uint8)
    rgba = np.dstack([v, v, v, np.full_like(v, 255)])
    img = skia.Image.fromarray(np.ascontiguousarray(rgba), skia.kRGBA_8888_ColorType)
    s = skia.Surface.MakeRasterN32Premul(n, n)
    s.getCanvas().drawImage(img, 0, 0)
    return s.makeImageSnapshot()  # native pixel format: no per-draw conversion


class PaperTheatre(Painter):
    name = "C"
    joint_style = "pin"

    def __init__(self, seed=0, shadow_px=7.0):
        super().__init__(seed)
        self.grain = _grain_image(seed + 5)
        self.grain_shader = self.grain.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat)
        self.shadow_px = shadow_px
        self._fills = {}
        self._shadow_paint = skia.Paint(AntiAlias=True, Color=skia.Color(30, 14, 10, 105))
        self._shadow_paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 4.0, False))

    def _shadow(self, c, path):
        m = c.getTotalMatrix()
        inv = skia.Matrix()
        if not m.invert(inv):
            return
        v = inv.mapVector(self.shadow_px * 0.6, self.shadow_px)
        c.save()
        c.translate(v.x(), v.y())
        c.drawPath(path, self._shadow_paint)
        c.restore()

    def fill(self, c, path, color, shade_k=0.32, kind="cloth", alpha=255, light=(-0.5, -0.8), shadow=True, edge=True):
        if shadow and kind != "detail":
            self._shadow(c, path)
        key = (color, alpha)
        p = self._fills.get(key)
        if p is None:  # building a shader that references the grain image costs ~15 ms in skia-python: cache per colour
            p = skia.Paint(AntiAlias=True)
            base = skia.Shaders.Color(rgb(shade(color, 0.04), alpha))
            p.setShader(skia.Shaders.Blend(skia.BlendMode.kMultiply, base, self.grain_shader))
            self._fills[key] = p
        c.drawPath(path, p)
        if edge and kind not in ("detail", "skin_detail"):
            m = c.getTotalMatrix()
            sc = math.sqrt(abs(m.getScaleX() * m.getScaleY() - m.getSkewX() * m.getSkewY())) or 1.0
            self.stroke(c, path, mix(PALETTE["gold"], color, 0.25), 1.6 / sc, alpha=int(alpha * 0.85))

    def joint(self, c, x, y, r, color):
        """Brass split-pin: a small domed brass head with a highlight and the pin's slot."""
        rr = max(4.5, r * 0.42)
        self._shadow(c, circle(x, y, rr))
        p = skia.Paint(AntiAlias=True)
        p.setShader(skia.GradientShader.MakeRadial((x - rr * 0.35, y - rr * 0.4), rr * 1.3,
                                                   [rgb("#fff1b0"), rgb("#d9a73a"), rgb("#7a5210")], [0.0, 0.45, 1.0]))
        c.drawCircle(x, y, rr, p)
        sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=rr * 0.22, Color=rgb("#5a3a0c", 200))
        c.drawLine(x - rr * 0.55, y, x + rr * 0.55, y, sp)

    def raster_part(self, c, img, x, y, paint):
        sp = skia.Paint(paint)
        sp.setImageFilter(skia.ImageFilters.DropShadow(4, 7, 3, 3, skia.Color(30, 14, 10, 110)))
        c.drawImage(img, x, y, sp)

    def layer_shadow(self):
        return self.shadow_px
