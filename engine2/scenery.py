"""Vector scenery: props + two parametric scene builders (exterior, interior). Every story location is a dict of
builder parameters (see LOCATIONS), so a new location is data, and both art directions draw the same scene.
Heartwick look: New England seasons, warm light inside / cool outside, jewel accents."""
import math
import numpy as np
import skia
from . import fx
from .scene import Scene, Layer, ACTORS
from .style import rgb, shade, mix, circle, ellipse, capsule, poly, path_op, PALETTE

SKIES = {  # top, middle, horizon
    "dawn": ("#6f86b5", "#c9a9c4", "#f6cfa6"), "morning": ("#8ea7c2", "#c3cdd6", "#ece4d6"),
    "day": ("#5f9fd8", "#9cc6e8", "#f2ead6"), "golden": ("#6a8fc8", "#e9b98a", "#ffd9a0"),
    "dusk": ("#2f3770", "#a2557a", "#f0935e"), "night": ("#0d1a30", "#1a3050", "#2c4a6a"),
    "storm": ("#121a2e", "#1f2a46", "#34405e"),
}
GROUND = {"autumn": ("#4f7a3e", "#2c4a2a"), "winter": ("#eef2f6", "#c9d3df"), "spring": ("#6c9a48", "#3c6a32"),
          "summer": ("#5f9444", "#365f2c")}
CANOPY = {"autumn": ["#b3243a", "#d9572a", "#e9a53a", "#8c1a2c", "#f2c14e", "#c8332b"],
          "spring": ["#b8432f", "#d46a5a", "#8da67e", "#6c9a48", "#c86450"],
          "summer": ["#3c7a3e", "#4f8f44", "#2f6a36", "#6aa050"], "winter": []}


def _noise_line(x0, x1, y, amp, seed, n=9):
    r = np.random.default_rng(seed)
    xs = np.linspace(x0, x1, n)
    return [(float(x), float(y + r.uniform(-amp, amp))) for x in xs]


# ---------------------------------------------------------------- props
def sky(c, P, w, h, tod):
    top, mid, hor = SKIES[tod]
    p = skia.Paint()
    p.setShader(skia.GradientShader.MakeLinear([(0, -300), (0, h * 0.75)], [rgb(top), rgb(mid), rgb(hor)], [0.0, 0.55, 1.0]))
    c.drawRect(skia.Rect(-400, -400, w + 400, h + 400), p)
    if P.name == "C":  # paper sky: grain over it
        g = skia.Paint()
        g.setShader(P.grain_shader)
        g.setBlendMode(skia.BlendMode.kMultiply)
        c.drawRect(skia.Rect(-400, -400, w + 400, h + 400), g)


def hills(c, P, w, y, amp, color, seed, layer_shade=0.0):
    pts = [(-400, y + 900)] + _noise_line(-400, w + 400, y, amp, seed) + [(w + 400, y + 900)]
    P.fill(c, poly(pts, smooth=True), shade(color, layer_shade), 0.25, kind="land")


def mountains(c, P, w, y, color, seed):
    r = np.random.default_rng(seed)
    for layer, (col, dy, hmax, step) in enumerate([(shade(color, 0.18), -40, 330, (380, 620)), (color, 60, 230, (260, 460))]):
        pts = [(-400, y + 900), (-400, y + dy + 60)]
        x = -400 + r.uniform(0, 200)
        while x < w + 400:  # irregular ridges: a long shoulder, a peak, a saddle
            sp = r.uniform(*step)
            pk = y + dy - r.uniform(0.45, 1.0) * hmax
            pts += [(x + sp * 0.3, y + dy - hmax * r.uniform(0.15, 0.4)), (x + sp * r.uniform(0.45, 0.6), pk),
                    (x + sp * 0.8, pk + hmax * r.uniform(0.25, 0.5)), (x + sp, y + dy + r.uniform(0, 50))]
            x += sp
        pts += [(w + 400, y + dy + 60), (w + 400, y + 900)]
        P.fill(c, poly(pts, smooth=True), col, 0.22, kind="land")


def clouds(c, P, w, y, color, seed, n=4):
    r = np.random.default_rng(seed)
    for i in range(n):
        cx, cy = r.uniform(0, w), y + r.uniform(-80, 80)
        p = skia.Path()
        for k in range(5):
            p.addCircle(cx + (k - 2) * 46 + r.uniform(-10, 10), cy + r.uniform(-18, 10) - (20 if k in (1, 2, 3) else 0), r.uniform(36, 58))
        p = path_op(p, skia.Path())
        P.fill(c, p, color, 0.18, kind="cloud", alpha=225)


def house(c, P, x, y, s=1.0, wall="#ece4d2", roof="#7a1328", trim="#fff4e2", lit=True, porch=True, windows=2):
    """White clapboard New England house, feet of the wall at (x, y)."""
    W, H = 300 * s, 190 * s
    body = poly([(x, y), (x + W, y), (x + W, y - H), (x, y - H)])
    P.fill(c, body, wall, 0.18, kind="wall")
    lp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=max(1.0, 2 * s), Color=rgb(shade(wall, -0.18), 150))
    for k in range(1, 9):
        yy = y - H * k / 9
        c.drawLine(x + 2, yy, x + W - 2, yy, lp)
    P.fill(c, poly([(x - 24 * s, y - H + 4), (x + W / 2, y - H - 150 * s), (x + W + 24 * s, y - H + 4)]), roof, 0.3, kind="roof")
    P.fill(c, poly([(x + W * 0.66, y - H - 80 * s), (x + W * 0.66, y - H - 130 * s), (x + W * 0.76, y - H - 130 * s), (x + W * 0.76, y - H - 50 * s)]), "#8a4a3a", 0.3)
    for k in range(windows):
        wx = x + W * (0.18 + 0.5 * k)
        wy = y - H * 0.72
        win = poly([(wx, wy), (wx + 60 * s, wy), (wx + 60 * s, wy + 74 * s), (wx, wy + 74 * s)])
        P.fill(c, win, "#ffc56b" if lit else "#5a6a80", 0.2, kind="detail")
        P.stroke(c, win, trim, 5 * s)
        c.drawLine(wx + 30 * s, wy, wx + 30 * s, wy + 74 * s, skia.Paint(AntiAlias=True, StrokeWidth=4 * s, Color=rgb(trim)))
        if lit:
            fx.glow(c, wx + 30 * s, wy + 37 * s, 90 * s, "#ffc56b", 0.45)
    door = poly([(x + W * 0.42, y), (x + W * 0.58, y), (x + W * 0.58, y - 110 * s), (x + W * 0.42, y - 110 * s)])
    P.fill(c, door, PALETTE["sapphire"], 0.25)
    if porch:
        P.fill(c, poly([(x - 10 * s, y), (x + W + 10 * s, y), (x + W + 10 * s, y + 16 * s), (x - 10 * s, y + 16 * s)]), shade(wall, -0.1), 0.2)


def church(c, P, x, y, s=1.0, wall="#f1ece2", roof="#4a5a74"):
    W, H = 160 * s, 170 * s
    P.fill(c, poly([(x, y), (x + W, y), (x + W, y - H), (x, y - H)]), wall, 0.15, kind="wall")
    P.fill(c, poly([(x - 14 * s, y - H), (x + W / 2, y - H - 90 * s), (x + W + 14 * s, y - H)]), roof, 0.3)
    tx = x + W / 2
    P.fill(c, poly([(tx - 34 * s, y - H - 40 * s), (tx + 34 * s, y - H - 40 * s), (tx + 34 * s, y - H - 200 * s), (tx - 34 * s, y - H - 200 * s)]), wall, 0.15, kind="wall")
    P.fill(c, poly([(tx - 42 * s, y - H - 200 * s), (tx, y - H - 420 * s), (tx + 42 * s, y - H - 200 * s)]), roof, 0.3)
    P.fill(c, ellipse(tx, y - H - 130 * s, 16 * s, 22 * s), "#2a3a54", 0.2, kind="detail")


def fence(c, P, x0, x1, y, h=90, color="#f4efe6"):
    k = x0
    while k < x1:
        P.fill(c, poly([(k, y), (k + 22, y), (k + 22, y - h), (k + 11, y - h - 16), (k, y - h)]), color, 0.2, kind="wood")
        k += 40
    P.fill(c, poly([(x0, y - h * 0.3), (x1, y - h * 0.3), (x1, y - h * 0.3 - 12), (x0, y - h * 0.3 - 12)]), shade(color, -0.08), 0.15, kind="wood")
    P.fill(c, poly([(x0, y - h * 0.75), (x1, y - h * 0.75), (x1, y - h * 0.75 - 12), (x0, y - h * 0.75 - 12)]), shade(color, -0.08), 0.15, kind="wood")


def trunk(c, P, x, y, h, w=90, color="#4a2e1e", lean=0.0):
    pts = [(x - w * 0.75, y + 10), (x - w * 0.45, y - h * 0.3), (x - w * 0.35 + lean, y - h * 0.75), (x - w * 0.9 + lean, y - h),
           (x - w * 0.6 + lean, y - h * 1.02), (x + lean * 0.9, y - h * 0.8), (x + w * 0.7 + lean, y - h * 1.0),
           (x + w * 0.9 + lean, y - h * 0.97), (x + w * 0.35 + lean, y - h * 0.7), (x + w * 0.45, y - h * 0.3), (x + w * 0.85, y + 10)]
    P.fill(c, poly(pts, smooth=True), color, 0.4, kind="wood")
    bark = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3, Color=rgb(shade(color, -0.3), 150))
    for k in range(5):
        xx = x - w * 0.3 + k * w * 0.15
        c.drawLine(xx, y - 10, xx + lean * 0.3, y - h * 0.6, bark)


class Canopy:
    """Leaf clusters that sway as a whole and jiggle individually (drawn per frame)."""

    def __init__(self, x, y, rx, ry, colors, seed, n=60, sway=2.2, bare=False):
        r = np.random.default_rng(seed)
        self.x, self.y, self.sway, self.bare = x, y, sway, bare
        self.cols = colors or ["#5a4030"]
        self.cl = []
        if not bare:  # clusters uniformly inside the crown ellipse, bigger toward the middle
            while len(self.cl) < n:
                u, v = r.uniform(-1, 1, 2)
                if u * u + v * v > 1:
                    continue
                rr = r.uniform(20, 34) * (1.25 - 0.45 * (u * u + v * v))
                self.cl.append((x + u * rx, y + v * ry, rr, self.cols[int(r.integers(len(self.cols)))], r.uniform(0, 6.28)))
            self.cl.sort(key=lambda q: q[1])
        mass = skia.Path()  # the crown's solid mass behind the clusters (no sky holes)
        for _ in range(9):  # kept inside the cluster region so its edge never shows
            cx_, cy_ = x + r.uniform(-0.35, 0.35) * rx, y + r.uniform(-0.3, 0.3) * ry
            mass.addOval(skia.Rect.MakeXYWH(cx_ - rx * 0.55, cy_ - ry * 0.55, rx * 1.1, ry * 1.1))
        self.mass = path_op(mass, skia.Path())
        self.branches = [(x + r.uniform(-rx * 0.8, rx * 0.8), y + r.uniform(-ry * 0.5, ry * 0.3)) for _ in range(9)]

    def draw(self, c, t, P):
        c.save()
        c.rotate(self.sway * math.sin(2 * math.pi * 0.18 * t), self.x, self.y + 600)
        bp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=14, Color=rgb("#4a2e1e"))
        bp.setStrokeCap(skia.Paint.kRound_Cap)
        for bx, by in self.branches:
            c.drawLine(self.x, self.y + 260, bx, by, bp)
        if self.cl:
            P.fill(c, self.mass, shade(self.cols[0], -0.35), 0.3, kind="leaf")
        kw = {"shadow": False, "edge": False} if P.name == "C" else {}
        for (x, y, r, col, ph) in self.cl:
            jx = 3.0 * math.sin(1.3 * t + ph)
            jy = 2.0 * math.cos(1.1 * t + ph * 1.3)
            P.fill(c, circle(x + jx, y + jy, r), col, 0.28, kind="leaf", **kw)
        c.restore()


def scatter_leaves(c, P, x0, x1, y0, y1, n, seed, colors=None):
    r = np.random.default_rng(seed)
    cols = colors or fx.LEAF_COLS
    pa = skia.Paint(AntiAlias=True)
    for _ in range(n):
        c.save()
        c.translate(r.uniform(x0, x1), r.uniform(y0, y1))
        c.rotate(r.uniform(0, 360))
        s = r.uniform(0.9, 1.8)
        c.scale(s, s * 0.55)
        pa.setColor(rgb(cols[int(r.integers(len(cols)))]))
        c.drawPath(fx._LEAF, pa)
        c.restore()


def grass_tufts(c, P, x0, x1, y, color, seed, n=14, h=90):
    r = np.random.default_rng(seed)
    for _ in range(n):
        x = r.uniform(x0, x1)
        hh = r.uniform(0.6, 1.3) * h
        p = poly([(x - 30, y), (x - 18, y - hh * 0.7), (x - 8, y - 6), (x, y - hh), (x + 8, y - 6), (x + 20, y - hh * 0.8), (x + 30, y)])
        P.fill(c, p, color, 0.3, kind="leaf")


def window_frame(c, P, x, y, w, h, tod, trim="#fff4e2", season="autumn", sash_open=0.0, outside_tree=True):
    """An interior window looking out: sky + a branch outside, lit by the time of day."""
    top, mid, hor = SKIES[tod]
    rect = poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    p = skia.Paint(AntiAlias=True)
    p.setShader(skia.GradientShader.MakeLinear([(x, y), (x, y + h)], [rgb(top), rgb(mid), rgb(hor)], [0.0, 0.5, 1.0]))
    c.drawPath(rect, p)
    c.save()
    c.clipPath(rect, skia.ClipOp.kIntersect, True)
    if outside_tree:
        bp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=12, Color=rgb("#3a2618"))
        c.drawLine(x + w * 0.9, y + h, x + w * 0.55, y + h * 0.2, bp)
        cols = CANOPY.get(season) or ["#e8eef4"]
        r = np.random.default_rng(int(x + y))
        for _ in range(14):
            P.fill(c, circle(x + w * r.uniform(0.4, 1.0), y + h * r.uniform(0.05, 0.6), r.uniform(18, 34)), cols[int(r.integers(len(cols)))], 0.2, kind="leaf")
    c.restore()
    P.stroke(c, rect, trim, 16)
    c.drawLine(x + w / 2, y, x + w / 2, y + h, skia.Paint(AntiAlias=True, StrokeWidth=10, Color=rgb(trim)))
    c.drawLine(x, y + h / 2, x + w, y + h / 2, skia.Paint(AntiAlias=True, StrokeWidth=10, Color=rgb(trim)))
    P.fill(c, poly([(x - 30, y + h + 8), (x + w + 30, y + h + 8), (x + w + 30, y + h + 30), (x - 30, y + h + 30)]), trim, 0.2, kind="wood")


def table(c, P, x, y, w=420, color="#8a5a32"):
    P.fill(c, poly([(x - w / 2, y - 260), (x + w / 2, y - 260), (x + w / 2, y - 236), (x - w / 2, y - 236)]), color, 0.3, kind="wood")
    for lx in (x - w / 2 + 30, x + w / 2 - 44):
        P.fill(c, poly([(lx, y - 236), (lx + 14, y - 236), (lx + 14, y), (lx, y)]), shade(color, -0.15), 0.3, kind="wood")


def mug(c, P, x, y, color=PALETTE["cream"], band=PALETTE["ruby"]):
    P.fill(c, poly([(x - 20, y - 52), (x + 20, y - 52), (x + 18, y), (x - 18, y)]), color, 0.25)
    P.fill(c, poly([(x - 20, y - 38), (x + 20, y - 38), (x + 20, y - 30), (x - 20, y - 30)]), band, 0.2, kind="detail")
    P.stroke(c, poly([(x + 20, y - 44), (x + 34, y - 36), (x + 19, y - 20)], smooth=True, closed=False), color, 6)


def chair(c, P, x, y, color="#6a4024", facing=1):
    f = facing
    P.fill(c, poly([(x - 70, y - 250), (x + 70, y - 250), (x + 70, y - 232), (x - 70, y - 232)]), color, 0.3, kind="wood")
    P.fill(c, poly([(x - 70 * f, y - 250), (x - 56 * f, y - 250), (x - 56 * f, y - 520), (x - 70 * f, y - 520)]), color, 0.3, kind="wood")
    for lx in (x - 66, x + 54):
        P.fill(c, poly([(lx, y - 232), (lx + 12, y - 232), (lx + 12, y), (lx, y)]), shade(color, -0.15), 0.3, kind="wood")


def armchair(c, P, x, y, color=PALETTE["ruby"]):
    P.fill(c, poly([(x - 150, y - 560), (x + 10, y - 560), (x + 30, y - 280), (x - 160, y - 240)], smooth=True), color, 0.35)
    P.fill(c, poly([(x - 170, y - 300), (x + 150, y - 300), (x + 160, y - 60), (x - 170, y - 60)], smooth=True), shade(color, 0.05), 0.35)
    P.pattern_dots(c, poly([(x - 170, y - 300), (x + 150, y - 300), (x + 160, y - 60), (x - 170, y - 60)], smooth=True), PALETTE["gold"], 22, 3, 160)
    for lx in (x - 150, x + 120):
        P.fill(c, poly([(lx, y - 60), (lx + 24, y - 60), (lx + 20, y), (lx + 4, y)]), "#4a2e1e", 0.3, kind="wood")


def lamp(c, P, x, y, lit=True):
    P.fill(c, poly([(x - 6, y), (x + 6, y), (x + 6, y - 330), (x - 6, y - 330)]), "#6a4a2a", 0.3)
    P.fill(c, poly([(x - 70, y - 330), (x + 70, y - 330), (x + 44, y - 420), (x - 44, y - 420)]), PALETTE["honey"], 0.25)
    if lit:
        fx.glow(c, x, y - 360, 340, "#ffc56b", 0.5)


def geraniums(c, P, x, y, n=2):
    for k in range(n):
        px = x + k * 90
        P.fill(c, poly([(px - 30, y - 60), (px + 30, y - 60), (px + 22, y), (px - 22, y)]), "#b8432f", 0.3)
        for j in range(5):
            P.fill(c, circle(px + (j - 2) * 14, y - 80 - (j % 2) * 18, 16), "#3c7a3e", 0.25, kind="leaf")
        for j in range(3):
            P.fill(c, circle(px + (j - 1) * 16, y - 108 - (j % 2) * 10, 12), "#d9334a", 0.25, kind="leaf")


def wallpaper(c, P, w, y0, y1, color, dot):
    P.flat(c, poly([(-400, y0), (w + 400, y0), (w + 400, y1), (-400, y1)]), color)
    P.pattern_dots(c, poly([(-400, y0), (w + 400, y0), (w + 400, y1), (-400, y1)]), dot, 46, 5, 120)


def floor(c, P, w, y, color="#8a5a32", h=700):
    P.fill(c, poly([(-400, y - 10), (w + 400, y - 10), (w + 400, y + h), (-400, y + h)]), color, 0.25, kind="wood")
    lp = skia.Paint(AntiAlias=True, StrokeWidth=3, Color=rgb(shade(color, -0.25), 160))
    for k in range(1, 9):
        c.drawLine(-400, y + k * 60, w + 400, y + k * 60, lp)


# ---------------------------------------------------------------- paper-theatre extras (style C only)
def proscenium(P):
    cache = {}

    def draw(c, t, W, H):
        if P.name != "C":
            return
        if (W, H) not in cache:  # static in screen space: draw once, blit every frame
            s = skia.Surface.MakeRasterN32Premul(W, H)
            s.getCanvas().clear(skia.ColorTRANSPARENT)
            _draw(s.getCanvas(), W, H)
            cache[(W, H)] = s.makeImageSnapshot()
        c.drawImage(cache[(W, H)], 0, 0)

    def _draw(c, W, H):
        k = W / 1080.0
        gold, ruby = "#d9a73a", "#7a1328"
        arch = skia.Path()
        arch.addRect(skia.Rect(-10, -10, W + 10, H + 10))
        inner = skia.Path()
        inner.addRRect(skia.RRect.MakeRectXY(skia.Rect(44 * k, 120 * k, W - 44 * k, H - 70 * k), 260 * k, 220 * k))
        frame = path_op(arch, inner, "difference")
        P.fill(c, frame, ruby, 0.3, kind="frame")
        rim = skia.Path()
        rim.addRRect(skia.RRect.MakeRectXY(skia.Rect(44 * k, 120 * k, W - 44 * k, H - 70 * k), 260 * k, 220 * k))
        P.stroke(c, rim, gold, 9 * k)
        for i in range(14):  # gilded beads along the arch
            a = math.pi * (0.08 + 0.84 * i / 13)
            bx, by = W / 2 - math.cos(a) * (W / 2 - 60 * k), 150 * k + (1 - math.sin(a)) * 40 * k
            P.joint(c, bx, by + 20 * k, 16 * k, gold)
        # curtain swags in the top corners
        for s in (-1, 1):
            x0 = 0 if s < 0 else W
            sw = poly([(x0, 0), (x0 - s * -300 * k, 0), (x0 - s * -170 * k, 150 * k), (x0 - s * -40 * k, 420 * k), (x0, 520 * k)], smooth=True)
            P.fill(c, sw, "#8c1a2c", 0.35, kind="cloth")
    return draw


def paper_grade(P):
    def grade(c, W, H, t):
        # warm footlight glow from the bottom + subtle full-frame grain (paper look)
        p = skia.Paint()
        p.setShader(skia.GradientShader.MakeLinear([(0, H), (0, H * 0.6)], [rgb("#ffb860", 70), rgb("#ffb860", 0)]))
        p.setBlendMode(skia.BlendMode.kScreen)
        c.drawRect(skia.Rect(0, 0, W, H), p)
    return grade


# ---------------------------------------------------------------- builders
def exterior(P, name, tod="morning", season="autumn", tall=False, house_at=None, maple_at=0.72, church_at=None,
             fence_y=True, fx_list=("motes",), weather=None, mist=False, seed=1, ground_frac=0.8, bend=0.0, rays=True,
             maple_bare=None):
    Ww, Wh = 1400, (3000 if tall else 2100)
    gy = Wh * ground_frac
    gcol = GROUND["winter" if season == "winter" else season]
    cols = CANOPY[season]
    r = np.random.default_rng(seed)
    canopy = Canopy(Ww * maple_at, gy - 940, 400, 250, cols or ["#7a6a5a"], seed + 3, n=150 if cols else 0,
                    sway=2.0 + (5.0 if weather == "storm" else 0), bare=not cols) if maple_at is not None else None
    twinkle = [(r.uniform(0, Ww), r.uniform(0, Wh * 0.4), r.uniform(1, 2.6), r.uniform(0, 6.28)) for _ in range(60)] if tod in ("night", "dusk", "storm") else []

    def sky_static(c):
        sky(c, P, Ww, gy, tod)
        if tod in ("day", "golden", "morning", "dawn"):
            fx.glow(c, Ww * 0.22, gy - 1250, 420, "#fff1c9", 0.55 if tod != "morning" else 0.3)
            P.fill(c, circle(Ww * 0.22, gy - 1250, 70), "#fff4d6", 0.1, kind="detail", alpha=235)
        if tod in ("night", "dusk"):
            fx.glow(c, Ww * 0.78, gy - 1400, 260, "#fff3c9", 0.45)
            P.fill(c, circle(Ww * 0.78, gy - 1400, 48), "#fff6d8", 0.05, kind="detail")
        clouds(c, P, Ww, gy - 1500, shade(SKIES[tod][1], 0.35) if tod not in ("storm", "night") else "#2a3352", seed + 5, n=5)

    def sky_dyn(c, t):
        pa = skia.Paint(AntiAlias=True)
        for x, y, rr, ph in twinkle:
            a = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t + ph))
            pa.setColor(rgb("#f7e3a8", int(255 * a)))
            c.drawCircle(x, y, rr, pa)
        if weather == "storm" and (t % 4.7) < 0.12:  # lightning flash
            pa.setColor(rgb("#dfe8ff", 90))
            c.drawRect(skia.Rect(-400, -400, Ww + 400, Wh), pa)
        if tod not in ("night", "storm"):  # a small flock drifts across the sky, wings flapping
            bp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=4, Color=rgb("#3a3046", 200))
            bp.setStrokeCap(skia.Paint.kRound_Cap)
            span = Ww + 1000
            for k in range(5):
                x = (t * 70 + k * 60 + seed * 137) % span - 500
                y = gy - 1350 + 40 * math.sin(k * 1.7) + 12 * math.sin(0.5 * t + k)
                f = 14 * math.sin(2 * math.pi * 2.2 * t + k)
                wing = poly([(x - 20, y - f), (x, y), (x + 20, y - f)], closed=False)
                c.drawPath(wing, bp)

    def far(c):
        mountains(c, P, Ww, gy - 520, mix("#5b6f96", SKIES[tod][2], 0.35) if tod not in ("night", "storm") else "#1f2c4a", seed + 7)
        hills(c, P, Ww, gy - 360, 50, "#4f6f8f" if tod not in ("night", "storm") else "#1c2a40", seed + 8)
        if church_at is not None:
            church(c, P, Ww * church_at, gy - 300, 0.8)

    def mid(c):
        hills(c, P, Ww, gy - 180, 36, shade(gcol[0], -0.15), seed + 9)
        if house_at is not None:
            house(c, P, Ww * house_at, gy - 60, 1.15, lit=tod in ("dawn", "dusk", "night", "storm", "morning"))
        if fence_y:
            fence(c, P, -60, Ww + 60, gy - 20, 80)

    def mid_dyn(c, t):
        if canopy is not None:
            canopy.draw(c, t, P)

    def ground_static(c):
        P.fill(c, poly([(-400, gy - 30), (Ww + 400, gy - 30), (Ww + 400, Wh + 400), (-400, Wh + 400)]), gcol[0], 0.25, kind="land")
        if maple_at is not None:
            trunk(c, P, Ww * maple_at, gy + 20, 760, 110, lean=bend)
        if season == "autumn":
            scatter_leaves(c, P, -300, Ww + 300, gy - 10, Wh, 160, seed + 11)
        if weather == "after_storm":
            bx = Ww * (maple_at or 0.7)
            P.fill(c, capsule(bx + 60, gy + 30, bx + 380, gy + 60, 16, 10), "#4a2e1e", 0.3, kind="wood")
            P.fill(c, capsule(bx + 220, gy + 50, bx + 280, gy - 10, 8, 5), "#4a2e1e", 0.3, kind="wood")
            scatter_leaves(c, P, bx + 40, bx + 400, gy - 10, gy + 70, 30, seed + 12)

    def fore(c):
        grass_tufts(c, P, -300, Ww + 300, Wh * 0.99, shade(gcol[1], -0.1), seed + 13, n=18, h=130)
        if season == "autumn":
            scatter_leaves(c, P, -300, Ww + 300, Wh * 0.95, Wh * 1.05, 26, seed + 14)

    parts = list(fx_list)
    pfx = [fx.Particles(k, (-200, 0, Ww + 200, Wh), seed=seed + i) for i, k in enumerate(parts) if k in ("leaves", "motes", "snow", "rain", "fireflies", "petals")]

    def light(c, t):
        if rays and tod in ("day", "golden", "morning", "dawn"):
            fx.god_rays(c, t, (Ww * 0.22, gy - 1250), 62, 2200, spread=34, n=7, alpha=0.14 if tod != "morning" else 0.1, seed=seed)
        if mist or "mist" in parts:
            fx.mist(c, t, gy - 120, Ww, 200, alpha=0.32, seed=seed)
        if "rain" in parts:
            fx.splashes(c, t, -100, Ww + 100, gy - 20, n=60, seed=seed)
        for p in pfx:
            p.draw(c, t)

    layers = [Layer("sky", 0.08, sky_static, sky_dyn), Layer("far", 0.3, far), Layer("mid", 0.6, mid, mid_dyn),
              Layer("ground", 0.92, ground_static), Layer(ACTORS, 1.0), Layer("fore", 1.25, fore), Layer("light", 1.05, None, light)]
    spots = {"under_maple": (Ww * maple_at - 160 if maple_at else Ww * 0.6, gy + 110), "left": (Ww * 0.3, gy + 60), "center": (Ww * 0.5, gy + 80),
             "right": (Ww * 0.68, gy + 80), "canopy": (Ww * (maple_at or 0.5), gy - 1000), "sky": (Ww * 0.5, gy - 1500)}
    return Scene(name, (Ww, Wh), layers, gy + 80, P, spots, overlay=proscenium(P) if P.name == "C" else None, tall=tall)


def interior(P, name, tod="dawn", season="autumn", wall="#e9dcc4", dot="#c9a87a", window_at=0.62, items=(), lamp_lit=True,
             fx_list=("motes",), seed=1, sash_open=0.0, big_window=False, ramp=False):
    """A cozy room: wallpaper, wainscot, a curtained window to the outside (time of day), rug, floor, and items:
    table, mug, mug2, chair, chair2, armchair, lamp, geraniums, counter, shelf, clock, phone, box.
    big_window: a large low window whose sill is a stage (spot "sill") for small characters."""
    Ww, Wh = 1400, 2100
    gy = Wh * 0.8
    warm = tod in ("dawn", "night", "dusk")
    pfx = [fx.Particles(k, (-200, 0, Ww + 200, Wh), seed=seed + i) for i, k in enumerate(fx_list) if k in ("motes", "snow", "fireflies", "leaves")]
    wwid, whgt = (620, 760) if big_window else (380, 470)
    wx, wy = Ww * window_at - wwid / 2, (gy - 1330 if big_window else gy - 1250)
    sill_y = wy + whgt + 8

    def back(c):
        wallpaper(c, P, Ww, -400, gy, wall, dot)
        P.fill(c, poly([(-400, gy - 380), (Ww + 400, gy - 380), (Ww + 400, gy), (-400, gy)]), shade(wall, -0.22), 0.15, kind="wood")
        P.fill(c, poly([(-400, gy - 396), (Ww + 400, gy - 396), (Ww + 400, gy - 374), (-400, gy - 374)]), PALETTE["cream"], 0.15, kind="wood")
        for k in range(-1, 9):  # wainscot panels
            P.stroke(c, poly([(k * 190 + 30, gy - 340), (k * 190 + 190, gy - 340), (k * 190 + 190, gy - 40), (k * 190 + 30, gy - 40)]), shade(wall, -0.32), 4)
        if not big_window and "shelf" not in items:  # a framed picture: a little maple on a hill
            fx_, fy_ = Ww * 0.16, gy - 1180
            P.fill(c, poly([(fx_, fy_), (fx_ + 200, fy_), (fx_ + 200, fy_ + 150), (fx_, fy_ + 150)]), "#8a5a32", 0.3, kind="wood")
            P.fill(c, poly([(fx_ + 16, fy_ + 16), (fx_ + 184, fy_ + 16), (fx_ + 184, fy_ + 134), (fx_ + 16, fy_ + 134)]), "#cfe0ea", 0.1, kind="detail")
            P.fill(c, ellipse(fx_ + 100, fy_ + 140, 90, 40), "#6c9a48", 0.2, kind="detail")
            P.fill(c, circle(fx_ + 110, fy_ + 70, 30), "#c8332b", 0.2, kind="detail")
        window_frame(c, P, wx, wy, wwid, whgt, tod, season=season)
        if ramp:  # Nell's little wooden ramp from the sill out to the maple
            P.fill(c, poly([(wx + wwid * 0.55, sill_y - 4), (wx + wwid + 160, sill_y - 120), (wx + wwid + 170, sill_y - 100), (wx + wwid * 0.58, sill_y + 14)]), "#b07a48", 0.3, kind="wood")
        if "counter" in items:
            P.fill(c, poly([(-400, gy - 330), (Ww * 0.42, gy - 330), (Ww * 0.42, gy), (-400, gy)]), "#7a8a6e", 0.3, kind="wood")
            P.fill(c, poly([(-400, gy - 350), (Ww * 0.45, gy - 350), (Ww * 0.45, gy - 324), (-400, gy - 324)]), "#d9cbb0", 0.2, kind="wood")
            for k in range(3):
                P.stroke(c, poly([(k * 190 - 40, gy - 300), (k * 190 + 130, gy - 300), (k * 190 + 130, gy - 30), (k * 190 - 40, gy - 30)]), shade("#7a8a6e", -0.2), 4)
        if "shelf" in items:
            for k in range(3):
                P.fill(c, poly([(Ww * 0.06, gy - 900 + k * 150), (Ww * 0.34, gy - 900 + k * 150), (Ww * 0.34, gy - 886 + k * 150), (Ww * 0.06, gy - 886 + k * 150)]), "#8a5a32", 0.3, kind="wood")
                for j in range(5):
                    P.fill(c, poly([(Ww * 0.08 + j * 70, gy - 900 + k * 150), (Ww * 0.08 + j * 70 + 40, gy - 900 + k * 150), (Ww * 0.08 + j * 70 + 40, gy - 990 + k * 150), (Ww * 0.08 + j * 70, gy - 990 + k * 150)]),
                           [PALETTE["ruby"], PALETTE["sapphire"], PALETTE["emerald"], PALETTE["marigold"], "#7a3b22"][(j + k) % 5], 0.3)
        if "clock" in items:
            P.fill(c, circle(Ww * 0.28, gy - 1180, 62), PALETTE["cream"], 0.2)
            P.stroke(c, circle(Ww * 0.28, gy - 1180, 62), "#6a4a2a", 8)
            P.stroke(c, poly([(Ww * 0.28, gy - 1180), (Ww * 0.28, gy - 1222)], closed=False), "#2a1a14", 6)
            P.stroke(c, poly([(Ww * 0.28, gy - 1180), (Ww * 0.28 + 30, gy - 1180)], closed=False), "#2a1a14", 6)
        if "phone" in items:
            P.fill(c, poly([(Ww * 0.1, gy - 760), (Ww * 0.16, gy - 760), (Ww * 0.16, gy - 640), (Ww * 0.1, gy - 640)]), "#e8e2d6", 0.3)
        floor(c, P, Ww, gy)
        P.fill(c, ellipse(Ww * 0.5, gy + 150, 560, 90), PALETTE["sapphire"], 0.25, kind="cloth")  # rug
        P.pattern_dots(c, ellipse(Ww * 0.5, gy + 150, 540, 80), PALETTE["marigold"], 26, 4, 180)
        if "geraniums" in items:
            geraniums(c, P, wx + (20 if big_window else 70), sill_y, 2 if big_window else 3)

    def furn(c):
        if "armchair" in items:
            armchair(c, P, Ww * 0.36, gy + 40)
        if "lamp" in items:
            lamp(c, P, Ww * 0.14, gy + 30, lamp_lit)
        if "table" in items:
            table(c, P, Ww * 0.62, gy + 60)
            if "mug" in items:
                mug(c, P, Ww * 0.6, gy - 200)
            if "mug2" in items:
                mug(c, P, Ww * 0.72, gy - 200, band=PALETTE["sapphire"])
            if "box" in items:
                P.fill(c, poly([(Ww * 0.52, gy - 200), (Ww * 0.7, gy - 200), (Ww * 0.7, gy - 290), (Ww * 0.52, gy - 290)]), "#c9a87a", 0.3)
                P.fill(c, ellipse(Ww * 0.61, gy - 290, 90, 16), PALETTE["cream"], 0.2)
        if "chair" in items:
            chair(c, P, Ww * 0.36, gy + 60)
        if "chair2" in items:
            chair(c, P, Ww * 0.84, gy + 60, facing=-1)

    def light(c, t):
        fx.god_rays(c, t, (Ww * window_at, gy - 1020), 118, 1500, spread=26, n=5, alpha=0.12 if not warm else 0.07, seed=seed,
                    color="#fff1c9" if not warm else "#ffd28a")
        if warm and lamp_lit:
            fx.glow(c, Ww * 0.5, gy - 700, 900, "#ffb860", 0.12 + 0.02 * math.sin(t * 0.8))
        for p in pfx:
            p.draw(c, t)

    def fore(c):
        if "table_fore" in items:
            table(c, P, Ww * 0.5, Wh * 1.06, 900)

    breeze = 1.8 if (sash_open or big_window) else 1.0

    def curtains(c, t):
        """Curtains breathe in the draught from the window (big, slow secondary motion)."""
        for s_ in (-1, 1):
            cx0 = wx - 40 if s_ < 0 else wx + wwid + 40
            sw = breeze * (16 * math.sin(2 * math.pi * 0.23 * t + (0 if s_ < 0 else 1.7)) + 7 * math.sin(2 * math.pi * 0.61 * t + s_))
            cur = poly([(cx0, wy - 60), (cx0 - s_ * 110, wy - 60), (cx0 - s_ * 70 + sw * 0.5, wy + whgt * 0.5),
                        (cx0 - s_ * 120 + sw, wy + whgt + 60), (cx0 + sw * 0.8, wy + whgt + 60)], smooth=True)
            P.fill(c, cur, PALETTE["ruby"] if not warm else "#8c1a2c", 0.35, kind="cloth")
            P.pattern_dots(c, cur, PALETTE["gold"], 24, 3, 150)
        P.fill(c, poly([(wx - 80, wy - 80), (wx + wwid + 80, wy - 80), (wx + wwid + 80, wy - 58), (wx - 80, wy - 58)]), "#6a4a2a", 0.3, kind="wood")

    layers = [Layer("room", 0.55, back, curtains), Layer("furniture", 0.85, furn), Layer(ACTORS, 1.0), Layer("fore", 1.2, fore),
              Layer("light", 1.0, None, light)]
    spots = {"left": (Ww * 0.32, gy + 60), "center": (Ww * 0.5, gy + 70), "right": (Ww * 0.68, gy + 70),
             "window": (Ww * window_at, wy + whgt / 2), "chair": (Ww * 0.36, gy + 60)}
    if "table" in items:
        spots["table"] = (Ww * 0.62, gy - 236)
    if big_window:
        spots["sill"] = (Ww * window_at + wwid * 0.18, sill_y)
    return Scene(name, (Ww, Wh), layers, gy + 70, P, spots, overlay=proscenium(P) if P.name == "C" else None)


# ---------------------------------------------------------------- the Pip story's locations -> builder parameters
LOCATIONS = {
    "village_dawn": ("exterior", dict(tod="dawn", season="autumn", tall=True, house_at=0.08, church_at=0.6, maple_at=0.82, mist=True)),
    "kitchen_dawn": ("interior", dict(tod="dawn", items=("counter", "geraniums", "table", "mug", "chair"))),
    "kitchen_table": ("interior", dict(tod="morning", items=("table", "mug", "chair", "chair2", "geraniums"))),
    "cupboard": ("interior", dict(tod="morning", items=("shelf", "counter", "geraniums"))),
    "house_quiet": ("exterior", dict(tod="dusk", season="autumn", house_at=0.18, maple_at=0.78)),
    "storm_night": ("exterior", dict(tod="storm", season="autumn", tall=True, house_at=0.1, maple_at=0.74, weather="storm", bend=60, fx_list=("rain", "leaves"))),
    "storm_maple": ("exterior", dict(tod="storm", season="autumn", maple_at=0.5, weather="storm", bend=90, fx_list=("rain", "leaves"))),
    "yard_morning": ("exterior", dict(tod="morning", season="autumn", house_at=0.02, maple_at=0.74, weather="after_storm", fx_list=("motes", "leaves"))),
    "yard_tall": ("exterior", dict(tod="morning", season="autumn", tall=True, house_at=0.02, maple_at=0.74, weather="after_storm", fx_list=("motes", "leaves"))),
    "leaves_close": ("exterior", dict(tod="morning", season="autumn", maple_at=None, fence_y=False, ground_frac=0.55, fx_list=("motes",))),
    "branches_up": ("exterior", dict(tod="morning", season="autumn", tall=True, maple_at=0.5, fence_y=False, fx_list=("leaves",))),
    "kitchen_warm": ("interior", dict(tod="day", items=("counter", "table", "chair", "phone", "geraniums"))),
    "porch_door": ("exterior", dict(tod="day", season="autumn", house_at=0.28, maple_at=0.9, fx_list=("leaves",))),
    "kitchen_night": ("interior", dict(tod="night", items=("table", "box", "chair", "clock"), lamp_lit=True)),
    "living_room": ("interior", dict(tod="day", season="winter", items=("armchair", "lamp", "shelf"), fx_list=("motes",))),
    "winter_window": ("interior", dict(tod="day", season="winter", items=("geraniums", "counter"), fx_list=("motes",))),
    "village_winter": ("exterior", dict(tod="day", season="winter", tall=True, house_at=0.12, church_at=0.62, maple_at=0.86, fx_list=("snow",))),
    "kitchen_spring": ("interior", dict(tod="day", season="spring", items=("table", "mug", "mug2", "chair", "geraniums"))),
    "night_vigil": ("interior", dict(tod="night", items=("armchair", "lamp", "box"), lamp_lit=True)),
    "window_ramp": ("interior", dict(tod="day", season="spring", items=("geraniums",), sash_open=1.0, big_window=True, ramp=True)),
    "maple_trunk": ("exterior", dict(tod="day", season="spring", tall=True, maple_at=0.5, house_at=0.86, fence_y=False)),
    "window_dusk": ("interior", dict(tod="dusk", season="summer", items=("geraniums",), big_window=True)),
    "porch_dusk": ("exterior", dict(tod="dusk", season="summer", house_at=0.3, maple_at=0.88, fx_list=("fireflies",))),
    "living_evening": ("interior", dict(tod="dusk", season="autumn", items=("armchair", "lamp"))),
    "snowy_sill": ("interior", dict(tod="morning", season="winter", items=("geraniums",), fx_list=("snow",), big_window=True)),
    "kitchen_two_mugs": ("interior", dict(tod="dusk", season="spring", items=("table", "mug", "mug2", "chair", "chair2"))),
    "open_window_golden": ("exterior", dict(tod="golden", season="spring", tall=True, house_at=0.05, maple_at=0.62, fx_list=("motes", "petals"))),
}


def build_location(name, painter, fx_override=None, seed=1, scenes_root=None, art="c", plate_dir=None):
    """Painted plates from assets/scenes when the location has some (engine2/assets.py), else the vector scene."""
    kind, kw = LOCATIONS.get(name, ("exterior", dict(tod="day")))
    kw = dict(kw)
    if fx_override:
        kw["fx_list"] = tuple(f for f in fx_override if f != "lightning")
    from . import assets
    plates = None if scenes_root == "" or art == "v1" else assets.scene_dir(name, scenes_root=scenes_root)
    v1 = None
    if art == "v1":  # art source A: the V1 painted plate split into depth layers
        from .v1art import load_v1_scene
        v1 = load_v1_scene(name, painter, kw, seed=seed, fx_override=[f for f in (fx_override or []) if f != "lightning"] or None, plate_dir=plate_dir)
    if v1 is not None:
        sc = v1
    elif plates:
        sc = assets.load_plates(plates, painter, seed=seed, fx_override=[f for f in (fx_override or []) if f != "lightning"] or None)
        sc.overlay = proscenium(painter) if painter.name == "C" else None
    else:
        sc = exterior(painter, name, seed=seed, **kw) if kind == "exterior" else interior(painter, name, seed=seed, **kw)
    from .post import mood_of
    sc.mood = kw.get("mood") or mood_of(kind, kw)
    tod = kw.get("tod", "day")
    Ww = sc.world[0]
    if kind == "interior" and "window" in sc.spots:
        sc.light = (sc.spots["window"], 0.55)
    elif kind == "interior":
        sc.light = ((Ww * 0.62, sc.ground - 1000), 0.55)
    else:
        gy = sc.ground - 80
        sc.light = ((Ww * 0.78, gy - 1400) if tod in ("night", "dusk") else (Ww * 0.5, gy - 1800) if tod == "storm"
                    else (Ww * 0.22, gy - 1250), 0.08)
    sc.haze = mix(SKIES[tod][2], SKIES[tod][1], 0.4)
    return sc
