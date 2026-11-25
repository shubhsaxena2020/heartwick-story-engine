"""Heartwick layered backgrounds (rich edition). Each scene = 5-6 static depth layers (PNG, world px * Q) + a
scene.json the loader (assets/scenes/load_scene.py) turns into an engine2.scene.Scene with the engine's own
particles, rays, mist and twinkles on top. World size and layer depths follow engine2/scenery.py
(1400 x 2100 world, tall scenes 1400 x 3000; ground line at 0.8 of the height; margin 260 around).

Our own New England folk-tale world: clapboard houses, white steeples, red barns, a covered bridge, sugar maples,
stone walls, Green-Mountain ridges, quilts and folk-painted interiors. No candles or flames (the stove is closed,
the lamps are electric-style glass shades).
python assets/src/scenes.py [name ...]"""
import json, math, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from paint import *

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scenes"))
MARGIN = 260
Q = 1.25


# ---------------------------------------------------------------- shapes
def maple_leaf(x, y, s, rot=0.0):
    pts = [(0, -16), (3, -8), (9, -12), (8, -4), (15, -6), (11, 1), (14, 5), (6, 5), (5, 10), (1, 7), (0, 16), (-1, 7), (-5, 10),
           (-6, 5), (-14, 5), (-11, 1), (-15, -6), (-8, -4), (-9, -12), (-3, -8)]
    p = poly([(px * s, py * s) for px, py in pts])
    m = skia.Matrix()
    m.setRotate(rot)
    m.postTranslate(x, y)
    p.transform(m)
    return p


LEAF = ["#9e1b34", "#c8332b", "#e0602a", "#ee8a2a", "#f2b23c", "#f6cf5a"]


def leaf(c, x, y, s, rnd, cols=LEAF, a=255):
    col = rnd.choice(cols)
    p = maple_leaf(x, y, s, rnd.uniform(0, 360))
    b = bounds(p)
    c.drawPath(p, paint(shader=lin((b.left(), b.top()), (b.right(), b.bottom()), [(0, shade(col, 0.25)), (1, shade(col, -0.3))]), a=a))
    stroke(c, poly([(x, y - 12 * s), (x, y + 10 * s)], closed=False), shade(col, -0.45), 0.6 * s, int(a * 0.6))


def canopy(c, cx, cy, rx, ry, seed, cols=LEAF, light=(-0.5, -0.7), n=70, edge_leaves=160, dark="#5a1020"):
    """A sugar-maple crown: clusters of lit leaf-masses (gold on the lit side, ruby in shadow) + leaves on the rim."""
    rnd = random.Random(seed)
    clusters = []
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi)
        rr = math.sqrt(rnd.random())
        x, y = cx + math.cos(a) * rx * rr * 0.92, cy + math.sin(a) * ry * rr * 0.9
        clusters.append((x, y, rnd.uniform(0.16, 0.28) * rx))
    clusters.sort(key=lambda q: (q[1] - cy) * -light[1] + (q[0] - cx) * -light[0], reverse=True)  # shadow side first
    silhouette = skia.Path()
    for x, y, r in clusters:
        silhouette.addCircle(x, y, r * 1.05)
    c.drawPath(silhouette, paint(dark, 255))
    # soft volume first (shadow core, lit dome), then the crown is packed with real maple-leaf shapes, shadow side first
    c.drawPath(silhouette, paint(shader=rad((cx + light[0] * rx * 0.4, cy + light[1] * ry * 0.4), max(rx, ry) * 1.2,
                                          [(0, cols[-2]), (0.5, cols[len(cols) // 2]), (1, dark)])))
    ls = max(1.0, rx / 150)  # leaf size scales with the tree
    n_leaves = int(30 * n * (1 if rx > 200 else 0.5))
    pts = []
    for _ in range(n_leaves):
        x, y, r = clusters[rnd.randrange(len(clusters))]
        a = rnd.uniform(0, 2 * math.pi)
        rr = r * math.sqrt(rnd.random()) * 1.1
        pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
    pts.sort(key=lambda q: (q[0] - cx) * light[0] + (q[1] - cy) * light[1])
    for x, y in pts:
        u = ((x - cx) * light[0] + (y - cy) * light[1]) / max(rx, ry)
        k = max(0, min(len(cols) - 1, int((u + 0.85) / 1.7 * len(cols) + rnd.uniform(-0.9, 0.9))))
        leaf(c, x, y, ls * rnd.uniform(0.8, 1.25), rnd, [cols[k]])
    for _ in range(edge_leaves):  # ragged rim of single leaves
        a = rnd.uniform(0, 2 * math.pi)
        x, y = cx + math.cos(a) * rx * rnd.uniform(0.9, 1.1), cy + math.sin(a) * ry * rnd.uniform(0.9, 1.1)
        leaf(c, x, y, ls * rnd.uniform(0.6, 1.0), rnd, cols)


def trunk(c, x, y, h, w, seed, col="#4a2e1e", lean=0.0, branches=True):
    rnd = random.Random(seed)
    top = (x + lean, y - h)
    p = smooth([(x - w * 0.62, y + 10), (x - w * 0.48, y - h * 0.15), (x - w * 0.28 + lean * 0.6, y - h * 0.6), (top[0] - w * 0.2, top[1]),
                (top[0] + w * 0.2, top[1]), (x + w * 0.3 + lean * 0.6, y - h * 0.6), (x + w * 0.5, y - h * 0.15), (x + w * 0.66, y + 10)])
    b = bounds(p)
    c.drawPath(p, paint(shader=lin((b.left(), 0), (b.right(), 0), [(0, shade(col, -0.45)), (0.3, shade(col, 0.25)), (0.55, col), (1, shade(col, -0.5))])))
    with clip(c, p):  # bark furrows
        for k in range(16):
            xx = b.left() + rnd.uniform(0, b.width())
            stroke(c, smooth([(xx, y + 10), (xx + rnd.uniform(-6, 6), y - h * 0.4), (xx + lean * 0.5 + rnd.uniform(-8, 8), y - h)], closed=False),
                   shade(col, -0.5), rnd.uniform(1.5, 3.5), 150)
    if branches:
        for k, (dx, dy, ang, L) in enumerate([(-0.1, 0.72, -140, 0.45), (0.1, 0.78, -40, 0.5), (0.0, 0.92, -110, 0.35), (0.05, 0.9, -60, 0.4)]):
            bx, by = x + lean * dy + dx * w, y - h * dy
            ex, ey = bx + math.cos(math.radians(ang)) * h * L, by + math.sin(math.radians(ang)) * h * L
            br = capsule(bx, by, ex, ey, w * 0.16, w * 0.05)
            c.drawPath(br, paint(shader=lin((bx - 20, by), (bx + 20, by), [(0, shade(col, -0.3)), (1, shade(col, 0.1))])))
    # roots
    for k in (-1, 1):
        stroke(c, smooth([(x + k * w * 0.4, y - 10), (x + k * w * 0.9, y + 6), (x + k * w * 1.3, y + 14)], closed=False), shade(col, -0.2), w * 0.18)


def ridge(c, W, y, amp, seed, col, forest=None, n=7, top_col=None, haze=None):
    """A mountain ridge with an autumn forest texture (dots of colour that thin out toward the top)."""
    rnd = random.Random(seed)
    pts = [(-MARGIN - 50, y + 400)]
    xs = [-MARGIN - 50 + i * (W + 2 * MARGIN + 100) / (n * 3) for i in range(n * 3 + 1)]
    for i, x in enumerate(xs):
        pk = math.sin(i / 3.0 * math.pi + seed) * 0.5 + 0.5
        pts.append((x, y - amp * (0.35 + 0.65 * pk) - rnd.uniform(0, amp * 0.18)))
    pts.append((W + MARGIN + 50, y + 400))
    p = smooth(pts[1:-1], closed=False)
    p.lineTo(W + MARGIN + 50, y + 1600)
    p.lineTo(-MARGIN - 50, y + 1600)
    p.close()
    c.drawPath(p, paint(shader=lin((0, y - amp), (0, y + 300), [(0, top_col or shade(col, 0.15)), (1, shade(col, -0.25))])))
    if forest:
        with clip(c, p):
            for _ in range(int((W + 2 * MARGIN) * 0.9)):
                x = rnd.uniform(-MARGIN, W + MARGIN)
                yy = rnd.uniform(y - amp, y + 260)
                if not p.contains(x, yy - 6):
                    continue
                r = rnd.uniform(3, 7) * (1 + (yy - y) / 600)
                fc = rnd.choice(forest)
                c.drawCircle(x, yy, r, paint(shader=rad((x - r * 0.3, yy - r * 0.4), r * 1.3, [(0, shade(fc, 0.25)), (1, shade(fc, -0.25))])))
    if haze:
        c.drawPath(p, paint(shader=lin((0, y - amp), (0, y + 300), [(0, haze, 0), (1, haze, 150)])))


def clouds(c, W, y, seed, lit="#fff1dc", body="#c9d3e6", shadow="#8c98b8", n=5, s=1.0, a=255):
    rnd = random.Random(seed)
    for k in range(n):
        cx = rnd.uniform(-100, W + 100)
        cy = y + rnd.uniform(-160, 160)
        w = rnd.uniform(200, 380) * s
        puffs = [circle(cx + (i - 3) * w * 0.16, cy - math.sin(i / 6 * math.pi) * w * 0.18 * rnd.uniform(0.8, 1.2), w * rnd.uniform(0.13, 0.2)) for i in range(7)]
        base = rect(cx - w * 0.52, cy - w * 0.05, cx + w * 0.52, cy + w * 0.08)
        cl = union(*puffs, base)
        b = bounds(cl)
        c.drawPath(cl, paint(shader=lin((0, b.top()), (0, b.bottom()), [(0, lit), (0.45, body), (1, shadow)]), a=a))
        with clip(c, cl):
            c.drawPath(offset(cl, 8, 10), paint(shadow, int(a * 0.5), blur=14))


def stars(c, W, H, seed, n=500, big=40):
    rnd = random.Random(seed)
    for _ in range(n):
        x, y = rnd.uniform(-MARGIN, W + MARGIN), rnd.uniform(-MARGIN, H)
        c.drawCircle(x, y, rnd.uniform(0.6, 1.8), paint(rnd.choice(["#ffffff", "#fff2c8", "#d8e4ff"]), int(rnd.uniform(90, 230))))
    for _ in range(big):
        x, y = rnd.uniform(-MARGIN, W + MARGIN), rnd.uniform(-MARGIN, H * 0.8)
        r = rnd.uniform(2, 4)
        glow(c, x, y, r * 6, "#fff3cf", 110)
        for ang in (0, 90):
            m = skia.Matrix()
            spark = poly([(x - r * 4, y), (x, y - r * 0.6), (x + r * 4, y), (x, y + r * 0.6)])
            m.setRotate(ang, x, y)
            spark.transform(m)
            c.drawPath(spark, paint("#fff8e0", 220))


# ---------------------------------------------------------------- buildings (New England)
def clapboard(c, path, col, step=10):
    b = bounds(path)
    with clip(c, path):
        c.drawPath(path, paint(shader=lin((b.left(), 0), (b.right(), 0), [(0, shade(col, 0.15)), (0.6, col), (1, shade(col, -0.2))])))
        y = b.top()
        while y < b.bottom():
            c.drawRect(skia.Rect(b.left(), y + step - 2, b.right(), y + step), paint(shade(col, -0.25), 150))
            c.drawRect(skia.Rect(b.left(), y + step, b.right(), y + step + 1.2), paint(shade(col, 0.35), 140))
            y += step


def window(c, x, y, w, h, lit=True, glow_col="#ffc86a", shutters="#1f6a52", frame="#fff4e2", night=False):
    if shutters:
        for sx in (x - w * 0.42, x + w):
            sp = rect(sx, y, sx + w * 0.42, y + h)
            c.drawPath(sp, paint(shader=lin((sx, 0), (sx + w * 0.42, 0), [(0, shade(shutters, 0.2)), (1, shade(shutters, -0.25))])))
            for k in range(int(h / 9)):
                stroke(c, poly([(sx + 2, y + 5 + k * 9), (sx + w * 0.42 - 2, y + 5 + k * 9)], closed=False), shade(shutters, -0.4), 1.2, 160)
    pane = rect(x, y, x + w, y + h)
    if lit:
        c.drawPath(pane, paint(shader=lin((0, y), (0, y + h), [(0, shade(glow_col, 0.35)), (1, shade(glow_col, -0.15))])))
        if night:
            glow(c, x + w / 2, y + h / 2, max(w, h) * 1.6, glow_col, 120)
    else:
        c.drawPath(pane, paint(shader=lin((x, y), (x + w, y + h), [(0, "#cfe0ee"), (0.5, "#8aa6c4"), (1, "#5a7090")])))
        stroke(c, poly([(x + w * 0.2, y + h * 0.8), (x + w * 0.7, y + h * 0.2)], closed=False), "#ffffff", 3, 90)
    stroke(c, pane, frame, 4)
    stroke(c, poly([(x + w / 2, y), (x + w / 2, y + h)], closed=False), frame, 2.5)
    stroke(c, poly([(x, y + h / 2), (x + w, y + h / 2)], closed=False), frame, 2.5)
    c.drawRect(skia.Rect(x - 6, y + h, x + w + 6, y + h + 7), paint(shade(frame, -0.1)))


def house(c, x, y, s, wall="#f2ead8", roof="#7a1f2e", trim="#fff4e2", door="#9e1b34", shutters="#1f6a52", lit=True, night=False, porch=True):
    """A New England cape/farmhouse: clapboard, steep gable, chimney, shuttered windows, porch with columns,
    gingerbread trim in gold, a fan light over the door."""
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    W, H = 360, 220
    body = rect(-W / 2, -H, W / 2, 0)
    clapboard(c, body, wall)
    # roof
    roofp = poly([(-W / 2 - 24, -H + 4), (0, -H - 170), (W / 2 + 24, -H + 4)])
    c.drawPath(roofp, paint(shader=lin((0, -H - 170), (0, -H), [(0, shade(roof, 0.2)), (1, shade(roof, -0.3))])))
    with clip(c, roofp):
        for k in range(16):
            yy = -H - 170 + k * 12
            for j in range(-20, 20):
                c.drawRect(skia.Rect(j * 22 + (k % 2) * 11, yy, j * 22 + (k % 2) * 11 + 20, yy + 10), paint(shade(roof, -0.25 + 0.04 * (j % 3)), 60))
    gable = poly([(-W / 2 + 8, -H), (0, -H - 150), (W / 2 - 8, -H)])
    clapboard(c, gable, wall)
    gold_line(c, poly([(-W / 2 - 24, -H + 4), (0, -H - 170), (W / 2 + 24, -H + 4)], closed=False), 3.2, beads=14)
    # attic round window
    c.drawCircle(0, -H - 62, 24, paint(trim))
    window(c, -16, -H - 78, 32, 32, lit=lit, shutters=None, night=night)
    # chimney
    ch = rect(W * 0.22, -H - 150, W * 0.22 + 36, -H - 60)
    c.drawPath(ch, paint(shader=lin((W * 0.22, 0), (W * 0.22 + 36, 0), [(0, "#a8483a"), (1, "#6a2a20")])))
    for k in range(7):
        stroke(c, poly([(W * 0.22, -H - 150 + k * 13), (W * 0.22 + 36, -H - 150 + k * 13)], closed=False), "#4a1a14", 1.2, 140)
    c.drawRect(skia.Rect(W * 0.22 - 4, -H - 158, W * 0.22 + 40, -H - 148), paint("#5a2a20"))
    # corner boards, windows, door
    for xx in (-W / 2, W / 2 - 12):
        c.drawRect(skia.Rect(xx, -H, xx + 12, 0), paint(trim))
    window(c, -W * 0.36, -H * 0.82, 58, 80, lit, shutters=shutters, night=night)
    window(c, W * 0.2, -H * 0.82, 58, 80, lit, shutters=shutters, night=night)
    d = rect(-30, -120, 30, 0)
    c.drawPath(d, paint(shader=lin((-30, 0), (30, 0), [(0, shade(door, 0.15)), (1, shade(door, -0.3))])))
    for k in range(2):
        for j in range(2):
            stroke(c, rect(-22 + j * 24, -108 + k * 52, -4 + j * 24, -66 + k * 52), shade(door, -0.4), 2)
    fan = skia.Path()
    fan.addArc(skia.Rect(-34, -150, 34, -90), 180, 180)
    c.drawPath(fan, paint(trim))
    window(c, -24, -142, 48, 20, lit, shutters=None, night=night)
    bead(c, 20, -58, 4)
    if porch:
        pr = poly([(-W / 2 - 30, -H * 0.55), (W / 2 + 30, -H * 0.55), (W / 2 + 10, -H * 0.62), (-W / 2 - 10, -H * 0.62)])
        c.drawPath(pr, paint(shader=lin((0, -H * 0.62), (0, -H * 0.55), [(0, shade(roof, 0.1)), (1, shade(roof, -0.35))])))
        for k in range(5):
            xx = -W / 2 - 20 + k * (W + 40) / 4
            col_ = rect(xx - 7, -H * 0.55, xx + 7, 0)
            cyl(c, col_, trim, depth=0.5, rim=False)
            gold_fill(c, rect(xx - 10, -H * 0.55, xx + 10, -H * 0.55 + 8))
        # gingerbread scallops
        for k in range(18):
            xx = -W / 2 - 26 + k * (W + 52) / 18
            arc = skia.Path()
            arc.addArc(skia.Rect(xx, -H * 0.55 - 6, xx + (W + 52) / 18, -H * 0.55 + 14), 0, 180)
            stroke(c, arc, "#e2ad3a", 2.2)
        c.drawRect(skia.Rect(-W / 2 - 34, -14, W / 2 + 34, 4), paint(shade(trim, -0.15)))
        for k in range(30):
            xx = -W / 2 - 30 + k * (W + 60) / 30
            c.drawRect(skia.Rect(xx, -56, xx + 4, -14), paint(trim))
        c.drawRect(skia.Rect(-W / 2 - 34, -60, W / 2 + 34, -52), paint(trim))
    c.restore()


def church(c, x, y, s, wall="#f6f1e6", roof="#3a4a66"):
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    body = rect(-110, -240, 110, 0)
    clapboard(c, body, wall, step=9)
    c.drawPath(poly([(-130, -236), (0, -330), (130, -236)]), paint(shader=lin((0, -330), (0, -236), [(0, shade(roof, 0.2)), (1, shade(roof, -0.3))])))
    tower = rect(-46, -440, 46, -236)
    clapboard(c, tower, wall, step=9)
    belfry = rect(-40, -520, 40, -440)
    clapboard(c, belfry, wall)
    for k in range(3):
        a = skia.Path()
        a.addRRect(skia.RRect.MakeRectXY(skia.Rect(-32 + k * 23, -506, -16 + k * 23, -456), 8, 8))
        c.drawPath(a, paint("#2a3248"))
    spire = poly([(-44, -520), (0, -760), (44, -520)])
    c.drawPath(spire, paint(shader=lin((-44, 0), (44, 0), [(0, shade(roof, 0.25)), (1, shade(roof, -0.35))])))
    gold_line(c, poly([(0, -760), (0, -800)], closed=False), 3)
    gem(c, 0, -804, 6, JEWEL["gold"])
    c.drawCircle(0, -380, 26, paint(shader=rad((-6, -386), 30, [(0, "#fff3c0"), (0.6, "#e2ad3a"), (1, "#8a5a14")])))
    c.drawCircle(0, -380, 20, paint("#fff8ea"))
    stroke(c, poly([(0, -380), (0, -394)], closed=False), "#2a1a14", 2.4)
    stroke(c, poly([(0, -380), (10, -380)], closed=False), "#2a1a14", 2.4)
    for k in (-1, 1):
        a = skia.Path()
        a.addRRect(skia.RRect.MakeRectXY(skia.Rect(k * 70 - 18, -200, k * 70 + 18, -90), 18, 18))
        c.drawPath(a, paint(shader=lin((0, -200), (0, -90), [(0, "#ffd890"), (1, "#c8742a")])))
        stroke(c, a, "#fff4e2", 3)
    dd = rect(-22, -110, 22, 0)
    c.drawPath(dd, paint("#7a1f2e"))
    c.restore()


def barn(c, x, y, s, col="#9e2a26"):
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    body = poly([(-160, 0), (-160, -170), (-120, -250), (0, -290), (120, -250), (160, -170), (160, 0)])
    b = bounds(body)
    c.drawPath(body, paint(shader=lin((-160, 0), (160, 0), [(0, shade(col, 0.15)), (1, shade(col, -0.3))])))
    with clip(c, body):
        for k in range(-16, 17):
            stroke(c, poly([(k * 10, -300), (k * 10, 0)], closed=False), shade(col, -0.35), 1.2, 150)
    c.drawPath(poly([(-176, -166), (-128, -258), (0, -300), (128, -258), (176, -166), (160, -164), (120, -246), (0, -284), (-120, -246), (-160, -164)]), paint("#3a3a44"))
    dr = rect(-70, -150, 70, 0)
    c.drawPath(dr, paint(shade(col, -0.1)))
    for (a, b_) in (((-70, -150), (70, 0)), ((-70, 0), (70, -150))):
        stroke(c, poly([a, b_], closed=False), "#fff4e2", 7)
    stroke(c, dr, "#fff4e2", 7)
    window(c, -22, -240, 44, 44, lit=False, shutters=None)
    rosette(c, 0, -205, 22, "#e2ad3a", center=JEWEL["ruby"])  # a painted barn star (our folk hex-sign)
    c.restore()


def covered_bridge(c, x, y, s, col="#8e2630"):
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    body = rect(-260, -120, 260, 0)
    c.drawPath(body, paint(shader=lin((0, -120), (0, 0), [(0, shade(col, 0.1)), (1, shade(col, -0.3))])))
    with clip(c, body):
        for k in range(-26, 27):
            stroke(c, poly([(k * 10, -120), (k * 10, 0)], closed=False), shade(col, -0.35), 1, 140)
    c.drawPath(poly([(-280, -116), (-250, -170), (250, -170), (280, -116)]), paint("#3a3a44"))
    a = skia.Path()
    a.addRRect(skia.RRect.MakeRectXY(skia.Rect(-250, -100, -190, 0), 26, 26))
    c.drawPath(a, paint("#1a1414"))
    for k in range(5):
        window(c, -130 + k * 60, -86, 26, 20, lit=False, shutters=None)
    c.restore()


def stone_wall(c, x0, x1, y, h, seed):
    rnd = random.Random(seed)
    x = x0
    while x < x1:
        for row in range(3):
            w = rnd.uniform(26, 46)
            yy = y - h + row * h / 3
            st = ellipse(x + w / 2 + (row % 2) * 12, yy + h / 6, w / 2, h / 6 + 2)
            col = rnd.choice(["#8a8478", "#a39c8e", "#6f6a62", "#9a8f80"])
            c.drawPath(st, paint(shader=rad((x + w * 0.35, yy), w * 0.7, [(0, shade(col, 0.3)), (1, shade(col, -0.35))])))
        x += rnd.uniform(26, 40)


def picket_fence(c, x0, x1, y, h=90, col="#f6f1e6"):
    x = x0
    while x < x1:
        p = poly([(x, y), (x, y - h + 10), (x + 7, y - h), (x + 14, y - h + 10), (x + 14, y)])
        c.drawPath(p, paint(shader=lin((x, 0), (x + 14, 0), [(0, shade(col, 0.1)), (1, shade(col, -0.25))])))
        x += 24
    for yy in (y - h * 0.7, y - h * 0.25):
        c.drawRect(skia.Rect(x0, yy, x1, yy + 8), paint(shade(col, -0.15)))


def pumpkin(c, x, y, r):
    for k, dx in enumerate((-0.45, 0.45, 0.0)):
        e = ellipse(x + dx * r, y, r * 0.62, r * 0.78)
        c.drawPath(e, paint(shader=rad((x + dx * r - r * 0.2, y - r * 0.3), r, [(0, "#ffc060"), (0.6, "#e8731e"), (1, "#8a3410")])))
    c.drawPath(capsule(x, y - r * 0.7, x + r * 0.15, y - r * 1.05, r * 0.1), paint("#4a5a22"))


def grass(c, x0, x1, y, h, col, seed, n=60, a=255):
    rnd = random.Random(seed)
    for _ in range(n):
        x = rnd.uniform(x0, x1)
        hh = h * rnd.uniform(0.5, 1.2)
        for k in range(5):
            ang = math.radians(-90 + (k - 2) * 14 + rnd.uniform(-6, 6))
            bl = smooth([(x - 3, y), (x + math.cos(ang) * hh * 0.5, y + math.sin(ang) * hh * 0.5), (x + math.cos(ang) * hh, y + math.sin(ang) * hh)], closed=False)
            stroke(c, bl, shade(col, rnd.uniform(-0.3, 0.25)), rnd.uniform(2.5, 4.5), a)


def ground_fill(c, W, gy, H, top, bottom, seed, path_col=None):
    p = rect(-MARGIN, gy - 30, W + MARGIN, H + MARGIN)
    c.drawPath(p, paint(shader=lin((0, gy - 30), (0, H), [(0, top), (1, bottom)])))
    rnd = random.Random(seed)
    with clip(c, p):
        for _ in range(500):  # mottled grass texture
            x, y = rnd.uniform(-MARGIN, W + MARGIN), rnd.uniform(gy - 20, H + MARGIN)
            c.drawPath(ellipse(x, y, rnd.uniform(10, 30), rnd.uniform(3, 7)), paint(shade(top, rnd.uniform(-0.25, 0.2)), 70, blur=3))


# ---------------------------------------------------------------- scene writer
class SceneOut:
    def __init__(self, name, W, H, tall=False, ground_frac=0.8):
        self.name, self.W, self.H, self.tall = name, W, H, tall
        self.gy = H * ground_frac
        self.dir = os.path.join(OUT, name)
        os.makedirs(self.dir, exist_ok=True)
        self.layers = []
        self.cfg = {"name": name, "world": [W, H], "ground": self.gy + 80, "tall": tall, "margin": MARGIN, "q": Q,
                    "spots": {"left": [W * 0.3, self.gy + 60], "center": [W * 0.5, self.gy + 80], "right": [W * 0.68, self.gy + 80]},
                    "fx": {}}

    def layer(self, lname, depth, draw, sway=None, fmt="png", haze=None):
        w, h = int((self.W + 2 * MARGIN) * Q), int((self.H + 2 * MARGIN) * Q)
        s = skia.Surface(w, h)
        c = s.getCanvas()
        c.clear(skia.ColorTRANSPARENT)
        c.scale(Q, Q)
        c.translate(MARGIN, MARGIN)
        draw(c)
        if haze:  # aerial perspective: tint what is already painted toward the light/sky colour
            col, a = haze
            c.drawRect(skia.Rect(-MARGIN, -MARGIN, self.W + MARGIN, self.H + MARGIN),
                       paint(shader=lin((0, self.gy - 1200), (0, self.gy + 200), [(0, col, int(a * 0.4)), (1, col, a)]), blend="srcatop"))
        fn = f"{len(self.layers):02d}_{lname}.{fmt}"
        img = s.makeImageSnapshot()
        if fmt == "jpg":
            img.save(os.path.join(self.dir, fn), skia.kJPEG, 92)
        else:
            from PIL import Image
            arr = img.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
            Image.fromarray(arr, "RGBA").save(os.path.join(self.dir, fn), optimize=True)
        L = {"name": lname, "depth": depth, "image": fn}
        if sway:
            L["sway"] = sway
        self.layers.append(L)

    def actors(self):
        self.layers.append({"name": "actors", "depth": 1.0})

    def save(self):
        self.cfg["layers"] = self.layers
        json.dump(self.cfg, open(os.path.join(self.dir, "scene.json"), "w"), indent=1)
        print(self.name, [l["name"] for l in self.layers])


# ---------------------------------------------------------------- the scenes
HAZE = {"village_autumn": "#f6d0a0", "maple_yard": "#f6d6a8", "night_sky": "#3a6a7a", "storm": "#4a6a7a", "castle_court": "#f8c898"}
AUTUMN_FOREST = ["#9e1b34", "#c8332b", "#e0602a", "#ee8a2a", "#f2b23c", "#3a6a3a", "#2a5a3a", "#b8432f"]


def village_autumn():
    S = SceneOut("village_autumn", 1400, 3000, tall=True)
    W, H, gy = S.W, S.H, S.gy

    def sky(c):
        c.drawRect(skia.Rect(-MARGIN, -MARGIN, W + MARGIN, H + MARGIN), paint(shader=lin((0, -MARGIN), (0, gy), [
            (0, "#1f3f7a"), (0.35, "#4f78b8"), (0.65, "#a8c4e0"), (0.85, "#f6d6a8"), (1, "#f8b878")])))
        glow(c, W * 0.24, gy - 1150, 520, "#fff0c8", 170)
        c.drawCircle(W * 0.24, gy - 1150, 78, paint(shader=rad((W * 0.24, gy - 1160), 80, [(0, "#fffbe8"), (1, "#ffe7a8")])))
        clouds(c, W, gy - 1900, 3, n=4, s=1.2)
        clouds(c, W, gy - 1450, 5, lit="#ffe8c8", body="#e8c8c0", shadow="#a89ab8", n=5)
    S.layer("sky", 0.08, sky, fmt="jpg")

    def far(c):
        ridge(c, W, gy - 900, 260, 1, "#5a6e9a", top_col="#8aa0c8", haze="#c8d6ec")
        ridge(c, W, gy - 640, 220, 2, "#4a6a7a", forest=["#b8603a", "#c87a3a", "#8a8a4a", "#a84a3a", "#6a7a5a"], haze="#d8c8c0")
        ridge(c, W, gy - 420, 160, 3, "#5a5a3a", forest=AUTUMN_FOREST)
    S.layer("far", 0.3, far, haze=(HAZE[S.name], 110))

    def mid(c):
        rnd = random.Random(9)
        church(c, W * 0.62, gy - 300, 0.75)
        covered_bridge(c, W * 0.18, gy - 250, 0.55)
        for k in range(14):  # far village maples
            x = rnd.uniform(-100, W + 100)
            trunk(c, x, gy - 230, 120, 18, k, branches=False)
            canopy(c, x, gy - 380, 90, 80, 100 + k, n=18, edge_leaves=30)
        barn(c, W * 0.9, gy - 210, 0.7)
        house(c, W * 0.38, gy - 190, 0.7, wall="#f2ead8", roof="#3a4a66", door="#1f3f9a", shutters="#7a1f2e")
    S.layer("mid", 0.6, mid, haze=(HAZE[S.name], 50))

    def ground(c):
        ground_fill(c, W, gy - 120, H, "#7a8a3a", "#4a5a2a", 11)
        road = smooth([(W * 0.42, gy - 120), (W * 0.52, gy - 120), (W * 0.9, H + 200), (W * 0.1, H + 200)])
        c.drawPath(road, paint(shader=lin((0, gy - 120), (0, H), [(0, "#c8b08a"), (1, "#8a6a4a")])))
        stone_wall(c, -MARGIN, W * 0.36, gy + 20, 60, 13)
        stone_wall(c, W * 0.62, W + MARGIN, gy + 30, 64, 14)
        house(c, W * 0.12, gy + 10, 1.05, wall="#fbf4e4", roof="#7a1f2e")
        trunk(c, W * 0.8, gy + 40, 820, 120, 21, lean=-30)
        canopy(c, W * 0.78, gy - 900, 430, 300, 22)
        rnd = random.Random(23)
        for _ in range(260):
            leaf(c, rnd.uniform(-MARGIN, W + MARGIN), rnd.uniform(gy + 30, H + 100), rnd.uniform(1.2, 2.2), rnd)
        for k, x in enumerate((W * 0.6, W * 0.66, W * 0.7)):
            pumpkin(c, x, gy + 70 + k * 6, 26 + k * 4)
        grass(c, -MARGIN, W + MARGIN, gy + 40, 40, "#6a7a32", 25, n=80)
    S.layer("ground", 0.92, ground)
    S.actors()

    def fore(c):
        rnd = random.Random(31)
        grass(c, -MARGIN, W + MARGIN, H + 60, 150, "#4a5a22", 33, n=60)
        for _ in range(40):
            leaf(c, rnd.uniform(-MARGIN, W + MARGIN), rnd.uniform(H - 60, H + 120), rnd.uniform(2.4, 3.6), rnd)
        # a framing maple branch in the top-left corner
        br = capsule(-MARGIN, -40, 380, 260, 26, 8)
        c.drawPath(br, paint(shader=lin((0, -40), (0, 260), [(0, "#6a4228"), (1, "#2a160c")])))
        for _ in range(90):
            leaf(c, rnd.uniform(-MARGIN, 420), rnd.uniform(-MARGIN, 320), rnd.uniform(2.2, 3.4), rnd)
    S.layer("fore", 1.25, fore)
    S.cfg["spots"].update({"under_maple": [W * 0.62, gy + 110], "canopy": [W * 0.78, gy - 900], "sky": [W * 0.5, gy - 1500]})
    S.cfg["fx"] = {"particles": ["leaves", "motes"], "rays": {"origin": [W * 0.24, gy - 1150], "angle": 62, "length": 2600, "alpha": 0.12},
                   "mist": {"y": gy - 200, "alpha": 0.22}, "birds": True}
    S.save()


def maple_yard():
    S = SceneOut("maple_yard", 1400, 2100)
    W, H, gy = S.W, S.H, S.gy

    def sky(c):
        c.drawRect(skia.Rect(-MARGIN, -MARGIN, W + MARGIN, H + MARGIN), paint(shader=lin((0, -MARGIN), (0, gy), [
            (0, "#3f6ab0"), (0.5, "#8fb4dc"), (0.85, "#f2e2c4"), (1, "#f6cf9a")])))
        glow(c, W * 0.2, gy - 1250, 460, "#fff0c8", 160)
        c.drawCircle(W * 0.2, gy - 1250, 64, paint("#fff8e0"))
        clouds(c, W, gy - 1350, 41, n=5)
    S.layer("sky", 0.08, sky, fmt="jpg")

    def far(c):
        ridge(c, W, gy - 560, 200, 4, "#5a6e9a", top_col="#8aa0c8", haze="#d8e0ee")
        ridge(c, W, gy - 360, 140, 5, "#5a5a3a", forest=AUTUMN_FOREST, haze="#e8d8c8")
        church(c, W * 0.86, gy - 300, 0.45)
    S.layer("far", 0.3, far, haze=(HAZE[S.name], 110))

    def mid(c):
        ground_fill(c, W, gy - 180, gy + 60, "#8a9a46", "#6a7a36", 43)
        barn(c, W * 0.66, gy - 160, 0.55)
        picket_fence(c, -MARGIN, W + MARGIN, gy - 30, 84)
        rnd = random.Random(45)
        for x in (W * 0.42, W * 0.95):
            trunk(c, x, gy - 140, 160, 24, int(x), branches=False)
            canopy(c, x, gy - 330, 120, 100, int(x) + 1, n=22, edge_leaves=40)
    S.layer("mid", 0.6, mid, haze=(HAZE[S.name], 50))

    def ground(c):
        ground_fill(c, W, gy - 30, H, "#7a8a3a", "#4a5a2a", 47)
        house(c, W * 0.02, gy + 20, 1.25, wall="#fbf4e4", roof="#7a1f2e", lit=True)
        trunk(c, W * 0.74, gy + 30, 800, 116, 49, lean=-20)
        canopy(c, W * 0.72, gy - 870, 430, 300, 50, n=80, edge_leaves=200)
        rnd = random.Random(51)
        pile = smooth([(W * 0.5, gy + 120), (W * 0.62, gy + 40), (W * 0.86, gy + 30), (W * 1.0, gy + 120)])
        c.drawPath(pile, paint("#8a2a1e"))
        for _ in range(220):
            leaf(c, rnd.uniform(W * 0.52, W * 0.98), rnd.uniform(gy + 30, gy + 120), rnd.uniform(1.4, 2.2), rnd)
        for _ in range(300):
            leaf(c, rnd.uniform(-MARGIN, W + MARGIN), rnd.uniform(gy + 30, H + 100), rnd.uniform(1.2, 2.0), rnd)
        # a fallen branch from last night's storm
        c.drawPath(capsule(W * 0.78, gy + 70, W * 1.06, gy + 100, 16, 9), paint("#4a2e1e"))
        rake = capsule(W * 0.3, gy - 160, W * 0.34, gy + 90, 4)
        c.drawPath(rake, paint("#a8743e"))
        grass(c, -MARGIN, W + MARGIN, gy + 30, 40, "#6a7a32", 53, n=70)
    S.layer("ground", 0.92, ground)
    S.actors()

    def fore(c):
        rnd = random.Random(55)
        grass(c, -MARGIN, W + MARGIN, H + 60, 140, "#4a5a22", 57, n=50)
        for _ in range(30):
            leaf(c, rnd.uniform(-MARGIN, W + MARGIN), rnd.uniform(H - 40, H + 100), rnd.uniform(2.6, 3.6), rnd)
    S.layer("fore", 1.25, fore)
    S.cfg["spots"].update({"under_maple": [W * 0.58, gy + 110], "canopy": [W * 0.72, gy - 870]})
    S.cfg["fx"] = {"particles": ["leaves", "motes"], "rays": {"origin": [W * 0.2, gy - 1250], "angle": 62, "length": 2200, "alpha": 0.12}, "birds": True}
    S.save()


def night_sky():
    S = SceneOut("night_sky", 1400, 3000, tall=True)
    W, H, gy = S.W, S.H, S.gy

    def sky(c):
        c.drawRect(skia.Rect(-MARGIN, -MARGIN, W + MARGIN, H + MARGIN), paint(shader=lin((0, -MARGIN), (0, gy), [
            (0, "#060c1e"), (0.45, "#0f2448"), (0.8, "#1f4e5c"), (1, "#3a6a6a")])))
        # a Milky Way band: soft teal/rose haze with dense small stars (kept teal/rose, away from indigo-and-gold)
        band = capsule(-200, gy - 2300, W + 200, gy - 900, 220)
        c.drawPath(band, paint("#5aa0a0", 50, blur=90))
        c.drawPath(capsule(-200, gy - 2250, W + 200, gy - 950, 90), paint("#d8a8b0", 40, blur=60))
        rnd = random.Random(61)
        for _ in range(1600):
            t = rnd.random()
            x = -200 + t * (W + 400) + rnd.gauss(0, 60)
            y = gy - 2300 + t * 1400 + rnd.gauss(0, 120)
            c.drawCircle(x, y, rnd.uniform(0.5, 1.4), paint("#ffffff", int(rnd.uniform(60, 200))))
        stars(c, W, gy - 300, 63, n=700, big=30)
        glow(c, W * 0.74, gy - 1700, 380, "#e8f0ff", 120)
        moon = circle(W * 0.74, gy - 1700, 90)
        c.drawPath(moon, paint(shader=rad((W * 0.72, gy - 1720), 100, [(0, "#fffcf0"), (0.8, "#f2ecd8"), (1, "#d8d0b8")])))
        with clip(c, moon):
            for (dx, dy, r) in ((-30, -20, 18), (20, 30, 12), (34, -30, 9), (-10, 40, 14)):
                c.drawCircle(W * 0.74 + dx, gy - 1700 + dy, r, paint("#c8c0a8", 70))
        clouds(c, W, gy - 1300, 65, lit="#5a7090", body="#2a3a5a", shadow="#141c30", n=4, a=200)
    S.layer("sky", 0.08, sky, fmt="jpg")

    def far(c):
        ridge(c, W, gy - 500, 240, 6, "#14243a", top_col="#2a4058")
        ridge(c, W, gy - 330, 160, 7, "#0f1c2c", forest=["#1a2a30", "#22343a", "#2a2030"])
        church(c, W * 0.3, gy - 280, 0.5, wall="#9aa4b8", roof="#1a2438")
    S.layer("far", 0.3, far, haze=(HAZE[S.name], 110))

    def mid(c):
        ground_fill(c, W, gy - 170, gy + 80, "#1f3a3a", "#142a2a", 67)
        rnd = random.Random(69)
        for k, x in enumerate((W * 0.08, W * 0.5, W * 0.9)):
            house(c, x, gy - 140, 0.55, wall="#aab4c4", roof="#3a2a3a", lit=True, night=True, porch=False)
        for k in range(6):
            x = rnd.uniform(0, W)
            trunk(c, x, gy - 140, 140, 20, 70 + k, col="#1a1414", branches=False)
            canopy(c, x, gy - 300, 100, 90, 80 + k, cols=["#3a1420", "#5a2028", "#6a3030", "#7a4a3a", "#8a5a40", "#9a6a4a"], n=18, edge_leaves=30, dark="#1a0a10")
    S.layer("mid", 0.6, mid, haze=(HAZE[S.name], 50))

    def ground(c):
        ground_fill(c, W, gy - 30, H, "#1f3a36", "#0c1a1a", 71)
        house(c, W * 0.74, gy + 20, 1.1, wall="#b8c0d0", roof="#4a2030", lit=True, night=True)
        picket_fence(c, -MARGIN, W * 0.5, gy + 40, 80, col="#a8b0c0")
        grass(c, -MARGIN, W + MARGIN, gy + 40, 40, "#1f3a30", 73, n=60)
    S.layer("ground", 0.92, ground)
    S.actors()

    def fore(c):
        grass(c, -MARGIN, W + MARGIN, H + 60, 150, "#0f2420", 75, n=60)
    S.layer("fore", 1.25, fore)
    S.cfg["fx"] = {"particles": ["fireflies"], "twinkle": {"n": 80, "y_max": gy - 600}, "mist": {"y": gy - 160, "alpha": 0.18}}
    S.save()


def storm():
    S = SceneOut("storm", 1400, 2100)
    W, H, gy = S.W, S.H, S.gy

    def sky(c):
        c.drawRect(skia.Rect(-MARGIN, -MARGIN, W + MARGIN, H + MARGIN), paint(shader=lin((0, -MARGIN), (0, gy), [
            (0, "#0a1420"), (0.5, "#1c3040"), (1, "#2c4a50")])))
        rnd = random.Random(81)
        for k in range(9):  # swirling storm-cloud banks with moonlit rims
            y = -100 + k * 150
            clouds(c, W, y, 82 + k, lit="#6a8aa0", body="#2a3c4c", shadow="#0c141c", n=4, s=1.6)
        bolt = poly([(W * 0.3, -MARGIN), (W * 0.36, 200), (W * 0.31, 210), (W * 0.4, 520), (W * 0.35, 525), (W * 0.46, 900)], closed=False)
        stroke(c, bolt, "#cfe0ff", 22, 60, blur=16)
        stroke(c, bolt, "#f4f8ff", 5, 230)
    S.layer("sky", 0.08, sky, fmt="jpg")

    def far(c):
        ridge(c, W, gy - 480, 220, 8, "#0e1a24", top_col="#24384a")
        ridge(c, W, gy - 320, 150, 9, "#0a141c", forest=["#14202a", "#1a2830", "#24201c"])
    S.layer("far", 0.3, far, haze=(HAZE[S.name], 110))

    def mid(c):
        ground_fill(c, W, gy - 160, gy + 80, "#1c2e2a", "#121e1c", 85)
        house(c, W * 0.14, gy - 130, 0.8, wall="#8a96a8", roof="#3a1a24", lit=True, night=True)
        picket_fence(c, -MARGIN, W + MARGIN, gy - 40, 80, col="#7a8494")
    S.layer("mid", 0.6, mid, haze=(HAZE[S.name], 50))

    def ground(c):
        ground_fill(c, W, gy - 30, H, "#1f3430", "#0c1814", 87)
        trunk(c, W * 0.7, gy + 30, 780, 112, 89, col="#2a1a14", lean=-120)
        canopy(c, W * 0.6, gy - 820, 400, 260, 90, cols=["#4a1020", "#6a1a24", "#8a2a28", "#9a3a2a", "#a8502e", "#b8683a"], dark="#1a0810")
        rnd = random.Random(91)
        for _ in range(200):
            leaf(c, rnd.uniform(-MARGIN, W + MARGIN), rnd.uniform(gy + 30, H + 100), rnd.uniform(1.2, 2.0), rnd,
                 cols=["#4a1020", "#6a1a24", "#8a2a28", "#9a3a2a"])
        with clip(c, rect(-MARGIN, gy + 40, W + MARGIN, H + MARGIN)):  # puddles reflecting the lit window
            for (x, y, rx) in ((W * 0.3, gy + 160, 160), (W * 0.62, gy + 300, 220)):
                pu = ellipse(x, y, rx, rx * 0.16)
                c.drawPath(pu, paint(shader=lin((0, y - rx * 0.16), (0, y + rx * 0.16), [(0, "#4a6a80"), (1, "#1a2a34")])))
                c.drawPath(ellipse(x - rx * 0.4, y, rx * 0.12, rx * 0.04), paint("#ffc86a", 150, blur=4))
        grass(c, -MARGIN, W + MARGIN, gy + 40, 46, "#1f3a30", 93, n=70)
    S.layer("ground", 0.92, ground)
    S.actors()

    def fore(c):
        grass(c, -MARGIN, W + MARGIN, H + 60, 160, "#0c1e18", 95, n=60)
    S.layer("fore", 1.25, fore)
    S.cfg["spots"].update({"under_maple": [W * 0.55, gy + 110]})
    S.cfg["fx"] = {"particles": ["rain", "leaves"], "lightning": {"period": 4.7, "flash": 0.12}, "sway": 7.0}
    S.save()


def wallpaper(c, x0, y0, x1, y1, base, motif, leafc):
    p = rect(x0, y0, x1, y1)
    c.drawPath(p, paint(shader=lin((0, y0), (0, y1), [(0, shade(base, 0.1)), (1, shade(base, -0.15))])))
    with clip(c, p):
        y, row = y0, 0
        while y < y1 + 60:
            x = x0 + (40 if row % 2 else 0)
            while x < x1 + 60:
                tulip(c, x, y, 1.6, motif, leaf=leafc, gold=True)
                c.drawCircle(x + 40, y + 34, 3, paint("#e2ad3a", 200))
                x += 80
            stroke(c, poly([(x0, y + 50), (x1, y + 50)], closed=False), shade(base, -0.12), 1.2, 90)
            y += 68
            row += 1


def kitchen():
    S = SceneOut("kitchen", 1400, 2100)
    W, H, gy = S.W, S.H, S.gy
    wx, wy, ww, wh = W * 0.62 - 220, gy - 1300, 440, 560

    def outside(c):  # what you see through the window (its own deep layer: parallax through the glass)
        c.drawRect(skia.Rect(wx - 80, wy - 80, wx + ww + 80, wy + wh + 80), paint(shader=lin((0, wy), (0, wy + wh), [(0, "#7fa8d8"), (0.7, "#f2dcb4"), (1, "#c8b080")])))
        ridge(c, W, wy + wh * 0.62, 90, 11, "#6a7a9a", haze="#d8e0ee")
        trunk(c, wx + ww * 0.7, wy + wh + 60, 360, 46, 101, branches=False)
        canopy(c, wx + ww * 0.64, wy + wh * 0.3, 220, 170, 102, n=40, edge_leaves=80)
    S.layer("outside", 0.3, outside)

    def room(c):
        wall = diff(rect(-MARGIN, -MARGIN, W + MARGIN, gy), rect(wx, wy, wx + ww, wy + wh))
        with clip(c, wall):
            wallpaper(c, -MARGIN, -MARGIN, W + MARGIN, gy - 420, "#f1e2c4", JEWEL["ruby_lt"], "#2f7a4a")
            # beadboard wainscot in sage with a gold chair rail
            wp = rect(-MARGIN, gy - 420, W + MARGIN, gy)
            c.drawPath(wp, paint(shader=lin((0, gy - 420), (0, gy), [(0, "#7a9a7e"), (1, "#4a6a50")])))
            for k in range(-20, 80):
                stroke(c, poly([(k * 22, gy - 410), (k * 22, gy)], closed=False), "#3a5a40", 1.6, 150)
            gold_fill(c, rect(-MARGIN, gy - 436, W + MARGIN, gy - 414))
        # window frame + sill + ruby curtains with gold tiebacks
        frame = diff(rect(wx - 22, wy - 22, wx + ww + 22, wy + wh + 22), rect(wx, wy, wx + ww, wy + wh))
        c.drawPath(frame, paint("#fff4e2"))
        stroke(c, poly([(wx + ww / 2, wy), (wx + ww / 2, wy + wh)], closed=False), "#fff4e2", 10)
        stroke(c, poly([(wx, wy + wh * 0.5), (wx + ww, wy + wh * 0.5)], closed=False), "#fff4e2", 10)
        c.drawRect(skia.Rect(wx - 50, wy + wh + 10, wx + ww + 50, wy + wh + 40), paint(shade("#fff4e2", -0.12)))
        for sgn in (-1, 1):
            x0 = wx - 30 if sgn < 0 else wx + ww + 30
            cur = smooth([(x0, wy - 70), (x0 - sgn * 150, wy - 70), (x0 - sgn * 110, wy + wh * 0.45), (x0 - sgn * 170, wy + wh + 80), (x0 + sgn * 20, wy + wh + 80),
                          (x0 + sgn * 10, wy + wh * 0.45)])
            c.drawPath(cur, paint("#8e1630"))
            pat_sprigs(c, cur, "#f2c24e", "#2f8a5a", 30, 0.9)
            shade_over(c, cur, strength=0.7)
            gold_line(c, smooth([(x0 - sgn * 120, wy + wh * 0.42), (x0 + sgn * 10, wy + wh * 0.48)], closed=False), 4, beads=10)
            gem(c, x0 - sgn * 60, wy + wh * 0.45, 7, JEWEL["emerald_lt"])
        rod = rect(wx - 200, wy - 86, wx + ww + 200, wy - 72)
        gold_fill(c, rod)
        # open shelf with blue-and-white plates, copper pots, jars
        sx0, sy = W * 0.04, gy - 1220
        for k in range(2):
            c.drawRect(skia.Rect(sx0 - 20, sy + k * 200, sx0 + 380, sy + k * 200 + 18), paint("#7a4a28"))
            for j in range(5):
                px, py = sx0 + 30 + j * 72, sy + k * 200 - 50
                if k == 0:
                    c.drawCircle(px, py, 42, paint(shader=rad((px - 10, py - 10), 46, [(0, "#ffffff"), (1, "#d8dce8")])))
                    stroke(c, circle(px, py, 34), JEWEL["sapphire"], 4)
                    rosette(c, px, py, 16, JEWEL["sapphire_lt"], center=JEWEL["sapphire"], ring=False)
                else:
                    jar = smooth([(px - 22, py - 50), (px + 22, py - 50), (px + 26, py + 50), (px - 26, py + 50)])
                    col = [JEWEL["ruby"], "#e9a53a", JEWEL["emerald"], "#b8432f", JEWEL["plum"]][j]
                    c.drawPath(jar, paint(shader=lin((px - 26, 0), (px + 26, 0), [(0, shade(col, -0.3)), (0.35, shade(col, 0.3)), (1, shade(col, -0.35))])))
                    c.drawRect(skia.Rect(px - 24, py - 62, px + 24, py - 48), paint("#e2ad3a"))
                    c.drawPath(rect(px - 16, py - 10, px + 16, py + 18), paint("#fff4e2", 220))
        # hanging copper pots
        for k, x in enumerate((W * 0.86, W * 0.94, W * 1.02)):
            stroke(c, poly([(x, gy - 1400), (x, gy - 1150 + k * 30)], closed=False), "#3a2a20", 3)
            pot = ellipse(x, gy - 1100 + k * 30, 46 - k * 6, 40 - k * 4)
            c.drawPath(pot, paint(shader=lin((x - 50, 0), (x + 50, 0), [(0, "#6a2a10"), (0.35, "#f0a060"), (0.6, "#b8582a"), (1, "#5a200c")])))
        # wall clock
        cx, cy = W * 0.24, gy - 1430
        c.drawCircle(cx, cy, 70, paint(shader=rad((cx - 10, cy - 10), 80, [(0, "#a8743e"), (1, "#4a2a14")])))
        c.drawCircle(cx, cy, 56, paint("#fff8ea"))
        for k in range(12):
            a = k * math.pi / 6
            c.drawCircle(cx + math.cos(a) * 46, cy + math.sin(a) * 46, 3, paint("#2a1a14"))
        stroke(c, poly([(cx, cy), (cx, cy - 36)], closed=False), "#2a1a14", 4)
        stroke(c, poly([(cx, cy), (cx + 26, cy + 8)], closed=False), "#2a1a14", 4)
        # floor: warm boards + braided oval rug
        fl = rect(-MARGIN, gy, W + MARGIN, H + MARGIN)
        c.drawPath(fl, paint(shader=lin((0, gy), (0, H), [(0, "#a86a3a"), (1, "#5a3418")])))
        for k in range(-10, 40):
            stroke(c, poly([(W / 2 + (k - 15) * 20, gy), (W / 2 + (k - 15) * 90, H + MARGIN)], closed=False), "#4a2a14", 2, 130)
        for k in range(9):
            r = 600 - k * 52
            col = [JEWEL["ruby"], "#e9a53a", JEWEL["emerald"], JEWEL["sapphire"], "#f4e8d0"][k % 5]
            e = ellipse(W * 0.5, gy + 180, r, r * 0.17)
            c.drawPath(e, paint(col))
            stroke(c, e, shade(col, -0.35), 3, 160)
    S.layer("room", 0.55, room)

    def furniture(c):
        # cast-iron cookstove (doors closed: no flame shown) on the left
        st = rect(W * 0.04, gy - 420, W * 0.34, gy - 60)
        c.drawPath(st, paint(shader=lin((W * 0.04, 0), (W * 0.34, 0), [(0, "#1a1a20"), (0.4, "#4a4a56"), (1, "#141418")])))
        for k in range(2):
            d = rect(W * 0.07 + k * 200, gy - 340, W * 0.07 + k * 200 + 160, gy - 200)
            c.drawPath(d, paint("#2a2a32"))
            stroke(c, d, "#c8a050", 3)
            bead(c, W * 0.07 + k * 200 + 140, gy - 270, 7)
        c.drawRect(skia.Rect(W * 0.03, gy - 440, W * 0.35, gy - 410), paint("#2a2a32"))
        gold_line(c, poly([(W * 0.03, gy - 442), (W * 0.35, gy - 442)], closed=False), 3)
        for k in range(4):
            c.drawRect(skia.Rect(W * 0.05 + k * 120, gy - 60, W * 0.05 + k * 120 + 22, gy + 20), paint("#141418"))
        kettle = ellipse(W * 0.2, gy - 480, 70, 52)
        c.drawPath(kettle, paint(shader=lin((W * 0.13, 0), (W * 0.27, 0), [(0, "#0f3a6a"), (0.35, "#6a9ad8"), (1, "#0a2244")])))
        pat_dots(c, kettle, "#ffffff", 14, 2, 160)
        stroke(c, smooth([(W * 0.15, gy - 520), (W * 0.2, gy - 580), (W * 0.25, gy - 520)], closed=False), "#1a1a20", 6)
        # table with a quilted runner, two mugs and a pie
        tx = W * 0.64
        top = rect(tx - 300, gy - 260, tx + 300, gy - 230)
        c.drawPath(top, paint(shader=lin((0, gy - 260), (0, gy - 230), [(0, "#c88a50"), (1, "#7a4a28")])))
        for lx in (tx - 270, tx + 250):
            leg_ = rect(lx, gy - 230, lx + 22, gy + 80)
            cyl(c, leg_, "#8a5a30", depth=0.8, rim=False)
        run = rect(tx - 120, gy - 262, tx + 120, gy - 226)
        c.drawPath(run, paint(JEWEL["emerald"]))
        pat_quilt(c, run, "#e2ad3a", 14, 140)
        for k, mx in enumerate((tx - 200, tx + 170)):
            mg = rect(mx, gy - 330, mx + 50, gy - 262)
            col = [JEWEL["ruby"], JEWEL["sapphire"]][k]
            c.drawPath(mg, paint(shader=lin((mx, 0), (mx + 50, 0), [(0, shade(col, -0.2)), (0.4, shade(col, 0.3)), (1, shade(col, -0.3))])))
            stroke(c, smooth([(mx + 50, gy - 318), (mx + 68, gy - 300), (mx + 50, gy - 280)], closed=False), col, 6)
            heart(c, mx + 25, gy - 296, 1.4, "#fff4e2", gold=False)
        pie = ellipse(tx, gy - 272, 70, 18)
        c.drawPath(pie, paint(shader=lin((0, gy - 290), (0, gy - 254), [(0, "#f0b868"), (1, "#a8642a")])))
        for k in range(6):
            stroke(c, poly([(tx - 50 + k * 20, gy - 284), (tx - 60 + k * 20, gy - 262)], closed=False), "#7a3a1a", 2)
        # geraniums on the sill
        for k in range(3):
            px = wx + 70 + k * 150
            pot = poly([(px - 34, wy + wh + 10), (px + 34, wy + wh + 10), (px + 26, wy + wh - 50), (px - 26, wy + wh - 50)])
            c.drawPath(pot, paint(shader=lin((px - 34, 0), (px + 34, 0), [(0, "#a8483a"), (0.4, "#e8865a"), (1, "#7a2a1a")])))
            rnd = random.Random(110 + k)
            for _ in range(8):
                c.drawCircle(px + rnd.uniform(-40, 40), wy + wh - 90 + rnd.uniform(-30, 20), rnd.uniform(14, 20), paint("#2f7a4a"))
            for _ in range(5):
                rosette(c, px + rnd.uniform(-30, 30), wy + wh - 120 + rnd.uniform(-30, 10), rnd.uniform(9, 13), "#e2283a", center="#ffd0a0", ring=False)
    S.layer("furniture", 0.85, furniture)
    S.actors()

    def fore(c):
        # a hanging bunch of dried herbs and a basket of apples in the near corners
        rnd = random.Random(121)
        for k in range(6):
            x = W * 0.05 + k * 18
            stroke(c, smooth([(W * 0.1, -MARGIN), (x, 120), (x + 6, 260)], closed=False), "#6a8a4a", 5)
            for _ in range(5):
                leaf_sprig(c, x + rnd.uniform(-8, 8), rnd.uniform(140, 260), 3.0, "#7a9a5a", rnd.uniform(-30, 30))
        bx, by = W * 0.96, H + 20
        bk = smooth([(bx - 200, by - 120), (bx + 200, by - 120), (bx + 170, by + 60), (bx - 170, by + 60)])
        c.drawPath(bk, paint("#b8864a"))
        pat_weave(c, bk, "#5a3418", 8, 160)
        for k in range(6):
            ax = bx - 150 + k * 60
            c.drawCircle(ax, by - 140, 46, paint(shader=rad((ax - 14, by - 156), 50, [(0, "#ff7060"), (0.6, "#c8202e"), (1, "#5a0a14")])))
    S.layer("fore", 1.2, fore)
    S.cfg["spots"].update({"window": [wx + ww / 2, wy + wh / 2], "table": [W * 0.64, gy - 260], "sill": [wx + ww * 0.5, wy + wh + 10]})
    S.cfg["fx"] = {"particles": ["motes"], "rays": {"origin": [wx + ww / 2, wy + wh * 0.4], "angle": 118, "length": 1600, "alpha": 0.12, "color": "#fff1c9"},
                   "glow": {"at": [W * 0.5, gy - 700], "r": 900, "color": "#ffb860", "alpha": 0.08}}
    S.save()


def castle():
    """For the folk-tale cast: a hilltop castle town at golden hour, banners, and a courtyard with a fountain."""
    S = SceneOut("castle_court", 1400, 2100)
    W, H, gy = S.W, S.H, S.gy

    def sky(c):
        c.drawRect(skia.Rect(-MARGIN, -MARGIN, W + MARGIN, H + MARGIN), paint(shader=lin((0, -MARGIN), (0, gy), [
            (0, "#2a4f8a"), (0.45, "#7aa0cc"), (0.8, "#f4c89a"), (1, "#f0a070")])))
        glow(c, W * 0.8, gy - 900, 520, "#ffe0a8", 170)
        clouds(c, W, gy - 1300, 131, lit="#fff0d0", body="#f0c0b0", shadow="#a888a8", n=5, s=1.1)
    S.layer("sky", 0.08, sky, fmt="jpg")

    def far(c):
        ridge(c, W, gy - 620, 260, 12, "#6a7aa0", top_col="#9aa8c8", haze="#f0d8c8")
        # hilltop castle silhouette with towers, conical roofs and banners
        cx, cy = W * 0.5, gy - 700
        hill = smooth([(cx - 700, cy + 300), (cx - 200, cy + 20), (cx + 200, cy + 10), (cx + 700, cy + 300)], closed=False)
        hill.lineTo(cx + 700, cy + 800)
        hill.lineTo(cx - 700, cy + 800)
        hill.close()
        c.drawPath(hill, paint(shader=lin((0, cy), (0, cy + 400), [(0, "#7a8a5a"), (1, "#4a5a3a")])))
        wallc = "#e8d8c0"
        c.drawPath(rect(cx - 260, cy - 120, cx + 260, cy + 30), paint(shader=lin((cx - 260, 0), (cx + 260, 0), [(0, shade(wallc, 0.1)), (1, shade(wallc, -0.3))])))
        for k in range(14):
            c.drawRect(skia.Rect(cx - 260 + k * 38, cy - 140, cx - 260 + k * 38 + 20, cy - 118), paint(wallc))
        for k, (tx, th, rr) in enumerate([(-240, 300, 44), (-110, 420, 54), (0, 520, 62), (120, 380, 50), (250, 280, 42)]):
            tw = rect(cx + tx - rr, cy - th, cx + tx + rr, cy + 20)
            c.drawPath(tw, paint(shader=lin((cx + tx - rr, 0), (cx + tx + rr, 0), [(0, shade(wallc, 0.2)), (0.5, wallc), (1, shade(wallc, -0.35))])))
            roofc = [JEWEL["sapphire"], JEWEL["ruby"], JEWEL["emerald"]][k % 3]
            cone = poly([(cx + tx - rr - 12, cy - th), (cx + tx, cy - th - rr * 2.6), (cx + tx + rr + 12, cy - th)])
            c.drawPath(cone, paint(shader=lin((cx + tx - rr, 0), (cx + tx + rr, 0), [(0, shade(roofc, 0.25)), (1, shade(roofc, -0.35))])))
            gold_line(c, poly([(cx + tx - rr - 12, cy - th), (cx + tx + rr + 12, cy - th)], closed=False), 3)
            fx, fy = cx + tx, cy - th - rr * 2.6
            stroke(c, poly([(fx, fy), (fx, fy - 60)], closed=False), "#5a3a14", 3)
            flag = smooth([(fx, fy - 60), (fx + 60, fy - 52), (fx + 44, fy - 44), (fx + 60, fy - 36), (fx, fy - 32)])
            c.drawPath(flag, paint([JEWEL["ruby"], "#e9a53a", JEWEL["sapphire"]][k % 3]))
            for j in range(3):
                window(c, cx + tx - 10, cy - th + 60 + j * 90, 20, 36, lit=True, shutters=None, frame="#c8a060")
        gate = skia.Path()
        gate.addRRect(skia.RRect.MakeRectXY(skia.Rect(cx - 50, cy - 90, cx + 50, cy + 30), 50, 50))
        c.drawPath(gate, paint("#3a2414"))
    S.layer("far", 0.3, far, haze=(HAZE[S.name], 110))

    def mid(c):
        # town houses: half-timbered gables in jewel colours
        rnd = random.Random(141)
        for k in range(8):
            x = -100 + k * 220
            y = gy - 200 + rnd.uniform(-20, 20)
            col = rnd.choice(["#f2e6cc", "#e8d4b0", "#f6ecd8"])
            body = rect(x - 90, y - 220, x + 90, y)
            c.drawPath(body, paint(shader=lin((x - 90, 0), (x + 90, 0), [(0, shade(col, 0.1)), (1, shade(col, -0.25))])))
            for (a, b) in [((x - 90, y - 110), (x + 90, y - 110)), ((x - 30, y - 220), (x - 30, y)), ((x + 30, y - 220), (x + 30, y)),
                           ((x - 90, y - 220), (x - 30, y - 110)), ((x + 90, y - 220), (x + 30, y - 110))]:
                stroke(c, poly([a, b], closed=False), "#5a3418", 7)
            roofc = rnd.choice([JEWEL["ruby"], JEWEL["sapphire"], JEWEL["emerald"], "#b8432f"])
            c.drawPath(poly([(x - 110, y - 216), (x, y - 360), (x + 110, y - 216)]), paint(shader=lin((0, y - 360), (0, y - 216), [(0, shade(roofc, 0.2)), (1, shade(roofc, -0.35))])))
            window(c, x - 70, y - 190, 34, 50, True, shutters=None, frame="#5a3418")
            window(c, x + 36, y - 190, 34, 50, True, shutters=None, frame="#5a3418")
            for j in range(4):
                c.drawCircle(x - 60 + j * 40, y - 100, 12, paint(rnd.choice(["#e2283a", "#f2b23c", "#d23a8a"])))
        # bunting strung across the square
        for row in range(2):
            line = smooth([(-MARGIN, gy - 520 + row * 120), (W * 0.5, gy - 420 + row * 120), (W + MARGIN, gy - 520 + row * 120)], closed=False)
            stroke(c, line, "#5a3a20", 2.5)
            for k, (x, y, ang) in enumerate(along(line, 46, 20)):
                fl = poly([(x - 16, y), (x + 16, y), (x, y + 36)])
                c.drawPath(fl, paint([JEWEL["ruby"], "#f2b23c", JEWEL["emerald"], JEWEL["sapphire"], "#fff4e2"][k % 5]))
    S.layer("mid", 0.6, mid, haze=(HAZE[S.name], 50))

    def ground(c):
        p = rect(-MARGIN, gy - 40, W + MARGIN, H + MARGIN)
        c.drawPath(p, paint(shader=lin((0, gy - 40), (0, H), [(0, "#d8c0a0"), (1, "#8a7058")])))
        with clip(c, p):  # cobblestones
            rnd = random.Random(151)
            for row in range(40):
                y = gy - 30 + row * row * 0.9 + row * 8
                for k in range(-4, 60):
                    x = k * (24 + row * 1.6) + (row % 2) * 12
                    col = rnd.choice(["#c8b090", "#b8a080", "#d8c4a4", "#a89070"])
                    st = ellipse(x, y, 10 + row * 0.7, 4 + row * 0.35)
                    c.drawPath(st, paint(shader=rad((x - 3, y - 2), 12 + row, [(0, shade(col, 0.3)), (1, shade(col, -0.3))])))
        # a stone fountain with gilded spout
        fx, fy = W * 0.28, gy + 80
        basin = ellipse(fx, fy, 230, 50)
        c.drawPath(rect(fx - 230, fy, fx + 230, fy + 80), paint(shader=lin((fx - 230, 0), (fx + 230, 0), [(0, "#e8dcc8"), (0.4, "#fff8ea"), (1, "#8a8070")])))
        c.drawPath(ellipse(fx, fy + 80, 230, 50), paint("#a89c88"))
        c.drawPath(basin, paint(shader=lin((0, fy - 50), (0, fy + 50), [(0, "#6ab0c8"), (1, "#2a6a8a")])))
        stroke(c, basin, "#fff8ea", 10)
        c.drawPath(rect(fx - 26, fy - 200, fx + 26, fy), paint(shader=lin((fx - 26, 0), (fx + 26, 0), [(0, "#e8dcc8"), (0.5, "#fff8ea"), (1, "#8a8070")])))
        gold_fill(c, ellipse(fx, fy - 210, 60, 18))
        for k in range(-2, 3):
            stroke(c, smooth([(fx, fy - 220), (fx + k * 50, fy - 260), (fx + k * 100, fy - 20)], closed=False), "#cfeaff", 4, 170)
    S.layer("ground", 0.92, ground)
    S.actors()

    def fore(c):
        # a stone archway framing the shot, with a hanging lantern-free garland of flowers
        arch = diff(rect(-MARGIN, -MARGIN, W + MARGIN, H + MARGIN), smooth([(60, H + MARGIN), (60, 400), (W * 0.5, 60), (W - 60, 400), (W - 60, H + MARGIN)]))
        c.drawPath(arch, paint(shader=lin((0, 0), (W, 0), [(0, "#6a5a4a"), (0.15, "#a89478"), (0.85, "#a89478"), (1, "#5a4a3a")])))
        with clip(c, arch):
            for k in range(60):
                y = -MARGIN + k * 50
                stroke(c, poly([(-MARGIN, y), (W + MARGIN, y)], closed=False), "#4a3a2a", 2, 160)
        gar = smooth([(60, 420), (W * 0.5, 160), (W - 60, 420)], closed=False)
        rnd = random.Random(161)
        for x, y, _ in along(gar, 26):
            c.drawCircle(x, y, 16, paint("#2f7a4a"))
            rosette(c, x + rnd.uniform(-8, 8), y + rnd.uniform(-8, 8), 10, rnd.choice(["#e2283a", "#f2b23c", "#d23a8a", "#fff4e2"]), ring=False)
    S.layer("fore", 1.2, fore)
    S.cfg["fx"] = {"particles": ["petals", "motes"], "rays": {"origin": [W * 0.8, gy - 900], "angle": 118, "length": 2200, "alpha": 0.1}, "birds": True}
    S.save()


SCENES = {"village_autumn": village_autumn, "maple_yard": maple_yard, "night_sky": night_sky, "storm": storm, "kitchen": kitchen,
          "castle_court": castle}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(SCENES):
        SCENES[n]()
