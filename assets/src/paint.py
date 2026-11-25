"""Heartwick part painter: the drawing vocabulary for the upgraded vector-cel art (direction B).

Everything is drawn with skia in a part's own canonical frame (limbs point +y, the body points -y), so shading that
is baked into a part turns with it, like the reference's airbrushed cut-out puppets. Our own Western folk vocabulary:
Pennsylvania-Dutch / Scandinavian folk embroidery (tulips, hearts, six-petal rosettes, vines), Fair-Isle and cable
knits, quilt and tartan weaves, gold braid, gem cabochons. No candles, no flames, nothing from the reference's
devotional iconography.
"""
import math
import skia

# ---------------------------------------------------------------- colour
JEWEL = {
    "ruby": "#9e1b34", "ruby_lt": "#d23a4e", "emerald": "#13735a", "emerald_lt": "#2fa37a", "sapphire": "#1f3f9a",
    "sapphire_lt": "#3f6fd0", "saffron": "#f0a020", "marigold": "#e9a53a", "gold": "#e2ad3a", "gold_lt": "#ffe59a",
    "gold_dk": "#8a5a14", "cream": "#fff4e2", "ivory": "#f6ead2", "plum": "#5a2350", "teal": "#1f6a72",
    "night": "#10233a", "ink": "#2a1a14",
}


def hexrgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def C(h, a=255):
    r, g, b = hexrgb(h)
    return skia.Color(r, g, b, int(max(0, min(255, a))))


def mix(c1, c2, u):
    a, b = hexrgb(c1), hexrgb(c2)
    return "#" + "".join(f"{int(round(x + (y - x) * u)):02x}" for x, y in zip(a, b))


def shade(c, k):
    """k<0 darken toward a warm plum shadow, k>0 lighten toward warm cream."""
    return mix(c, "#24101f", -k) if k < 0 else mix(c, "#fff6e6", k)


# ---------------------------------------------------------------- paths
def P():
    return skia.Path()


def circle(x, y, r):
    p = skia.Path()
    p.addCircle(x, y, r)
    return p


def ellipse(x, y, rx, ry, rot=0.0):
    p = skia.Path()
    p.addOval(skia.Rect(x - rx, y - ry, x + rx, y + ry))
    if rot:
        m = skia.Matrix()
        m.setRotate(rot, x, y)
        p.transform(m)
    return p


def rect(x0, y0, x1, y1):
    p = skia.Path()
    p.addRect(skia.Rect(x0, y0, x1, y1))
    return p


def capsule(x0, y0, x1, y1, r0, r1=None):
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


def smooth(pts, closed=True, tension=1.0):
    """Catmull-Rom spline through points."""
    p = skia.Path()
    n = len(pts)
    p.moveTo(*pts[0])
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        if not closed:
            p0, p3 = pts[max(0, i - 1)], pts[min(n - 1, i + 2)]
        t = tension / 6
        c1 = (p1[0] + (p2[0] - p0[0]) * t, p1[1] + (p2[1] - p0[1]) * t)
        c2 = (p2[0] - (p3[0] - p1[0]) * t, p2[1] - (p3[1] - p1[1]) * t)
        p.cubicTo(*c1, *c2, *p2)
    if closed:
        p.close()
    return p


def poly(pts, closed=True):
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    if closed:
        p.close()
    return p


def union(*paths):
    out = paths[0]
    for q in paths[1:]:
        try:
            out = skia.Op(out, q, skia.PathOp.kUnion_PathOp) or out
        except RuntimeError:
            pass
    return out


def diff(a, b):
    try:
        return skia.Op(a, b, skia.PathOp.kDifference_PathOp) or a
    except RuntimeError:
        return a


def inter(a, b):
    try:
        return skia.Op(a, b, skia.PathOp.kIntersect_PathOp) or a
    except RuntimeError:
        return a


def offset(path, dx, dy):
    q = skia.Path(path)
    q.offset(dx, dy)
    return q


def along(path, step, start=0.0):
    """[(x, y, angle_deg)] every `step` along the path."""
    m = skia.PathMeasure(path, False)
    out = []
    while True:
        L = m.getLength()
        d = start
        while d <= L:
            pos, tan = m.getPosTan(d)
            out.append((pos.x(), pos.y(), math.degrees(math.atan2(tan.y(), tan.x()))))
            d += step
        if not m.nextContour():
            break
    return out


# ---------------------------------------------------------------- paint helpers
def paint(color=None, a=255, shader=None, blend=None, blur=0.0, stroke=0.0, cap=True):
    p = skia.Paint(AntiAlias=True)
    if color is not None:
        p.setColor(C(color, a))
    if shader is not None:
        p.setShader(shader)
        if a < 255:
            p.setAlphaf(a / 255)
    if blend:
        p.setBlendMode({"multiply": skia.BlendMode.kMultiply, "screen": skia.BlendMode.kScreen, "overlay": skia.BlendMode.kOverlay,
                        "plus": skia.BlendMode.kPlus, "srcatop": skia.BlendMode.kSrcATop, "softlight": skia.BlendMode.kSoftLight,
                        "dstout": skia.BlendMode.kDstOut}[blend])
    if blur:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        if cap:
            p.setStrokeCap(skia.Paint.kRound_Cap)
            p.setStrokeJoin(skia.Paint.kRound_Join)
    return p


def lin(p0, p1, stops):
    """stops: [(pos, '#hex', alpha?)]"""
    cols = [C(s[1], s[2] if len(s) > 2 else 255) for s in stops]
    return skia.GradientShader.MakeLinear([p0, p1], cols, [s[0] for s in stops])


def rad(c, r, stops):
    cols = [C(s[1], s[2] if len(s) > 2 else 255) for s in stops]
    return skia.GradientShader.MakeRadial(c, r, cols, [s[0] for s in stops])


def flat(c, path, color, a=255, blur=0.0):
    c.drawPath(path, paint(color, a, blur=blur))


def stroke(c, path, color, w, a=255, blur=0.0):
    c.drawPath(path, paint(color, a, stroke=w, blur=blur))


class clip:
    """with clip(c, path): ... draws only inside path."""

    def __init__(self, c, path):
        self.c, self.path = c, path

    def __enter__(self):
        self.c.save()
        self.c.clipPath(self.path, skia.ClipOp.kIntersect, True)
        return self.c

    def __exit__(self, *a):
        self.c.restore()


def bounds(path):
    return path.computeTightBounds()


# ---------------------------------------------------------------- shading (airbrushed cut-out look)
def cyl(c, path, base, axis="x", light=0.36, depth=1.0, rim=True, hi=0.22, x0=None, x1=None):
    """Fill `path` with a cylinder-style airbrushed gradient across `axis` (x: a vertical tube lit from the left)."""
    b = bounds(path)
    if axis == "x":
        a, z = (x0 if x0 is not None else b.left()), (x1 if x1 is not None else b.right())
        p0, p1 = (a, 0), (z, 0)
    else:
        a, z = b.top(), b.bottom()
        p0, p1 = (0, a), (0, z)
    st = [(0.0, shade(base, -0.42 * depth)), (0.12, shade(base, -0.12 * depth)), (light, shade(base, hi)),
          (min(0.95, light + 0.22), base), (0.84, shade(base, -0.3 * depth)), (1.0, shade(base, -0.5 * depth))]
    c.drawPath(path, paint(shader=lin(p0, p1, st)))
    if rim:  # cool bounce light along the shadow edge
        with clip(c, path):
            q = offset(path, -6 if axis == "x" else 0, -6 if axis == "y" else 0)
            ring = diff(path, q)
            c.drawPath(ring, paint(shade(base, 0.35), 70, blur=2.5))


def shade_over(c, path, axis="x", strength=0.55, light=0.36, rim=True, a=255):
    """Shade whatever is already drawn inside `path` (patterns included): multiply shadow + screen highlight."""
    b = bounds(path)
    if axis == "x":
        p0, p1 = (b.left(), 0), (b.right(), 0)
    elif axis == "y":
        p0, p1 = (0, b.top()), (0, b.bottom())
    else:  # diagonal light from top-left
        p0, p1 = (b.left(), b.top()), (b.right(), b.bottom())
    s = strength
    with clip(c, path):
        sh = lin(p0, p1, [(0.0, mix("#ffffff", "#5a3550", 0.75 * s)), (0.14, mix("#ffffff", "#5a3550", 0.25 * s)),
                          (light, "#ffffff"), (light + 0.25, "#ffffff"), (0.86, mix("#ffffff", "#4a2a48", 0.55 * s)),
                          (1.0, mix("#ffffff", "#3a2040", 0.85 * s))])
        c.drawPath(path, paint(shader=sh, blend="multiply", a=a))
        hl = lin(p0, p1, [(0.0, "#000000"), (max(0.0, light - 0.14), "#000000"), (light, mix("#000000", "#fff1d6", 0.32 * s)),
                          (light + 0.16, "#000000"), (1.0, "#000000")])
        c.drawPath(path, paint(shader=hl, blend="screen", a=a))
        if rim:
            q = offset(path, -5, -3)
            c.drawPath(diff(path, q), paint("#cfe0ff", int(55 * s), blur=2.0, blend="screen"))


def ao(c, path, x, y, r, a=90, color="#2a1222"):
    """Ambient-occlusion blob (contact shadow) clipped into a part, e.g. under a collar or at a joint."""
    with clip(c, path):
        c.drawCircle(x, y, r, paint(shader=rad((x, y), r, [(0, color, a), (1, color, 0)])))


def glow(c, x, y, r, color, a=140):
    c.drawCircle(x, y, r, paint(shader=rad((x, y), r, [(0, color, a), (0.45, color, int(a * 0.35)), (1, color, 0)]), blend="screen"))


def sheen(c, path, x0, y0, x1, y1, w, a=90, color="#fff6e0"):
    """A soft specular streak (airbrush highlight) inside a shape."""
    with clip(c, path):
        c.drawPath(capsule(x0, y0, x1, y1, w), paint(color, a, blur=w * 0.6, blend="screen"))


# ---------------------------------------------------------------- metal, gems, trims
GOLD_STOPS = [(0.0, "#7a4c10"), (0.22, "#d9a23a"), (0.42, "#fff0b0"), (0.55, "#f2c24e"), (0.8, "#a8701c"), (1.0, "#6a400c")]


def gold_fill(c, path, axis="y"):
    b = bounds(path)
    p0, p1 = ((0, b.top()), (0, b.bottom())) if axis == "y" else ((b.left(), 0), (b.right(), 0))
    c.drawPath(path, paint(shader=lin(p0, p1, GOLD_STOPS)))
    stroke(c, path, "#6a400c", 0.9, 200)


def gold_line(c, path, w, beads=0.0, bead_r=None, a=255):
    """Gold braid / piping along an open path: dark edge, gold body, bright core, optional beads."""
    stroke(c, path, "#5a360a", w + 2.2, int(a * 0.9))
    stroke(c, path, "#c8902c", w, a)
    q = offset(path, -w * 0.12, -w * 0.18)
    stroke(c, q, "#ffe9a6", max(0.8, w * 0.38), int(a * 0.9))
    if beads:
        br = bead_r or w * 0.75
        for x, y, _ in along(path, beads, beads * 0.5):
            bead(c, x, y, br)


def bead(c, x, y, r, color="#e8b440"):
    c.drawCircle(x, y, r, paint(shader=rad((x - r * 0.35, y - r * 0.4), r * 1.4, [(0, "#fff6c8"), (0.35, color), (1, "#6a400c")])))


def gem(c, x, y, r, color, bezel=True):
    if bezel:
        c.drawCircle(x, y, r * 1.32, paint(shader=rad((x - r * 0.4, y - r * 0.5), r * 1.8, [(0, "#fff3b8"), (0.4, "#d9a23a"), (1, "#5e3a0a")])))
    c.drawCircle(x, y, r, paint(shader=rad((x + r * 0.25, y + r * 0.3), r * 1.2, [(0, shade(color, 0.35)), (0.55, color), (1, shade(color, -0.6))])))
    c.drawCircle(x - r * 0.35, y - r * 0.38, r * 0.28, paint("#ffffff", 235))
    c.drawCircle(x + r * 0.3, y + r * 0.35, r * 0.16, paint("#ffffff", 90))


def button(c, x, y, r):
    c.drawCircle(x + r * 0.15, y + r * 0.25, r * 1.05, paint("#2a1420", 90, blur=r * 0.35))
    c.drawCircle(x, y, r, paint(shader=rad((x - r * 0.35, y - r * 0.4), r * 1.5, [(0, "#fff5c0"), (0.3, "#e8b440"), (0.75, "#a36c18"), (1, "#5a360a")])))
    stroke(c, circle(x, y, r * 0.62), "#7a4c10", r * 0.14, 160)


def stitch_line(c, path, color, w=1.4, dash=6, gap=4, a=200):
    p = paint(color, a, stroke=w)
    p.setPathEffect(skia.DashPathEffect.Make([dash, gap], 0))
    c.drawPath(path, p)


# ---------------------------------------------------------------- folk motifs (our own Western folk-art vocabulary)
def tulip(c, x, y, s, color, leaf="#2f7a4a", rot=0.0, gold=True):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    stem = smooth([(0, 10), (0, 2), (0, -4)], closed=False)
    stroke(c, stem, leaf, 1.6)
    for sgn in (-1, 1):
        lf = smooth([(0, 9), (sgn * 7, 4), (sgn * 9, -3), (sgn * 3, 3)])
        flat(c, lf, leaf)
    cup = smooth([(-6, -4), (-7, -11), (-3.5, -9), (0, -15), (3.5, -9), (7, -11), (6, -4), (0, -1)])
    c.drawPath(cup, paint(shader=lin((0, -15), (0, 0), [(0, shade(color, 0.3)), (1, shade(color, -0.25))])))
    if gold:
        stroke(c, cup, "#f2c24e", 0.8, 220)
    c.restore()


def heart(c, x, y, s, color, gold=True):
    c.save()
    c.translate(x, y)
    c.scale(s, s)
    h = smooth([(0, 7), (-7, 0), (-6.5, -5), (-3, -7), (0, -4.5), (3, -7), (6.5, -5), (7, 0)])
    c.drawPath(h, paint(shader=lin((0, -7), (0, 7), [(0, shade(color, 0.28)), (1, shade(color, -0.3))])))
    if gold:
        stroke(c, h, "#f2c24e", 0.8, 230)
    c.restore()


def rosette(c, x, y, r, petal, center="#f2c24e", n=6, ring=True):
    """Six-petal folk rosette (barn-star / hex-sign flavour)."""
    for k in range(n):
        a = 2 * math.pi * k / n
        px, py = x + math.cos(a) * r * 0.5, y + math.sin(a) * r * 0.5
        e = ellipse(px, py, r * 0.5, r * 0.22, math.degrees(a))
        c.drawPath(e, paint(shader=rad((x, y), r, [(0, shade(petal, 0.3)), (1, shade(petal, -0.25))])))
    if ring:
        stroke(c, circle(x, y, r * 1.02), "#f2c24e", max(0.8, r * 0.08), 220)
    c.drawCircle(x, y, r * 0.22, paint(center))


def leaf_sprig(c, x, y, s, color, rot=0.0):
    c.save()
    c.translate(x, y)
    c.rotate(rot)
    c.scale(s, s)
    stroke(c, smooth([(0, 8), (1, 0), (0, -8)], closed=False), color, 1.2)
    for k, sgn in ((-5, -1), (-1, 1), (3, -1)):
        flat(c, ellipse(sgn * 3.5, k, 3.4, 1.6, sgn * 30), color)
    c.restore()


def vine_border(c, path, color, flower, step=26, s=1.0, a=255):
    """A running vine with alternating tulips/hearts along a path (embroidered hem)."""
    stroke(c, path, color, 1.6 * s, a)
    for i, (x, y, ang) in enumerate(along(path, step * s, step * s * 0.5)):
        if i % 2:
            tulip(c, x, y - 1 * s, 0.85 * s, flower, leaf=color, rot=0)
        else:
            heart(c, x, y, 0.7 * s, shade(flower, 0.1))


# ---------------------------------------------------------------- textiles (clip to a shape and fill)
def pat_dots(c, path, color, step=14, r=1.8, a=200, stagger=True):
    b = bounds(path)
    with clip(c, path):
        p = paint(color, a)
        y, row = b.top() - step, 0
        while y < b.bottom() + step:
            x = b.left() - step + (step / 2 if (stagger and row % 2) else 0)
            while x < b.right() + step:
                c.drawCircle(x, y, r, p)
                x += step
            y += step
            row += 1


def pat_sprigs(c, path, color, leaf, step=34, s=0.8):
    """Calico: tiny scattered flowers."""
    b = bounds(path)
    with clip(c, path):
        y, row = b.top() - step, 0
        while y < b.bottom() + step:
            x = b.left() - step + (step / 2 if row % 2 else 0)
            while x < b.right() + step:
                for k in range(5):
                    a = 2 * math.pi * k / 5 + row
                    c.drawCircle(x + math.cos(a) * 2.6 * s, y + math.sin(a) * 2.6 * s, 1.9 * s, paint(color, 220))
                c.drawCircle(x, y, 1.2 * s, paint("#f2c24e", 230))
                flat(c, ellipse(x + 5 * s, y + 4 * s, 2.6 * s, 1.2 * s, 35), leaf, 200)
                x += step
            y += step
            row += 1


def pat_knit(c, path, base, step=9, cable_every=5, a=120):
    """Knit stocking stitch (little Vs) with cable twists every few columns."""
    b = bounds(path)
    lt, dk = shade(base, 0.22), shade(base, -0.28)
    with clip(c, path):
        col = 0
        x = b.left() - step
        while x < b.right() + step:
            cable = (col % cable_every) == 0
            y = b.top() - step
            k = 0
            while y < b.bottom() + step:
                if cable:
                    ph = (k % 4) / 4
                    q = smooth([(x - step * 0.9, y), (x + step * (0.9 if ph < 0.5 else -0.9), y + step * 1.0), (x - step * 0.9, y + step * 2)], closed=False)
                    stroke(c, q, dk, step * 0.55, a)
                    stroke(c, offset(q, -1, -1), lt, step * 0.25, a)
                else:
                    v = poly([(x - step * 0.42, y), (x, y + step * 0.62), (x + step * 0.42, y)], closed=False)
                    stroke(c, v, dk, step * 0.3, int(a * 0.8))
                    stroke(c, offset(v, -0.6, -0.8), lt, step * 0.12, int(a * 0.7))
                y += step * (1.0 if not cable else 1.0)
                k += 1
            x += step * (1.4 if cable else 1.0)
            col += 1


def pat_tartan(c, path, base, lines, step=46, a=110):
    """lines: [(offset, width, colour)] repeated every `step` both ways."""
    b = bounds(path)
    with clip(c, path):
        c.drawPath(path, paint(base))
        for o, w, col in lines:
            x = b.left() - step + o
            while x < b.right() + step:
                c.drawRect(skia.Rect(x, b.top() - 10, x + w, b.bottom() + 10), paint(col, a))
                x += step
            y = b.top() - step + o
            while y < b.bottom() + step:
                c.drawRect(skia.Rect(b.left() - 10, y, b.right() + 10, y + w), paint(col, a))
                y += step
        # twill hatch
        p = paint("#000000", 22, stroke=1.0)
        k = b.left() - b.height()
        while k < b.right():
            c.drawLine(k, b.bottom(), k + b.height(), b.top(), p)
            k += 4


def pat_quilt(c, path, color, step=22, a=110, dot="#f2c24e"):
    """Diamond quilting with gold knots at the crossings."""
    b = bounds(path)
    with clip(c, path):
        p = paint(color, a, stroke=1.4)
        k = b.left() - b.height() - step
        while k < b.right() + step:
            c.drawLine(k, b.bottom(), k + b.height(), b.top(), p)
            c.drawLine(k, b.top(), k + b.height(), b.bottom(), p)
            k += step
        y = b.top()
        row = 0
        while y < b.bottom() + step:
            x = b.left() + (step / 2 if row % 2 else 0)
            while x < b.right() + step:
                c.drawCircle(x, y, 1.6, paint(dot, 220))
                x += step
            y += step / 2
            row += 1


def pat_fairisle(c, path, y0, base, colors, h=26):
    """A horizontal Fair-Isle yoke band starting at y0 (snowflake / heart / peerie rows)."""
    b = bounds(path)
    with clip(c, path):
        c.drawRect(skia.Rect(b.left() - 5, y0, b.right() + 5, y0 + h), paint(base))
        for i, x in enumerate(range(int(b.left()) - 12, int(b.right()) + 12, 12)):
            col = colors[i % len(colors)]
            c.drawRect(skia.Rect(x, y0 + 2, x + 4, y0 + 6), paint(colors[0], 220))
            heart(c, x + 6, y0 + h * 0.55, 0.62, col, gold=False)
            c.drawRect(skia.Rect(x + 6, y0 + h - 6, x + 10, y0 + h - 2), paint(colors[-1], 220))


def pat_weave(c, path, color, step=6, a=60):
    """Fine crosshatch for denim / linen texture."""
    b = bounds(path)
    with clip(c, path):
        p = paint(color, a, stroke=0.9)
        k = b.left() - b.height()
        while k < b.right():
            c.drawLine(k, b.bottom(), k + b.height(), b.top(), p)
            k += step
        p2 = paint(color, int(a * 0.5), stroke=0.6)
        y = b.top()
        while y < b.bottom():
            c.drawLine(b.left(), y, b.right(), y, p2)
            y += step * 0.75


def fur(c, path, base, n=400, length=9, seed=1, ang=0.0, a=90, light=True):
    """Short fur strokes scattered inside a shape (direction `ang`, degrees from +x)."""
    import random
    rnd = random.Random(seed)
    b = bounds(path)
    dk, lt = shade(base, -0.35), shade(base, 0.45)
    with clip(c, path):
        for _ in range(n):
            x = rnd.uniform(b.left(), b.right())
            y = rnd.uniform(b.top(), b.bottom())
            if not path.contains(x, y):
                continue
            a_ = math.radians(ang + rnd.uniform(-22, 22))
            L = length * rnd.uniform(0.6, 1.3)
            col = lt if (light and rnd.random() < 0.45) else dk
            c.drawLine(x, y, x + math.cos(a_) * L, y + math.sin(a_) * L, paint(col, int(a * rnd.uniform(0.5, 1.0)), stroke=rnd.uniform(0.8, 1.6)))


# ---------------------------------------------------------------- surfaces
class Surf:
    """A drawing surface in rig units: origin at (ox, oy) px, `q` px per unit."""

    def __init__(self, w_units, h_units, ox_units, oy_units, q=2.0):
        self.q = q
        self.w, self.h = int(w_units * q), int(h_units * q)
        self.s = skia.Surface(self.w, self.h)
        self.c = self.s.getCanvas()
        self.c.clear(skia.ColorTRANSPARENT)
        self.ox, self.oy = ox_units * q, oy_units * q
        self.c.translate(self.ox, self.oy)
        self.c.scale(q, q)

    def image(self):
        return self.s.makeImageSnapshot()

    def crop_save(self, path, pad=6):
        """Crop to opaque pixels; returns the anchor (the units origin) in the cropped image px."""
        import numpy as np
        from PIL import Image
        arr = self.image().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
        al = arr[..., 3]
        ys, xs = np.where(al > 2)
        if len(xs) == 0:
            raise ValueError(f"empty part {path}")
        x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + 1 + pad)
        y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + 1 + pad)
        Image.fromarray(arr[y0:y1, x0:x1], "RGBA").save(path, optimize=True)
        return (float(self.ox - x0), float(self.oy - y0)), (int(x1 - x0), int(y1 - y0))
