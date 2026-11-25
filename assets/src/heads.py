"""Profile heads (facing right), drawn in the head bone's frame: pivot (0, 0) at the top of the neck, -y up.
Head height ~124 rig units (8 heads tall on a 1000-unit figure)."""
import math
from paint import *


def head_shape(spec):
    """Silhouette of skull + face + jaw + a neck stub, with per-character nose/chin/brow tweaks."""
    n = spec.get("nose", "soft")
    nx = {"soft": 0, "long": 3, "button": -4, "hooked": 2, "strong": 4}[n]
    ny = {"soft": 0, "long": 2, "button": -2, "hooked": 3, "strong": 1}[n]
    ch = spec.get("chin", 0.0)  # + = stronger chin
    jaw = spec.get("jaw", 0.0)  # + = heavier jaw (men)
    child = spec.get("child", False)
    p = skia.Path()
    p.moveTo(-46, -20)
    p.cubicTo(-56, -36, -60, -52, -59, -72)
    p.cubicTo(-58, -110, -20, -127, 8, -126)
    p.cubicTo(38, -125, 57, -106, 58, -86)
    p.cubicTo(59, -79, 61, -74, 59, -69)
    if child:
        p.cubicTo(58, -66, 62, -58, 65, -53)
        p.cubicTo(68, -49, 70, -46, 67, -44)
        p.cubicTo(65, -42, 62, -42, 61, -41)
    else:
        p.cubicTo(58, -66, 62 + nx * 0.4, -60, 66 + nx * 0.8, -55 + ny * 0.3)
        p.cubicTo(70 + nx, -50 + ny * 0.5, 76 + nx, -47 + ny, 74 + nx * 0.8, -43 + ny)
        if n == "hooked":
            p.cubicTo(72 + nx * 0.6, -39 + ny, 66, -41, 63, -40)
        else:
            p.cubicTo(72 + nx * 0.5, -40 + ny * 0.5, 66, -40, 63, -40)
    p.cubicTo(64, -37, 66, -35, 65, -33)
    p.cubicTo(63, -31, 62, -30, 63, -28)
    p.cubicTo(65, -26, 64, -23, 61, -21)
    p.cubicTo(58, -19, 60 + ch, -14, 61 + ch, -11 + jaw * 0.3)
    p.cubicTo(62 + ch, -4 + jaw, 54 + jaw, 2 + jaw, 44 + jaw * 0.5, 3 + jaw * 0.6)
    p.cubicTo(36, 4, 28, 6, 24, 10)
    p.cubicTo(22, 16, 10, 22, 2, 22)
    p.cubicTo(-8, 22, -18, 18, -20, 10)
    p.cubicTo(-30, 0, -40, -8, -46, -20)
    p.close()
    return p


def skin_fill(c, path, skin):
    c.drawPath(path, paint(shader=rad((34, -74), 120, [(0, shade(skin, 0.22)), (0.38, skin), (0.8, shade(skin, -0.2)), (1, shade(skin, -0.36))])))
    with clip(c, path):  # under-jaw and back-of-neck shadow, front-face sheen
        c.drawPath(smooth([(60, -4), (30, 4), (-10, 6), (-30, 14), (-30, 40), (60, 40)]), paint(shade(skin, -0.38), 110, blur=6))
        c.drawPath(ellipse(52, -86, 10, 20, -10), paint("#fff3e2", 55, blur=6, blend="screen"))


def ear(c, skin, x=-4, y=-58, s=1.0, earring=None):
    e = ellipse(x, y, 9 * s, 14 * s, 8)
    c.drawPath(e, paint(shader=rad((x + 3, y - 3), 16 * s, [(0, shade(skin, 0.1)), (1, shade(skin, -0.28))])))
    stroke(c, smooth([(x + 3 * s, y - 9 * s), (x - 3 * s, y - 6 * s), (x - 4 * s, y + 3 * s), (x + 1 * s, y + 7 * s)], closed=False), shade(skin, -0.42), 1.6 * s, 200)
    if earring:
        gem(c, x + 1, y + 16 * s, 2.6 * s, earring)


def eye(c, spec, blink=False):
    ex, ey = spec.get("eye_x", 44), spec.get("eye_y", -66)
    s = spec.get("eye_s", 1.0)
    ink = spec.get("ink", "#2a1a14")
    if blink:
        stroke(c, smooth([(ex - 10 * s, ey - 1), (ex - 1, ey + 2.5 * s), (ex + 7 * s, ey)], closed=False), ink, 2.2)
        if spec.get("lashes"):
            for k in range(3):
                stroke(c, poly([(ex + (1 + 2.4 * k) * s, ey + 1.6), (ex + (2.5 + 3 * k) * s, ey + 4.5 * s)], closed=False), ink, 1.2)
        return
    white = smooth([(ex - 10 * s, ey - 1), (ex - 3 * s, ey - 5.5 * s), (ex + 6 * s, ey - 4 * s), (ex + 8 * s, ey), (ex + 4 * s, ey + 3.5 * s), (ex - 4 * s, ey + 3.5 * s)])
    flat(c, white, "#fbf6ee")
    with clip(c, white):
        ic = spec.get("iris", "#4a3020")
        ix = ex + 3.2 * s + spec.get("look", 0)
        c.drawCircle(ix, ey - 0.5, 4.6 * s, paint(shader=rad((ix, ey), 4.6 * s, [(0, shade(ic, 0.35)), (0.7, ic), (1, shade(ic, -0.5))])))
        c.drawCircle(ix + 0.4, ey - 0.5, 2.1 * s, paint("#120a08"))
        c.drawCircle(ix - 1.2 * s, ey - 2.2 * s, 1.3 * s, paint("#ffffff", 240))
        c.drawPath(white, paint(shader=lin((0, ey - 6 * s), (0, ey), [(0, "#6a4a3a", 120), (1, "#6a4a3a", 0)])))
    # upper lid (thick ink line), crease, lower lid
    lid = smooth([(ex - 11 * s, ey - 0.5), (ex - 3 * s, ey - 6.2 * s), (ex + 6 * s, ey - 4.6 * s), (ex + 9 * s, ey - 0.2)], closed=False)
    stroke(c, lid, ink, 2.4 * s)
    stroke(c, offset(lid, 0, -3.2 * s), shade(spec["skin"], -0.3), 1.1, 170)
    stroke(c, smooth([(ex - 4 * s, ey + 3.8 * s), (ex + 3 * s, ey + 3.8 * s), (ex + 7 * s, ey + 1)], closed=False), shade(spec["skin"], -0.35), 1.0, 150)
    if spec.get("lashes"):
        for k in range(3):
            x0 = ex + (2.5 + 2.6 * k) * s
            stroke(c, poly([(x0, ey - 4.6 * s + k * 0.8), (x0 + 3.2 * s, ey - 7.5 * s + k * 1.4)], closed=False), ink, 1.3)


def brow(c, spec):
    ex, ey = spec.get("eye_x", 44), spec.get("eye_y", -66)
    col = spec.get("brow", shade(spec.get("hair", "#5a3a2a"), -0.3))
    w = spec.get("brow_w", 2.8)
    b = smooth([(ex - 10, ey - 9), (ex - 1, ey - 13), (ex + 9, ey - 11.5)], closed=False)
    stroke(c, b, col, w)
    stroke(c, offset(b, 0, -0.8), shade(col, 0.25), w * 0.35, 140)


def mouth(c, spec):
    lip = spec.get("lip", mix(spec["skin"], "#b8404a", 0.45))
    up = smooth([(57, -32.5), (61, -34.5), (64.6, -33.2), (63.2, -31), (58, -30.6)])
    lo = smooth([(58, -30.2), (63, -29.6), (63.4, -26.4), (60, -25.4), (57.5, -28)])
    flat(c, up, shade(lip, -0.1), 230)
    flat(c, lo, lip, 230)
    sm = spec.get("smile", 0.5)
    stroke(c, smooth([(55.5, -29 - sm * 2.4), (58.5, -30.4), (63, -30.4)], closed=False), shade(lip, -0.55), 1.4, 220)
    c.drawCircle(61.5, -27.6, 1.1, paint("#ffffff", 90))


def age_lines(c, spec):
    sk = spec["skin"]
    a = spec.get("age_lines", 0)
    if not a:
        return
    col = shade(sk, -0.38)
    ex, ey = spec.get("eye_x", 44), spec.get("eye_y", -66)
    stroke(c, smooth([(58, -46), (54.5, -40), (55, -33)], closed=False), col, 1.5, int(150 * a))  # nasolabial
    for k in range(3):  # crow's feet
        stroke(c, poly([(ex - 12, ey - 1 + k * 3), (ex - 17, ey - 3 + k * 4.5)], closed=False), col, 1.1, int(130 * a))
    stroke(c, smooth([(ex - 6, ey + 6), (ex + 1, ey + 7.5), (ex + 6, ey + 5)], closed=False), col, 1.1, int(130 * a))  # eye bag
    stroke(c, smooth([(30, -98), (42, -101), (50, -99)], closed=False), col, 1.0, int(90 * a))  # forehead
    stroke(c, smooth([(52, -20), (55, -16)], closed=False), col, 1.0, int(110 * a))


def glasses(c, spec):
    ex, ey = spec.get("eye_x", 44), spec.get("eye_y", -66)
    r = 10.5
    lens = circle(ex + 2, ey - 0.5, r)
    c.drawPath(lens, paint(shader=lin((ex - r, ey - r), (ex + r, ey + r), [(0, "#ffffff", 70), (0.5, "#ffffff", 10), (1, "#cfe4ff", 50)])))
    gold_line(c, lens, 1.5)
    arm = smooth([(ex - r + 1, ey - 2), (ex - 22, ey - 3), (-1, -62)], closed=False)
    gold_line(c, arm, 1.3)


# ---------------------------------------------------------------- hair styles (all drawn over the head shape)
def hair_strands(c, path, base, n=14, seed=3, ang=None, a=110):
    """Strands for free-form hair masses (beards, back hair): gently curved top-to-bottom lines."""
    import random
    rnd = random.Random(seed)
    b = bounds(path)
    with clip(c, path):
        for k in range(n):
            x0 = b.left() + (k + 0.5) / n * b.width() + rnd.uniform(-3, 3)
            bend = rnd.uniform(-6, 6)
            pts = [(x0, b.top() - 4), (x0 + bend, b.top() + b.height() * 0.4), (x0 + bend * 0.4 - 2, b.top() + b.height() * 0.75), (x0 - 4, b.bottom() + 4)]
            stroke(c, smooth(pts, closed=False), shade(base, 0.4 if k % 2 else -0.28), rnd.uniform(0.7, 1.3), a)


def flow(c, clip_path, inner, outer, base, n=12, a=120, seed=1):
    """Hair strands that follow the hair's flow: curves interpolated between an inner and an outer edge polyline."""
    import random
    rnd = random.Random(seed)
    with clip(c, clip_path):
        for k in range(n):
            t = (k + 0.5) / n + rnd.uniform(-0.02, 0.02)
            pts = [(i[0] + (o[0] - i[0]) * t, i[1] + (o[1] - i[1]) * t) for i, o in zip(inner, outer)]
            col = shade(base, 0.42) if k % 3 == 0 else shade(base, -0.3) if k % 3 == 1 else shade(base, 0.15)
            stroke(c, smooth(pts, closed=False), col, rnd.uniform(0.7, 1.4), a)


HAIRLINES = {  # (outer edge, inner edge) front -> back, for flow strands
    "bun": ([(56, -100), (50, -118), (30, -129), (4, -133), (-26, -127), (-50, -108), (-62, -80), (-60, -52), (-48, -30)],
            [(46, -98), (38, -94), (26, -92), (12, -88), (0, -80), (-10, -68), (-16, -54), (-22, -40), (-30, -24)]),
    "braid": ([(56, -100), (48, -120), (28, -130), (2, -134), (-28, -128), (-52, -108), (-62, -80), (-58, -50), (-46, -26)],
              [(44, -100), (36, -96), (24, -94), (10, -90), (-2, -80), (-10, -66), (-16, -52), (-24, -38), (-34, -22)]),
    "regal": ([(56, -100), (46, -120), (26, -130), (0, -134), (-28, -128), (-54, -108), (-64, -78), (-62, -46), (-52, -20)],
              [(52, -98), (44, -96), (30, -96), (12, -92), (-2, -80), (-10, -66), (-16, -52), (-24, -38), (-32, -20)]),
    "long_white": ([(40, -110), (38, -122), (22, -131), (-2, -134), (-30, -128), (-56, -108), (-66, -78), (-64, -46), (-54, -18)],
                   [(36, -104), (28, -98), (18, -95), (6, -90), (-4, -80), (-10, -66), (-16, -52), (-24, -38), (-32, -20)]),
}


def hair_cap(style):
    outer, inner = HAIRLINES[style]
    return smooth(outer + list(reversed(inner))), inner, outer


def _cap_fill(c, cap, hc, lift=0.4):
    c.drawPath(cap, paint(shader=lin((50, -134), (-58, -24), [(0, shade(hc, lift)), (0.45, hc), (1, shade(hc, -0.36))])))


def draw_hair(c, spec, stage):
    """stage 'back' (behind the head) or 'front'."""
    st = spec.get("hair_style")
    hc = spec["hair"]
    if st == "bun":  # Nell: silver hair swept back into a braided low bun with a tortoiseshell comb
        if stage == "back":
            bun = circle(-56, -56, 23)
            c.drawPath(bun, paint(shader=rad((-62, -64), 30, [(0, shade(hc, 0.4)), (0.7, hc), (1, shade(hc, -0.38))])))
            for k in range(6):  # braid wrapped around the bun
                arc = skia.Path()
                rr = 20 - k * 2.6
                arc.addArc(skia.Rect(-56 - rr, -56 - rr, -56 + rr, -56 + rr), k * 61, 140)
                stroke(c, arc, shade(hc, -0.34 if k % 2 else 0.45), 1.5, 180)
            return
        cap, inner, outer = hair_cap("bun")
        _cap_fill(c, cap, hc)
        flow(c, cap, inner, outer, hc, n=14, seed=5)
        sheen(c, cap, 28, -122, -34, -108, 6, 120)
        wave = smooth([(54, -97), (44, -92), (32, -97), (20, -90), (8, -94), (-2, -84)], closed=False)
        stroke(c, wave, shade(hc, 0.5), 2.6, 210)
        comb = smooth([(-44, -84), (-35, -80), (-37, -60), (-46, -63)])
        c.drawPath(comb, paint(shader=lin((-46, -84), (-34, -62), [(0, "#a8642a"), (0.5, "#5a2c12"), (1, "#c88440")])))
        for k in range(4):
            gem(c, -40.5, -79 + k * 5, 1.4, JEWEL["emerald_lt"], bezel=False)
    elif st == "braid":  # Dee: hair pulled back; the long braid hangs from the nape as spring parts
        if stage == "back":
            return
        cap, inner, outer = hair_cap("braid")
        _cap_fill(c, cap, hc, 0.3)
        flow(c, cap, inner, outer, hc, n=16, seed=7)
        sheen(c, cap, 32, -124, -30, -110, 5, 100)
        c.drawPath(ellipse(-46, -28, 9, 7, -20), paint(shader=rad((-48, -31), 10, [(0, JEWEL["ruby_lt"]), (1, JEWEL["ruby"])])))
    elif st == "messy":  # Eli: sandy messy tufts
        if stage == "back":
            return
        cap, inner, outer = hair_cap("braid")
        tufts = [cap]
        for k in range(8):
            a = math.radians(-175 + k * 22)
            tufts.append(ellipse(2 + math.cos(a) * 56, -80 + math.sin(a) * 48, 17, 10, math.degrees(a) + 90))
        tufts.append(smooth([(50, -104), (64, -96), (54, -90), (44, -96)]))
        tufts.append(smooth([(40, -112), (58, -112), (50, -102)]))
        hp = union(*tufts)
        _cap_fill(c, hp, hc, 0.45)
        flow(c, hp, inner, outer, hc, n=18, seed=9)
        sheen(c, hp, 28, -124, -26, -114, 6, 110)
    elif st == "regal":  # King: shoulder-length wavy hair under the crown
        if stage == "back":
            back = smooth([(-26, -66), (-62, -76), (-72, -34), (-66, 4), (-48, 22), (-30, 10), (-24, -30)])
            c.drawPath(back, paint(shader=lin((-70, -70), (-20, 20), [(0, shade(hc, 0.25)), (1, shade(hc, -0.42))])))
            hair_strands(c, back, hc, n=9, seed=11)
            return
        cap, inner, outer = hair_cap("regal")
        _cap_fill(c, cap, hc, 0.35)
        flow(c, cap, inner, outer, hc, n=14, seed=13)
    elif st == "long_white":  # wise elder: long white hair over the shoulders
        if stage == "back":
            back = smooth([(-22, -72), (-64, -78), (-76, -24), (-70, 30), (-50, 52), (-30, 26), (-24, -30)])
            c.drawPath(back, paint(shader=lin((-70, -70), (-20, 40), [(0, shade(hc, 0.15)), (1, shade(hc, -0.38))])))
            hair_strands(c, back, hc, n=12, seed=15)
            return
        cap, inner, outer = hair_cap("long_white")
        _cap_fill(c, cap, hc, 0.3)
        flow(c, cap, inner, outer, hc, n=16, seed=17, a=100)


def beard(c, spec, stage="beard"):
    b = spec.get("beard")
    if not b:
        return
    col, L = b["color"], b.get("length", 30)
    if stage == "beard":  # from the sideburn along the jaw, around the chin, down to the tip
        shape = smooth([(-4, -72), (8, -76), (16, -60), (32, -48), (50, -43), (61, -37), (66, -22), (66, -6), (60, L * 0.55),
                        (48, L * 0.92), (32, L), (16, L * 0.78), (4, L * 0.4), (-12, 6), (-18, -22), (-12, -50)])
        c.drawPath(shape, paint(shader=lin((60, -50), (10, L), [(0, shade(col, 0.42)), (0.55, col), (1, shade(col, -0.38))])))
        inner = [(8, -70), (18, -50), (28, -36), (38, -20), (40, 0), (36, L * 0.5), (30, L * 0.9)]
        outer = [(2, -60), (4, -40), (8, -20), (14, 0), (18, L * 0.3), (22, L * 0.7), (28, L)]
        front = [(30, -52), (50, -40), (60, -24), (62, -6), (58, L * 0.3), (50, L * 0.7), (36, L)]
        flow(c, shape, outer, front, col, n=int(12 + L / 8), seed=21, a=150)
        sheen(c, shape, 50, -30, 40, L * 0.5, 4, 90)
    else:  # moustache drawn over the mouth
        mo = smooth([(67, -38), (62, -32.5), (54, -30), (46, -33), (50, -38), (58, -40)])
        c.drawPath(mo, paint(shader=lin((66, -42), (46, -30), [(0, shade(col, 0.5)), (1, shade(col, -0.25))])))
        stroke(c, smooth([(64, -37), (56, -34), (49, -34)], closed=False), shade(col, -0.35), 1.0, 170)


def crown(c, spec):
    if not spec.get("crown"):
        return
    # a band that wraps the head (front higher, back lower), with five fleur-free points tipped with gems
    band = smooth([(52, -102), (56, -124), (0, -130), (-56, -116), (-58, -94), (0, -104)])
    pts = [(-56, -114)]
    tips = []
    for k in range(5):
        x0 = -56 + k * 22.4
        y0 = -116 - (k * 2.4)
        tip = (x0 + 11.2, y0 - 30 + (5 if k % 2 else 0))
        tips.append(tip)
        pts += [(x0 + 4, y0 - 2), tip, (x0 + 18.4, y0 - 2)]
    pts += [(56, -124)]
    gold_fill(c, union(band, poly(pts)), "y")
    for k, (x, y) in enumerate(tips):
        gem(c, x, y - 2, 3.0, [JEWEL["ruby_lt"], JEWEL["emerald_lt"], JEWEL["sapphire_lt"]][k % 3])
    gold_line(c, smooth([(-57, -98), (0, -107), (53, -105)], closed=False), 1.8, beads=7)
    for k, (x, y) in enumerate([(-34, -108), (-6, -114), (24, -116), (48, -114)]):
        gem(c, x, y, 3.6, [JEWEL["sapphire_lt"], JEWEL["ruby_lt"], JEWEL["emerald_lt"], JEWEL["ruby_lt"]][k])



def hood(c, spec, stage):
    """Wise elder's deep-green hood (folded back) with embroidered border."""
    h = spec.get("hood")
    if not h:
        return
    col = h["color"]
    if stage == "back":
        hd = smooth([(-10, -128), (-56, -118), (-80, -70), (-78, -10), (-50, 26), (-16, 18), (-24, -40), (-12, -90)])
        c.drawPath(hd, paint(shader=lin((-80, -120), (-20, 20), [(0, shade(col, 0.2)), (0.6, col), (1, shade(col, -0.45))])))
        edge = smooth([(-10, -128), (-24, -90), (-30, -40), (-16, 18)], closed=False)
        gold_line(c, edge, 2.2, beads=9)


def head_part(spec, blink=False):
    """Draw function for the head part (pivot = neck top)."""
    sc = spec.get("head_scale", 1.0)

    def draw(c):
        c.save()
        c.scale(sc, sc)
        hood(c, spec, "back")
        draw_hair(c, spec, "back")
        hs = head_shape(spec)
        skin_fill(c, hs, spec["skin"])
        ear(c, spec["skin"], earring=spec.get("earring"))
        age_lines(c, spec)
        c.drawPath(ellipse(44, -48, 13, 9), paint(spec.get("blush", "#ff8a6a"), spec.get("blush_a", 60), blur=5))
        mouth(c, spec)
        eye(c, spec, blink)
        brow(c, spec)
        if spec.get("freckles"):
            for k in range(7):
                c.drawCircle(38 + 3.2 * k, -54 + 2.5 * (k % 2), 1.0, paint("#b8683a", 170))
        beard(c, spec)
        if spec.get("beard"):
            mouth(c, spec)
            beard(c, spec, "moustache")
        draw_hair(c, spec, "front")
        if spec.get("glasses"):
            glasses(c, spec)
        crown(c, spec)
        c.restore()
    return draw
