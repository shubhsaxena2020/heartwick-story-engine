"""Animals on engine v2's squirrel skeleton (engine2/characters.py squirrel(): body along +x, paws/legs hang +y,
tail bones run along -x). Pip is an Eastern grey squirrel drawn from life: agouti grey with a rusty wash on the
back and feet, white belly and eye ring, ear tufts, a long bushy tail with a frosted silver fringe.
python assets/src/animals.py [pip pip_kit hare]"""
import math, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from paint import *
from rigkit import Rigkit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "characters"))


def sq_bones(K, kit=False, t2=None, t3=None):
    K.bone("body", None)
    K.bone("head", "body", (128, -14))
    K.bone("paw_f", "body", (104, 18))
    K.bone("paw_n", "body", (112, 22))
    K.bone("leg_f", "body", (14, 20))
    K.bone("leg_n", "body", (22, 24))
    K.bone("tail1", "body", (-22, -6), 90)
    K.bone("tail2", "tail1", (t2 if t2 is not None else (-40 if kit else -90), 0), 85)
    K.bone("tail3", "tail2", (t3 if t3 is not None else (-36 if kit else -85), 0), 80)


def furry(c, path, base, belly=None, wash=None, n=900, L=8, seed=1, ang=200, a=110, dark_back=True):
    """Fur-shaded fill: radial body shading, rusty wash on the back, then hair strokes."""
    b = bounds(path)
    c.drawPath(path, paint(shader=rad((b.centerX() - b.width() * 0.1, b.centerY() - b.height() * 0.3), max(b.width(), b.height()) * 0.75,
                                      [(0, shade(base, 0.25)), (0.6, base), (1, shade(base, -0.4))])))
    with clip(c, path):
        if wash:
            c.drawPath(ellipse(b.centerX(), b.top() + b.height() * 0.15, b.width() * 0.45, b.height() * 0.28), paint(wash, 120, blur=b.height() * 0.12))
        if belly:
            c.drawPath(ellipse(b.centerX() + b.width() * 0.12, b.bottom() - b.height() * 0.1, b.width() * 0.42, b.height() * 0.3), paint(belly, 255, blur=b.height() * 0.08))
    fur(c, path, base, n=n, length=L, seed=seed, ang=ang, a=a)
    shade_over(c, path, axis="y", strength=0.45, light=0.25, rim=False)


def tail_seg(r0, r1, L, base, fringe, seed, tip=False, kit=False):
    """One bushy segment from the pivot along -x: an overlapping fluffy mass with frosted edge hairs."""
    def draw(c):
        rnd = random.Random(seed)
        pts = [(-L * k / 10, 0) for k in range(11)]
        mass = skia.Path()
        for i, (x, y) in enumerate(pts):
            u = i / 10
            r = r0 + (r1 - r0) * u
            if tip:
                r = r1 * math.sin(math.pi * min(1.0, 0.25 + u * 0.75)) + 4
            mass.addCircle(x, y, r)
        mass = union(mass, circle(0, 0, r0))
        # fringe: long silver hairs radiating out of the mass edge
        hp = paint(fringe, 200, stroke=1.6)
        for i in range(180 if not kit else 70):
            u = rnd.random()
            x = -L * u
            r = (r0 + (r1 - r0) * u) if not tip else (r1 * math.sin(math.pi * min(1.0, 0.25 + u * 0.75)) + 4)
            a = rnd.uniform(0, 2 * math.pi)
            x0, y0 = x + math.cos(a) * r * 0.75, math.sin(a) * r * 0.75
            ext = r * rnd.uniform(0.25, 0.55)
            hp.setAlpha(int(rnd.uniform(110, 220)))
            c.drawLine(x0, y0, x0 + math.cos(a) * ext - 6, y0 + math.sin(a) * ext, hp)
        c.drawPath(mass, paint(shader=lin((0, -r1), (0, r1), [(0, shade(fringe, -0.1)), (0.25, base), (0.6, shade(base, -0.2)), (1, shade(base, -0.45))])))
        with clip(c, mass):  # a dark band inside the frosted edge, like a real grey squirrel's tail
            inner = skia.Path()
            for i, (x, y) in enumerate(pts):
                r = (r0 + (r1 - r0) * i / 10) * 0.55 if not tip else r1 * 0.4
                inner.addCircle(x, y, r)
            c.drawPath(inner, paint(shade(base, -0.4), 120, blur=6))
        fur(c, mass, base, n=500 if not kit else 160, length=12 if not kit else 6, seed=seed + 3, ang=180, a=130)
        # frosted silver rim
        with clip(c, mass):
            for dy in (-1, 1):
                c.drawPath(diff(mass, offset(mass, 0, -dy * 7)), paint(fringe, 150, blur=2.5))
    return draw


def squirrel(kit=False):
    name = "pip_kit" if kit else "pip"
    K = Rigkit(name, kind="squirrel", units=200.0 if kit else 260.0, q=3.0, pelvis_h=60.0)
    sq_bones(K, kit)
    base = "#8f9196" if not kit else "#9a9ca2"
    wash = "#9a6a48"
    belly = "#f4ecdc"
    s = 0.92 if kit else 1.0

    def body(c):
        c.save()
        c.scale(s, s)
        p = smooth([(-24, 8), (-14, -30), (20, -52), (70, -56), (118, -40), (146, -8), (140, 22), (110, 44), (70, 54), (30, 54), (-6, 44)])
        furry(c, p, base, belly=belly, wash=wash, n=1400 if not kit else 900, L=9 if not kit else 5, seed=3, ang=195)
        haunch = ellipse(22, 14, 46, 44)
        with clip(c, p):
            c.drawPath(haunch, paint(shader=rad((14, 0), 50, [(0, shade(base, 0.12)), (0.8, shade(base, -0.12)), (1, shade(base, -0.4))])))
            fur(c, haunch, base, n=300, length=8 if not kit else 4, seed=5, ang=220, a=110)
            stroke(c, smooth([(-18, 30), (-12, -12), (20, -28), (58, -10)], closed=False), shade(base, -0.45), 2.0, 90)
            c.drawPath(ellipse(100, 30, 40, 18), paint(belly, 230, blur=6))
        c.restore()
    K.part("body", 10, body, (-60, -90, 180, 90))

    def head(c):
        c.save()
        c.scale(s, s)
        hs = 1.14 if kit else 1.0
        c.scale(hs, hs)
        ear_h = 0.75 if kit else 1.0
        for dx, dk in ((-6, 0.25), (6, 0.0)):  # far ear then near ear
            e = smooth([(dx - 14, -30), (dx - 12, -30 - 34 * ear_h), (dx - 2, -30 - 44 * ear_h), (dx + 6, -30 - 30 * ear_h), (dx + 6, -28)])
            c.drawPath(e, paint(shader=lin((0, -74), (0, -28), [(0, shade(wash, -dk)), (1, shade(base, -0.2 - dk))])))
            c.drawPath(smooth([(dx - 9, -34), (dx - 7, -30 - 30 * ear_h), (dx - 1, -30 - 34 * ear_h), (dx + 2, -32)]), paint("#d8a898", 170 if not dk else 110))
            if not kit:  # winter ear tufts
                for k in range(6):
                    stroke(c, poly([(dx - 4, -30 - 40 * ear_h), (dx - 4 + k * 2 - 6, -30 - 52 * ear_h - k)], closed=False), shade(base, -0.35), 1.2, 180)
        hp = smooth([(-14, 14), (-18, -16), (0, -40), (36, -46), (64, -30), (86, -6), (90, 4), (80, 14), (56, 22), (20, 26)])
        furry(c, hp, base, belly=None, wash=wash, n=700 if not kit else 400, L=6 if not kit else 4, seed=7, ang=190)
        with clip(c, hp):
            c.drawPath(smooth([(30, 10), (60, 4), (86, 6), (78, 18), (40, 26)]), paint(belly, 235, blur=4))  # pale cheek and chin
        flat(c, ellipse(86, -2, 6.5, 5), "#3a2828")  # nose
        c.drawCircle(84.5, -3.5, 1.6, paint("#ffffff", 150))
        stroke(c, smooth([(86, 4), (82, 10), (76, 12)], closed=False), "#4a3030", 1.6, 200)  # mouth line
        ex, ey = 50, -16
        if not kit:
            c.drawCircle(ex, ey, 14, paint("#f8f2e8", 245))  # white eye ring
        if kit:
            stroke(c, smooth([(ex - 9, ey), (ex, ey + 3.5), (ex + 9, ey)], closed=False), "#2a1a14", 3)
            c.drawPath(ellipse(ex, ey + 1.2, 7, 2.2), paint("#1a1210", 230))
            c.drawCircle(ex + 3, ey + 0.6, 1.0, paint("#ffffff", 200))
        else:
            c.drawCircle(ex, ey, 10.5, paint(shader=rad((ex + 3, ey + 3), 11, [(0, "#3a2a22"), (0.7, "#120c0a"), (1, "#000000")])))
            c.drawCircle(ex - 3.4, ey - 4, 3.4, paint("#ffffff", 240))
            c.drawCircle(ex + 3.5, ey + 3.5, 1.6, paint("#ffffff", 120))
        for k in range(5):  # whiskers
            a = math.radians(-20 + k * 12)
            stroke(c, poly([(76, 2), (76 + math.cos(a) * 46, 2 + math.sin(a) * 30)], closed=False), "#2a2020", 0.9, 140)
        c.restore()
    K.part("head", 14, head, (-40, -110, 130, 50))

    def paw(dk):
        def draw(c):
            c.save()
            c.scale(s, s)
            col = shade(base, -dk)
            arm = capsule(0, 0, 0, 30, 10, 7)
            furry(c, arm, col, wash=None, n=120, L=4, seed=11, ang=90, a=100)
            hand = smooth([(-7, 26), (7, 26), (10, 36), (6, 42), (-4, 42), (-9, 36)])
            c.drawPath(hand, paint(shader=lin((0, 26), (0, 42), [(0, shade(wash, -dk * 0.5)), (1, shade(wash, -0.3 - dk))])))
            for k in range(3):
                stroke(c, poly([(-4 + k * 4, 40), (-5 + k * 4, 46)], closed=False), "#2a1a14", 1.2, 200)
            c.restore()
        return draw
    K.part("paw_f", 6, paw(0.22), (-25, -20, 25, 60))
    K.part("paw_n", 16, paw(0.0), (-25, -20, 25, 60))

    def leg(dk):
        def draw(c):
            c.save()
            c.scale(s, s)
            col = shade(base, -dk)
            shin = capsule(0, 0, 4, 34, 16, 10)
            furry(c, shin, col, n=160, L=5, seed=13, ang=100, a=100)
            foot = smooth([(-6, 34), (20, 32), (48, 38), (50, 46), (-4, 46)])
            c.drawPath(foot, paint(shader=lin((0, 32), (0, 46), [(0, shade(wash, -dk * 0.5)), (1, shade(wash, -0.35 - dk))])))
            fur(c, foot, wash, n=60, length=4, seed=17, ang=0, a=90)
            for k in range(3):
                stroke(c, poly([(44 + k * 2, 44), (50 + k * 2, 47)], closed=False), "#2a1a14", 1.1, 200)
            c.restore()
        return draw
    K.part("leg_f", 5, leg(0.22), (-30, -25, 70, 60))
    K.part("leg_n", 12, leg(0.0), (-30, -25, 70, 60))
    fringe = "#e8eaee"
    if kit:
        K.part("tail1", 3, tail_seg(10, 14, 40, base, fringe, 21, kit=True), (-70, -40, 30, 40))
        K.part("tail2", 3.1, tail_seg(14, 15, 36, base, fringe, 23, kit=True), (-70, -40, 30, 40))
        K.part("tail3", 3.2, tail_seg(15, 12, 30, base, fringe, 25, tip=True, kit=True), (-60, -40, 30, 40))
    else:
        K.part("tail1", 3, tail_seg(20, 40, 90, base, fringe, 21), (-150, -80, 40, 80))
        K.part("tail2", 3.1, tail_seg(40, 50, 85, base, fringe, 23), (-150, -90, 60, 90))
        K.part("tail3", 3.2, tail_seg(50, 46, 92, base, fringe, 25, tip=True), (-150, -90, 60, 90))
    K.contacts = [("leg_n", (22, 46 if not kit else 42)), ("leg_f", (22, 46 if not kit else 42)), ("body", (62, 54 if not kit else 50)),
                  ("paw_n", (4, 42 if not kit else 38)), ("body", (16, 56)), ("body", (-14, 44))]
    K.springs = [{"bones": ["tail2", "tail3"], "freq": 0.9, "zeta": 0.3, "wind": 9 if not kit else 4, "hang": 0.0,
                  "rest": -25.0 if kit else 38.0, "drag": 0.06}]
    return K


def hare():
    """Folk-tale cast animal: a brown hare (sits upright like the squirrel rig expects; long ears, long hind feet)."""
    K = Rigkit("hare", kind="squirrel", units=320.0, q=3.0, pelvis_h=60.0)
    sq_bones(K, t2=-12, t3=-10)
    base = "#a07850"
    belly = "#f2e6d0"
    wash = "#c89a68"

    def body(c):
        p = smooth([(-30, 10), (-20, -36), (24, -60), (80, -58), (124, -40), (150, -6), (142, 26), (108, 48), (60, 58), (20, 60), (-14, 48)])
        furry(c, p, base, belly=belly, wash="#7a5432", n=1500, L=8, seed=31, ang=195)
        with clip(c, p):
            haunch = ellipse(18, 12, 54, 50)
            c.drawPath(haunch, paint(shader=rad((8, -4), 56, [(0, shade(base, 0.15)), (0.8, shade(base, -0.1)), (1, shade(base, -0.42))])))
            fur(c, haunch, base, n=360, length=8, seed=33, ang=220, a=110)
            # a sapphire ribbon collar with a small gold bell: the folk-tale touch
        stroke(c, smooth([(126, -40), (138, -6), (134, 22)], closed=False), JEWEL["sapphire"], 9)
        gold_line(c, smooth([(126, -40), (138, -6), (134, 22)], closed=False), 1.2)
        bead(c, 140, 8, 6)
    K.part("body", 10, body, (-70, -100, 190, 100))

    def head(c):
        for dx, dk in ((-10, 0.25), (6, 0.0)):  # long ears with black tips
            e = smooth([(dx - 10, -30), (dx - 22, -90), (dx - 16, -136), (dx - 2, -140), (dx + 8, -96), (dx + 8, -30)])
            c.drawPath(e, paint(shader=lin((0, -140), (0, -30), [(0, "#1a1410"), (0.12, "#2a2018"), (0.2, shade(base, -dk)), (1, shade(base, -0.2 - dk))])))
            c.drawPath(smooth([(dx - 6, -40), (dx - 15, -92), (dx - 10, -122), (dx + 1, -96), (dx + 2, -42)]), paint("#e8b8a8", 160 if not dk else 100))
            fur(c, e, base, n=80, length=5, seed=41, ang=270, a=80)
        hp = smooth([(-16, 14), (-20, -16), (0, -40), (38, -44), (68, -26), (92, -2), (90, 10), (70, 22), (24, 28)])
        furry(c, hp, base, wash=wash, n=700, L=6, seed=43, ang=190)
        with clip(c, hp):
            c.drawPath(smooth([(36, 10), (66, 4), (90, 6), (80, 20), (40, 26)]), paint(belly, 230, blur=4))
        flat(c, ellipse(89, 0, 5, 4.4), "#8a4a48")
        stroke(c, smooth([(90, 5), (86, 11), (80, 12)], closed=False), "#4a3030", 1.5, 200)
        ex, ey = 50, -14
        c.drawCircle(ex, ey, 13.5, paint("#f2e6d0", 220))
        c.drawCircle(ex, ey, 10.5, paint(shader=rad((ex + 3, ey + 3), 11, [(0, "#7a5020"), (0.55, "#3a2410"), (1, "#0a0604")])))
        c.drawCircle(ex + 1, ey, 5, paint("#0a0604"))
        c.drawCircle(ex - 3.4, ey - 4, 3.2, paint("#ffffff", 240))
        for k in range(5):
            a = math.radians(-20 + k * 12)
            stroke(c, poly([(80, 4), (80 + math.cos(a) * 50, 4 + math.sin(a) * 32)], closed=False), "#2a2020", 0.9, 140)
    K.part("head", 14, head, (-40, -160, 130, 50))

    def paw(dk):
        def draw(c):
            col = shade(base, -dk)
            arm = capsule(0, 0, 0, 36, 11, 8)
            furry(c, arm, col, n=140, L=4, seed=45, ang=90, a=100)
            flat(c, ellipse(2, 40, 10, 6), shade(belly, -dk - 0.1))
        return draw
    K.part("paw_f", 6, paw(0.22), (-25, -20, 25, 60))
    K.part("paw_n", 16, paw(0.0), (-25, -20, 25, 60))

    def leg(dk):
        def draw(c):
            col = shade(base, -dk)
            shin = capsule(0, 0, 6, 36, 18, 11)
            furry(c, shin, col, n=180, L=5, seed=47, ang=100, a=100)
            foot = smooth([(-8, 34), (30, 32), (70, 38), (72, 48), (-6, 48)])
            c.drawPath(foot, paint(shader=lin((0, 32), (0, 48), [(0, shade(belly, -dk - 0.05)), (1, shade(base, -0.3 - dk))])))
            fur(c, foot, base, n=80, length=4, seed=49, ang=0, a=90)
        return draw
    K.part("leg_f", 5, leg(0.22), (-35, -25, 90, 60))
    K.part("leg_n", 12, leg(0.0), (-35, -25, 90, 60))

    def cotton(c):
        m = union(circle(-10, -6, 16), circle(-18, 4, 12), circle(-2, 6, 12))
        c.drawPath(m, paint(shader=rad((-14, -8), 22, [(0, "#ffffff"), (1, "#d8d0c4")])))
        fur(c, m, "#f4efe6", n=120, length=4, seed=51, ang=180, a=100)
    K.part("tail1", 3, cotton, (-50, -40, 30, 40))
    K.part("tail2", 2.9, lambda c: c.drawCircle(0, 0, 0.6, paint("#f4efe6", 10)), (-6, -6, 6, 6))
    K.part("tail3", 2.8, lambda c: c.drawCircle(0, 0, 0.6, paint("#f4efe6", 10)), (-6, -6, 6, 6))
    K.contacts = [("leg_n", (30, 48)), ("leg_f", (30, 48)), ("body", (62, 58)), ("paw_n", (2, 44)), ("body", (14, 62)), ("body", (-20, 50))]
    K.springs = []
    return K


def blink_variant(K, kit):
    """Head with the eye closed: re-run the head drawing, then a fur lid over the eye."""
    hd = next(p for p in K.parts if p["name"] == "head")
    base = "#8f9196" if not kit else "#9a9ca2"
    s = 0.92 if kit else 1.0
    hs = 1.14 if kit else 1.0

    def draw(c):
        hd["draw"](c)
        c.save()
        c.scale(s * hs, s * hs)
        c.drawCircle(50, -16, 11.5, paint(shade(base, 0.08)))
        stroke(c, smooth([(40, -17), (50, -12), (60, -17)], closed=False), "#1a1210", 2.6)
        c.restore()
    K.variants["head"] = {"blink": (draw, hd["box"])}


if __name__ == "__main__":
    for n in sys.argv[1:] or ["pip", "pip_kit", "hare"]:
        if n == "hare":
            K = hare()
        else:
            K = squirrel(kit=(n == "pip_kit"))
            blink_variant(K, n == "pip_kit")
        J = K.build(os.path.join(ROOT, n, "rig"))
        print(n, len(J["parts"]), "parts")
