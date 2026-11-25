"""Vector puppet characters (the SVG/vector part-library source). Same skeleton for every humanoid, outfits are
data, so a new character is a dict. Tube limbs with round joint caps, profile heads ~1/8 of the height, like the
reference's puppets; drawn in Heartwick's own designs and palette.

build(name, painter) -> (rig, springs). Raster (AI-image) rigs come from engine2/rig.load_partsheet instead and
use the same bone names, so poses, scheduler and springs drive them unchanged.
"""
import math
import skia
from .rig import Bone, Skeleton, Part, Rig
from .style import (rgb, shade, mix, circle, ellipse, capsule, poly, hull, path_op, PALETTE)

GOLD = PALETTE["gold"]


class D:  # proportions (rig units, total height ~1000)
    def __init__(self, **kw):
        self.thigh, self.shin, self.spine, self.chest, self.neck = 232, 222, 172, 150, 38
        self.head_r, self.uarm, self.farm, self.limb = 64, 168, 148, 1.0
        self.width = 1.0
        self.__dict__.update(kw)
        self.pelvis_h = self.thigh + self.shin + 14


def human_skeleton(d):
    B = Bone
    sh_y = -d.chest + 24
    return Skeleton([
        B("pelvis", None),
        B("spine", "pelvis", (0, 0), d.spine), B("chest", "spine", (0, -d.spine), d.chest),
        B("neck", "chest", (8, -d.chest), d.neck), B("head", "neck", (2, -d.neck)),
        B("uarm_f", "chest", (-10, sh_y), d.uarm), B("farm_f", "uarm_f", (0, d.uarm), d.farm), B("hand_f", "farm_f", (0, d.farm)),
        B("uarm_n", "chest", (10, sh_y), d.uarm), B("farm_n", "uarm_n", (0, d.uarm), d.farm), B("hand_n", "farm_n", (0, d.farm)),
        B("thigh_f", "pelvis", (-12, 4), d.thigh), B("shin_f", "thigh_f", (0, d.thigh), d.shin), B("foot_f", "shin_f", (0, d.shin)),
        B("thigh_n", "pelvis", (12, 4), d.thigh), B("shin_n", "thigh_n", (0, d.thigh), d.shin), B("foot_n", "shin_n", (0, d.shin)),
        B("scarf1", "chest", (26, -d.chest + 14), 64), B("scarf2", "scarf1", (0, 64), 58), B("scarf3", "scarf2", (0, 58), 52),
        B("hair1", "head", (-50, -46), 34), B("hair2", "hair1", (0, 34), 30),
    ])


# ---------------------------------------------------------------- humanoid parts
def _hand(c, P, skin, shape, side_dark=0.0, r=1.0):
    sk = shade(skin, -side_dark)
    if shape == "fist":
        P.fill(c, circle(0, 18 * r, 17 * r), sk, 0.25, kind="skin")
        return
    palm = ellipse(0, 24 * r, 15 * r, 21 * r)
    P.fill(c, palm, sk, 0.25, kind="skin")
    if shape == "point":
        P.fill(c, capsule(4 * r, 34 * r, 6 * r, 70 * r, 5.5 * r), sk, 0.2, kind="skin")
    elif shape == "cup":
        P.fill(c, capsule(10 * r, 30 * r, 18 * r, 44 * r, 6 * r), sk, 0.2, kind="skin")
    else:  # open: fingers hint
        P.fill(c, capsule(0, 34 * r, 2 * r, 50 * r, 9 * r, 7 * r), sk, 0.2, kind="skin")
    P.fill(c, ellipse(12 * r, 14 * r, 6 * r, 10 * r), sk, 0.2, kind="skin")  # thumb


def _arm_parts(spec, d, side, z0):
    far = side == "f"
    dk = 0.22 if far else 0.0
    top = shade(spec["top"], -dk)
    skin = spec["skin"]
    cuff = spec.get("cuff", shade(spec["top"], 0.25))
    sleeve_len = spec.get("sleeve", 1.0)  # 1 = long sleeves to the wrist

    def uarm(c, x):
        P = x.P
        P.fill(c, capsule(0, 0, 0, d.uarm, 25 * d.width, 21 * d.width), top, 0.34)
        if spec.get("top_dots"):
            P.pattern_dots(c, capsule(0, 0, 0, d.uarm, 25 * d.width, 21 * d.width), shade(spec["top_dots"], -dk), 15, 2.2, 170)
        P.joint(c, 0, 2, 24 * d.width, top)

    def farm(c, x):
        P = x.P
        L = d.farm * sleeve_len
        if sleeve_len < 1:
            P.fill(c, capsule(0, 0, 0, d.farm, 15, 12), shade(skin, -dk), 0.25, kind="skin")
        P.fill(c, capsule(0, 0, 0, L, 20 * d.width, 16 * d.width), top, 0.34)
        if spec.get("top_dots"):
            P.pattern_dots(c, capsule(0, 0, 0, L, 20 * d.width, 16 * d.width), shade(spec["top_dots"], -dk), 15, 2.2, 170)
        if sleeve_len >= 1:
            P.fill(c, capsule(0, L - 16, 0, L - 4, 18 * d.width), cuff, 0.2)
        P.joint(c, 0, 0, 20 * d.width, top)

    def hand(c, x):
        shape = x.face.get(f"hand_{side}", "open")
        _hand(c, x.P, skin, shape, dk)
        prop = x.extra.get(f"prop_{side}")
        if prop and not far:
            PROPS[prop](c, x)

    return [Part(f"uarm_{side}", z0, uarm), Part(f"farm_{side}", z0 + 1, farm), Part(f"hand_{side}", z0 + 2, hand)]


def _leg_parts(spec, d, side, z0):
    far = side == "f"
    dk = 0.25 if far else 0.0
    trousers = spec.get("bottom") == "trousers"
    leg = shade(spec["bottom_color"] if trousers else spec.get("stockings", "#6b4a3a"), -dk)
    shoe = shade(spec.get("shoes", "#3a2418"), -dk)

    def thigh(c, x):
        x.P.fill(c, capsule(0, 0, 0, d.thigh, 31, 25), leg, 0.3)
        if trousers:
            x.P.joint(c, 0, d.thigh, 24, leg)

    def shin(c, x):
        x.P.fill(c, capsule(0, 0, 0, d.shin, 22 if trousers else 16, 18 if trousers else 13), leg, 0.3)
        if spec.get("boots"):
            x.P.fill(c, capsule(0, d.shin * 0.45, 0, d.shin, 21, 20), shoe, 0.3)

    def foot(c, x):
        p = poly([(-16, -8), (10, -12), (44, -2), (50, 10), (-18, 12)], smooth=True)
        x.P.fill(c, p, shoe, 0.35)
        if spec.get("sneakers"):
            x.P.flat(c, poly([(-18, 6), (50, 6), (50, 12), (-18, 12)]), "#f4efe6")

    return [Part(f"thigh_{side}", z0, thigh), Part(f"shin_{side}", z0 + 1, shin), Part(f"foot_{side}", z0 + 2, foot)]


def _skirt(spec, d):
    col = spec["bottom_color"]
    band = spec.get("band")

    def draw(c, x):
        pts = [(-58, -34), (60, -34), (80, 40), (-82, 40)]
        for leg in ("n", "f"):  # the cloth hangs from each knee down to mid-shin, flaring out
            k = x.local(f"shin_{leg}")
            m = x.local(f"shin_{leg}", (0, d.shin * 0.6))
            dx, dy = m[0] - k[0], m[1] - k[1]
            L = math.hypot(dx, dy) or 1
            nx, ny = -dy / L, dx / L
            pts += [(k[0] + nx * 30, k[1] + ny * 30), (k[0] - nx * 30, k[1] - ny * 30),
                    (m[0] + nx * 46, m[1] + ny * 46), (m[0] - nx * 46, m[1] - ny * 46)]
        path = poly(hull(pts), smooth=True)
        x.P.fill(c, path, col, 0.36)
        if band:  # embroidered hem band: the skirt minus itself shifted up = a strip along the lower edge
            up = skia.Path(path)
            up.offset(0, -30)
            strip = path_op(path, up, "difference")
            if strip is not None:
                x.P.fill(c, strip, band, 0.2, kind="detail")
                x.P.pattern_dots(c, strip, spec.get("band_dots", PALETTE["ruby"]), 13, 3.2, 230)
    return Part("pelvis", 20, draw, "skirt")


def _torso(spec, d):
    top, blouse = spec["top"], spec.get("blouse", PALETTE["cream"])
    w = d.width

    def lower(c, x):  # waist, attached to spine
        p = poly([(-56 * w, 10), (58 * w, 10), (64 * w, -d.spine * 0.55), (50 * w, -d.spine - 10), (-50 * w, -d.spine - 10), (-62 * w, -d.spine * 0.55)], smooth=True)
        x.P.fill(c, p, spec.get("lower", top), 0.34)
        if spec.get("top_dots") and not spec.get("lower"):
            x.P.pattern_dots(c, p, spec["top_dots"], 15, 2.2, 160)
        if spec.get("belt"):
            x.P.fill(c, poly([(-60 * w, -2), (62 * w, -2), (62 * w, 14), (-60 * w, 14)]), spec["belt"], 0.2)

    def upper(c, x):  # chest, shoulders, cardigan opening + buttons
        br = 1 + 0.012 * x.face.get("breath", 0)
        c.save()
        c.scale(br, br)
        p = poly([(-52 * w, 6), (56 * w, 6), (62 * w, -d.chest * 0.5), (44 * w, -d.chest + 8), (8, -d.chest - 6), (-40 * w, -d.chest + 6), (-58 * w, -d.chest * 0.5)], smooth=True)
        x.P.fill(c, p, top, 0.34)
        if spec.get("top_dots"):
            x.P.pattern_dots(c, p, spec["top_dots"], 15, 2.2, 170)
        if spec.get("vest"):
            v = poly([(-50 * w, 4), (30 * w, 4), (36 * w, -d.chest * 0.55), (20, -d.chest + 4), (-30 * w, -d.chest + 6), (-56 * w, -d.chest * 0.5)], smooth=True)
            x.P.fill(c, v, spec["vest"], 0.3)
        # blouse V and buttons on the front edge
        x.P.fill(c, poly([(14, -d.chest + 2), (42, -d.chest + 6), (30, -d.chest + 58)], smooth=False), blouse, 0.15, kind="detail")
        if spec.get("buttons", True):
            for k in range(3):
                x.P.fill(c, circle(40 * w, -d.chest + 62 + k * 36, 5.2), GOLD, 0.4, kind="detail")
        if spec.get("plaid"):
            c.save()
            c.clipPath(p, skia.ClipOp.kIntersect, True)
            pl = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=5, Color=rgb(spec["plaid"], 110))
            for k in range(-4, 6):
                c.drawLine(-80, -d.chest + k * 26, 80, -d.chest + k * 26, pl)
                c.drawLine(-60 + k * 26, -d.chest - 20, -60 + k * 26, 20, pl)
            c.restore()
        c.restore()

    return [Part("spine", 30, lower, "waist"), Part("chest", 31, upper, "upper")]


def _head(spec, d):
    skin, hair = spec["skin"], spec["hair"]
    r = d.head_r
    cx, cy = 14, -r - 2

    def neck(c, x):
        x.P.fill(c, capsule(0, 4, 0, -d.neck - 6, 17, 15), shade(skin, -0.12), 0.25, kind="skin")

    def head(c, x):
        P, f = x.P, x.face
        style = spec.get("hair_style", "bun")
        # back hair / bun / braid
        if style == "bun":
            P.fill(c, circle(cx - r * 0.62, cy - r * 0.78, r * 0.44), hair, 0.3, kind="hair")
            P.stroke(c, poly([(cx - r * 0.9, cy - r * 0.75), (cx - r * 0.4, cy - r * 0.9)], closed=False), shade(hair, -0.3), 3)
        elif style == "braid":
            for k in range(4):
                P.fill(c, ellipse(cx - r * 0.78, cy + r * (0.2 + k * 0.42), r * 0.2, r * 0.26), hair, 0.3, kind="hair")
        P.fill(c, ellipse(cx - r * 0.82, cy + 2, r * 0.2, r * 0.27), shade(skin, -0.12), 0.2, kind="skin")  # ear
        face = skia.Path()
        face.addOval(skia.Rect(cx - r, cy - r * 1.02, cx + r * 0.95, cy + r * 1.02))
        nose = skia.Path()
        nose.addOval(skia.Rect(cx + r * 0.8, cy - r * 0.2, cx + r * 1.12, cy + r * 0.14))
        chin = skia.Path()
        chin.addOval(skia.Rect(cx + r * 0.1, cy + r * 0.35, cx + r * 0.85, cy + r * 1.02))
        head_p = path_op(path_op(face, nose), chin)
        P.fill(c, head_p, skin, 0.28, kind="skin")
        # hair cap over the top and back of the head
        cap = poly([(cx + r * 0.72, cy - r * 0.62), (cx + r * 0.3, cy - r * 1.08), (cx - r * 0.45, cy - r * 1.06),
                    (cx - r * 1.04, cy - r * 0.45), (cx - r * 0.98, cy + r * 0.35), (cx - r * 0.62, cy + r * 0.55),
                    (cx - r * 0.55, cy - r * 0.05), (cx - r * 0.1, cy - r * 0.55), (cx + r * 0.42, cy - r * 0.5)], smooth=True)
        if style == "messy":
            for k in range(6):
                a = math.radians(-160 + k * 26)
                P.fill(c, ellipse(cx + math.cos(a) * r * 0.8, cy - r * 0.35 + math.sin(a) * r * 0.75, r * 0.28, r * 0.2), hair, 0.3, kind="hair")
        P.fill(c, cap, hair, 0.3, kind="hair")
        # face details
        ex, ey = cx + r * 0.52, cy - r * 0.12
        blink = f.get("blink", 1.0)
        lk = f.get("look", 0.0)
        P.fill(c, ellipse(cx + r * 0.42, cy + r * 0.32, r * 0.2, r * 0.14), PALETTE["ember"], 0.1, alpha=70, kind="detail")  # cheek
        P.flat(c, ellipse(ex + lk * 1.5, ey, r * 0.085, max(0.6, r * 0.11 * blink)), "#2a1a14")
        if blink > 0.6:
            P.flat(c, circle(ex + lk * 1.5 + 1.6, ey - 2.4, 1.8), "#ffffff", 220)
        brow = shade(hair, -0.35) if spec.get("age") != "old" else "#9a9fb2"
        P.stroke(c, poly([(ex - r * 0.16, ey - r * 0.2), (ex + r * 0.02, ey - r * 0.27), (ex + r * 0.16, ey - r * 0.21)], smooth=True, closed=False), brow, 3.2)
        sm = f.get("smile", spec.get("smile", 0.6))
        mx, my = cx + r * 0.66, cy + r * 0.5
        P.stroke(c, poly([(mx - r * 0.13, my - 1), (mx, my + r * 0.07 * sm), (mx + r * 0.1, my - 2)], smooth=True, closed=False), "#8a3a3a", 3.0)
        if spec.get("age") == "old":
            P.stroke(c, poly([(ex - r * 0.3, ey + r * 0.05), (ex - r * 0.22, ey + r * 0.14)], closed=False), shade(skin, -0.25), 1.8)
        if spec.get("freckles"):
            for k in range(5):
                P.flat(c, circle(cx + r * (0.3 + 0.07 * k), cy + r * (0.15 + 0.05 * (k % 2)), 1.6), "#c27a52", 190)
        if spec.get("glasses"):
            gp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3.0, Color=rgb("#c9962a"))
            c.drawCircle(ex, ey, r * 0.24, gp)
            c.drawLine(ex - r * 0.24, ey - 2, cx - r * 0.7, cy - r * 0.08, gp)
        if spec.get("hair_style") == "bun":  # front fringe wave
            P.fill(c, poly([(cx + r * 0.62, cy - r * 0.7), (cx + r * 0.2, cy - r * 0.98), (cx - r * 0.2, cy - r * 0.72), (cx + r * 0.25, cy - r * 0.6)], smooth=True), shade(hair, 0.15), 0.2, kind="hair")

    def wisp(c, x):
        x.P.fill(c, capsule(0, 0, 0, 34, 9, 5), spec["hair"], 0.3, kind="hair")

    def wisp2(c, x):
        x.P.fill(c, capsule(0, 0, 0, 30, 5, 2.5), spec["hair"], 0.3, kind="hair")

    parts = [Part("neck", 32, neck), Part("head", 34, head)]
    if spec.get("hair_style") in ("bun", "braid", "messy"):
        parts += [Part("hair1", 33, wisp), Part("hair2", 33, wisp2)]
    return parts


def _scarf(spec):
    sc = spec.get("scarf")
    if not sc:
        return []
    col, dots = sc["color"], sc.get("dots", GOLD)

    def wrap(c, x):  # collar wrap on the chest
        p = poly([(-30, -142), (34, -150), (44, -126), (-26, -118)], smooth=True)
        x.P.fill(c, p, col, 0.3)
        x.P.pattern_dots(c, p, dots, 14, 2.4, 220)

    def seg(L, r0, r1, end=False):
        def draw(c, x):
            p = capsule(0, 0, 0, L, r0, r1)
            x.P.fill(c, p, col, 0.3)
            x.P.pattern_dots(c, p, dots, 14, 2.4, 220)
            if end:
                for k in range(5):
                    a = -14 + k * 7
                    tp = capsule(-r1 + k * r1 / 2, L, -r1 + k * r1 / 2 + math.sin(math.radians(a)) * 20, L + 22, 2.2)
                    x.P.fill(c, tp, GOLD, 0.2, kind="detail")
                    x.P.fill(c, circle(-r1 + k * r1 / 2 + math.sin(math.radians(a)) * 20, L + 24, 3.6), GOLD, 0.3, kind="detail")
        return draw

    return [Part("chest", 36, wrap, "scarf_wrap"), Part("scarf1", 35, seg(64, 17, 16)), Part("scarf2", 35, seg(58, 16, 15)),
            Part("scarf3", 35, seg(52, 15, 14, end=True))]


def humanoid(spec, painter=None):
    d = D(**spec.get("dims", {}))
    sk = human_skeleton(d)
    parts = []
    parts += _leg_parts(spec, d, "f", 10)
    parts += _arm_parts(spec, d, "f", 12)
    parts += _leg_parts(spec, d, "n", 16)
    if spec.get("bottom") == "skirt":
        parts.append(_skirt(spec, d))
    parts += _torso(spec, d)
    parts += _head(spec, d)
    parts += _scarf(spec)
    parts += _arm_parts(spec, d, "n", 40)
    contacts = [("foot_n", (44, 10)), ("foot_n", (-16, 12)), ("foot_f", (44, 10)), ("foot_f", (-16, 12)),
                ("shin_n", (0, 0)), ("shin_f", (0, 0)), ("shin_f", (0, d.shin * 0.6))]
    rig = Rig(spec["name"], sk, parts, contacts, units=1000.0, kind="humanoid", meta={"pelvis_h": d.pelvis_h, "source": "vector"})
    springs = [{"bones": ["hair1", "hair2"], "freq": 1.3, "zeta": 0.22, "wind": 7, "drag": 0.03}]
    if spec.get("scarf"):
        springs.append({"bones": ["scarf1", "scarf2", "scarf3"], "freq": 0.75, "zeta": 0.2, "wind": 7, "drag": 0.05})
    return rig, springs


# ---------------------------------------------------------------- props held in the near hand (hand frame: +y along the arm)
def _mug(c, x):
    P = x.P
    P.fill(c, poly([(-14, 30), (22, 30), (20, 66), (-12, 66)], smooth=False), PALETTE["cream"], 0.25)
    P.stroke(c, poly([(22, 38), (34, 44), (21, 58)], smooth=True, closed=False), PALETTE["cream"], 5)
    P.fill(c, poly([(-14, 40), (22, 40), (22, 47), (-14, 47)]), PALETTE["ruby"], 0.2, kind="detail")
    t = x.t
    for k in range(2):  # steam
        a = 0.35 + 0.25 * math.sin(t * 2 + k)
        P.stroke(c, poly([(0 + k * 9, 26), (6 + k * 9 + 6 * math.sin(t * 3 + k), 8), (0 + k * 9, -12)], smooth=True, closed=False), "#ffffff", 3, int(255 * a))


def _bundle(c, x):
    P = x.P
    p = ellipse(10, 46, 40, 30)
    P.fill(c, p, "#b3243a", 0.3)
    c.save()
    c.clipPath(p, skia.ClipOp.kIntersect, True)
    pl = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=4, Color=rgb("#1d1b2e", 120))
    for k in range(-3, 5):
        c.drawLine(-40, 16 + k * 14, 60, 16 + k * 14, pl)
        c.drawLine(-30 + k * 14, 10, -30 + k * 14, 80, pl)
    c.restore()
    P.fill(c, circle(24, 30, 13), "#e9b7a6", 0.25, kind="skin")  # tiny face peeking out
    P.stroke(c, poly([(26, 28), (31, 29)], closed=False), "#4a2a22", 2)


def _phone(c, x):
    x.P.fill(c, poly([(-8, 4), (10, 4), (12, 70), (-6, 70)]), "#e8e2d6", 0.3)


def _book(c, x):
    P = x.P
    P.fill(c, poly([(-30, 30), (36, 22), (40, 60), (-26, 66)]), PALETTE["sapphire"], 0.3)
    P.fill(c, poly([(-26, 34), (32, 27), (34, 56), (-22, 61)]), PALETTE["cream"], 0.1, kind="detail")


def _can(c, x):
    P = x.P
    P.fill(c, poly([(-16, 30), (20, 30), (18, 70), (-14, 70)]), PALETTE["emerald"], 0.3)
    P.stroke(c, poly([(18, 40), (52, 22)], closed=False), PALETTE["emerald"], 6)


def _walnut(c, x):
    x.P.fill(c, circle(0, 22, 13), "#8a5a32", 0.35)


PROPS = {"mug": _mug, "bundle": _bundle, "phone": _phone, "book": _book, "can": _can, "walnut": _walnut}


# ---------------------------------------------------------------- squirrels
def squirrel(kind="adult"):
    """Side-view squirrel. Bones run along +x (head direction); paws/legs hang along +y; tail runs along -x and
    positive angles curl it upward."""
    kit = kind == "kit"
    B = Bone
    sk = Skeleton([
        B("body", None), B("head", "body", (128, -14)),
        B("paw_f", "body", (104, 18)), B("paw_n", "body", (112, 22)),
        B("leg_f", "body", (14, 20)), B("leg_n", "body", (22, 24)),
        B("tail1", "body", (-22, -6), 90), B("tail2", "tail1", (-40 if kit else -90, 0), 85), B("tail3", "tail2", (-36 if kit else -85, 0), 80),
    ])
    fur = "#e8b8a8" if kit else "#9aa0aa"
    fur_dk = "#c98f80" if kit else "#6e7480"
    belly = "#f4cfc4" if kit else "#f3e6cf"

    def body(c, x):
        P = x.P
        br = 1 + 0.03 * x.face.get("breath", 0)
        c.save()
        c.scale(1, br)
        P.fill(c, ellipse(62, 0, 82, 50 if kit else 56), fur, 0.32, kind="fur")
        P.fill(c, ellipse(84, 14, 50, 30 if kit else 34), belly, 0.15, kind="fur")
        P.fill(c, ellipse(20, 10, 46, 46), fur_dk, 0.3, kind="fur")  # haunch
        if kit:
            P.pattern_dots(c, ellipse(62, -10, 78, 36), "#b9b4bd", 11, 1.4, 120)
        c.restore()

    def head(c, x):
        P, f = x.P, x.face
        r = 50 if kit else 44
        P.fill(c, ellipse(-8, -54 if not kit else -40, 12, 20) if not kit else ellipse(-6, -36, 8, 10), fur_dk, 0.3, kind="fur")  # ear
        P.fill(c, ellipse(30, -6, r * 1.05, r * 0.95), fur, 0.3, kind="fur")
        P.fill(c, ellipse(64, 6, r * 0.42, r * 0.36), belly, 0.15, kind="fur")  # muzzle
        P.flat(c, circle(80, 2, 5), "#3a2420")
        ex, ey = 46, -14
        if kit:  # eyes barely open: a closed lid line that lifts a little when it 'opens'
            o = 1 - x.extra.get("eyes_closed", 1.0)
            P.stroke(c, poly([(ex - 10, ey), (ex, ey + 3), (ex + 10, ey)], smooth=True, closed=False), "#6a4040", 3)
            if o > 0.05:
                P.flat(c, ellipse(ex, ey + 1, 6, 4 * o), "#2a1a14")
        else:
            P.flat(c, ellipse(ex, ey, 10, 11 * max(0.1, f.get("blink", 1))), "#1a1210")
            P.flat(c, circle(ex + 3, ey - 4, 3.2), "#ffffff", 230)
        if not kit:
            P.fill(c, ellipse(-12, -50, 9, 22), fur_dk, 0.3, kind="fur")
            P.fill(c, poly([(-14, -70), (-8, -92), (-2, -70)]), fur_dk, 0.3, kind="fur")  # ear tuft

    def paw(dk):
        def draw(c, x):
            col = shade("#e8a898" if kit else fur, -dk)
            x.P.fill(c, capsule(0, 0, 0, 34 if not kit else 22, 9 if not kit else 7, 7), col, 0.3, kind="fur")
            x.P.fill(c, ellipse(4, 36 if not kit else 24, 9, 6), shade(belly, -dk), 0.2, kind="fur")
            if x.extra.get("prop_paw") and dk == 0:
                PROPS[x.extra["prop_paw"]](c, x)
        return draw

    def leg(dk):
        def draw(c, x):
            col = shade(fur_dk, -dk)
            x.P.fill(c, capsule(0, 0, 0, 34 if not kit else 22, 15 if not kit else 10, 9), col, 0.3, kind="fur")
            x.P.fill(c, ellipse(22, 38 if not kit else 26, 26 if not kit else 16, 8), col, 0.3, kind="fur")
        return draw

    def tail(c, x):
        """One bushy shape through the whole spring chain (drawn in tail1's frame)."""
        pts = [(0.0, 0.0), x.local("tail2"), x.local("tail3"), x.local("tail3", (-80, 0))]
        if kit:
            pts = [(0.0, 0.0), x.local("tail2"), x.local("tail3"), x.local("tail3", (-30, 0))]
        # resample the Catmull-Rom curve and lay overlapping discs that swell toward the middle (fluffy tail)
        curve = []
        for i in range(len(pts) - 1):
            p0, p1, p2, p3 = pts[max(0, i - 1)], pts[i], pts[i + 1], pts[min(len(pts) - 1, i + 2)]
            for k in range(8):
                u = k / 8
                u2, u3 = u * u, u * u * u
                curve.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * u + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * u2
                                         + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * u3) for j in (0, 1)))
        curve.append(pts[-1])
        path = skia.Path()
        n = len(curve)
        for i, (px, py) in enumerate(curve):
            u = i / (n - 1)
            r = (4 + 3 * (1 - u)) if kit else (26 + 30 * math.sin(math.pi * min(1, u * 1.15)) + 6 * (1 - u))
            path.addCircle(px, py, r)
        path = path_op(path, skia.Path())
        x.P.fill(c, path, "#e3a99a" if kit else fur, 0.3, kind="fur")
        if not kit:  # pale fur edge on the outer curve
            hp = skia.Path()
            for i, (px, py) in enumerate(curve[2:-2]):
                hp.addCircle(px - 6, py - 10, 10 + 10 * math.sin(math.pi * i / max(1, n - 5)))
            x.P.fill(c, hp, "#c9ccd4", 0.15, kind="detail", alpha=170)

    parts = [Part("tail1", 3, tail, "tail"),
             Part("leg_f", 5, leg(0.2)), Part("paw_f", 6, paw(0.2)), Part("body", 10, body), Part("leg_n", 12, leg(0)),
             Part("head", 14, head), Part("paw_n", 16, paw(0))]
    contacts = [("leg_n", (22, 44 if not kit else 32)), ("leg_f", (22, 44 if not kit else 32)), ("body", (62, 50 if kit else 56)),
                ("paw_n", (4, 40 if not kit else 28))]
    rig = Rig(f"squirrel_{kind}", sk, parts, contacts, units=260.0 if not kit else 200.0, kind="squirrel", meta={"pelvis_h": 60.0, "source": "vector"})
    springs = [{"bones": ["tail2", "tail3"], "freq": 0.9, "zeta": 0.3, "wind": 9 if not kit else 4, "hang": 0.0,
                "rest": -25.0 if kit else 38.0, "drag": 0.06}]
    return rig, springs


# ---------------------------------------------------------------- the cast (Heartwick's own designs, from stories/pip.md)
CAST = {
    "nell": dict(name="nell", skin="#f2c6a6", hair="#e6e4ee", hair_style="bun", glasses=True, age="old",
                 top=PALETTE["emerald"], top_dots="#5fb48e", blouse=PALETTE["cream"], bottom="skirt", bottom_color="#7a3b22",
                 band=PALETTE["marigold"], band_dots=PALETTE["ruby"], stockings="#5a3a2e", shoes="#3a2418",
                 scarf={"color": PALETTE["sapphire"], "dots": PALETTE["gold"]}, dims={"width": 1.02}),
    "dee": dict(name="dee", skin="#e3b08a", hair="#6b4226", hair_style="braid", age="adult", top="#b8432f", plaid="#2a1e3a",
                vest="#7d838e", blouse="#f1e3cf", buttons=False, bottom="trousers", bottom_color="#3a5a8a", shoes="#3f6b3a",
                boots=True, smile=0.4),
    "eli": dict(name="eli", skin="#f3cba9", hair="#d9b36a", hair_style="messy", age="child", freckles=True, top="#c8332b",
                blouse="#f4efe6", buttons=False, bottom="trousers", bottom_color="#3a5a8a", shoes="#e9e4da", sneakers=True,
                smile=1.0, dims={"thigh": 205, "shin": 196, "spine": 160, "chest": 140, "head_r": 78, "uarm": 150, "farm": 132, "width": 0.92}),
}


def build(name, painter=None):
    if name in ("pip", "squirrel"):
        return squirrel("adult")
    if name in ("pip_kit", "kit"):
        return squirrel("kit")
    return humanoid(CAST[name], painter)
