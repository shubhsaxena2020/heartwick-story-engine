"""Heartwick cast, rich edition: rig-ready part sheets for engine v2 (raster part-sheet format).

Same skeleton, bone names and proportions as engine2/characters.py (so every pose in engine2/poses.py fits), drawn
with the airbrushed jewel-tone vocabulary in paint.py. Run:  python assets/src/chars.py [name ...]
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from paint import *
from heads import head_part
from rigkit import Rigkit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "characters"))


class D:
    def __init__(self, **kw):
        self.thigh, self.shin, self.spine, self.chest, self.neck = 232, 222, 172, 150, 38
        self.uarm, self.farm, self.w = 168, 148, 1.0
        self.__dict__.update(kw)
        self.pelvis_h = self.thigh + self.shin + 14


def human_bones(K, d, chains=()):
    sh_y = -d.chest + 24
    K.bone("pelvis", None)
    K.bone("spine", "pelvis", (0, 0), d.spine)
    K.bone("chest", "spine", (0, -d.spine), d.chest)
    K.bone("neck", "chest", (8, -d.chest), d.neck)
    K.bone("head", "neck", (2, -d.neck))
    for s, x in (("f", -10), ("n", 10)):
        K.bone(f"uarm_{s}", "chest", (x, sh_y), d.uarm)
        K.bone(f"farm_{s}", f"uarm_{s}", (0, d.uarm), d.farm)
        K.bone(f"hand_{s}", f"farm_{s}", (0, d.farm))
    for s, x in (("f", -12), ("n", 12)):
        K.bone(f"thigh_{s}", "pelvis", (x, 4), d.thigh)
        K.bone(f"shin_{s}", f"thigh_{s}", (0, d.thigh), d.shin)
        K.bone(f"foot_{s}", f"shin_{s}", (0, d.shin))
    if "scarf" in chains:
        K.bone("scarf1", "chest", (30, -d.chest + 22), 70)
        K.bone("scarf2", "scarf1", (0, 70), 64)
        K.bone("scarf3", "scarf2", (0, 64), 58)
    if "cape" in chains:
        K.bone("cape1", "chest", (-34, -d.chest + 16), 150)
        K.bone("cape2", "cape1", (0, 150), 150)
        K.bone("cape3", "cape2", (0, 150), 140)
    if "braid" in chains:
        K.bone("hair1", "head", (-46, -28), 64)
        K.bone("hair2", "hair1", (0, 64), 60)
    K.contacts = [("foot_n", (44, 10)), ("foot_n", (-16, 12)), ("foot_f", (44, 10)), ("foot_f", (-16, 12)),
                  ("shin_n", (0, 0)), ("shin_f", (0, 0)), ("shin_f", (0, d.shin * 0.6))]


# ---------------------------------------------------------------- garment surface: base colour + pattern + shading
def garment(c, path, g, axis="x", far=False, light=0.36):
    """g: {"color", "pattern": (kind, args...)}; far-side parts are a little darker (they sit behind the body)."""
    base = shade(g["color"], -0.2) if far else g["color"]
    c.drawPath(path, paint(base))
    pat = g.get("pattern")
    if pat:
        kind = pat[0]
        if kind == "knit":
            pat_knit(c, path, base, *pat[1:])
        elif kind == "tartan":
            pat_tartan(c, path, base, *pat[1:])
        elif kind == "quilt":
            pat_quilt(c, path, *pat[1:])
        elif kind == "sprigs":
            pat_sprigs(c, path, *pat[1:])
        elif kind == "dots":
            pat_dots(c, path, *pat[1:])
        elif kind == "weave":
            pat_weave(c, path, *pat[1:])
        elif kind == "damask":
            damask(c, path, *pat[1:])
    shade_over(c, path, axis=axis, strength=0.75 if far else 0.62, light=light)


def damask(c, path, color, step=40, s=1.0):
    """Gold folk-damask: alternating rosettes and tulip pairs on a diagonal lattice."""
    b = bounds(path)
    with clip(c, path):
        y, row = b.top() - step, 0
        while y < b.bottom() + step:
            x = b.left() - step + (step / 2 if row % 2 else 0)
            while x < b.right() + step:
                if (row + int(x / step)) % 2:
                    rosette(c, x, y, 7 * s, color, center=shade(color, 0.4), ring=False)
                else:
                    tulip(c, x, y + 4 * s, 0.7 * s, color, leaf=color, gold=False)
                x += step
            y += step * 0.75
            row += 1


def tube_part(L, r0, r1, g, far=False, cap_top=True, cuff=None, trim=None):
    """A limb segment along +y from the pivot: garment tube, optional cuff band and gold trim."""
    def draw(c):
        p = capsule(0, 0, 0, L, r0, r1)
        garment(c, p, g, far=far)
        ao(c, p, 0, -2, r0 * 1.1, 70)
        if cuff:
            band = inter(p, rect(-r1 * 2, L - cuff["h"], r1 * 2, L + r1))
            garment(c, band, cuff, far=far)
            if cuff.get("gold"):
                gold_line(c, poly([(-r1 - 1, L - cuff["h"]), (r1 + 1, L - cuff["h"])], closed=False), 2.0, beads=7 if cuff.get("beads") else 0)
        if trim:
            trim(c, p)
    return draw


def skin_tube(L, r0, r1, skin, far=False):
    def draw(c):
        p = capsule(0, 0, 0, L, r0, r1)
        cyl(c, p, shade(skin, -0.18) if far else skin, depth=0.8)
    return draw


# ---------------------------------------------------------------- hands (hand frame: +y along the forearm, palm faces the body side)
def hand_draw(skin, shape="open", far=False, s=1.0, glove=None, prop=None):
    col = shade(skin, -0.2) if far else skin

    def draw(c):
        c.save()
        c.scale(s, s)
        dk = shade(col, -0.32)
        if prop == "staff_back":
            prop_staff(c, back=True)
        if shape == "fist":
            fist = smooth([(-13, 2), (12, 0), (17, 14), (16, 30), (8, 36), (-10, 34), (-15, 18)])
            cyl(c, fist, col, depth=0.7)
            for k in range(3):
                stroke(c, smooth([(4, 14 + k * 7), (14, 15 + k * 7)], closed=False), dk, 1.2, 150)
            thumb = smooth([(10, 6), (17, 10), (18, 20), (12, 22)])
            cyl(c, thumb, shade(col, 0.05), depth=0.6)
        elif shape == "point":
            palm = smooth([(-12, 0), (12, 0), (14, 22), (8, 34), (-8, 34), (-14, 18)])
            cyl(c, palm, col, depth=0.7)
            finger = capsule(4, 30, 7, 70, 5.2, 4.2)
            cyl(c, finger, col, depth=0.6)
            curled = smooth([(-10, 28), (6, 30), (8, 42), (-6, 44)])
            cyl(c, curled, shade(col, -0.08), depth=0.6)
            thumb = smooth([(10, 10), (18, 16), (17, 26), (11, 24)])
            cyl(c, thumb, col, depth=0.6)
        elif shape == "cup":
            palm = smooth([(-12, 0), (12, 0), (16, 20), (22, 34), (18, 46), (4, 40), (-10, 30), (-14, 14)])
            cyl(c, palm, col, depth=0.7)
            stroke(c, smooth([(16, 34), (12, 40), (2, 36)], closed=False), dk, 1.2, 150)
            thumb = smooth([(10, 6), (20, 12), (22, 22), (14, 22)])
            cyl(c, thumb, col, depth=0.6)
        else:  # open, relaxed: palm, fingers together, thumb forward
            hand = smooth([(-12, 0), (11, 0), (14, 18), (13, 36), (9, 54), (2, 58), (-5, 54), (-10, 36), (-14, 16)])
            cyl(c, hand, col, depth=0.7)
            for k in range(3):
                stroke(c, smooth([(-4 + k * 5, 34), (-3 + k * 5, 52)], closed=False), dk, 0.9, 110)
            thumb = smooth([(9, 6), (19, 14), (21, 28), (15, 30), (11, 20)])
            cyl(c, thumb, shade(col, 0.05), depth=0.6)
        if glove:
            pass
        if prop == "staff":
            prop_staff(c)
        c.restore()
    return draw


def prop_staff(c, back=False):
    """Wise elder's walking staff, held in the fist: carved spiral, gold ferrule, a sapphire set in a gold cradle."""
    if back:
        return
    shaft = capsule(2, -150, 2, 330, 7, 6)
    c.drawPath(shaft, paint(shader=lin((-6, 0), (10, 0), [(0, "#3a2212"), (0.35, "#a8743e"), (0.6, "#7a4a22"), (1, "#2a160a")])))
    for k in range(14):
        y = -140 + k * 32
        stroke(c, smooth([(-5, y), (2, y + 8), (9, y + 14)], closed=False), "#3a2212", 1.4, 160)
    knob = smooth([(-12, -150), (-14, -176), (-4, -196), (12, -192), (16, -170), (10, -150)])
    c.drawPath(knob, paint(shader=lin((-14, 0), (16, 0), [(0, "#3a2212"), (0.4, "#b07a44"), (1, "#2a160a")])))
    gold_line(c, smooth([(-12, -152), (2, -148), (12, -152)], closed=False), 2.4)
    gold_line(c, smooth([(-10, -192), (2, -200), (14, -190)], closed=False), 2.0, beads=5)
    gem(c, 2, -176, 7, JEWEL["sapphire_lt"])
    gold_fill(c, rect(-5, 316, 9, 334))


def fist_over_staff(skin, far=False):
    """The hand wraps the staff: staff behind, then fingers in front."""
    def draw(c):
        prop_staff(c)
        hand_draw(skin, "fist", far)(c)
    return draw


# ---------------------------------------------------------------- feet
def shoe(kind, color, far=False, sole="#2a1a12", accent=None):
    col = shade(color, -0.22) if far else color

    def draw(c):
        if kind == "sneaker":
            p = smooth([(-18, -10), (8, -16), (34, -8), (54, 2), (54, 12), (-20, 13), (-22, 0)])
            c.drawPath(p, paint(shader=lin((0, -16), (0, 13), [(0, shade(col, 0.2)), (0.7, col), (1, shade(col, -0.25))])))
            flat(c, poly([(-22, 7), (55, 7), (55, 14), (-22, 14)]), "#e9e2d6")
            if accent:
                stroke(c, smooth([(-6, -2), (20, -6), (40, 2)], closed=False), accent, 4)
            for k in range(3):
                stroke(c, poly([(6 + k * 7, -12), (12 + k * 7, -9)], closed=False), shade(col, -0.4), 1.4)
        elif kind == "rubber":  # tall rubber boot foot (the shaft is on the shin)
            p = smooth([(-20, -26), (16, -26), (24, -10), (50, -2), (54, 12), (-22, 13)])
            c.drawPath(p, paint(shader=lin((-20, 0), (54, 0), [(0, shade(col, -0.3)), (0.35, shade(col, 0.25)), (1, shade(col, -0.2))])))
            sheen(c, p, -8, -20, 34, -4, 3, 120)
            flat(c, poly([(-22, 8), (55, 8), (55, 14), (-22, 14)]), sole)
        else:  # leather boot / shoe with laces
            p = smooth([(-18, -24), (12, -24), (18, -8), (44, -2), (52, 8), (50, 13), (-20, 13)])
            c.drawPath(p, paint(shader=lin((0, -24), (0, 13), [(0, shade(col, 0.25)), (0.6, col), (1, shade(col, -0.3))])))
            sheen(c, p, 12, -6, 44, 2, 3, 100)
            flat(c, poly([(-21, 9), (52, 9), (51, 14), (-21, 14)]), sole)
            for k in range(3):
                stroke(c, poly([(2 + k * 5, -18 + k * 5), (10 + k * 5, -16 + k * 5)], closed=False), shade(col, 0.45), 1.3, 200)
    return draw


# ---------------------------------------------------------------- torso pieces
def torso_paths(d, female=True, belly=0.0):
    w = d.w
    ch = d.chest
    if female:
        chest = smooth([(-42 * w, 30), (-46 * w, -40), (-45 * w, -100), (-36 * w, -ch + 12), (-14, -ch - 2), (12, -ch - 4), (28 * w, -ch + 4),
                        (42 * w, -ch + 30), (55 * w, -ch + 58), (54 * w, -ch + 80), (44 * w, -ch + 100), (44 * w, -20), (42 * w, 30)])
    else:
        chest = smooth([(-44 * w, 30), (-48 * w, -40), (-48 * w, -100), (-38 * w, -ch + 12), (-14, -ch - 2), (12, -ch - 4), (30 * w, -ch + 4),
                        (46 * w, -ch + 26), (52 * w, -ch + 60), (50 * w, -ch + 100), (48 * w + belly, -30), (46 * w, 30)])
    waist = smooth([(-50 * w, 34), (-46 * w, -60), (-44 * w, -d.spine - 30), (44 * w, -d.spine - 30), (44 * w + belly, -70), (52 * w, 34)])
    hips = smooth([(-54 * w, -36), (54 * w, -36), (58 * w, 10), (40 * w, 44), (-40 * w, 44), (-58 * w, 10)])
    return chest, waist, hips


def skirt_panels(K, d, g, hem, z_back=9.5, z_front=21, length=0.85):
    """A long skirt/robe as rigid cloth panels: a hip bell on the pelvis, one panel per thigh, one hem panel per shin."""
    w = d.w
    hip = smooth([(-48 * w, -70), (48 * w, -70), (54 * w, -10), (62 * w, 70), (-62 * w, 70), (-54 * w, -10)])

    def hip_draw(c):
        garment(c, hip, g, axis="x")
    K.part("pelvis", 20, hip_draw, (-100, -90, 100, 90), name="skirt_hip")
    # inner drape: hidden inside the panels when standing, it fills the space under the lap when kneeling/sitting
    DL = d.pelvis_h * 0.62
    drape = smooth([(-56, -40), (56, -40), (72, DL * 0.5), (84, DL), (40, DL + 8), (0, DL - 4), (-40, DL + 8), (-84, DL), (-72, DL * 0.5)])

    def drape_draw(c):
        garment(c, drape, g, far=True)
        with clip(c, drape):
            for k in range(-3, 4):
                c.drawPath(capsule(k * 24, 0, k * 28, DL, 6), paint("#2a1020", 50, blur=7))
        if hem:
            band = inter(drape, rect(-100, DL - hem["h"] * 0.8, 100, DL + 20))
            garment(c, band, hem, far=True)
    K.part("pelvis", 12.5, drape_draw, (-110, -60, 110, DL + 40), name="skirt_drape")
    for s, z in (("f", z_back), ("n", z_front)):
        far = s == "f"
        bk = 0 if far else 14  # the near panel overlaps the far one so a standing figure reads as one skirt
        th = smooth([(-54 - bk, -50), (56 + bk, -50), (60 + bk, d.thigh * 0.5), (62 + bk, d.thigh + 26), (0, d.thigh + 40),
                     (-60 - bk, d.thigh + 26), (-58 - bk, d.thigh * 0.5)])

        def th_draw(c, th=th, far=far):
            garment(c, th, g, far=far)
            with clip(c, th):  # soft vertical folds
                for k in range(-3, 4):
                    c.drawPath(capsule(k * 22, -20, k * 26, d.thigh + 30, 5), paint("#2a1020", 40, blur=6))
            ao(c, th, 0, -40, 50, 60)
        K.part(f"thigh_{s}", z, th_draw, (-120, -70, 120, d.thigh + 60), name=f"skirt_{s}")
        L = d.shin * length
        sh = smooth([(-68 - bk, -40), (72 + bk, -40), (84 + bk, L * 0.5), (96 + bk, L), (0, L + 6), (-92 - bk, L), (-80 - bk, L * 0.5)])

        def sh_draw(c, sh=sh, far=far, L=L):
            garment(c, sh, g, far=far)
            with clip(c, sh):
                for k in range(-3, 4):
                    c.drawPath(capsule(k * 26, -30, k * 30, L, 5), paint("#2a1020", 40, blur=6))
            if hem:
                band = inter(sh, rect(-130, L - hem["h"], 130, L + 12))
                garment(c, band, hem, far=far)
                c.save()
                c.clipPath(sh, skia.ClipOp.kIntersect, True)
                gold_line(c, poly([(-110, L - hem["h"]), (112, L - hem["h"])], closed=False), 2.2, beads=11)
                gold_line(c, poly([(-110, L - 3), (112, L - 3)], closed=False), 1.6)
                if hem.get("vine"):
                    vine_border(c, poly([(-96, L - hem["h"] * 0.5), (100, L - hem["h"] * 0.5)], closed=False), hem["vine"][0], hem["vine"][1], step=24, s=1.3)
                c.restore()
                shade_over(c, band, strength=0.5)
        K.part(f"shin_{s}", z - 0.5, sh_draw, (-130, -70, 130, L + 30), name=f"hem_{s}")


def scarf_parts(K, g, fringe=True, z=35, wrap_draw=None):
    for i, (L, r0, r1) in enumerate(((70, 18, 17), (64, 17, 16), (58, 16, 15))):
        def draw(c, L=L, r0=r0, r1=r1, last=(i == 2)):
            p = capsule(0, 0, 0, L, r0, r1)
            garment(c, p, g)
            if g.get("stripe"):
                with clip(c, p):
                    for k in range(2):
                        c.drawRect(skia.Rect(-30, L * (0.3 + 0.35 * k), 30, L * (0.3 + 0.35 * k) + 6), paint(g["stripe"], 230))
                shade_over(c, p, strength=0.4)
            if last and fringe:
                for k in range(6):
                    x = -r1 + 2 + k * (2 * r1 - 4) / 5
                    stroke(c, poly([(x, L), (x + 1, L + 22)], closed=False), g.get("fringe", "#e2ad3a"), 2.4)
                    bead(c, x + 1, L + 24, 2.6)
        K.part(f"scarf{i + 1}", z, draw, (-40, -30, 40, L + 40))
    if wrap_draw:
        K.part("chest", z + 1, wrap_draw, (-80, -200, 90, 0), name="scarf_wrap")


# ---------------------------------------------------------------- characters
def nell():
    d = D(w=1.02)
    K = Rigkit("nell", pelvis_h=d.pelvis_h)
    human_bones(K, d, chains=("scarf",))
    skin = "#f0c4a4"
    card = {"color": "#17694f", "pattern": ("knit", 9, 5, 115)}
    blouse = {"color": "#fbf1df", "pattern": ("sprigs", "#d23a4e", "#2f8a5a", 22, 0.55)}
    skirt = {"color": "#7e1f33", "pattern": ("dots", "#a8354a", 16, 1.5, 160)}
    hem = {"color": "#1f3f9a", "h": 46, "vine": ("#e2ad3a", "#d23a4e")}
    shawl = {"color": "#24479c", "pattern": ("quilt", "#e2ad3a", 18, 120), "fringe": "#e2ad3a"}
    spec = dict(skin=skin, hair="#e6e3ec", hair_style="bun", glasses=True, age_lines=1.0, nose="hooked", lashes=True,
                iris="#4f6f96", brow="#b9b8c6", smile=0.65, earring=JEWEL["emerald_lt"])
    chest, waist, hips = torso_paths(d, female=True)

    def chest_draw(c):
        garment(c, chest, card)
        # cream blouse V + lace collar, button band with gold buttons, cameo brooch
        v = smooth([(10, -d.chest - 4), (32, -d.chest + 2), (26, -d.chest + 60)])
        garment(c, v, blouse, axis="x")
        for k in range(7):
            c.drawCircle(14 + k * 3.2, -d.chest - 2 + k * 1.2, 3.2, paint("#fffaf0"))
        band = smooth([(28, -d.chest + 2), (44, -d.chest + 30), (46, 30), (36, 30), (34, -d.chest + 40), (22, -d.chest + 10)])
        garment(c, band, {"color": "#13574a"})
        for k in range(5):
            button(c, 39, -d.chest + 56 + k * 34, 5.4)
        ao(c, chest, -10, -d.chest + 30, 40, 50)
    K.part("chest", 31, chest_draw, (-80, -200, 90, 60))

    def waist_draw(c):
        garment(c, waist, card)
        band = smooth([(34, -d.spine - 30), (46, -d.spine - 30), (50, 34), (38, 34)])
        garment(c, band, {"color": "#13574a"})
        pocket = smooth([(-4, -60), (34, -60), (36, -18), (-2, -18)])
        garment(c, pocket, {"color": "#13574a", "pattern": ("knit", 7, 3, 100)})
        stitch_line(c, poly([(-4, -58), (34, -58)], closed=False), "#e2ad3a", 1.4)
    K.part("spine", 30, waist_draw, (-80, -240, 90, 60))

    def neck_draw(c):
        p = capsule(0, 6, 0, -d.neck - 10, 15, 14)
        cyl(c, p, skin, depth=0.8)
        ao(c, p, 6, 6, 22, 80)
    K.part("neck", 32, neck_draw, (-30, -70, 30, 30))
    K.part("head", 34, head_part(spec), (-100, -200, 110, 60))
    K.variant("head", "blink", head_part(spec, blink=True), (-100, -200, 110, 60))
    for s, zu in (("f", 12), ("n", 40)):
        far = s == "f"
        K.part(f"uarm_{s}", zu, tube_part(d.uarm, 26, 22, card, far=far), (-40, -40, 40, d.uarm + 40))
        K.part(f"farm_{s}", zu + 1, tube_part(d.farm, 22, 18, card, far=far, cuff={"color": "#13574a", "h": 22, "gold": True}), (-40, -40, 40, d.farm + 40))
        K.part(f"hand_{s}", zu + 2, hand_draw(skin, "open", far), (-40, -20, 50, 80))
        for v in ("point", "fist", "cup"):
            K.variant(f"hand_{s}", v, hand_draw(skin, v, far), (-40, -20, 50, 90))
    for s, zl in (("f", 10), ("n", 16)):
        far = s == "f"
        K.part(f"thigh_{s}", zl, skin_tube(d.thigh, 28, 22, "#5a3a2e", far), (-40, -40, 40, d.thigh + 40))
        K.part(f"shin_{s}", zl + 1, skin_tube(d.shin, 20, 15, "#5a3a2e", far), (-30, -30, 30, d.shin + 30))
        K.part(f"foot_{s}", zl + 2, shoe("boot", "#6a3a1e", far), (-30, -40, 64, 24))
    skirt_panels(K, d, skirt, hem)

    def wrap_draw(c):  # shawl wrapped over the shoulders, tied at the front
        p = smooth([(-44, -d.chest + 4), (-20, -d.chest - 12), (14, -d.chest - 8), (40, -d.chest + 14), (44, -d.chest + 40),
                    (30, -d.chest + 34), (6, -d.chest + 14), (-30, -d.chest + 30), (-48, -d.chest + 26)])
        garment(c, p, shawl)
        gold_line(c, smooth([(-48, -d.chest + 26), (-30, -d.chest + 30), (6, -d.chest + 14), (30, -d.chest + 34), (44, -d.chest + 40)], closed=False), 2.0, beads=10)
        knot = ellipse(32, -d.chest + 26, 12, 10, 20)
        garment(c, knot, shawl)
        gem(c, 32, -d.chest + 26, 4.5, JEWEL["ruby_lt"])
    scarf_parts(K, shawl, wrap_draw=wrap_draw)
    K.springs = [{"bones": ["scarf1", "scarf2", "scarf3"], "freq": 0.75, "zeta": 0.2, "wind": 7, "drag": 0.05}]
    return K


def dee():
    d = D()
    K = Rigkit("dee", pelvis_h=d.pelvis_h)
    human_bones(K, d, chains=("braid",))
    skin = "#e3b08a"
    shirt = {"color": "#a3203a", "pattern": ("tartan", [(0, 10, "#2a1830"), (18, 4, "#2a1830"), (30, 3, "#e2ad3a"), (36, 12, "#5a1020")], 44, 120)}
    vest = {"color": "#2c4a6e", "pattern": ("quilt", "#8fb0d8", 16, 110, "#e2ad3a")}
    jeans = {"color": "#2f4f86", "pattern": ("weave", "#9ab4e0", 5, 55)}
    spec = dict(skin=skin, hair="#6b4226", hair_style="braid", age_lines=0.5, nose="soft", lashes=True, iris="#4a6a3a", smile=0.45,
                earring=JEWEL["sapphire_lt"])
    chest, waist, hips = torso_paths(d, female=True)

    def chest_draw(c):
        garment(c, chest, shirt)
        collar = smooth([(4, -d.chest - 6), (30, -d.chest), (40, -d.chest + 26), (20, -d.chest + 12)])
        garment(c, collar, shirt)
        vest_p = smooth([(-44, 30), (-46, -60), (-42, -d.chest + 20), (-24, -d.chest + 4), (0, -d.chest + 10), (20, -d.chest + 70),
                         (30, 30)])
        garment(c, vest_p, vest)
        stroke(c, smooth([(0, -d.chest + 10), (20, -d.chest + 70), (30, 30)], closed=False), "#e2ad3a", 2.4)
        rosette(c, -18, -d.chest + 64, 9, JEWEL["ruby_lt"])  # embroidered folk patch on the vest
        ao(c, chest, -10, -d.chest + 30, 40, 50)
    K.part("chest", 31, chest_draw, (-80, -200, 90, 60))

    def waist_draw(c):
        garment(c, waist, shirt)
        vp = smooth([(-46, 34), (-46, -60), (-44, -d.spine - 30), (30, -d.spine - 30), (32, -40), (34, 20)])
        garment(c, vp, vest)
        stroke(c, poly([(30, -d.spine - 30), (32, -40), (34, 20)], closed=False), "#e2ad3a", 2.4)
        belt = rect(-52, 14, 54, 34)
        garment(c, belt, {"color": "#5a3418"})
        gold_fill(c, rect(34, 12, 50, 36))
    K.part("spine", 30, waist_draw, (-80, -240, 90, 60))

    def hips_draw(c):
        garment(c, hips, jeans)
        stitch_line(c, smooth([(30, -30), (40, 10), (36, 40)], closed=False), "#e2ad3a", 1.3)
    K.part("pelvis", 29, hips_draw, (-80, -60, 80, 70))

    def neck_draw(c):
        p = capsule(0, 6, 0, -d.neck - 10, 15, 14)
        cyl(c, p, skin, depth=0.8)
    K.part("neck", 32, neck_draw, (-30, -70, 30, 30))
    K.part("head", 34, head_part(spec), (-100, -200, 110, 60))
    K.variant("head", "blink", head_part(spec, blink=True), (-100, -200, 110, 60))
    for s, zu in (("f", 12), ("n", 40)):
        far = s == "f"
        K.part(f"uarm_{s}", zu, tube_part(d.uarm, 25, 21, shirt, far=far), (-40, -40, 40, d.uarm + 40))
        K.part(f"farm_{s}", zu + 1, tube_part(d.farm, 21, 17, shirt, far=far, cuff={"color": "#7a1428", "h": 16}), (-40, -40, 40, d.farm + 40))
        K.part(f"hand_{s}", zu + 2, hand_draw(skin, "open", far), (-40, -20, 50, 80))
        for v in ("point", "fist", "cup"):
            K.variant(f"hand_{s}", v, hand_draw(skin, v, far), (-40, -20, 50, 90))
    for s, zl in (("f", 10), ("n", 16)):
        far = s == "f"
        K.part(f"thigh_{s}", zl, tube_part(d.thigh, 31, 25, jeans, far=far), (-45, -45, 45, d.thigh + 40))
        boot = {"color": "#1f7a4c"}

        def shin_draw(c, far=far):
            p = capsule(0, 0, 0, d.shin, 23, 19)
            garment(c, p, jeans, far=far)
            b = smooth([(-24, d.shin * 0.42), (24, d.shin * 0.42), (22, d.shin + 4), (-22, d.shin + 4)])
            col = shade(boot["color"], -0.22) if far else boot["color"]
            c.drawPath(b, paint(shader=lin((-24, 0), (24, 0), [(0, shade(col, -0.3)), (0.35, shade(col, 0.3)), (1, shade(col, -0.25))])))
            sheen(c, b, -8, d.shin * 0.48, -6, d.shin * 0.9, 3, 130)
            stroke(c, poly([(-24, d.shin * 0.42 + 4), (24, d.shin * 0.42 + 4)], closed=False), shade(col, -0.35), 3)
        K.part(f"shin_{s}", zl + 1, shin_draw, (-35, -35, 35, d.shin + 30))
        K.part(f"foot_{s}", zl + 2, shoe("rubber", "#1f7a4c", far), (-30, -40, 64, 24))
    braid_col = "#6b4226"
    for i, L in enumerate((64, 60)):
        def bdraw(c, L=L, last=(i == 1)):
            for k in range(int(L / 12) + 1):
                y = k * 12
                e = ellipse(-2 if k % 2 else 2, y + 6, 9 - k * 0.15 - (2 if last else 0) * k / 5, 8)
                c.drawPath(e, paint(shader=rad((-3, y + 2), 12, [(0, shade(braid_col, 0.35)), (1, shade(braid_col, -0.35))])))
                stroke(c, smooth([(-6, y + 2), (0, y + 8), (6, y + 10)], closed=False), shade(braid_col, -0.45), 1.0, 160)
            if last:
                c.drawPath(ellipse(0, L + 4, 7, 5), paint(JEWEL["ruby"]))
                c.drawPath(smooth([(-6, L + 8), (6, L + 8), (4, L + 26), (0, L + 30), (-4, L + 26)]), paint(shader=lin((0, L), (0, L + 30), [(0, braid_col), (1, shade(braid_col, -0.3))])))
        K.part(f"hair{i + 1}", 33, bdraw, (-20, -20, 20, L + 40))
    K.springs = [{"bones": ["hair1", "hair2"], "freq": 1.1, "zeta": 0.25, "wind": 5, "drag": 0.03}]
    return K


def eli():
    d = D(thigh=205, shin=196, spine=160, chest=140, uarm=150, farm=132, w=0.92)
    K = Rigkit("eli", pelvis_h=d.pelvis_h)
    human_bones(K, d)
    skin = "#f3cba9"
    jacket = {"color": "#c22a2e", "pattern": ("dots", "#e2533e", 14, 1.2, 90)}
    jeans = {"color": "#2f4f86", "pattern": ("weave", "#9ab4e0", 5, 55)}
    spec = dict(skin=skin, hair="#d9b36a", hair_style="messy", child=True, freckles=True, eye_s=1.15, iris="#3a6aa0", smile=1.0,
                head_scale=1.22, blush_a=80)
    chest, waist, hips = torso_paths(d, female=False)

    def chest_draw(c):
        garment(c, chest, jacket)
        # Fair-Isle yoke across the shoulders (a hand-knit sweater shows at the collar), zip with gold puller
        pat_fairisle(c, chest, -d.chest + 4, "#f4efe6", [JEWEL["sapphire"], JEWEL["ruby"], JEWEL["emerald"]], h=28)
        shade_over(c, inter(chest, rect(-80, -d.chest, 80, -d.chest + 34)), strength=0.5)
        stroke(c, poly([(38, -d.chest + 30), (42, 30)], closed=False), "#e2ad3a", 2.6)
        gold_fill(c, rect(36, -d.chest + 34, 46, -d.chest + 50))
        stroke(c, poly([(-46, -d.chest + 60), (-20, -d.chest + 20)], closed=False), "#f4efe6", 6, 220)
    K.part("chest", 31, chest_draw, (-80, -200, 90, 60))

    def waist_draw(c):
        garment(c, waist, jacket)
        hem = rect(-52, 6, 54, 30)
        garment(c, inter(waist, hem), {"color": "#8a1a22", "pattern": ("weave", "#000000", 4, 40)})
        stroke(c, poly([(42, -d.spine - 30), (44, 30)], closed=False), "#e2ad3a", 2.6)
    K.part("spine", 30, waist_draw, (-80, -240, 90, 60))

    def hips_draw(c):
        garment(c, hips, jeans)
    K.part("pelvis", 29, hips_draw, (-80, -60, 80, 70))

    def neck_draw(c):
        p = capsule(0, 6, 0, -d.neck - 10, 14, 13)
        cyl(c, p, skin, depth=0.8)
    K.part("neck", 32, neck_draw, (-30, -70, 30, 30))
    K.part("head", 34, head_part(spec), (-130, -240, 130, 60))
    K.variant("head", "blink", head_part(spec, blink=True), (-130, -240, 130, 60))
    for s, zu in (("f", 12), ("n", 40)):
        far = s == "f"

        def sleeve_stripes(c, p, far=far):
            with clip(c, p):
                for k in range(2):
                    c.drawRect(skia.Rect(-14 + k * 10, -10, -10 + k * 10, 200), paint("#f4efe6", 200 if not far else 150))
        K.part(f"uarm_{s}", zu, tube_part(d.uarm, 23, 20, jacket, far=far, trim=sleeve_stripes), (-40, -40, 40, d.uarm + 40))
        K.part(f"farm_{s}", zu + 1, tube_part(d.farm, 20, 16, jacket, far=far, cuff={"color": "#8a1a22", "h": 16}, trim=sleeve_stripes), (-40, -40, 40, d.farm + 40))
        K.part(f"hand_{s}", zu + 2, hand_draw(skin, "open", far, s=0.88), (-40, -20, 50, 80))
        for v in ("point", "fist", "cup"):
            K.variant(f"hand_{s}", v, hand_draw(skin, v, far, s=0.88), (-40, -20, 50, 90))
    for s, zl in (("f", 10), ("n", 16)):
        far = s == "f"
        K.part(f"thigh_{s}", zl, tube_part(d.thigh, 28, 23, jeans, far=far), (-45, -45, 45, d.thigh + 40))
        K.part(f"shin_{s}", zl + 1, tube_part(d.shin, 22, 19, jeans, far=far, cuff={"color": "#3f62a0", "h": 14}), (-35, -35, 35, d.shin + 30))
        K.part(f"foot_{s}", zl + 2, shoe("sneaker", "#f2eee6", far, accent=JEWEL["sapphire_lt"]), (-30, -30, 64, 24))
    return K


def king():
    d = D(w=1.05)
    K = Rigkit("king", pelvis_h=d.pelvis_h)
    human_bones(K, d, chains=("cape",))
    skin = "#e8b896"
    tunic = {"color": "#1f3f9a", "pattern": ("damask", "#e2ad3a", 34, 0.9)}
    hose = {"color": "#6a1a2e", "pattern": ("dots", "#8a2a40", 12, 1.2, 140)}
    cape = {"color": "#8e1630", "pattern": ("dots", "#b8304a", 18, 1.4, 120)}
    spec = dict(skin=skin, hair="#8a4a2a", hair_style="regal", nose="strong", jaw=3, chin=2, iris="#3a5a8a", smile=0.2,
                beard=dict(color="#9a5a36", length=40), crown=True, brow_w=3.6, head_scale=0.96, blush_a=35)
    chest, waist, hips = torso_paths(d, female=False, belly=4)

    def ermine(c, path):
        c.drawPath(path, paint("#fbf7ef"))
        b = bounds(path)
        with clip(c, path):
            y, row = b.top(), 0
            while y < b.bottom() + 10:
                x = b.left() + (9 if row % 2 else 0)
                while x < b.right() + 10:
                    c.drawPath(smooth([(x, y - 4), (x + 2, y + 2), (x, y + 5), (x - 2, y + 2)]), paint("#1a1414"))
                    x += 18
                y += 16
                row += 1
        shade_over(c, path, strength=0.4)

    def chest_draw(c):
        garment(c, chest, tunic)
        # gold chain of office with a ruby medallion
        chain = smooth([(-30, -d.chest + 4), (-6, -d.chest + 50), (24, -d.chest + 74), (44, -d.chest + 40)], closed=False)
        gold_line(c, chain, 3.0, beads=7, bead_r=3.4)
        gold_fill(c, circle(20, -d.chest + 80, 13))
        gem(c, 20, -d.chest + 80, 7, JEWEL["ruby_lt"])
        ao(c, chest, -10, -d.chest + 30, 40, 50)
    K.part("chest", 31, chest_draw, (-80, -200, 90, 60))

    def collar_draw(c):  # ermine collar over the cape's top, in front of the neck
        p = smooth([(-48, -d.chest + 30), (-40, -d.chest - 6), (-10, -d.chest - 14), (20, -d.chest - 12), (40, -d.chest + 2),
                    (36, -d.chest + 20), (10, -d.chest + 8), (-20, -d.chest + 14)])
        ermine(c, p)
        gold_line(c, smooth([(-48, -d.chest + 30), (-20, -d.chest + 14), (10, -d.chest + 8), (36, -d.chest + 20)], closed=False), 2.0)
    K.part("chest", 33, collar_draw, (-80, -200, 90, 0), name="collar")

    def waist_draw(c):
        garment(c, waist, tunic)
        belt = rect(-54, -40, 56, -14)
        gold_fill(c, belt)
        for k in range(5):
            gem(c, -40 + k * 22, -27, 4.6, [JEWEL["emerald_lt"], JEWEL["ruby_lt"], JEWEL["sapphire_lt"]][k % 3])
        skirt = smooth([(-54, -14), (58, -14), (68, 34), (-64, 34)])
        garment(c, skirt, tunic)
        gold_line(c, poly([(-64, 32), (68, 32)], closed=False), 3.0, beads=9)
    K.part("spine", 30, waist_draw, (-90, -240, 100, 60))

    def hips_draw(c):  # tunic skirt continues over the hips
        p = smooth([(-62, -40), (64, -40), (74, 70), (-72, 70)])
        garment(c, p, tunic)
        gold_line(c, poly([(-72, 64), (74, 64)], closed=False), 3.2, beads=9)
        gold_line(c, poly([(-71, 52), (73, 52)], closed=False), 1.4)
    K.part("pelvis", 29, hips_draw, (-100, -60, 100, 100))

    def neck_draw(c):
        p = capsule(0, 6, 0, -d.neck - 10, 17, 16)
        cyl(c, p, skin, depth=0.8)
    K.part("neck", 32, neck_draw, (-30, -70, 30, 30))
    K.part("head", 34, head_part(spec), (-110, -230, 110, 110))
    K.variant("head", "blink", head_part(spec, blink=True), (-110, -230, 110, 110))
    for s, zu in (("f", 12), ("n", 40)):
        far = s == "f"
        K.part(f"uarm_{s}", zu, tube_part(d.uarm, 27, 23, tunic, far=far), (-40, -40, 40, d.uarm + 40))
        K.part(f"farm_{s}", zu + 1, tube_part(d.farm, 23, 19, tunic, far=far, cuff={"color": "#e2ad3a", "h": 18, "gold": True, "beads": True}), (-40, -40, 40, d.farm + 40))
        K.part(f"hand_{s}", zu + 2, hand_draw(skin, "open", far), (-40, -20, 50, 80))
        for v in ("point", "fist", "cup"):
            K.variant(f"hand_{s}", v, hand_draw(skin, v, far), (-40, -20, 50, 90))
    for s, zl in (("f", 10), ("n", 16)):
        far = s == "f"
        K.part(f"thigh_{s}", zl, tube_part(d.thigh, 30, 24, hose, far=far), (-45, -45, 45, d.thigh + 40))

        def shin_draw(c, far=far):
            p = capsule(0, 0, 0, d.shin, 22, 17)
            garment(c, p, hose, far=far)
            b = smooth([(-25, d.shin * 0.35), (25, d.shin * 0.35), (22, d.shin + 4), (-22, d.shin + 4)])
            col = "#5a3418" if not far else shade("#5a3418", -0.2)
            c.drawPath(b, paint(shader=lin((-25, 0), (25, 0), [(0, shade(col, -0.3)), (0.35, shade(col, 0.35)), (1, shade(col, -0.25))])))
            gold_line(c, poly([(-25, d.shin * 0.35 + 3), (25, d.shin * 0.35 + 3)], closed=False), 2.4, beads=8)
        K.part(f"shin_{s}", zl + 1, shin_draw, (-35, -35, 35, d.shin + 30))
        K.part(f"foot_{s}", zl + 2, shoe("boot", "#5a3418", far), (-30, -40, 64, 24))
    # the cape: three hanging panels behind everything (spring chain)
    for i, (L, w0, w1) in enumerate(((150, 52, 62), (150, 62, 70), (140, 70, 74))):
        def cdraw(c, L=L, w0=w0, w1=w1, i=i):
            p = smooth([(-w0, -14), (w0, -14), (w1, L + 18), (0, L + 24), (-w1, L + 18)])
            garment(c, p, cape, axis="x")
            if i == 2:
                band = inter(p, rect(-90, L - 18, 90, L + 40))
                ermine(c, band)
                gold_line(c, poly([(-w1, L - 18), (w1, L - 18)], closed=False), 2.4)
            for sgn in (-1, 1):  # ermine edge down the front opening
                pass
        K.part(f"cape{i + 1}", 5 - i * 0.1, cdraw, (-100, -40, 100, L + 50))
    K.springs = [{"bones": ["cape1", "cape2", "cape3"], "freq": 0.6, "zeta": 0.3, "wind": 4, "drag": 0.06}]
    return K


def elder():
    d = D(w=0.98)
    K = Rigkit("elder", pelvis_h=d.pelvis_h)
    human_bones(K, d)
    skin = "#d9a888"
    robe = {"color": "#1d5a40", "pattern": ("sprigs", "#e2ad3a", "#7ac08a", 30, 0.7)}
    hem = {"color": "#5a2350", "h": 40, "vine": ("#e2ad3a", "#e9a53a")}
    spec = dict(skin=skin, hair="#f2efe8", hair_style="long_white", nose="long", age_lines=1.0, iris="#6a5030", smile=0.55,
                beard=dict(color="#f4f1ea", length=110), hood=dict(color="#16492f"), brow="#e8e4dc", brow_w=4.2, head_scale=0.97, blush_a=40)
    chest, waist, hips = torso_paths(d, female=False)

    def chest_draw(c):
        garment(c, chest, robe)
        # embroidered front border down the robe's opening + satchel strap
        edge = smooth([(14, -d.chest - 2), (34, -d.chest + 40), (40, 30)], closed=False)
        stroke(c, edge, "#5a2350", 14)
        vine_border(c, edge, "#e2ad3a", "#e9a53a", step=20, s=0.9)
        strap = smooth([(-36, -d.chest + 6), (0, -d.chest + 60), (40, 10)], closed=False)
        stroke(c, strap, "#6a3a1e", 9)
        stitch_line(c, strap, "#e2ad3a", 1.2)
    K.part("chest", 31, chest_draw, (-80, -200, 90, 60))

    def waist_draw(c):
        garment(c, waist, robe)
        edge = poly([(40, -d.spine - 30), (44, 34)], closed=False)
        stroke(c, edge, "#5a2350", 14)
        vine_border(c, edge, "#e2ad3a", "#e9a53a", step=20, s=0.9)
        rope = smooth([(-50, -30), (50, -34)], closed=False)
        stroke(c, rope, "#c8a060", 8)
        for k in range(8):
            stroke(c, poly([(-46 + k * 12, -36), (-40 + k * 12, -26)], closed=False), "#7a5a30", 1.6)
        tassel = smooth([(30, -30), (38, -30), (40, 30), (28, 30)])
        c.drawPath(tassel, paint(shader=lin((28, 0), (40, 0), [(0, "#a07838"), (0.5, "#e8c880"), (1, "#806028")])))
        gem(c, 34, -24, 4, JEWEL["ruby_lt"])
    K.part("spine", 30, waist_draw, (-80, -240, 90, 60))

    def neck_draw(c):
        p = capsule(0, 6, 0, -d.neck - 10, 15, 14)
        cyl(c, p, skin, depth=0.8)
    K.part("neck", 32, neck_draw, (-30, -70, 30, 30))
    K.part("head", 34, head_part(spec), (-110, -200, 110, 150))
    K.variant("head", "blink", head_part(spec, blink=True), (-110, -200, 110, 150))
    for s, zu in (("f", 12), ("n", 40)):
        far = s == "f"
        # wide bell sleeves
        K.part(f"uarm_{s}", zu, tube_part(d.uarm, 27, 26, robe, far=far), (-40, -40, 40, d.uarm + 40))

        def farm_draw(c, far=far):
            p = smooth([(-24, -10), (24, -10), (38, d.farm - 6), (30, d.farm + 14), (-30, d.farm + 14), (-30, d.farm - 6)])
            garment(c, p, robe, far=far)
            band = inter(p, rect(-50, d.farm - 14, 50, d.farm + 20))
            garment(c, band, {"color": "#5a2350"}, far=far)
            gold_line(c, poly([(-32, d.farm - 14), (36, d.farm - 14)], closed=False), 2.0, beads=7)
        K.part(f"farm_{s}", zu + 1, farm_draw, (-50, -30, 50, d.farm + 40))
        if s == "n":
            K.part("hand_n", zu + 2, fist_over_staff(skin, far), (-40, -230, 50, 360))
            for v in ("open", "point", "cup"):
                K.variant("hand_n", v, hand_draw(skin, v, far), (-40, -20, 50, 90))
        else:
            K.part("hand_f", zu + 2, hand_draw(skin, "open", far), (-40, -20, 50, 80))
            for v in ("point", "fist", "cup"):
                K.variant("hand_f", v, hand_draw(skin, v, far), (-40, -20, 50, 90))
    for s, zl in (("f", 10), ("n", 16)):
        far = s == "f"
        K.part(f"thigh_{s}", zl, skin_tube(d.thigh, 28, 22, "#3a2a20", far), (-40, -40, 40, d.thigh + 40))
        K.part(f"shin_{s}", zl + 1, skin_tube(d.shin, 20, 15, "#3a2a20", far), (-30, -30, 30, d.shin + 30))
        K.part(f"foot_{s}", zl + 2, shoe("boot", "#4a2a14", far), (-30, -40, 64, 24))
    skirt_panels(K, d, robe, hem, length=0.9)
    K.meta["props"] = {"hand_n": "staff (built into hand_n; use the hand_n variants for an empty hand)"}
    return K


CAST = {"nell": nell, "dee": dee, "eli": eli, "king": king, "elder": elder}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CAST)
    for n in names:
        K = CAST[n]()
        J = K.build(os.path.join(ROOT, n, "rig"))
        print(n, len(J["parts"]), "parts", sum(len(v) for v in J["variants"].values()), "variants")
