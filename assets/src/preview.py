"""Load each part sheet with engine v2's own loader (engine2.rig.load_partsheet) and pose it with engine v2's pose
library, so the preview proves the assets rig correctly. Writes assets/characters/<name>/preview.png.
python assets/src/preview.py [name ...]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
import skia
from engine2.rig import load_partsheet, mcopy
from engine2 import poses as PO
from engine2.style import VectorCel

CHARS = os.path.join(REPO, "assets", "characters")
HUMAN_POSES = ["stand", "point", "heart", "arms_open", "kneel_reach", "sit_hold", "walk"]
SQ_POSES = ["sit", "sit_look_up", "nibble", "alert", "crouch", "look_back", "run"]


def root_for(rig, angs, sem, x, ground, h, facing=1):
    s = h / rig.units
    m = skia.Matrix()
    m.setTranslate(x, ground)
    m.preScale(facing * s, s)
    pel = rig.meta.get("pelvis_h", 470.0)
    auto = mcopy(m)
    auto.preTranslate(0, -pel)
    dy = rig.ground_offset(angs, auto, ground) / s
    y = -sem["root_y"] if "root_y" in sem else -pel + dy
    m.preTranslate(0, y)
    return m


def pose(kind, name):
    if kind == "humanoid":
        sem = PO.walk(0.25) if name == "walk" else dict(PO.HUMAN[name])
        return sem, PO.to_bones(sem)
    sem = PO.run_cycle(0.3) if name == "run" else dict(PO.SQUIRREL[name])
    a = PO.sq_to_bones(sem)
    a.setdefault("tail2", 20 if name != "run" else 6)
    a.setdefault("tail3", 24 if name != "run" else 4)
    return sem, a


def sheet(name, cell=(420, 760), h=620, bg="#efe6d6", out=None):
    folder = os.path.join(CHARS, name, "rig")
    rig = load_partsheet(folder)
    kind = rig.kind
    names = HUMAN_POSES if kind == "humanoid" else SQ_POSES
    W, H = cell[0] * len(names), cell[1]
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    c.clear(skia.Color(*[int(bg[i:i + 2], 16) for i in (1, 3, 5)]))
    P = VectorCel()
    for i, pn in enumerate(names):
        sem, angs = pose(kind, pn)
        ground = H - 60
        x = cell[0] * i + cell[0] * (0.42 if kind == "humanoid" else 0.5)
        hh = h if kind == "humanoid" else h * 0.42
        root = root_for(rig, angs, sem, x, ground, hh)
        sh = skia.Paint(AntiAlias=True, Color=skia.Color(40, 20, 10, 50))
        sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 8))
        c.drawOval(skia.Rect(x - 90, ground - 10, x + 90, ground + 10), sh)
        face = {k: v for k, v in sem.items() if k.startswith("hand_")}
        rig.draw(c, angs, root, P, 0.0, face)
        tp = skia.Paint(AntiAlias=True, Color=skia.Color(90, 60, 40))
        c.drawString(pn, cell[0] * i + 14, 30, skia.Font(None, 22), tp)
    out = out or os.path.join(CHARS, name, "preview.png")
    surf.makeImageSnapshot().save(out, skia.kPNG)
    return out


if __name__ == "__main__":
    for n in sys.argv[1:] or sorted(d for d in os.listdir(CHARS) if os.path.isdir(os.path.join(CHARS, d, "rig"))):
        print(sheet(n))
