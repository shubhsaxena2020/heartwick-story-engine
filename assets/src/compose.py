"""Render test frames through engine v2's own Scene renderer: a layered scene from assets/scenes + posed part-sheet
puppets from assets/characters. Used for the before/after comparisons in assets/README.md.
python assets/src/compose.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
import skia
from engine2.rig import load_partsheet
from engine2.style import VectorCel
from assets.scenes.load_scene import load_scene
from preview import root_for, pose

CH = os.path.join(REPO, "assets", "characters")
SC = os.path.join(REPO, "assets", "scenes")
OUT = os.path.join(REPO, "assets", "previews")


class Puppet:
    def __init__(self, name, pose_name, x, ground, h, facing=1, t2=None):
        self.rig = load_partsheet(os.path.join(CH, name, "rig"))
        self.sem, self.angs = pose(self.rig.kind, pose_name)
        self.x, self.ground, self.h, self.facing = x, ground, h, facing

    def draw(self, c, t):
        root = root_for(self.rig, self.angs, self.sem, self.x, self.ground, self.h, self.facing)
        sh = skia.Paint(AntiAlias=True, Color=skia.Color(20, 12, 10, 70))
        sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 10))
        w = self.h * (0.3 if self.rig.kind == "humanoid" else 0.5)
        c.drawOval(skia.Rect(self.x - w, self.ground - w * 0.12, self.x + w, self.ground + w * 0.12), sh)
        face = {k: v for k, v in self.sem.items() if k.startswith("hand_")}
        self.rig.draw(c, self.angs, root, VectorCel(), t, face)


def frame(scene_name, cast, cam, t=1.0, out=None, W=1080, H=1920):
    P = VectorCel()
    sc = load_scene(os.path.join(SC, "kitchen" if scene_name == "closeup" else scene_name), P)
    actors = []
    for (name, pn, fx, dy, h, facing) in cast:
        a = Puppet(name, pn, sc.world[0] * fx, sc.ground + dy, h, facing)
        actors.append(a)
    s = skia.Surface(W, H)
    c = s.getCanvas()
    sc.render(c, t, cam, actors, W, H)
    os.makedirs(OUT, exist_ok=True)
    out = out or os.path.join(OUT, f"{scene_name}.png")
    s.makeImageSnapshot().save(out, skia.kPNG)
    return out


SHOTS = {
    "maple_yard": ([("nell", "kneel_reach", 0.4, 0, 760, 1), ("pip_kit", "curl", 0.62, 10, 170, -1)], (700, 1380, 1.25)),
    "village_autumn": ([("eli", "point", 0.3, 0, 700, 1), ("dee", "talk", 0.55, 0, 860, -1)], (700, 1900, 1.0)),
    "kitchen": ([("nell", "hold", 0.42, 0, 900, 1), ("pip", "sit", 0.62, -340, 220, -1)], (700, 1250, 1.15)),
    "night_sky": ([("elder", "point_up", 0.4, 0, 900, 1), ("hare", "sit_look_up", 0.62, 20, 300, -1)], (700, 1700, 1.0)),
    "storm": ([("pip", "crouch", 0.5, 30, 260, 1)], (700, 1300, 1.0)),
    "closeup": ([("nell", "heart", 0.42, 0, 900, 1), ("pip", "sit_look_up", 0.6, -340, 220, -1)], (640, 1080, 2.2)),
    "castle_court": ([("king", "arms_open", 0.6, 0, 920, -1), ("elder", "bow", 0.36, 0, 900, 1)], (700, 1300, 1.0)),
}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(SHOTS):
        cast, cam = SHOTS[n]
        print(frame(n, cast, cam))
