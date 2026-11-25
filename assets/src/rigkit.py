"""Builds raster part sheets in engine v2's format (engine2/rig.py load_partsheet) from vector-drawn parts.

Every part is drawn in its bone's canonical frame (pivot at 0,0; limbs along +y, body/neck/head along -y), at
q px per rig unit, then cropped. Because every rest angle is 0, rig.json pivots are just the rest-pose positions,
so engine v2's pose library drives these puppets unchanged (same bone names and proportions as the vector cast).

Extra keys the current loader ignores but documents for the engine (assets/characters/README.md):
  "springs":  spring chains (same dicts as engine2/characters.py) for scarf / cape / braid / tail bones;
  "variants": alternative images per part (hand shapes open/point/fist/cup, head blink).
"""
import json, os
from paint import Surf


class Rigkit:
    def __init__(self, name, kind="humanoid", units=1000.0, q=2.0, pelvis_h=468.0):
        self.name, self.kind, self.units, self.q, self.pelvis_h = name, kind, units, q, pelvis_h
        self.bones = []  # (name, parent, pivot(units, parent frame), length)
        self.parts = []  # dict(bone, name, z, draw, box)
        self.variants = {}  # part name -> {variant: (draw, box)}
        self.contacts = []
        self.springs = []
        self.meta = {}

    def bone(self, name, parent, pivot=(0.0, 0.0), length=0.0):
        self.bones.append((name, parent, tuple(pivot), length))

    def part(self, bone, z, draw, box, name=None):
        """box: (x0, y0, x1, y1) in units around the pivot that safely contains the drawing."""
        self.parts.append(dict(bone=bone, name=name or bone, z=z, draw=draw, box=box))

    def variant(self, part_name, vname, draw, box):
        self.variants.setdefault(part_name, {})[vname] = (draw, box)

    def abs_pos(self):
        pos = {}
        for n, par, pv, L in self.bones:
            if par is None:
                pos[n] = (0.0, 0.0)
            else:
                px, py = pos[par]
                pos[n] = (px + pv[0], py + pv[1])
        return pos

    def _render(self, draw, box, path):
        x0, y0, x1, y1 = box
        s = Surf(x1 - x0, y1 - y0, -x0, -y0, self.q)
        draw(s.c)
        return s.crop_save(path)

    def build(self, out_dir, origin=(1000.0, 2400.0)):
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(out_dir, f))
        q = self.q
        pos = self.abs_pos()
        parts = []
        for p in self.parts:
            fn = f"{p['name']}.png"
            anchor, size = self._render(p["draw"], p["box"], os.path.join(out_dir, fn))
            parts.append({"bone": p["bone"], "name": p["name"], "image": fn, "anchor_px": [round(anchor[0], 2), round(anchor[1], 2)],
                          "z": p["z"], "size_px": list(size)})
        variants = {}
        for pn, vs in self.variants.items():
            bone = next(p["bone"] for p in self.parts if p["name"] == pn)
            for vn, (draw, box) in vs.items():
                fn = f"{pn}__{vn}.png"
                anchor, size = self._render(draw, box, os.path.join(out_dir, fn))
                variants.setdefault(pn, {})[vn] = {"bone": bone, "image": fn, "anchor_px": [round(anchor[0], 2), round(anchor[1], 2)]}
        lengths = {n: L for n, _, _, L in self.bones}
        bones = [{"name": n, "parent": par, "pivot_px": [round(origin[0] + pos[n][0] * q, 3), round(origin[1] + pos[n][1] * q, 3)],
                  "rest_deg": 0.0, "length": round(lengths[n] * q, 3)} for n, par, _, _ in self.bones]
        J = {"name": self.name, "kind": self.kind, "units": self.units, "units_per_px": 1.0 / q, "pelvis_h": self.pelvis_h,
             "source": "heartwick vector part library (assets/src), rendered at %.1f px per rig unit" % q,
             "bones": bones, "parts": sorted(parts, key=lambda p: p["z"]),
             "contacts": [[b, [round(x * q, 3), round(y * q, 3)]] for b, (x, y) in self.contacts],
             "springs": self.springs, "variants": variants}
        J.update(self.meta)
        json.dump(J, open(os.path.join(out_dir, "rig.json"), "w"), indent=1)
        return J
