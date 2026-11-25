"""Skeleton + forward kinematics + drawable parts. One rig format, two part sources:

* vector parts: Python callables that draw skia paths in the bone's local frame (engine2/characters.py);
* raster parts: PNG cut-outs with a joint anchor (a "part sheet" made by engine2/autorig.py from an AI image).

Every bone frame is canonical: its origin is the joint pivot, limbs point along +y ("down") and the spine/neck/head
point along -y ("up") at rest. Pose angles (engine2/poses.py) are therefore the same numbers for both sources.
Angles are degrees, positive = clockwise on screen (skia's y axis points down).
"""
import json, math, os
import skia


def mcopy(m):
    n = skia.Matrix()
    n.set9(m.get9())
    return n


class Bone:
    __slots__ = ("name", "parent", "pivot", "length", "bias")

    def __init__(self, name, parent, pivot=(0.0, 0.0), length=0.0, bias=0.0):
        self.name, self.parent, self.pivot, self.length, self.bias = name, parent, tuple(pivot), length, bias


class Skeleton:
    def __init__(self, bones):
        seen = set()
        for b in bones:
            if b.parent is not None and b.parent not in seen:
                raise ValueError(f"bone {b.name}: parent {b.parent} must come first")
            seen.add(b.name)
        self.bones = bones
        self.by = {b.name: b for b in bones}

    def world(self, angles, root):
        """angles: {bone: deg}. root: skia.Matrix placing the root bone. -> {bone: skia.Matrix} (bone frame -> screen)."""
        out = {}
        for b in self.bones:
            m = mcopy(root if b.parent is None else out[b.parent])
            m.preTranslate(b.pivot[0], b.pivot[1])
            m.preRotate(angles.get(b.name, 0.0) + b.bias)
            out[b.name] = m
        return out

    def world_angle(self, angles, name):
        a = 0.0
        while name is not None:
            b = self.by[name]
            a += angles.get(name, 0.0) + b.bias
            name = b.parent
        return a


def map_pt(m, x, y):
    p = m.mapXY(x, y)
    return p.x(), p.y()


class Ctx:
    """What a part's draw function gets: canvas helpers, style painter, time, face state, other bones' positions."""

    def __init__(self, rig, world, painter, t, face, bone, extra=None):
        self.rig, self.world, self.P, self.t, self.face, self.bone = rig, world, painter, t, face, bone
        self.extra = extra or {}

    def local(self, other, pt=(0.0, 0.0)):
        """Point `pt` of bone `other`, expressed in the current part's bone frame (for skirts, straps, deformers)."""
        x, y = map_pt(self.world[other], *pt)
        inv = skia.Matrix()
        self.world[self.bone].invert(inv)
        return map_pt(inv, x, y)


class Part:
    """Vector part: draw(canvas, ctx) in the bone's local frame. z orders parts (low = behind)."""

    def __init__(self, bone, z, draw, name=None):
        self.bone, self.z, self.draw, self.name = bone, z, draw, name or bone


class RasterPart(Part):
    """Raster part: image with the joint pivot at `anchor` (px). `rot` rotates the image into the canonical frame."""

    def __init__(self, bone, z, image, anchor, scale=1.0, rot=0.0, name=None):
        super().__init__(bone, z, self._draw, name)
        self.image, self.anchor, self.scale, self.rot = image, anchor, scale, rot
        self.paint = skia.Paint(AntiAlias=True)
        self.paint.setFilterQuality(skia.FilterQuality.kMedium_FilterQuality)

    def _draw(self, c, ctx):
        c.save()
        c.rotate(self.rot)
        c.scale(self.scale, self.scale)
        ctx.P.raster_part(c, self.image, -self.anchor[0], -self.anchor[1], self.paint)
        c.restore()
        if self.bone in ("hand_n", "paw_n"):  # held props (mug, bundle, walnut...) work on raster hands too
            prop = ctx.extra.get("prop_n" if self.bone == "hand_n" else "prop_paw")
            if prop:
                from .characters import PROPS
                PROPS[prop](c, ctx)


class Rig:
    def __init__(self, name, skeleton, parts, contacts=(), units=1000.0, kind="humanoid", meta=None):
        self.name, self.skel, self.kind = name, skeleton, kind
        self.parts = sorted(parts, key=lambda p: p.z)
        self.contacts = list(contacts)  # (bone, (x, y)) points that may touch the ground
        self.units = units  # rig height in its own units (feet to top of head)
        self.meta = meta or {}

    def ground_offset(self, angles, root, ground_y):
        """Screen px to move the rig vertically so its lowest contact point sits on the ground line ground_y."""
        if not self.contacts:
            return 0.0
        w = self.skel.world(angles, root)
        low = max(map_pt(w[b], *p)[1] for b, p in self.contacts)
        return ground_y - low

    def draw(self, canvas, angles, root, painter, t=0.0, face=None, extra=None):
        w = self.skel.world(angles, root)
        for p in self.parts:
            canvas.save()
            canvas.concat(w[p.bone])
            p.draw(canvas, Ctx(self, w, painter, t, face or {}, p.bone, extra))
            canvas.restore()
        return w


# ---------------------------------------------------------------- raster part sheets (AI-image source)
def load_partsheet(folder, contacts=None):
    """Load a part sheet written by engine2/autorig.py: rig.json + one PNG per part.

    rig.json: {"name", "units_per_px", "bones": [{"name", "parent", "pivot_px": [x, y], "rest_deg"}],
               "parts": [{"bone", "image", "anchor_px": [x, y], "z"}], "contacts": [[bone, [x, y]], ...]}
    pivot_px are absolute pixel positions in the source image; rest_deg is the bone's world angle in the image
    relative to its canonical direction (limbs: down, spine/head: up). Pivots are converted into canonical
    parent frames and each image is counter-rotated by its rest angle, so poses written for vector rigs drive
    raster rigs unchanged. image_pose_angles(folder) gives the angles that reproduce the source image.
    """
    J = json.load(open(os.path.join(folder, "rig.json")))
    k = J.get("units_per_px", 1.0)
    rest = {b["name"]: b.get("rest_deg", 0.0) for b in J["bones"]}
    px = {b["name"]: b["pivot_px"] for b in J["bones"]}
    bones = []
    for b in J["bones"]:
        par = b.get("parent")
        if par is None:
            piv = (0.0, 0.0)
        else:
            dx, dy = (px[b["name"]][0] - px[par][0]) * k, (px[b["name"]][1] - px[par][1]) * k
            a = -math.radians(rest[par])  # rotate the image offset into the parent's canonical frame
            piv = (dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a))
        bones.append(Bone(b["name"], par, piv, b.get("length", 0.0) * k))
    sk = Skeleton(bones)
    parts = []
    for p in J["parts"]:
        img = skia.Image.open(os.path.join(folder, p["image"]))
        parts.append(RasterPart(p["bone"], p.get("z", 0), img, p["anchor_px"], scale=k, rot=-rest[p["bone"]], name=p.get("name")))
    cts = contacts or [(c[0], tuple(c[1])) for c in J.get("contacts", [])]
    cts = [(b, tuple(_rot(p[0] * k, p[1] * k, -rest[b]))) for b, p in cts]  # contact points given in image px offsets
    return Rig(J.get("name", os.path.basename(folder)), sk, parts, cts, units=J.get("units", 1000.0),
               kind=J.get("kind", "humanoid"), meta={"source": "raster", "folder": folder, "pelvis_h": J.get("pelvis_h", 470.0)})


def _rot(x, y, deg):
    a = math.radians(deg)
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)


def image_pose_angles(folder):
    """Local bone angles that put a raster rig back into the pose of its source image."""
    J = json.load(open(os.path.join(folder, "rig.json")))
    rest = {b["name"]: b.get("rest_deg", 0.0) for b in J["bones"]}
    return {b["name"]: rest[b["name"]] - (rest[b["parent"]] if b.get("parent") else 0.0) for b in J["bones"]}
