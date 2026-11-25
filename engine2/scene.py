"""Layered parallax scenes. Back to front (reference: 5-7 layers): sky, far, mid, ground, ACTORS, foreground,
light (rays/glow/particles), then the vignette in screen space.

Static layer content is rasterised once per scene into a cached skia.Image (the fix for the prototype's
0.84 s/frame SVG re-rasterisation); only moving bits (swaying canopy, twinkles, particles, rays) are drawn per frame.
"""
import skia
from .fx import vignette_image

ACTORS = "actors"


class Layer:
    def __init__(self, name, depth, static=None, dynamic=None, blend=None):
        self.name, self.depth, self.static, self.dynamic, self.blend = name, depth, static, dynamic, blend


class Scene:
    """world: (w, h) design px. layers: list of Layer, one of them named ACTORS (depth 1.0).
    ground: world y of the main floor line. spots: named (x, y) world positions (e.g. "under_maple")."""

    def __init__(self, name, world, layers, ground, painter, spots=None, margin=260, overlay=None, tall=False,
                 mood="day", light=None, haze="#c9d2de"):
        self.name, self.world, self.layers, self.ground, self.P = name, world, layers, ground, painter
        self.mood = mood            # engine2/post.py MOODS: grade, key light, rim, bloom
        self.light = light or ((world[0] * 0.25, world[1] * 0.2), 0.3)  # (world xy, parallax depth) of the light source
        self.haze = haze
        self.spots = spots or {}
        self.margin = margin
        self.overlay = overlay  # screen-space draw(c, t, W, H) after the vignette (e.g. paper proscenium)
        self.tall = tall
        self._cache = {}
        self._vig = {}

    def anchor(self):
        return self.world[0] / 2, self.world[1] / 2

    def view(self, cam, depth, W, H):
        """Matrix from layer coords to screen for a camera (cx, cy, zoom) and a parallax depth."""
        cx, cy, z = cam
        ax, ay = self.anchor()
        lcx, lcy = ax + (cx - ax) * depth, ay + (cy - ay) * depth
        zd = (1 + (z - 1) * depth) * (W / 1080.0)
        m = skia.Matrix()
        m.setTranslate(W / 2, H / 2)
        m.preScale(zd, zd)
        m.preTranslate(-lcx, -lcy)
        return m

    def _static_image(self, layer, q):
        key = (layer.name, round(q, 2))
        if key not in self._cache:
            m = self.margin
            w, h = int((self.world[0] + 2 * m) * q), int((self.world[1] + 2 * m) * q)
            s = skia.Surface.MakeRasterN32Premul(w, h)
            c = s.getCanvas()
            c.clear(skia.ColorTRANSPARENT)
            c.scale(q, q)
            c.translate(m, m)
            layer.static(c)
            from .post import atmosphere  # far layers hazed + softened, foreground defocused: once, at cache time
            self._cache[key] = atmosphere(s.makeImageSnapshot(), layer.depth, self.haze, q)
        return self._cache[key]

    def light_xy(self, cam, W, H):
        (x, y), d = self.light
        p = self.view(cam, d, W, H).mapXY(x, y)
        return p.x(), p.y()

    def render(self, c, t, cam, actors, W, H, q=1.0, fin=None):
        self._img_paint = getattr(self, "_img_paint", None) or skia.Paint(AntiAlias=True)
        self._img_paint.setFilterQuality(skia.FilterQuality.kLow_FilterQuality)
        for L in self.layers:
            if L.name == ACTORS and fin is not None and actors:  # own layer: rim light on the lit edge
                ac = fin.actor_surf.getCanvas()
                ac.clear(skia.ColorTRANSPARENT)
                ac.save()
                ac.concat(self.view(cam, L.depth, W, H))
                for a in sorted(actors, key=lambda a: a.ground):
                    a.draw(ac, t)
                ac.restore()
                img = fin.actor_surf.makeImageSnapshot()
                c.drawImage(img, 0, 0)
                fin.rim(c, img, self.mood, self.light_xy(cam, W, H))
                continue
            c.save()
            c.concat(self.view(cam, L.depth, W, H))
            if L.name == ACTORS:
                for a in sorted(actors, key=lambda a: a.ground):
                    a.draw(c, t)
            else:
                if L.blend is not None:
                    c.saveLayer(None, _blend_paint(L.blend))
                if L.static is not None:
                    img = self._static_image(L, q)
                    c.save()
                    c.translate(-self.margin, -self.margin)
                    c.scale(1 / q, 1 / q)
                    c.drawImage(img, 0, 0, self._img_paint)
                    c.restore()
                if L.dynamic is not None:
                    L.dynamic(c, t)
                if L.blend is not None:
                    c.restore()
            c.restore()
        key = (W, H)
        if key not in self._vig:
            self._vig[key] = vignette_image(W, H)
        c.drawImage(self._vig[key], 0, 0)
        if fin is not None:
            fin.key_light(c, self.mood, self.light_xy(cam, W, H))
        if self.overlay:
            self.overlay(c, t, W, H)
        self.P.grade(c, W, H, t)


def _blend_paint(mode):
    p = skia.Paint()
    p.setBlendMode({"screen": skia.BlendMode.kScreen, "plus": skia.BlendMode.kPlus, "multiply": skia.BlendMode.kMultiply}[mode])
    return p
