"""Always-moving camera. Speeds are clamped to the reference's measured ranges (research/parabrahma/REPORT.md):
pans 0.2-0.44 px/frame at 1080 width (6-13 px/s at 30 fps), zoom 0.7-1.4 %/s, plus 4 s eased tilt reveals.
Positions are world px (scene design space, 1080 px = frame width at zoom 1)."""
import math
from .mathx import smooth, clamp

PAN_PX_S = (6.0, 13.2)
ZOOM_PCT_S = (0.7, 1.4)
TILT_S = 4.0


class Camera:
    def __init__(self, kind, t0, t1, world, focus, zoom=1.0, rng=None, W=1080, H=1920, tilt_from=None,
                 pan_px_s=None, zoom_pct_s=None, track=None, shakes=(), handheld=4.0, seed=0):
        self.kind, self.t0, self.t1 = kind, t0, t1
        self.world = world  # (w, h)
        self.focus = focus  # (x, y) world px the shot frames
        self.z0 = zoom
        self.W, self.H = W, H
        u = (lambda a, b: a + (b - a) * rng.random()) if rng is not None else (lambda a, b: (a + b) / 2)
        self.pan = pan_px_s or u(*PAN_PX_S)
        self.zr = (zoom_pct_s or u(*ZOOM_PCT_S)) / 100.0
        self.dir = 1 if (rng is None or rng.random() < 0.5) else -1
        self.tilt_from = tilt_from  # world y the tilt starts on (tall scenes)
        self.track = track  # follow: t -> (x, facing) of the actor the camera follows
        self.shakes = list(shakes)  # impact moments (storm lightning, dramatic beats): a short decaying shake
        self.handheld = handheld
        from .mathx import Noise1D
        self._nx, self._ny = Noise1D(seed * 7 + 1), Noise1D(seed * 7 + 2)

    def _clamp(self, cx, cy, z):
        hw, hh = self.W / 2 / z, self.H / 2 / z
        ww, wh = self.world
        cx = clamp(cx, hw, ww - hw) if ww > 2 * hw else ww / 2
        cy = clamp(cy, hh, wh - hh) if wh > 2 * hh else wh / 2
        return cx, cy

    def at(self, t):
        """-> (cx, cy, zoom) in world px."""
        d = t - self.t0
        fx, fy = self.focus
        z = self.z0
        k = self.kind
        if k == "push_in":
            z = self.z0 * (1 + self.zr * d)
            fx += self.dir * self.pan * 0.8 * d
        elif k == "pull_back":
            z = self.z0 * (1 + self.zr * (self.t1 - self.t0 - d))
            fx -= self.dir * self.pan * 0.8 * d
        elif k in ("pan_left", "pan_right"):
            s = -1 if k == "pan_left" else 1
            fx += s * self.pan * (d - (self.t1 - self.t0) / 2)
            z = self.z0 * (1 + 0.4 * self.zr * d)
        elif k == "drift":
            fx += self.dir * self.pan * 0.8 * (d - (self.t1 - self.t0) / 2)
            fy += self.pan * 0.35 * d
            z = self.z0 * (1 + 0.7 * self.zr * d)
        elif k in ("tilt_down", "tilt_up"):
            span = self.tilt_from if self.tilt_from is not None else fy + (-1 if k == "tilt_down" else 1) * self.H * 0.55
            u = smooth(min(1.0, d / TILT_S))  # smoothstep: peak speed 1.5x the mean (slow reveal)
            fy = span + (fy - span) * u
            z = self.z0 * (1 + 0.6 * self.zr * d)
            fx += self.dir * self.pan * 0.5 * d
            if d > TILT_S:  # keep creeping after the reveal so the frame never freezes
                fy += (1 if k == "tilt_down" else -1) * self.pan * 0.5 * (d - TILT_S)
        if k == "follow" and self.track is not None:  # follow the walker, a little behind, leading in its direction
            ax, fc = self.track(t - 0.18)
            fx = ax + 110 * (1 if fc >= 0 else -1)
            z = self.z0 * (1 + 0.5 * self.zr * d)
        # handheld drift (slow, a few px) + impact shakes
        fx += self.handheld * self._nx.fbm(t * 0.35)
        fy += self.handheld * 0.7 * self._ny.fbm(t * 0.3)
        for ts in self.shakes:
            if 0 <= t - ts < 0.45:
                e = (1 - (t - ts) / 0.45) ** 2
                fx += 14 * e * math.sin(2 * math.pi * 23 * (t - ts))
                fy += 10 * e * math.sin(2 * math.pi * 29 * (t - ts) + 1)
        cx, cy = self._clamp(fx, fy, z)
        return cx, cy, z

    def speeds(self, fps=30, dt=None):  # always per 1/30 s: the reference was measured at 30 fps
        """Measured px/frame (at 1080 width) and zoom %/s at the middle of the shot (QA)."""
        dt = dt or 1.0 / 30
        tm = (self.t0 + self.t1) / 2
        a, b = self.at(tm), self.at(tm + dt)
        px = math.hypot((b[0] - a[0]) * a[2], (b[1] - a[1]) * a[2])
        return px, (b[2] / a[2] - 1) * 100 / dt


FRAMING = {"wide": 1.0, "medium": 1.32, "close": 1.7}
