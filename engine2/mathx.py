"""Small deterministic math helpers: easing, smooth noise, damped springs."""
import math
import numpy as np


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_io(x):
    """Cubic ease-in-out (pose transitions)."""
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_out_back(x, s=1.4):
    """Overshoots a little then settles: puppet 'snap' into a pose."""
    x = clamp(x)
    x -= 1
    return x * x * ((s + 1) * x + s) + 1


def lerp(a, b, u):
    return a + (b - a) * u


class Noise1D:
    """Smooth value noise in [-1, 1]; same seed -> same curve (render workers agree)."""

    def __init__(self, seed=0, n=512):
        self.v = np.random.default_rng(seed).uniform(-1, 1, n)
        self.n = n

    def __call__(self, t):
        i = math.floor(t)
        f = t - i
        a, b = self.v[i % self.n], self.v[(i + 1) % self.n]
        return float(a + (b - a) * smooth(f))

    def fbm(self, t, octaves=2):
        s, amp, tot = 0.0, 1.0, 0.0
        for k in range(octaves):
            s += amp * self(t * (2 ** k) + 31.7 * k)
            tot += amp
            amp *= 0.5
        return s / tot


def spring_follow(target, fps, freq=1.0, zeta=0.3, x0=None):
    """Second-order damped spring that follows `target` (array, one value per frame).
    freq in Hz, zeta = damping ratio (<1 overshoots). Semi-implicit Euler with 4 substeps per frame."""
    target = np.asarray(target, dtype=np.float64)
    w = 2 * math.pi * freq
    out = np.empty_like(target)
    x = target[0] if x0 is None else x0
    v = 0.0
    dt = 1.0 / fps / 4
    for i, tg in enumerate(target):
        for _ in range(4):
            a = w * w * (tg - x) - 2 * zeta * w * v
            v += a * dt
            x += v * dt
        out[i] = x
    return out


def ease_back_io(x, c1=1.1):
    """Anime-style key-to-key: a small anticipation dip, a quick move, a slight overshoot that settles."""
    x = clamp(x)
    c2 = c1 * 1.525
    if x < 0.5:
        return ((2 * x) ** 2 * ((c2 + 1) * 2 * x - c2)) / 2
    return ((2 * x - 2) ** 2 * ((c2 + 1) * (x * 2 - 2) + c2) + 2) / 2