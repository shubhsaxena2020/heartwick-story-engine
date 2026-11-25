"""2.5D cut-out renderer: depth-parallax plates + puppet characters + particles + grade + captions + end card.
Pure numpy/OpenCV/PIL so it runs on Kaggle CPUs and (for low-res previews) on the laptop at low priority."""
import json, math, os, subprocess, sys, time
import numpy as np
import cv2
from PIL import Image
from . import brand
from .style import PALETTE, TINTS

cv2.setNumThreads(2)


def _ease(u):
    u = min(1.08, max(-0.08, u))
    return 0.65 * u + 0.35 * (0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, u))))


def _load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255


def _load_rgba(p):
    return np.asarray(Image.open(p).convert("RGBA"), np.float32) / 255


class Assets:
    def __init__(self, d, scale):
        self.d, self.scale, self.cache = d, scale, {}

    def bg(self, loc):
        k = ("bg", loc)
        if k not in self.cache:
            p = f"{self.d}/bg_{loc}.png"
            if os.path.exists(p):
                img = _load_rgb(p)
            else:  # graceful placeholder so a missing plate never kills a render
                img = np.ones((1344, 768, 3), np.float32) * np.array(PALETTE["dusk_teal"], np.float32) / 255
            if self.scale < 1:
                img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            dp = f"{self.d}/bg_{loc}_depth.png"
            if os.path.exists(dp):
                dep = np.asarray(Image.open(dp).convert("L"), np.float32) / 255
                dep = cv2.resize(dep, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
                lo, hi = np.percentile(dep, 3), np.percentile(dep, 97)
                dep = np.clip((dep - lo) / max(1e-3, hi - lo), 0, 1)
                k_ = int(img.shape[1] * 0.03) | 1
                dep = cv2.GaussianBlur(dep, (k_, k_), 0)
            else:
                yy = np.linspace(0, 1, img.shape[0], dtype=np.float32)[:, None]
                dep = np.repeat(yy ** 1.5, img.shape[1], axis=1)  # assume ground plane: lower = nearer
            self.cache[k] = (img, dep, img.reshape(-1, 3).mean(0))
        return self.cache[k]

    def cut(self, char, pose):
        k = ("cut", char, pose)
        if k not in self.cache:
            p = f"{self.d}/pose_{char}_{pose}_cut.png"
            if not os.path.exists(p):
                p2 = f"{self.d}/master_{char}_cut.png"
                p = p2 if os.path.exists(p2) else None
            self.cache[k] = _load_rgba(p) if p else None
        return self.cache[k]


class Particles:
    def __init__(self, fx, seed, W, H, rs):
        self.fx, self.W, self.H, self.rs = set(fx or []), W, H, rs
        r = np.random.default_rng(seed)
        self.motes = r.random((int(40), 5)).astype(np.float32)
        self.snow = r.random((110, 5)).astype(np.float32)
        self.rain = r.random((200, 4)).astype(np.float32)
        self.ff = r.random((22, 5)).astype(np.float32)
        self.leaves = r.random((11, 6)).astype(np.float32)
        self.flash_t = sorted(r.uniform(0.15, 0.85, 2))
        if "leaves" in self.fx:
            self.leaf_sprites = [self._leaf(int((34 + 40 * r.random()) * rs), c) for c in
                                 (PALETTE["maple"], PALETTE["ember"], (205, 92, 45), PALETTE["honey"])]
        if "mist" in self.fx:
            n = r.random((24, 64)).astype(np.float32)
            n = cv2.resize(n, (int(W / 4 * 2), int(H / 4 * 0.5)), interpolation=cv2.INTER_CUBIC)
            n = cv2.GaussianBlur(n, (0, 0), W / 4 * 0.06)
            n = (n - n.min()) / max(1e-6, n.max() - n.min())
            self.mist = np.clip((n - 0.35) * 1.6, 0, 1)

    @staticmethod
    def _leaf(size, col):
        S = size * 4
        im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        from PIL import ImageDraw
        d = ImageDraw.Draw(im)
        pts = []
        for k in range(360):  # 5-lobed maple-ish leaf in polar form
            a = math.radians(k)
            rr = 0.32 + 0.14 * abs(math.cos(2.5 * (a - math.pi / 2))) ** 0.6 + 0.05 * math.cos(10 * a)
            if math.sin(a) > 0.85:
                rr *= 0.45
            pts.append((S / 2 + rr * S * math.cos(a), S / 2 - rr * S * math.sin(a)))
        d.polygon(pts, fill=tuple(col) + (235,))
        d.line([(S / 2, S / 2), (S / 2, S * 0.92)], fill=(110, 50, 30, 230), width=max(1, S // 40))
        return np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255

    def draw(self, img, t, dur, pan):
        """Particles are accumulated at half resolution (one additive layer + one alpha layer), then composited once."""
        W, H, rs = self.W, self.H, self.rs
        px, py = pan
        if not self.fx:
            return img
        h2, w2 = H // 2, W // 2
        add = None
        if self.fx & {"motes", "fireflies", "rain"}:
            add = np.zeros((h2, w2, 3), np.float32)
        if "motes" in self.fx:
            col = np.array(PALETTE["honey"], np.float32) / 255
            for x, y, s, ph, sp in self.motes:
                yy = ((y - t * (0.01 + 0.02 * sp)) % 1.0) * H - py * 0.3 * (0.5 + s)
                xx = ((x + 0.015 * math.sin(t * 0.7 + ph * 6)) % 1.0) * W - px * 0.3 * (0.5 + s)
                a = 0.35 + 0.35 * math.sin(t * (1.5 + sp) + ph * 6.28)
                cv2.circle(add, (int(xx / 2), int(yy % H / 2)), max(1, int((1.5 + 3 * s) * rs)), (col * a).tolist(), -1, cv2.LINE_AA)
        if "fireflies" in self.fx:
            col = np.array((0.85, 0.95, 0.45), np.float32)
            for x, y, s, ph, sp in self.ff:
                xx = (x + 0.05 * math.sin(t * 0.5 * (1 + sp) + ph * 6)) * W
                yy = (0.35 + 0.6 * y + 0.04 * math.sin(t * 0.7 + ph * 4)) * H
                a = max(0.0, math.sin(t * (1.2 + sp) + ph * 6.28)) ** 2
                cv2.circle(add, (int(xx / 2), int(yy / 2)), max(1, int(3 * rs)), (col * a).tolist(), -1, cv2.LINE_AA)
        if add is not None and self.fx & {"motes", "fireflies"}:
            add += cv2.GaussianBlur(add, (0, 0), 3 * rs + 1) * 1.6
        if "rain" in self.fx:
            for x, y, s, ph in self.rain:
                L = (20 + 30 * s) * rs
                yy = ((y + t * (1.3 + s)) % 1.0) * h2
                xx = ((x + yy / h2 * 0.12) % 1.0) * w2
                v = float(0.18 + 0.22 * s)
                cv2.line(add, (int(xx), int(yy)), (int(xx - L * 0.25), int(yy - L)), (v * 0.75, v * 0.82, v * 0.95), max(1, int(rs)), cv2.LINE_AA)
        alpha = None
        if self.fx & {"snow", "mist"}:
            alpha = np.zeros((h2, w2), np.float32)
        if "snow" in self.fx:
            for x, y, s, ph, sp in self.snow:
                yy = ((y + t * (0.03 + 0.05 * sp)) % 1.0) * h2
                xx = ((x + 0.02 * math.sin(t * 0.9 + ph * 6)) % 1.0) * w2 - px * 0.15 * s
                cv2.circle(alpha, (int(xx), int(yy)), max(1, int((1 + 2.5 * s) * rs)), float(0.55 + 0.4 * s), -1, cv2.LINE_AA)
        if "mist" in self.fx:
            m = self.mist
            off = int((t * 0.012 * W / 4) % (m.shape[1] - W // 4 - 1)) if m.shape[1] > W // 4 + 1 else 0
            band = cv2.resize(m[:, off:off + W // 4], (w2, h2 // 2), interpolation=cv2.INTER_LINEAR) * 0.28
            fe = np.clip(np.sin(np.linspace(0, np.pi, band.shape[0], dtype=np.float32)), 0, 1) ** 1.5  # feather top/bottom: no hard band edge
            band = band * fe[:, None]
            y0 = int(h2 * 0.45)
            np.maximum(alpha[y0:y0 + band.shape[0]], band[:h2 - y0], out=alpha[y0:y0 + band.shape[0]])
        if add is not None:
            img += cv2.resize(add, (W, H), interpolation=cv2.INTER_LINEAR)
        if alpha is not None:
            a = cv2.resize(cv2.GaussianBlur(alpha, (0, 0), 0.8), (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
            tmp = np.subtract(np.float32(0.93), img)
            tmp *= a
            img += tmp
        if "leaves" in self.fx:
            for i, (x, y, s, ph, sp, c) in enumerate(self.leaves):
                spr = self.leaf_sprites[int(c * len(self.leaf_sprites)) % len(self.leaf_sprites)]
                sz = spr.shape[0]
                yy = ((y + t * (0.05 + 0.08 * sp)) % 1.15 - 0.08) * H
                xx = (x + 0.06 * math.sin(t * (0.8 + sp) + ph * 6)) * W - px * 0.5
                ang = math.degrees(t * (0.6 + 1.5 * sp) + ph * 6)
                sc = 0.6 + 0.8 * s
                M = cv2.getRotationMatrix2D((sz / 2, sz / 2), ang, sc)
                M[0, 2] += xx - sz / 2; M[1, 2] += yy - sz / 2
                x0, y0 = int(max(0, xx - sz)), int(max(0, yy - sz))
                x1, y1 = int(min(W, xx + sz)), int(min(H, yy + sz))
                if x1 <= x0 or y1 <= y0:
                    continue
                M[0, 2] -= x0; M[1, 2] -= y0
                w = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
                a = w[..., 3:4]
                img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + w[..., :3] * a
        if "lightning" in self.fx:
            u = t / max(dur, 1e-3)
            for ft in self.flash_t:
                k = (u - ft) * dur
                if 0 <= k < 0.35:
                    f = 0.45 * math.exp(-k * 12) + (0.25 if 0.12 < k < 0.18 else 0)
                    img += np.array([0.8, 0.85, 1.0], np.float32) * np.float32(f)
        return img


class ShotRenderer:
    """Renders one shot (or sub-shot) at time t."""

    def __init__(self, sh, tl, assets, W, H):
        self.sh, self.W, self.H = sh, W, H
        self.rs = W / 1080
        loc = tl["locations"].get(sh["bg"], {})
        self.tint = TINTS.get(loc.get("tint", "warm"), TINTS["warm"])
        self.img, self.dep, self.mean = assets.bg(sh["bg"])
        bh, bw = self.img.shape[:2]
        self.tall = bh / bw > H / W * 1.15
        self.s0 = (W / bw * 1.04) if self.tall else max(W / bw, H / bh) * 1.06
        self.Bw, self.Bh = bw * self.s0, bh * self.s0
        self.chars = []
        for c in sh.get("chars", []):
            cut = assets.cut(c["char"], c["pose"])
            if cut is None:
                continue
            # light wrap: pull the puppet slightly toward the plate's average colour
            rgb = cut[..., :3] * (0.88 + 0.12 * self.mean / max(1e-3, self.mean.mean()))
            cut = np.concatenate([np.clip(rgb, 0, 1), cut[..., 3:]], -1)
            fx, fy = c["x"] * W, c["y"] * H
            # depth of the ground under the feet -> parallax factor
            u, v = self._rest_to_bg(fx, fy)
            dv = float(self.dep[int(np.clip(v, 0, bh - 1)), int(np.clip(u, 0, bw - 1))])
            parts = [(cut, None)]
            if c.get("kind") == "human" and cut.shape[0] > 200:
                # puppet articulation: head layer pivots at the neck with a feathered seam
                hh = cut.shape[0]
                ys = np.arange(hh, dtype=np.float32)[:, None, None]
                m_head = np.clip((0.29 * hh - ys) / (0.05 * hh), 0, 1)   # opaque to 0.24h, fades out by 0.29h
                m_body = np.clip((ys - 0.19 * hh) / (0.02 * hh), 0, 1)   # body starts at ~0.2h (hidden under the head)
                head = cut.copy(); head[..., 3:] *= m_head
                body = cut.copy(); body[..., 3:] *= m_body
                top = cut[: int(0.24 * hh), :, 3]
                cols = np.where(top.sum(0) > 0.5)[0]
                px = float(cols.mean()) if len(cols) else cut.shape[1] / 2
                parts = [(body, None), (head, (px, 0.215 * hh))]
            self.chars.append(dict(c, cut=cut, parts=parts, p=0.6 + 0.4 * max(dv, 0.75), phase=(hash(c["char"] + c["pose"]) % 100) / 16))
        self.chars.sort(key=lambda c: c["y"])  # farther (higher feet) first
        self.parts = Particles(sh.get("fx"), hash((sh["bg"], round(sh["s"], 2))) % 2 ** 31, W, H, self.rs)
        # camera focus
        if sh.get("focus"):
            self.focus = (sh["focus"][0] * W, sh["focus"][1] * H)
        elif self.chars:
            self.focus = (np.mean([c["x"] * W for c in self.chars]), np.mean([(c["y"] - c["h"] * 0.6) * H for c in self.chars]))
        else:
            self.focus = (W / 2, H / 2)
        self.dur = sh["e"] - sh["s"]
        self.t0 = sh["s"] - sh.get("trans_in", 0) / 2
        self.t1 = sh["e"] + 0.35
        ys = np.arange(H, dtype=np.float32); xs = np.arange(W, dtype=np.float32)
        self.xs, self.ys = xs, ys

    def _rest_to_bg(self, X, Y):
        bh, bw = self.img.shape[:2]
        return (X - self.W / 2) / self.s0 + bw / 2, (Y - self.H / 2) / self.s0 + bh / 2

    def _lim(self, z):
        return max(0.0, self.Bw / 2 - self.W / (2 * z)), max(0.0, self.Bh / 2 - self.H / (2 * z))

    def camera(self, t):
        u = _ease((t - self.t0) / max(1e-3, self.t1 - self.t0))
        cam, W, H = self.sh["cam"], self.W, self.H
        fr = self.sh.get("framing", "wide")
        zf = {"wide": 1.0, "medium": 1.14, "close": 1.3}[fr]
        fx, fy = self.focus[0] - W / 2, self.focus[1] - H / 2
        if cam == "push_in":
            z = 1.0 + 0.11 * u; k = 0.15 + 0.55 * u; pan = (fx * k, fy * k)
        elif cam == "pull_back":
            z = 1.13 - 0.12 * u; k = 0.7 - 0.55 * u; pan = (fx * k, fy * k)
        elif cam in ("pan_right", "pan_left"):
            z = 1.1; lx, _ = self._lim(z * zf); d = (u - 0.5) * 2 * min(lx * 0.85, 160 * self.rs * 2)
            pan = (d if cam == "pan_right" else -d, fy * 0.3)
        elif cam in ("tilt_down", "tilt_up"):
            z = 1.0 if self.tall else 1.08; _, ly = self._lim(z * zf)
            d = (u - 0.5) * 2 * ly * 0.92
            pan = (0.0, d if cam == "tilt_down" else -d)
        else:  # drift
            z = 1.05 + 0.05 * u; pan = (fx * 0.3 + (u - 0.5) * 70 * self.rs, fy * 0.3 - (u - 0.5) * 40 * self.rs)
        z *= zf
        if zf > 1:
            pan = (pan[0] * 0.5 + fx * 0.6, pan[1] * 0.5 + fy * 0.6)
        lx, ly = self._lim(z)
        return z, (float(np.clip(pan[0], -lx, lx)), float(np.clip(pan[1], -ly, ly)))

    def _maps(self, z, pan, p):
        zp = 1 + p * (z - 1)
        bh, bw = self.img.shape[:2]
        X = (self.xs - self.W / 2) / zp + self.W / 2 + p * pan[0]
        Y = (self.ys - self.H / 2) / zp + self.H / 2 + p * pan[1]
        return (X - self.W / 2) / self.s0 + bw / 2, (Y - self.H / 2) / self.s0 + bh / 2

    def render(self, t):
        W, H = self.W, self.H
        z, pan = self.camera(t)
        uf, vf = self._maps(z, pan, 0.55)
        un, vn = self._maps(z, pan, 1.0)
        hx = np.ascontiguousarray(np.broadcast_to(un[None, ::2], (H // 2, W // 2)), np.float32)
        hy = np.ascontiguousarray(np.broadcast_to(vn[::2, None], (H // 2, W // 2)), np.float32)
        d = cv2.resize(cv2.remap(self.dep, hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), (W, H), interpolation=cv2.INTER_LINEAR)
        mx = np.multiply(d, (un - uf).astype(np.float32)[None, :]); mx += uf.astype(np.float32)[None, :]
        my = np.multiply(d, (vn - vf).astype(np.float32)[:, None]); my += vf.astype(np.float32)[:, None]
        img = cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for c in self.chars:
            img = self._draw_char(img, c, t, z, pan)
        img = self.parts.draw(img, t - self.sh["s"], self.dur, pan)
        mul, lift = self.tint
        img *= np.array(mul, np.float32)
        img += np.array(lift, np.float32) / 255
        return img

    def _draw_char(self, img, c, t, z, pan):
        W, H = self.W, self.H
        cut = c["cut"]; ch, cw = cut.shape[:2]
        lt = t - self.sh["s"]
        k = c["h"] * H / ch
        ph = c["phase"]
        breathe = 1 + 0.012 * math.sin(2 * math.pi * 0.27 * lt + ph)
        shear = 0.022 * math.sin(2 * math.pi * 0.16 * lt + ph) + 0.008 * math.sin(2 * math.pi * 0.41 * lt + ph * 2)
        ox, oy = 0.0, 0.0
        anim = c.get("anim", "idle")
        if anim == "tremble":
            ox = 1.6 * self.rs * math.sin(lt * 55) ; breathe += 0.01 * math.sin(lt * 31)
        elif anim == "hop":
            oy = -abs(math.sin(lt * 4.2)) * 0.06 * c["h"] * H; ox = (lt / max(self.dur, 1) - 0.5) * 0.5 * W
        elif anim == "climb":
            oy = -(lt / max(self.dur, 1)) * 0.45 * H + abs(math.sin(lt * 7)) * 6 * self.rs
        X0, Y0 = c["x"] * W + ox, c["y"] * H + oy
        # local: cut px -> rest frame
        A = np.array([[k / breathe ** 0.5, k * shear, X0 - k / breathe ** 0.5 * cw / 2 - k * shear * ch],
                      [0, k * breathe, Y0 - k * breathe * ch]], np.float64)
        p = c["p"]; zp = 1 + p * (z - 1)
        # rest -> screen for parallax p: S = (X - C - p*pan)*zp + C
        B = np.array([[zp, 0, (-W / 2 - p * pan[0]) * zp + W / 2], [0, zp, (-H / 2 - p * pan[1]) * zp + H / 2]])
        M = B[:, :2] @ A
        M = np.hstack([M[:, :2], (B[:, :2] @ A[:, 2:] + B[:, 2:])])
        corners = np.array([[0, 0, 1], [cw, 0, 1], [0, ch, 1], [cw, ch, 1]], np.float64) @ M.T
        x0, y0 = int(max(0, corners[:, 0].min() - 2)), int(max(0, corners[:, 1].min() - 2))
        x1, y1 = int(min(W, corners[:, 0].max() + 2)), int(min(H, corners[:, 1].max() + 2))
        if x1 <= x0 or y1 <= y0:
            return img
        # contact shadow
        fx_, fy_ = (M @ np.array([cw / 2, ch, 1.0]))
        sw, shh = (corners[:, 0].max() - corners[:, 0].min()) * 0.42, max(4, (y1 - y0) * 0.035)
        sx0, sy0, sx1, sy1 = int(max(0, fx_ - sw * 1.4)), int(max(0, fy_ - shh * 3)), int(min(W, fx_ + sw * 1.4)), int(min(H, fy_ + shh * 3))
        if sx1 > sx0 and sy1 > sy0:
            yy, xx = np.mgrid[sy0:sy1, sx0:sx1].astype(np.float32)
            sm = np.exp(-(((xx - fx_) / sw) ** 2 + ((yy - fy_) / shh) ** 2) * 1.6)[..., None] * 0.38
            img[sy0:sy1, sx0:sx1] *= (1 - sm)
        nod = math.radians(2.4 * math.sin(2 * math.pi * 0.19 * lt + ph * 1.7) + 1.2 * math.sin(2 * math.pi * 0.07 * lt + ph))
        for part, pivot in c["parts"]:
            Mp = M
            if pivot is not None:  # rotate head about the neck pivot (in cut-out pixel space) before M
                cx_, cy_ = pivot
                R = np.array([[math.cos(nod), -math.sin(nod), cx_ - cx_ * math.cos(nod) + cy_ * math.sin(nod)],
                              [math.sin(nod), math.cos(nod), cy_ - cx_ * math.sin(nod) - cy_ * math.cos(nod)]])
                Mp = np.hstack([M[:, :2] @ R[:, :2], (M[:, :2] @ R[:, 2:] + M[:, 2:])])
            M2 = Mp.copy(); M2[0, 2] -= x0; M2[1, 2] -= y0
            w = cv2.warpAffine(part, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            a = w[..., 3:4]
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + w[..., :3] * a
        return img


class Overlay:
    """Captions, series tag, hook title, grade (vignette + bloom)."""

    def __init__(self, tl, W, H):
        self.tl, self.W, self.H, self.rs = tl, W, H, W / 1080
        self.cap_cache = {}
        self.tag = np.asarray(brand.series_tag(tl.get("series_tag") or "", self.rs), np.float32) / 255 if tl.get("series_tag") else None
        hk = tl.get("hook")
        self.hook = np.asarray(brand.hook_sprite(hk["kicker"], hk["lines"], hk.get("highlight"), self.rs, W), np.float32) / 255 if hk else None
        self.hook_s = 2.4 if hk else 0.0
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.48) / (H * 0.62)) ** 2)
        self.vig = np.repeat((1 - 0.32 * np.clip(r - 0.45, 0, 1) ** 1.3)[..., None], 3, axis=2).astype(np.float32)
        self.cap_y = int(H * 0.70)
        self.speech_end = tl["speech_end"]

    @staticmethod
    def _blit(img, spr, x, y, alpha=1.0):
        H, W = img.shape[:2]
        h, w = spr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return img
        s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
        a = s[..., 3:4] * alpha
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + s[..., :3] * a
        return img

    def grade(self, img):
        lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
        img -= lum[..., None]
        img *= np.float32(1.16)          # a little more colour than the raw SDXL plates
        img += lum[..., None] * np.float32(1.04) - np.float32(0.02)
        small = cv2.resize(img, (self.W // 4, self.H // 4), interpolation=cv2.INTER_AREA)
        small -= 0.72
        np.maximum(small, 0, out=small)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (self.W, self.H))
        cv2.scaleAdd(bloom, 0.45, img, dst=img)
        img *= self.vig
        return img

    def apply(self, img, t):
        W, H, rs = self.W, self.H, self.rs
        if self.hook is not None and t < self.hook_s:
            a = min(1.0, (self.hook_s - t) / 0.35) * min(1.0, t / 0.15 + 0.3)
            img = self._blit(img, self.hook, 0, int(H * 0.10), a)
        if self.tag is not None and self.hook_s <= t < self.speech_end + 0.3:
            a = min(1.0, (t - self.hook_s) / 0.4)
            img = self._blit(img, self.tag, int(64 * rs), int(200 * rs), a)
        for i, c in enumerate(self.tl["captions"]):
            if c["s"] <= t < c["e"]:
                if i not in self.cap_cache:
                    self.cap_cache = {k: v for k, v in self.cap_cache.items() if k > i - 3}
                    self.cap_cache[i] = np.asarray(brand.caption_sprite(c["words"], c["hl"], rs), np.float32) / 255
                spr = self.cap_cache[i]
                k = min(1.0, (t - c["s"]) / 0.2)
                out = min(1.0, (c["e"] - t) / 0.12) if (i + 1 >= len(self.tl["captions"]) or self.tl["captions"][i + 1]["s"] > c["e"] + 0.05) else 1.0
                y = self.cap_y - spr.shape[0] // 2 + int((1 - k) * 10 * rs)
                img = self._blit(img, spr, W // 2 - spr.shape[1] // 2, y, k * out)
                break
        return img


def frame_at(t, tl, shots, overlay, endcard):
    """Composite frame at time t (float RGB 0..1)."""
    total, ec = tl["total"], tl["end_card_s"]
    ec_start = total - ec
    def shot_img(j):
        return shots(j).render(t)
    S = tl["shots"]
    j = 0
    while j + 1 < len(S) and S[j + 1]["s"] <= t:
        j += 1
    img = None
    if t < ec_start + 0.6:
        img = shot_img(j)
        if j > 0:  # fading in from previous
            tr = S[j].get("trans_in", 0)
            a = (t - (S[j]["s"] - tr / 2)) / tr if tr > 0 else 1
            if a < 1:
                img = shot_img(j - 1) * (1 - a) + img * a
        if j + 1 < len(S):
            tr = S[j + 1].get("trans_in", 0)
            a = (t - (S[j + 1]["s"] - tr / 2)) / tr if tr > 0 else 0
            if 0 < a < 1:
                img = img * (1 - a) + shot_img(j + 1) * a
        img = overlay.grade(img)
        img = overlay.apply(img, t)
    if t >= ec_start:
        e = endcard.frame(t - ec_start)
        a = min(1.0, (t - ec_start) / 0.6)
        img = e if img is None else img * (1 - a) + e * a
    return np.clip(img, 0, 1)


def _worker(args):
    tl_path, assets_dir, fr0, fr1, seg_path, W, H, fps, threads, low = args
    if low:
        _lower_priority()
    tl = json.load(open(tl_path))
    scale = W / 1080
    assets = Assets(assets_dir, 1.0 if scale >= 0.99 else max(0.5, scale * 1.05))
    cache = {}

    def shots(j):
        if j not in cache:
            for k in list(cache):
                if k < j - 1:
                    del cache[k]
            cache[j] = ShotRenderer(tl["shots"][j], tl, assets, W, H)
        return cache[j]

    overlay = Overlay(tl, W, H)
    endcard = brand.EndCard(W, H, tl.get("cta", "Follow for the next story"))
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", str(threads), seg_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if low:
        _lower_priority(p.pid)
    for f in range(fr0, fr1):
        img = frame_at(f / fps, tl, shots, overlay, endcard)
        p.stdin.write((img * 255 + 0.5).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    return seg_path


def _lower_priority(pid=None):
    try:
        import psutil
        pr = psutil.Process(pid) if pid else psutil.Process()
        pr.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def build_audio(tl, tts_dir, out_wav, music=None, music_db=-22.0, lufs=-15.0, low=False):
    """Narration track (+ optional ducked music bed) normalised to `lufs`."""
    import soundfile as sf
    sr = 24000
    n = int((tl["total"] + 0.5) * sr)
    voice = np.zeros(n, np.float32)
    for a in tl["audio"]:
        x, _ = sf.read(f"{tts_dir}/{a['id']}.wav", dtype="float32")
        i = int(a["t"] * sr)
        voice[i:i + len(x)] += x[:max(0, n - i)]
    raw = out_wav.replace(".wav", "_voice24k.wav")
    sf.write(raw, voice, sr)
    pre = ["ffmpeg", "-v", "error", "-y", "-threads", "4"]
    if music:
        fc = (f"[0:a]aresample=48000,pan=stereo|c0=c0|c1=c0[v];[1:a]aresample=48000,aloop=loop=-1:size=2e9,atrim=0:{tl['total']:.2f},"
              f"volume={music_db}dB,afade=t=in:d=1.5,afade=t=out:st={max(0, tl['total'] - 3):.2f}:d=3[m];"
              f"[m][v]sidechaincompress=threshold=0.05:ratio=4:attack=30:release=400[md];[v][md]amix=inputs=2:normalize=0[mix]")
        mix_cmd = pre + ["-i", raw, "-i", music, "-filter_complex", fc, "-map", "[mix]"]
    else:
        mix_cmd = pre + ["-i", raw, "-af", "aresample=48000,pan=stereo|c0=c0|c1=c0"]
    tmp = out_wav.replace(".wav", "_premix.wav")
    subprocess.run(mix_cmd + ["-ar", "48000", tmp], check=True)
    # two-pass loudnorm
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-threads", "4", "-i", tmp, "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    js = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
    af = (f"loudnorm=I={lufs}:TP=-1.5:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:"
          f"measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true")
    subprocess.run(pre + ["-i", tmp, "-af", af, "-ar", "48000", out_wav], check=True)
    return out_wav


def render_video(tl, assets_dir, tts_dir, out_path, cfg, workers=4, low=False, music=None):
    W, H, fps = tl["W"], tl["H"], tl["fps"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tl_path = out_path.replace(".mp4", "_timeline.json")
    json.dump(tl, open(tl_path, "w"))
    nfr = int(math.ceil(tl["total"] * fps))
    bounds = [round(nfr * k / workers) for k in range(workers + 1)]
    segs = [out_path.replace(".mp4", f"_seg{k}.mp4") for k in range(workers)]
    threads = 1 if workers > 1 else 4
    jobs = [(tl_path, assets_dir, bounds[k], bounds[k + 1], segs[k], W, H, fps, threads, low) for k in range(workers)]
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        # fork on Linux (Kaggle): spawn would re-import the kernel script (no main guard) in every worker
        with mp.get_context("fork" if os.name != "nt" else "spawn").Pool(workers) as pool:
            pool.map(_worker, jobs)
    else:
        _worker(jobs[0])
    print(f"frames rendered in {time.time() - t0:.0f}s", flush=True)
    lst = out_path.replace(".mp4", "_segs.txt")
    open(lst, "w").write("".join(f"file '{os.path.abspath(s)}'\n" for s in segs))
    wav = out_path.replace(".mp4", "_mix.wav")
    build_audio(tl, tts_dir, wav, music=music or cfg.get("music"), music_db=cfg.get("music_db", -22.0), lufs=cfg.get("lufs", -15.0))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "4", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    "-shortest", out_path], check=True)
    for s in segs:
        os.remove(s)
    os.remove(lst)
    print("wrote", out_path, flush=True)
    return out_path
