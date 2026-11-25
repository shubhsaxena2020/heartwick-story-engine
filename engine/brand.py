"""Heartwick brand drawing: emblem, caption chunk sprites, series tag, hook title, end card.
Everything is drawn in code (supersampled PIL) so it is ours and resolution-independent. No candle/flame imagery."""
import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from .style import PALETTE, FONT_CAPTION, FONT_TITLE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fonts = {}


def font(kind, size, weight=900, opsz=None, soft=100):
    key = (kind, size, weight, opsz, soft)
    if key in _fonts:
        return _fonts[key]
    path = os.path.join(ROOT, FONT_CAPTION if kind == "caption" else FONT_TITLE)
    f = ImageFont.truetype(path, size)
    try:
        if kind == "caption":
            f.set_variation_by_axes([weight])
        else:  # Fraunces axes: opsz, wght, SOFT, WONK
            f.set_variation_by_axes([opsz or min(144, max(9, size)), weight, soft, 0])
    except Exception:
        pass
    _fonts[key] = f
    return f


def _blob(w, h, color=PALETTE["night_pine"], alpha=150, blur=None):
    """Soft dark elliptical shadow blob (reference mechanic, our warm-dark tint)."""
    pad = int(h * 0.9)
    im = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), color + (0,))
    a = Image.new("L", im.size, 0)
    ImageDraw.Draw(a).ellipse([pad, pad, pad + w, pad + h], fill=alpha)
    a = a.filter(ImageFilter.GaussianBlur(blur or h * 0.32))
    im.putalpha(a)
    return im, pad


def caption_sprite(words, hl_index, scale=1.0):
    """words: list of display strings; hl_index: index of highlighted word or -1. Returns RGBA PIL image."""
    size = int(66 * scale)
    f = font("caption", size, 900)
    space = f.getlength(" ")
    widths = [f.getlength(w) for w in words]
    tw = int(sum(widths) + space * (len(words) - 1))
    asc, desc = f.getmetrics()
    th = asc + desc
    blob, pad = _blob(int(tw * 1.18 + 40 * scale), int(th * 1.55), alpha=140)
    W, H = blob.size
    im = blob.copy()
    txt = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(txt)
    x = (W - tw) / 2
    y = (H - th) / 2 - 2 * scale
    for i, (w, ww) in enumerate(zip(words, widths)):
        col = PALETTE["ember"] if i == hl_index else PALETTE["cream"]
        d.text((x, y), w, font=f, fill=col + (255,))
        x += ww + space
    # crisp soft drop shadow for legibility on bright plates
    sh = txt.split()[3].filter(ImageFilter.GaussianBlur(3 * scale))
    shadow = Image.new("RGBA", im.size, (20, 12, 8, 0)); shadow.putalpha(sh.point(lambda v: int(v * 0.55)))
    im = Image.alpha_composite(im, shadow.transform(im.size, Image.AFFINE, (1, 0, -2 * scale, 0, 1, -3 * scale)))
    return Image.alpha_composite(im, txt)


def series_tag(story_label, scale=1.0):
    f1 = font("caption", int(30 * scale), 900)
    f2 = font("title", int(48 * scale), 600, opsz=48)
    txt1 = "HEARTWICK STORIES"
    spacing = 3.2 * scale
    w1 = sum(f1.getlength(c) + spacing for c in txt1)
    w2 = f2.getlength(story_label)
    W, H = int(max(w1, w2) + 40 * scale), int(124 * scale)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 4 * scale
    # small emblem dot (heart) before the caps line
    for c in txt1:
        d.text((x, 4 * scale), c, font=f1, fill=PALETTE["cream"] + (225,))
        x += f1.getlength(c) + spacing
    d.text((2 * scale, 46 * scale), story_label, font=f2, fill=PALETTE["honey"] + (255,))
    d.line([(4 * scale, 42 * scale), (min(w1, 140 * scale), 42 * scale)], fill=PALETTE["ember"] + (240,), width=max(2, int(3 * scale)))
    a = im.split()[3]
    shadow = Image.new("RGBA", im.size, PALETTE["night_pine"] + (0,))
    shadow.putalpha(a.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(6 * scale)).point(lambda v: min(255, int(v * 1.1))))
    return Image.alpha_composite(Image.alpha_composite(shadow, shadow), im)


def hook_sprite(kicker, lines, highlight, scale=1.0, width=1080):
    fk = font("caption", int(28 * scale), 800)
    ft = font("title", int(88 * scale), 600, opsz=96)
    lh = int(98 * scale)
    H = int(60 * scale + lh * len(lines) + 60 * scale)
    W = width  # already in output pixels
    blob, pad = _blob(int(W * 0.86), int(H * 0.8), alpha=120)
    im = blob.resize((W, int(H + 2 * pad * 0.5)))
    d = ImageDraw.Draw(im)
    y0 = int(im.height / 2 - (H / 2) + 20 * scale)
    sp = 4 * scale
    kw = sum(fk.getlength(c) + sp for c in kicker)
    x = (W - kw) / 2
    for c in kicker:
        d.text((x, y0), c, font=fk, fill=PALETTE["honey"] + (255,)); x += fk.getlength(c) + sp
    y = y0 + int(44 * scale)
    for line in lines:
        parts = line.split(" ")
        widths = [ft.getlength(p) for p in parts]
        spc = ft.getlength(" ")
        lw = sum(widths) + spc * (len(parts) - 1)
        x = (W - lw) / 2
        for p, pw in zip(parts, widths):
            col = PALETTE["ember"] if p.strip(".,!?").lower() == (highlight or "").lower() else PALETTE["cream"]
            d.text((x, y), p, font=ft, fill=col + (255,))
            x += pw + spc
        y += lh
    return im


def emblem(size=360, progress=1.0):
    """Open storybook whose page-curls form a heart, maple-leaf bookmark, dotted ring. Drawn at 4x and downsampled."""
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx, cy = S / 2, S / 2
    cream, ember, honey, maple = PALETTE["cream"], PALETTE["ember"], PALETTE["honey"], PALETTE["maple"]
    # dotted ring
    r = S * 0.46
    n = 72
    for i in range(int(n * progress)):
        a = -math.pi / 2 + 2 * math.pi * i / n
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        rr = S * (0.006 if i % 3 else 0.010)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=honey + (230,))
    # heart (two lobes + point), formed above the book
    hs = S * 0.20
    hx, hy = cx, cy - S * 0.07
    heart = []
    for k in range(200):
        t = 2 * math.pi * k / 200
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        heart.append((hx + x * hs / 16, hy - y * hs / 16))
    d.polygon(heart, fill=ember + (255,))
    # inner highlight on heart
    sx, sy, sr = hx - hs * 0.52, hy - hs * 0.55, hs * 0.16
    d.ellipse([sx - sr, sy - sr * 0.7, sx + sr, sy + sr * 0.7], fill=(255, 214, 170, 230))
    # open book: two pages as curved quads
    bw, bh = S * 0.30, S * 0.12
    by = cy + S * 0.17
    def page(sign):
        pts = []
        for k in range(41):
            u = k / 40
            x = cx + sign * bw * u
            y = by - bh * 0.55 - math.sin(u * math.pi) * bh * 0.25 + u * u * bh * 0.15
            pts.append((x, y))
        for k in range(40, -1, -1):
            u = k / 40
            x = cx + sign * bw * u
            y = by + bh * 0.25 - math.sin(u * math.pi) * bh * 0.12 + u * u * bh * 0.05
            pts.append((x, y))
        return pts
    d.polygon(page(-1), fill=cream + (255,))
    d.polygon(page(1), fill=cream + (255,))
    for k in range(1, 4):  # page lines
        for sign in (-1, 1):
            pts = [(cx + sign * bw * u, by - bh * 0.42 + k * bh * 0.16 - math.sin(u * math.pi) * bh * 0.2) for u in np.linspace(0.12, 0.85, 20)]
            d.line(pts, fill=(200, 170, 140, 200), width=int(S * 0.006))
    d.line([(cx, by - bh * 0.6), (cx, by + bh * 0.32)], fill=(170, 130, 100, 255), width=int(S * 0.008))
    # maple-leaf bookmark ribbon hanging from the spine
    rib = [(cx + S * 0.02, by + bh * 0.2), (cx + S * 0.055, by + bh * 0.2), (cx + S * 0.055, by + S * 0.16), (cx + S * 0.0375, by + S * 0.14), (cx + S * 0.02, by + S * 0.16)]
    d.polygon(rib, fill=maple + (255,))
    return im.resize((size, size), Image.LANCZOS)


def wordmark(scale=1.0):
    f = font("title", int(92 * scale), 600, opsz=96)
    f2 = font("caption", int(26 * scale), 800)
    t1 = "Heartwick"; t2 = "S T O R I E S"
    w = int(f.getlength(t1) + 20 * scale)
    W = max(w, int(f2.getlength(t2) + 20))
    im = Image.new("RGBA", (W, int(160 * scale)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text(((W - f.getlength(t1)) / 2, 0), t1, font=f, fill=PALETTE["cream"] + (255,))
    d.text(((W - f2.getlength(t2)) / 2, int(112 * scale)), t2, font=f2, fill=PALETTE["honey"] + (255,))
    return im


def pill(text, scale=1.0):
    f = font("caption", int(34 * scale), 850)
    tw = f.getlength(text)
    W, H = int(tw + 70 * scale), int(74 * scale)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H // 2, fill=PALETTE["ember"] + (255,))
    d.text(((W - tw) / 2, H * 0.17), text, font=f, fill=PALETTE["night_pine"] + (255,))
    return im


class EndCard:
    """Animated end card. frame(t) -> float32 RGB (H,W,3) in 0..1."""

    def __init__(self, W, H, cta, tagline="Stories that stay with you", seed=7):
        self.W, self.H = W, H
        s = W / 1080
        self.s = s
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        g = (yy / H)[..., None]
        top = np.array(PALETTE["night_pine"], np.float32) / 255
        bot = np.array(PALETTE["dusk_teal"], np.float32) / 255 * 0.8
        base = top * (1 - g) + bot * g
        glow = np.exp(-(((xx - W / 2) / (W * 0.45)) ** 2 + ((yy - H * 0.40) / (H * 0.22)) ** 2))[..., None]
        self.bg = np.clip(base + glow * np.array([0.22, 0.13, 0.05], np.float32), 0, 1)
        self.emb = emblem(int(380 * s))
        self.wm = wordmark(s)
        self.tag = self._text(tagline, int(34 * s))
        self.cta = pill(cta, s)
        rng = np.random.default_rng(seed)
        self.motes = rng.random((70, 4)).astype(np.float32)  # x, y, size, phase

    def _text(self, t, size):
        f = font("title", size, 400, opsz=size)
        im = Image.new("RGBA", (int(f.getlength(t) + 10), int(size * 1.5)), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((5, 0), t, font=f, fill=PALETTE["cream"] + (220,))
        return im

    def frame(self, t):
        W, H, s = self.W, self.H, self.s
        canvas = Image.fromarray((self.bg * 255).astype(np.uint8)).convert("RGBA")
        over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        for x, y, sz, ph in self.motes:
            yy = (y - t * 0.03 * (0.5 + sz)) % 1.0
            a = int(120 + 100 * math.sin(t * 2.2 + ph * 6.28))
            r = (1.5 + 3.5 * sz) * s
            xx = (x + 0.01 * math.sin(t + ph * 9)) * W
            d.ellipse([xx - r, yy * H - r, xx + r, yy * H + r], fill=PALETTE["honey"] + (max(0, a),))
        over = over.filter(ImageFilter.GaussianBlur(1.2 * s))
        canvas = Image.alpha_composite(canvas, over)

        def ease(a, b):
            u = min(1, max(0, (t - a) / (b - a)))
            return 1 - (1 - u) ** 3

        k = ease(0.0, 0.9)
        sc = 0.85 + 0.15 * k + 0.02 * t / 4
        emb = self.emb.resize((max(1, int(self.emb.width * sc)), max(1, int(self.emb.height * sc))), Image.LANCZOS)
        emb.putalpha(emb.split()[3].point(lambda v: int(v * k)))
        canvas.alpha_composite(emb, (int(W / 2 - emb.width / 2), int(H * 0.34 - emb.height / 2)))
        for img, y, a0 in ((self.wm, 0.47, 0.35), (self.tag, 0.565, 0.6), (self.cta, 0.66, 0.9)):
            kk = ease(a0, a0 + 0.6)
            im2 = img.copy(); im2.putalpha(im2.split()[3].point(lambda v: int(v * kk)))
            canvas.alpha_composite(im2, (int(W / 2 - img.width / 2), int(H * y + (1 - kk) * 24 * s)))
        return np.asarray(canvas.convert("RGB"), np.float32) / 255


if __name__ == "__main__":
    out = os.path.join(ROOT, "brand")
    emblem(1024).save(os.path.join(out, "emblem_1024.png"))
    ec = EndCard(1080, 1920, "Follow for Part 2")
    Image.fromarray((ec.frame(3.0) * 255).astype(np.uint8)).save(os.path.join(out, "endcard_preview.png"))
    c = caption_sprite(["She", "set", "out", "one", "mug."], 3)
    bg = Image.new("RGBA", (1080, 400), (200, 170, 140, 255)); bg.alpha_composite(c, ((1080 - c.width) // 2, 60))
    bg.alpha_composite(series_tag("Pip · Part 1"), (64, 250))
    bg.save(os.path.join(out, "caption_preview.png"))
    h = hook_sprite("A HEARTWICK STORY", ["She found him", "after the storm"], "storm")
    bg2 = Image.new("RGBA", (1080, 600), (90, 110, 140, 255)); bg2.alpha_composite(h, (0, 0)); bg2.save(os.path.join(out, "hook_preview.png"))
    print("ok")
