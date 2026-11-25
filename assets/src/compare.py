"""Comparison sheets. before_after.jpg (committed): engine-v2 style-B test frames vs the new assets.
The 3-way sheet with reference frames is written under research/parabrahma/sheets/ (git-ignored: third-party media)."""
import os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PV = os.path.join(REPO, "assets", "previews")
CW, CH = 360, 640


def label(im, text):
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arialbd.ttf", 26)
    except OSError:
        f = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 44], fill=(20, 16, 24))
    d.text((14, 8), text, fill=(255, 214, 150), font=f)


def row(imgs, title):
    r = Image.new("RGB", (CW * len(imgs) + 6 * (len(imgs) - 1), CH + 44), (20, 16, 24))
    label(r, title)
    for i, im in enumerate(imgs):
        r.paste(im.convert("RGB").resize((CW, CH), Image.LANCZOS), (i * (CW + 6), 44))
    return r


def stack(rows, out):
    W = max(r.width for r in rows)
    S = Image.new("RGB", (W, sum(r.height for r in rows) + 8 * (len(rows) - 1)), (20, 16, 24))
    y = 0
    for r in rows:
        S.paste(r, (0, y))
        y += r.height + 8
    S.save(out, quality=88)
    return out


def old_b():
    raw = subprocess.run(["git", "-C", REPO, "show", "HEAD:out/style_tests/B_vs_C.jpg"], capture_output=True).stdout
    tmp = os.path.join(PV, "_old_B_vs_C.jpg")
    open(tmp, "wb").write(raw)
    im = Image.open(tmp)
    im.load()
    im = im.copy()
    os.remove(tmp)
    xs = [0, 366, 732, 1098]
    return [im.crop((x, 54, x + 360, 694)) for x in xs]


if __name__ == "__main__":
    new = [Image.open(os.path.join(PV, f + ".png")) for f in ("maple_yard", "kitchen", "village_autumn", "storm")]
    new2 = [Image.open(os.path.join(PV, f + ".png")) for f in ("night_sky", "castle_court")]
    print(stack([row(old_b(), "BEFORE  engine v2 style-B test (out/style_tests, 21 Nov)"),
                 row(new, "AFTER  rich vector-cel assets through engine v2's renderer"),
                 row(new2 + [Image.open(os.path.join(REPO, "assets", "characters", n, "preview.png")).crop((0, 40, 840, 760)) for n in ("king", "pip")],
                     "AFTER  night sky, castle court; king and Pip part sheets posed by engine v2")],
                os.path.join(PV, "before_after.jpg")))
    ref_dir = os.path.join(REPO, "research", "parabrahma", "frames")
    refs = [Image.open(os.path.join(ref_dir, f"Df1utU8sgp4_t{t}.png")) for t in (18, 74, 104, 130)]
    out = os.path.join(REPO, "research", "parabrahma", "sheets", "art_vs_reference.jpg")
    print(stack([row(refs, "REFERENCE  Parabrahma World (local research copy, not redistributed)"),
                 row(old_b(), "OURS BEFORE  engine v2 style-B test"), row(new, "OURS AFTER  rich assets")], out))
