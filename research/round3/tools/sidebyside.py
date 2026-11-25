"""Stack contact sheets (one row per variant) with a label strip: research/round3/compare/<axis>.jpg
usage: python sidebyside.py out.jpg "label 1=path/a_sheet.jpg" "label 2=path/b_sheet.jpg" ...
       python sidebyside.py --crops out.jpg t_sec x0 y0 w h "label=clip.mp4" ...   (1:1 crops of finished clips at time t)"""
import subprocess, sys
from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def rows(out, items, lw=300):
    ims = [(l, Image.open(p).convert("RGB")) for l, p in items]
    W = max(im.width for _, im in ims)
    ims = [(l, im if im.width == W else im.resize((W, int(im.height * W / im.width)))) for l, im in ims]
    H = sum(im.height for _, im in ims)
    S = Image.new("RGB", (W + lw, H), "white"); d = ImageDraw.Draw(S); y = 0; f = font(20)
    for l, im in ims:
        S.paste(im, (lw, y))
        yy = y + 8
        for line in l.split("|"):
            d.text((8, yy), line, fill="black", font=f); yy += 26
        y += im.height
    S.save(out, quality=88); print(out, S.size)


def crops(out, t, x0, y0, w, h, items):
    tiles = []
    for l, p in items:
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", p, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                           capture_output=True, check=True)
        import io
        im = Image.open(io.BytesIO(r.stdout)).convert("RGB").crop((x0, y0, x0 + w, y0 + h))
        d = ImageDraw.Draw(im); d.rectangle([0, 0, w, 30], fill="black"); d.text((6, 4), l, fill="white", font=font(20))
        tiles.append(im)
    S = Image.new("RGB", (w * len(tiles), h), "white")
    for i, im in enumerate(tiles):
        S.paste(im, (i * w, 0))
    S.save(out, quality=92); print(out, S.size)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--crops":
        out, t, x0, y0, w, h = a[1], float(a[2]), *map(int, a[3:7])
        crops(out, t, x0, y0, w, h, [s.split("=", 1) for s in a[7:]])
    else:
        rows(a[0], [s.split("=", 1) for s in a[1:]])
