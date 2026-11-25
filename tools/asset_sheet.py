"""Grid of asset thumbnails (plates, depth, cut-outs on a checkerboard) for visual review.
usage: python tools/asset_sheet.py <assets_dir> <out.jpg> [prefix ...]"""
import os, sys
from PIL import Image, ImageDraw

d, out = sys.argv[1], sys.argv[2]
prefixes = tuple(sys.argv[3:]) or ("bg_", "pose_", "master_")
files = sorted(f for f in os.listdir(d) if f.endswith(".png") and f.startswith(prefixes))
tw, th = 240, 400
cols = 8
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, max(1, rows) * (th + 18)), (25, 25, 25))
dr = ImageDraw.Draw(sheet)
chk = Image.new("RGB", (tw, th))
cd = ImageDraw.Draw(chk)
for y in range(0, th, 16):
    for x in range(0, tw, 16):
        cd.rectangle([x, y, x + 15, y + 15], fill=(200, 200, 200) if (x // 16 + y // 16) % 2 else (150, 150, 150))
for i, f in enumerate(files):
    im = Image.open(os.path.join(d, f))
    im.thumbnail((tw, th))
    x, y = (i % cols) * tw, (i // cols) * (th + 18)
    if im.mode == "RGBA":
        bg = chk.copy(); bg.paste(im, ((tw - im.width) // 2, (th - im.height) // 2), im); sheet.paste(bg, (x, y))
    else:
        sheet.paste(im.convert("RGB"), (x + (tw - im.width) // 2, y + (th - im.height) // 2))
    dr.text((x + 3, y + th + 2), f[:-4][:34], fill=(255, 255, 0))
sheet.save(out, quality=88)
print(len(files), "->", out)
