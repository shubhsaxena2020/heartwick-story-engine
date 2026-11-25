"""Draw OpenPose-18 skeleton frames from Blender's 2D joint export (control video for Wan-VACE repaint)."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
src, out, W, H, NF = sys.argv[1], Path(sys.argv[2]), 480, 832, 81
out.mkdir(parents=True, exist_ok=True)
d = json.load(open(src)); fr = d["frames"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COL = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170),
       (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
for i in range(NF):
    f = fr[round(i * (len(fr) - 1) / (NF - 1))]
    im = Image.new("RGB", (W, H), "black"); dr = ImageDraw.Draw(im)
    P = [(x * W, y * H) for x, y in f["j"]]
    for k, (a, b) in enumerate(LIMBS):
        c = tuple(int(v * 0.6) for v in COL[k])
        dr.line([P[a], P[b]], fill=c, width=6)
    for k, p in enumerate(P):
        dr.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=COL[k])
    im.save(out / f"{i:03d}.png")
print("ok", NF)
