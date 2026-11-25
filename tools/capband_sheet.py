"""Tile the caption band of a video over a time window, with timestamps (for reading caption chunking)."""
import subprocess, sys
import numpy as np
from PIL import Image, ImageDraw

src, start, dur, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
fps = float(sys.argv[5]) if len(sys.argv) > 5 else 4
y0 = int(sys.argv[6]) if len(sys.argv) > 6 else 1260
cw, ch = 360, 73
cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-ss", str(start), "-t", str(dur), "-i", src,
       "-vf", f"fps={fps},crop=1080:220:0:{y0},scale={cw}:{ch}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
raw = subprocess.run(cmd, capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, ch, cw, 3)
cols = 6
rows = (len(fr) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cw, rows * ch), "black")
d = ImageDraw.Draw(sheet)
for i, f in enumerate(fr):
    x, y = (i % cols) * cw, (i // cols) * ch
    sheet.paste(Image.fromarray(f), (x, y))
    d.text((x + 3, y + 2), f"{start + i / fps:.2f}", fill=(255, 255, 0))
sheet.save(out)
print(len(fr), "crops ->", out)
