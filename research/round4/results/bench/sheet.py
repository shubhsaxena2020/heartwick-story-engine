
import glob, os, imageio.v3 as iio
from PIL import Image, ImageDraw
for p in sorted(glob.glob("/kaggle/working/out/clips/*.mp4")) + sorted(glob.glob("/kaggle/working/out/la/*.mp4")):
    fr = list(iio.imiter(p)); n = len(fr); idx = [round(i * (n - 1) / 7) for i in range(8)]
    ims = [Image.fromarray(fr[i]) for i in idx]; w = 240; h = round(ims[0].height * w / ims[0].width)
    S = Image.new("RGB", (w * 8, h + 18), "white"); d = ImageDraw.Draw(S); d.text((4, 2), os.path.basename(p), fill="black")
    for i, im in enumerate(ims): S.paste(im.resize((w, h)), (i * w, 18))
    S.save(p[:-4] + "_sheet.jpg", quality=88)
