"""Before/after frames: the same story moments from an earlier render (mp4) and from the current engine.

    python tools/frame_compare.py stories/pip_short.md out/pip_short_C/pip_short_C.mp4 out/v2_preview/before_after.jpg 5,13,26,45,70,84
Renders only the listed frames (small, no Kaggle), using the cached narration for the story's voice.
"""
import os, subprocess, sys
import numpy as np
import skia
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine2 import pipeline, render as R  # noqa: E402


def main(story, before_mp4, out, times, style="C", W=360, ss=1.0):
    times = [float(x) for x in times.split(",")]
    meta, utts = pipeline.utterances(story, pipeline.DEFAULT_VOICE, 1.0)
    tdir, _ = pipeline.collect(meta.get("slug"), utts)
    H = W * 16 // 9
    F = R.Film(story, tdir, {"style": style, "W": W, "H": H, "fps": 30, "voice": "af_heart", "ss": ss})
    s = skia.Surface.MakeRasterN32Premul(F.W, F.H)
    o = skia.Surface.MakeRasterN32Premul(W, H)
    dp = skia.Paint(AntiAlias=True)
    dp.setFilterQuality(skia.FilterQuality.kHigh_FilterQuality)
    rows = []
    for t in times:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", before_mp4, "-frames:v", "1", "-vf", f"scale={W}:{H}",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        before = np.frombuffer(raw, np.uint8).reshape(H, W, 3) if len(raw) == W * H * 3 else np.zeros((H, W, 3), np.uint8)
        F.frame(s.getCanvas(), t, s)
        oc = o.getCanvas()
        oc.save()
        oc.scale(W / F.W, H / F.H)
        oc.drawImage(s.makeImageSnapshot(), 0, 0, dp)
        oc.restore()
        after = o.makeImageSnapshot().toarray()[..., [2, 1, 0]]
        rows.append(np.hstack([before, np.full((H, 6, 3), 20, np.uint8), after]))
    import cv2
    sheet = np.hstack([np.vstack([r, np.full((6, r.shape[1], 3), 20, np.uint8)]) for r in rows[: (len(rows) + 1) // 2]]) if False else None
    grid = np.vstack([np.hstack(rows[i:i + 3] + [np.zeros_like(rows[0])] * (3 - len(rows[i:i + 3]))) for i in range(0, len(rows), 3)])
    cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
