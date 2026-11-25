"""Small preview strips straight from the engine (no Kaggle, light): frames at a fixed spacing.

    python tools/preview_frames.py stories/walk_test.md out/v2_preview/walk_B.jpg --art c --style C --t 0:3.5:0.25 [--W 270] [--cols 8]
"""
import argparse, os, sys
import numpy as np
import skia
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine2 import pipeline, render as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("story")
    ap.add_argument("out")
    ap.add_argument("--art", default="c")
    ap.add_argument("--style", default="C")
    ap.add_argument("--t", default="0:4:0.25")
    ap.add_argument("--W", type=int, default=270)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--fps", type=int, default=60)
    a = ap.parse_args()
    meta, utts = pipeline.utterances(a.story, pipeline.DEFAULT_VOICE, 1.0)
    tdir, missing = pipeline.collect(meta.get("slug"), utts)
    if missing:
        sys.exit(f"{len(missing)} narration lines missing")
    W, H = a.W, a.W * 16 // 9
    F = R.Film(a.story, tdir, {"style": a.style, "art": a.art, "W": W, "H": H, "fps": a.fps, "voice": "af_heart"})
    t0, t1, dt = (float(x) for x in a.t.split(":"))
    s = skia.Surface.MakeRasterN32Premul(F.W, F.H)
    tiles = []
    import cv2
    for t in np.arange(t0, t1 + 1e-6, dt):
        F.frame(s.getCanvas(), float(t), s)
        im = s.makeImageSnapshot().toarray()[..., [2, 1, 0]].copy()
        cv2.putText(im, f"{t:.2f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    rows = [np.hstack(tiles[i:i + a.cols] + [np.zeros_like(tiles[0])] * (a.cols - len(tiles[i:i + a.cols]))) for i in range(0, len(tiles), a.cols)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    rep = F.motion_qa()
    print(a.out, rep)


if __name__ == "__main__":
    main()
