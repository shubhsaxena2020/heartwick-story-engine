"""Motion audit for a finished episode or a single clip (no models, numpy + OpenCV only).

For every shot span (timeline.json) or the whole clip, it decodes at low resolution and reports:
  - mean abs frame difference (MAD) between consecutive *unique* source frames, global
  - fraction of frames that are exact/near duplicates of the previous one (held frames)
  - longest run of near-static frames in seconds ("freeze")
  - optical-flow magnitude (Farneback) median and p95, global and in the face/mouth box if given
usage: python motion_audit.py VIDEO [--timeline timeline.json] [--fps 12] [--out report.json]
"""
import argparse, json, subprocess

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--timeline")
ap.add_argument("--fps", type=float, default=12.0)
ap.add_argument("--w", type=int, default=270)
ap.add_argument("--out")
ap.add_argument("--static_thr", type=float, default=0.35, help="mean |diff| (0-255) below which a frame counts as static")
a = ap.parse_args()

probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "json", a.video], capture_output=True, text=True).stdout)
W0, H0 = probe["streams"][0]["width"], probe["streams"][0]["height"]
w = a.w; h = int(round(H0 * w / W0 / 2) * 2)
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.video, "-vf", f"fps={a.fps},scale={w}:{h}", "-f", "rawvideo",
                      "-pix_fmt", "gray", "-"], capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w)
n = len(fr)
mad = np.zeros(n); flow = np.zeros(n); res = np.zeros(n)
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
X = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w, np.float32)], 1)
for i in range(1, n):
    a0, a1 = fr[i - 1].astype(np.float32), fr[i].astype(np.float32)
    mad[i] = np.abs(a1 - a0).mean()
    f = cv2.calcOpticalFlowFarneback(fr[i - 1], fr[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow[i] = np.median(np.linalg.norm(f, axis=2)) * (W0 / w)   # in source pixels per sample
    # camera-compensated motion: fit a global affine to the flow field, keep the residual (= subject motion)
    F2 = f.reshape(-1, 2)
    A, *_ = np.linalg.lstsq(X[::7], F2[::7], rcond=None)
    r = np.linalg.norm(F2 - X @ A, axis=1)
    res[i] = np.percentile(r, 97) * (W0 / w)


def stats(i0, i1):
    m = mad[max(i0, 1):i1]; fl = flow[max(i0, 1):i1]; rs = res[max(i0, 1):i1]
    if len(m) == 0:
        return {}
    static = m < a.static_thr
    run = best = 0
    for s in static:
        run = run + 1 if s else 0
        best = max(best, run)
    return {"mad_mean": round(float(m.mean()), 3), "mad_p10": round(float(np.percentile(m, 10)), 3),
            "flow_med_px": round(float(np.median(fl)), 2), "flow_p95_px": round(float(np.percentile(fl, 95)), 2),
            "subject_p97_med_px": round(float(np.median(rs)), 2), "subject_still_frac": round(float((rs < 1.0).mean()), 3),
            "static_frac": round(float(static.mean()), 3), "longest_static_s": round(best / a.fps, 2)}


rep = {"video": a.video, "fps_sampled": a.fps, "n": n, "all": stats(0, n), "shots": []}
if a.timeline:
    for t in json.load(open(a.timeline)):
        i0, i1 = int(t["t0"] * a.fps), int(t["t1"] * a.fps)
        rep["shots"].append(dict(id=t["id"], dur=round(t["t1"] - t["t0"], 2), **stats(i0, i1)))
print(json.dumps(rep["all"]))
for s in rep["shots"]:
    print(s)
if a.out:
    json.dump(rep, open(a.out, "w"), indent=1)
