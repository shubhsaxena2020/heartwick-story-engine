"""Mouth-motion audit for a clip (numpy + OpenCV optical flow, no ML models).

Given a head box and a mouth box (normalised x0,y0,x1,y1 on the first frame), it tracks the head with the median flow
of the head box and measures the mouth box's *residual* motion (flow minus the head's own motion), per frame.
A talking mouth shows up as repeated residual spikes; a closed mouth stays near the noise floor.
Also writes a strip of mouth crops (every 4th frame) so a human can look.
usage: python mouth_audit.py clip.mp4 --head x0 y0 x1 y1 --mouth x0 y0 x1 y1 [--strip out.jpg]
"""
import argparse, json, subprocess

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--head", nargs=4, type=float, required=True)
ap.add_argument("--mouth", nargs=4, type=float, required=True)
ap.add_argument("--strip")
a = ap.parse_args()
cap = cv2.VideoCapture(a.video)
fr = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    fr.append(f)
H, W = fr[0].shape[:2]
box = lambda b: (int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))
hx0, hy0, hx1, hy1 = box(a.head); mx0, my0, mx1, my1 = box(a.mouth)
off = np.zeros(2)   # tracked offset of the head since frame 0
res, crops = [], []
g0 = cv2.cvtColor(fr[0], cv2.COLOR_BGR2GRAY)
for i in range(1, len(fr)):
    g1 = cv2.cvtColor(fr[i], cv2.COLOR_BGR2GRAY)
    f = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 9, 3, 5, 1.1, 0)
    ox, oy = int(round(off[0])), int(round(off[1]))
    hb = f[max(0, hy0 + oy):hy1 + oy, max(0, hx0 + ox):hx1 + ox].reshape(-1, 2)
    head = np.median(hb, 0) if len(hb) else np.zeros(2)
    mb = f[max(0, my0 + oy):my1 + oy, max(0, mx0 + ox):mx1 + ox].reshape(-1, 2)
    r = np.linalg.norm(mb - head, axis=1) if len(mb) else np.zeros(1)
    res.append(float(np.percentile(r, 90)))
    off += head
    if i % 4 == 0:
        pad = 6
        c = fr[i][max(0, my0 + oy - pad):my1 + oy + pad, max(0, mx0 + ox - pad):mx1 + ox + pad]
        if c.size:
            crops.append(cv2.resize(c, (96, int(96 * c.shape[0] / max(1, c.shape[1])))))
    g0 = g1
res = np.array(res)
out = {"video": a.video, "frames": len(fr), "mouth_resid_p90_median": round(float(np.median(res)), 3),
       "mouth_resid_p90_p95": round(float(np.percentile(res, 95)), 3), "spikes_over_1px": int((res > 1.0).sum()),
       "head_drift_px": [round(float(off[0]), 1), round(float(off[1]), 1)]}
print(json.dumps(out))
if a.strip and crops:
    h = max(c.shape[0] for c in crops)
    crops = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for c in crops]
    rows = [np.hstack(crops[k:k + 16]) for k in range(0, len(crops), 16)]
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 2, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(a.strip, np.vstack(rows))
