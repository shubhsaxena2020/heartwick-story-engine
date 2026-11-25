"""Mouth lock: keep a narrated (non-speaking) character's mouth closed in an AI clip, as a deterministic post-pass.

Frame 0 of an I2V clip is the keyframe, where the mouth is closed. For every later frame we track the head with
sparse optical flow (corners inside the head box but outside the mouth box), fit a similarity transform frame0->frame_i,
warp frame 0's lower-face patch with it and blend it in with a feathered ellipse. Head turns, nods and drift are kept;
only the lips stop flapping. OpenCV only, no ML models; works on raw 16-24 fps clips before RIFE/ESRGAN.
usage: python mouth_lock.py in.mp4 out.mp4 --head x0 y0 x1 y1 --mouth x0 y0 x1 y1 [--fps 16]
"""
import argparse, subprocess

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--head", nargs=4, type=float, required=True)
ap.add_argument("--mouth", nargs=4, type=float, required=True)
ap.add_argument("--fps", type=float, default=16)
ap.add_argument("--feather", type=float, default=0.35, help="feather width as a fraction of the mouth box")
a = ap.parse_args()

cap = cv2.VideoCapture(a.src); fr = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    fr.append(f)
H, W = fr[0].shape[:2]
bx = lambda b: (int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))
hx0, hy0, hx1, hy1 = bx(a.head); mx0, my0, mx1, my1 = bx(a.mouth)
# feathered elliptical mask around the mouth box (a bit larger than the box)
cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2; rx, ry = (mx1 - mx0) * 0.75, (my1 - my0) * 0.9
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
mask0 = np.clip((1.0 - d) / a.feather + 1.0, 0, 1)   # 1 inside, ramps to 0 over the feather band
mask0 = cv2.GaussianBlur(mask0, (0, 0), 1.5)

g0 = cv2.cvtColor(fr[0], cv2.COLOR_BGR2GRAY)
roi = np.zeros_like(g0); roi[hy0:hy1, hx0:hx1] = 255
roi[int(cy - ry * 1.3):int(cy + ry * 1.3), int(cx - rx * 1.3):int(cx + rx * 1.3)] = 0   # no corners on the lips
p0 = cv2.goodFeaturesToTrack(g0, 80, 0.01, 4, mask=roi)
out = [fr[0]]; prev_g, prev_p, keep = g0, p0, np.ones(len(p0), bool)
pts0 = p0.copy()
M = np.float32([[1, 0, 0], [0, 1, 0]])
for i in range(1, len(fr)):
    g = cv2.cvtColor(fr[i], cv2.COLOR_BGR2GRAY)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, prev_p, None, winSize=(21, 21), maxLevel=3)
    keep &= st.ravel().astype(bool)
    if keep.sum() >= 6:
        M_new, inl = cv2.estimateAffinePartial2D(pts0[keep], p1[keep], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M_new is not None:
            M = M_new.astype(np.float32)
    patch = cv2.warpAffine(fr[0], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask0, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    # match the patch's brightness to the current frame around the mouth (light changes over the clip)
    ring = (m[..., 0] > 0.05) & (m[..., 0] < 0.6)
    if ring.sum() > 50:
        gain = (fr[i][ring].astype(np.float32).mean(0) + 1) / (patch[ring].astype(np.float32).mean(0) + 1)
        patch = np.clip(patch.astype(np.float32) * gain, 0, 255)
    o = (patch.astype(np.float32) * m + fr[i].astype(np.float32) * (1 - m)).round().astype(np.uint8)
    out.append(o)
    prev_g, prev_p = g, p1
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
                        "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", a.dst], stdin=subprocess.PIPE)
for o in out:
    enc.stdin.write(o.tobytes())
enc.stdin.close(); enc.wait()
print("tracked points kept:", int(keep.sum()), "of", len(keep))
