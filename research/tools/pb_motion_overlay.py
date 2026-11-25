"""Camera-compensated 'what moved on its own' overlay for a shot window.

For each consecutive frame pair: dense flow -> global similarity fit (RANSAC) -> residual flow = independent motion.
Accumulates residual magnitude per pixel, then paints it (red) over the middle frame, and writes a timeline of
residual motion energy per 0.1 s so gesture timing (pose changes) can be read off.
Usage: python pb_motion_overlay.py <mp4> <t0> <t1> <out_prefix>
"""
import json, subprocess, sys
import numpy as np
import cv2

W, H, FPS = 360, 640, 30


def main(mp4, t0, t1, out):
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-ss", str(t0), "-to", str(t1), "-i", mp4, "-vf", f"fps={FPS},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    fr = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, np.uint8).reshape(-1, H, W, 3)
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr]
    acc = np.zeros((H, W), np.float32)
    ys, xs = np.mgrid[0:H:6, 0:W:6]
    P = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    energy = []
    cap = np.zeros((H, W), bool); cap[int(H * .64):int(H * .78)] = True; cap[:int(H * .2), :int(W * .7)] = True
    for i in range(1, len(g)):
        fl = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 4, 21, 3, 7, 1.5, 0)
        V = fl[::6, ::6].reshape(-1, 2)
        A, _ = cv2.estimateAffinePartial2D(P, P + V, method=cv2.RANSAC, ransacReprojThreshold=0.8)
        if A is None:
            energy.append(0.0); continue
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        pred = np.stack([A[0, 0] * gx + A[0, 1] * gy + A[0, 2] - gx, A[1, 0] * gx + A[1, 1] * gy + A[1, 2] - gy], -1)
        res = np.linalg.norm(fl - pred, axis=2)
        res[cap] = 0
        res[res < 0.6] = 0
        acc += res
        energy.append(float(res.mean()))
    mid = fr[len(fr) // 2].astype(np.float32)
    a = np.clip(acc / (np.percentile(acc[acc > 0], 98) + 1e-6 if (acc > 0).any() else 1), 0, 1)
    over = mid * (1 - 0.75 * a[..., None]) + np.array([255, 30, 30], np.float32) * (0.75 * a[..., None])
    side = np.concatenate([mid, over], 1).astype(np.uint8)
    cv2.imwrite(out + ".jpg", cv2.cvtColor(side, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    e = np.array(energy)
    per = [round(float(e[k:k + 3].mean()), 3) for k in range(0, len(e), 3)]
    json.dump(dict(mp4=mp4, t0=t0, t1=t1, energy_per_0_1s=per), open(out + ".json", "w"))
    print(out, "peaks at", [round(t0 + 0.1 * k, 1) for k in np.argsort(per)[-8:][::-1]])


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
