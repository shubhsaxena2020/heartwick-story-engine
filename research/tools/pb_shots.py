"""Shot boundary detection for the Parabrahma reels (structure-based, palette-independent).

Hard cut: normalised cross-correlation (NCC) of consecutive frames (blurred 90x160 gray, caption band + tag masked)
drops below 0.75 while it stays high around it.
Dissolve: a window where frame t is well explained as a linear blend of the frames 0.4-1.0 s before and after,
and the two ends are structurally different (NCC < 0.6).
Usage: python pb_shots.py <research/parabrahma dir>   -> metrics/<id>_shots.json
"""
import glob, json, os, subprocess, sys
import numpy as np
import cv2

W, H, FPS = 90, 160, 30


def decode(path):
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def main(root):
    for mp4 in sorted(glob.glob(os.path.join(root, "media", "*.mp4"))):
        sid = os.path.basename(mp4)[:-4]
        fr = decode(mp4)
        m = np.ones((H, W), bool); m[int(H * .62):int(H * .80)] = False; m[:int(H * .2), :int(W * .65)] = False
        X = np.stack([cv2.GaussianBlur(f, (5, 5), 0)[m] for f in fr])
        X = X - X.mean(1, keepdims=True); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
        n = len(X)
        ncc1 = np.r_[1.0, (X[1:] * X[:-1]).sum(1)]
        cuts = []
        for i in range(1, n):
            ctx = np.r_[ncc1[max(1, i - 5):i], ncc1[i + 1:i + 6]]
            if ncc1[i] < 0.75 and (len(ctx) == 0 or np.median(ctx) > 0.9):
                cuts.append(i)
        # dissolves: blend test
        dis = []
        k = 15
        raw = np.stack([f[m] for f in fr])
        for i in range(k, n - k, 3):
            if any(abs(i - c) < k for c in cuts):
                continue
            a, b, c = raw[i - k], raw[i + k], raw[i]
            if float((X[i - k] * X[i + k]).sum()) > 0.6:
                continue
            d = a - b
            alpha = float(np.clip(((c - b) * d).sum() / ((d * d).sum() + 1e-6), 0, 1))
            err = np.abs(c - (alpha * a + (1 - alpha) * b)).mean()
            ref = min(np.abs(c - a).mean(), np.abs(c - b).mean())
            if 0.15 < alpha < 0.85 and err < 0.45 * ref:
                if dis and i - dis[-1][1] <= 6:
                    dis[-1][1] = i
                else:
                    dis.append([i, i])
        diss = [((s + e) / 2 / FPS, (e - s + 2 * k) / FPS) for s, e in dis]
        bounds = sorted([c / FPS for c in cuts] + [d[0] for d in diss])
        dur = n / FPS
        shots = np.diff([0] + bounds + [dur])
        out = dict(id=sid, duration=round(dur, 2), hard_cuts=[round(c / FPS, 2) for c in cuts],
                   dissolves=[(round(t, 2), round(l, 2)) for t, l in diss], n_shots=len(shots),
                   shots_per_min=round(len(shots) / dur * 60, 1), shot_median=round(float(np.median(shots)), 2),
                   shot_mean=round(float(shots.mean()), 2), shot_p10=round(float(np.percentile(shots, 10)), 2),
                   shot_p90=round(float(np.percentile(shots, 90)), 2), shot_max=round(float(shots.max()), 2))
        json.dump(out, open(os.path.join(root, "metrics", sid + "_shots.json"), "w"), indent=1)
        print(sid, {k: out[k] for k in ("n_shots", "shots_per_min", "shot_median", "shot_p10", "shot_p90", "shot_max")},
              len(cuts), "cuts", len(diss), "dissolves", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
