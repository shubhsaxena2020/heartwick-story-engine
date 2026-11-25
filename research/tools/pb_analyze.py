"""Per-reel reverse-engineering of the Parabrahma World reels (research only).

Light CPU: decodes a 180x320 proxy at 30 fps with ffmpeg -threads 2, BelowNormal priority.
For each reel in research/parabrahma/media/*.mp4 writes research/parabrahma/metrics/<id>.json and
a contact sheet (one frame every 2 s) to research/parabrahma/sheets/<id>_sheet.jpg.

Measures
- audio: speech/pause segmentation, pause stats, speech level vs bed level in pauses
- cuts: hard cuts (histogram jump in 1 frame) and dissolves (sustained blend)
- motion: global camera motion (affine fit of dense flow) vs residual LOCAL motion
  (what moves on its own, i.e. character animation), per second
Usage: python -I pb_analyze.py <research/parabrahma dir>
"""
import json, os, subprocess, sys, glob
import numpy as np
import cv2
import soundfile as sf

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    psutil = None

W, H, FPS = 180, 320, 30


def decode(path, w=W, h=H, fps=FPS, color=False):
    pix = "rgb24" if color else "gray"
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-vf", f"fps={fps},scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", pix, "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if psutil:
        try:
            psutil.Process(p.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass
    raw = p.stdout.read(); p.wait()
    c = 3 if color else 1
    a = np.frombuffer(raw, np.uint8)
    return a.reshape(-1, h, w, c) if color else a.reshape(-1, h, w)


def audio_stats(wav):
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x.mean(1)
    hop = int(sr * 0.02)
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    # speech threshold: 12 dB under the 90th percentile of loud frames
    loud = np.percentile(db, 90)
    thr = loud - 14
    sp = db > thr
    # smooth: fill speech gaps < 0.12 s, drop blips < 0.08 s
    def runs(b):
        r, s = [], 0
        for i in range(1, len(b) + 1):
            if i == len(b) or b[i] != b[s]:
                r.append((bool(b[s]), s, i)); s = i
        return r
    b = sp.copy()
    for v, s, e in runs(b):
        if not v and (e - s) * 0.02 < 0.12 and s > 0:
            b[s:e] = True
    pauses = [((e - s) * 0.02, s * 0.02) for v, s, e in runs(b) if not v]
    # trim leading/trailing silence
    inner = [p for p in pauses if p[1] > 0.3 and p[1] + p[0] < len(b) * 0.02 - 0.3]
    pl = np.array([p[0] for p in inner if p[0] >= 0.25])
    speech_db = np.median(db[b])
    bed_db = np.median(db[~b]) if (~b).sum() else None
    spoken = b.sum() * 0.02
    return dict(duration=round(len(x) / sr, 2), speech_rms_db=round(float(speech_db), 1),
                pause_floor_db=round(float(bed_db), 1) if bed_db is not None else None,
                bed_below_speech_db=round(float(speech_db - bed_db), 1) if bed_db is not None else None,
                spoken_seconds=round(float(spoken), 1),
                pauses_ge_0_25=int(len(pl)), pause_mean=round(float(pl.mean()), 2) if len(pl) else None,
                pause_median=round(float(np.median(pl)), 2) if len(pl) else None,
                pause_p90=round(float(np.percentile(pl, 90)), 2) if len(pl) else None,
                long_pauses_ge_0_7=[round(p[1], 2) for p in inner if p[0] >= 0.7],
                pause_list=[(round(p[1], 2), round(p[0], 2)) for p in inner if p[0] >= 0.25])


def video_stats(mp4):
    fr = decode(mp4)
    n = len(fr)
    mask = np.ones((H, W), np.uint8); mask[int(H * .62):int(H * .80)] = 0; mask[:int(H * .18), :int(W * .6)] = 0
    hs = []
    for f in fr:
        h = cv2.calcHist([f], [0], mask, [32], [0, 256]).ravel(); hs.append(h / h.sum())
    hd = np.array([0] + [0.5 * np.abs(hs[i] - hs[i - 1]).sum() for i in range(1, n)])
    # hard cut: single-frame jump that is large and isolated
    cuts = [i for i in range(1, n) if hd[i] > 0.30 and hd[i] > 4 * np.median(hd[max(0, i - 8):i + 8] + 1e-4)]
    # dissolves: windows of 8-40 frames where hist distance per frame is elevated & monotonic blend
    # proxy: distance between frame i-10 and i+10 large while each step small
    dis = []
    for i in range(10, n - 10, 2):
        far = 0.5 * np.abs(hs[i + 10] - hs[i - 10]).sum()
        step = hd[i - 9:i + 11].max()
        if far > 0.35 and step < 0.25:
            if not dis or i - dis[-1][-1] > 6:
                dis.append([i, i])
            else:
                dis[-1][-1] = i
    dissolves = [((a + b) / 2 / FPS, (b - a + 20) / FPS) for a, b in dis]
    # flow: global affine vs residual
    glob_mag, loc_frac, loc_mag, scale_rate, loc_maps = [], [], [], [], []
    cut_set = set(cuts)
    ys, xs = np.mgrid[0:H:4, 0:W:4]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    keep = mask[ys.ravel(), xs.ravel()] > 0
    heat = np.zeros((H // 4 + (H % 4 > 0), W // 4 + (W % 4 > 0)), np.float32)
    for i in range(1, n):
        if i in cut_set:
            glob_mag.append(np.nan); loc_frac.append(np.nan); loc_mag.append(np.nan); scale_rate.append(np.nan); continue
        fl = cv2.calcOpticalFlowFarneback(fr[i - 1], fr[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        v = fl[::4, ::4].reshape(-1, 2)
        P, V = pts[keep], v[keep]
        A, inl = cv2.estimateAffinePartial2D(P, P + V, method=cv2.RANSAC, ransacReprojThreshold=0.6)
        if A is None:
            glob_mag.append(np.nan); loc_frac.append(np.nan); loc_mag.append(np.nan); scale_rate.append(np.nan); continue
        pred = P @ A[:, :2].T + A[:, 2] - P
        res = np.linalg.norm(V - pred, axis=1)
        glob_mag.append(float(np.linalg.norm(pred, axis=1).mean()))
        s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
        scale_rate.append(s - 1)
        moving = res > 0.35  # px at 180 wide (~2 px at 1080)
        loc_frac.append(float(moving.mean())); loc_mag.append(float(res[moving].mean()) if moving.any() else 0.0)
        full = np.zeros(len(v), bool); full[np.where(keep)[0][moving]] = True
        heat += full.reshape(heat.shape)
    g = np.array(glob_mag); lf = np.array(loc_frac); sr_ = np.array(scale_rate)
    per_sec = []
    for s in range(int(n / FPS)):
        sl = slice(s * FPS, (s + 1) * FPS)
        per_sec.append(dict(t=s, cam_px=round(float(np.nanmean(g[sl])) * 6, 2),
                            zoom_pct_per_s=round(float(np.nansum(sr_[sl])) * 100, 2),
                            local_frac=round(float(np.nanmean(lf[sl])), 3)))
    static = float(np.nanmean((g < 0.05) & (lf < 0.01)))
    return fr, dict(frames=n, hard_cuts_s=[round(c / FPS, 2) for c in cuts],
                    dissolves_s=[(round(a, 2), round(b, 2)) for a, b in dissolves],
                    camera_px_per_frame_at1080_median=round(float(np.nanmedian(g)) * 6, 2),
                    local_motion_frac_median=round(float(np.nanmedian(lf)), 3),
                    frac_frames_local_motion_gt_2pct=round(float(np.nanmean(lf > 0.02)), 3),
                    near_static_frac=round(static, 3), per_second=per_sec), heat / max(1, n)


def contact_sheet(mp4, out, every=2.0, cols=8, tw=200):
    fr = decode(mp4, w=tw, h=int(tw * 16 / 9), fps=1 / every, color=True)
    th = fr.shape[1]
    rows = int(np.ceil(len(fr) / cols))
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, f in enumerate(fr):
        r, c = divmod(k, cols)
        im = f.copy()
        cv2.putText(im, f"{k * every:.0f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])


def main(root):
    os.makedirs(os.path.join(root, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(root, "sheets"), exist_ok=True)
    for mp4 in sorted(glob.glob(os.path.join(root, "media", "*.mp4"))):
        sid = os.path.basename(mp4)[:-4]
        outj = os.path.join(root, "metrics", sid + ".json")
        if os.path.exists(outj):
            continue
        a = audio_stats(os.path.join(root, "audio", sid + ".wav"))
        _, v, heat = video_stats(mp4)
        hm = cv2.applyColorMap(np.clip(heat / max(heat.max(), 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(root, "sheets", sid + "_localmotion_heat.png"), cv2.resize(hm, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST))
        json.dump(dict(id=sid, audio=a, video=v), open(outj, "w"), indent=1)
        contact_sheet(mp4, os.path.join(root, "sheets", sid + "_sheet.jpg"))
        print(sid, "done", a["duration"], len(v["hard_cuts_s"]), "cuts", len(v["dissolves_s"]), "dissolves", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
