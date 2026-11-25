"""Reverse-engineer pacing, captions, motion and audio of the reference reel.

Light CPU: decodes a 270x480 proxy at 10 fps with -threads 2 and BelowNormal priority.
Output: analysis/reference_metrics.json
"""
import json, subprocess, sys, os
import numpy as np
import cv2
import soundfile as sf
import pyloudnorm as pyln

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reference", "ref_reel.mp4")
WAV = os.path.join(ROOT, "reference", "ref_audio.wav")
OUT = os.path.join(ROOT, "analysis", "reference_metrics.json")
W, H, FPS = 270, 480, 10

try:
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def decode_proxy():
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", REF, "-vf", f"fps={FPS},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        import psutil
        psutil.Process(p.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    raw = p.stdout.read()
    p.wait()
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W)


def main():
    fr = decode_proxy()
    n = len(fr)
    dur = n / FPS
    # ---- shot cuts: full-frame histogram distance (excluding caption band)
    mask = np.ones((H, W), bool); mask[int(H * .62):int(H * .80)] = False
    def hist(f):
        h = cv2.calcHist([f], [0], mask.astype(np.uint8), [32], [0, 256]).ravel(); return h / h.sum()
    hs = [hist(f) for f in fr]
    d = np.array([0] + [0.5 * np.abs(hs[i] - hs[i - 1]).sum() for i in range(1, n)])
    # ---- motion per frame pair: global affine via phase correlation + scale via ECC-lite
    mot = []
    for i in range(1, n):
        a = fr[i - 1].astype(np.float32); b = fr[i].astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(a, b)
        mot.append((dx, dy, resp, float(np.abs(a - b).mean())))
    mot = np.array(mot)
    # ---- caption band activity: detect bright white text pixels in band
    band = fr[:, int(H * .66):int(H * .78), int(W * .08):int(W * .92)]
    white = (band > 225).mean(axis=(1, 2))
    # caption-change events: big change of white-pixel layout
    bw = (band > 225).astype(np.float32)
    cdiff = np.array([0] + [np.abs(bw[i] - bw[i - 1]).mean() for i in range(1, n)])
    cap_on = white > 0.004
    cap_events = [i / FPS for i in range(1, n) if cdiff[i] > 0.02 and cap_on[i]]
    # merge events closer than 0.25 s
    merged = []
    for t in cap_events:
        if not merged or t - merged[-1] > 0.25:
            merged.append(t)
    # ---- audio
    data, sr = sf.read(WAV)
    if data.ndim > 1:
        mono = data.mean(axis=1)
    else:
        mono = data
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(data)
    peak_db = 20 * np.log10(np.abs(data).max() + 1e-9)
    # short-term RMS (50 ms) to find speech pauses and the music floor in them
    hop = int(sr * 0.05)
    rms = np.array([np.sqrt(np.mean(mono[i:i + hop] ** 2)) for i in range(0, len(mono) - hop, hop)])
    rdb = 20 * np.log10(rms + 1e-9)
    p95 = np.percentile(rdb, 95)
    p10 = np.percentile(rdb, 10)
    # pauses = runs >= 0.3 s below (p95 - 20 dB)
    thr = p95 - 20
    low = rdb < thr
    pauses = []
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            if (j - i) * 0.05 >= 0.3:
                pauses.append((i * 0.05, (j - i) * 0.05, float(np.median(rdb[i:j]))))
            i = j
        else:
            i += 1
    speech_db = float(np.median(rdb[rdb > p95 - 10]))
    floor_in_pauses = float(np.median([p[2] for p in pauses])) if pauses else None
    # spectral check of the floor: music vs silence (spectral flatness inside pauses)
    res = {
        "duration_s": round(dur, 2), "frames_analysed": int(n),
        "histogram_cut_candidates": [round(i / FPS, 1) for i in range(1, n) if d[i] > 0.35],
        "hist_diff_top": sorted([(round(i / FPS, 1), round(float(d[i]), 3)) for i in range(1, n)], key=lambda x: -x[1])[:40],
        "motion": {
            "median_abs_dx_px_per_frame_at_270w": round(float(np.median(np.abs(mot[:, 0]))), 3),
            "median_abs_dy": round(float(np.median(np.abs(mot[:, 1]))), 3),
            "median_mean_absdiff": round(float(np.median(mot[:, 3])), 3),
            "frac_frames_static(<0.6 absdiff)": round(float((mot[:, 3] < 0.6).mean()), 3),
        },
        "captions": {
            "frac_time_caption_visible": round(float(cap_on.mean()), 3),
            "caption_change_events": len(merged),
            "mean_chunk_s": round(float(np.mean(np.diff(merged))), 3) if len(merged) > 2 else None,
            "median_chunk_s": round(float(np.median(np.diff(merged))), 3) if len(merged) > 2 else None,
            "events": [round(t, 2) for t in merged],
        },
        "audio": {
            "integrated_lufs": round(float(lufs), 2), "sample_peak_dbfs": round(float(peak_db), 2),
            "speech_rms_db": round(speech_db, 1), "pause_floor_rms_db": round(floor_in_pauses, 1) if floor_in_pauses else None,
            "music_below_voice_db": round(speech_db - floor_in_pauses, 1) if floor_in_pauses else None,
            "n_pauses_ge_0.3s": len(pauses), "mean_pause_s": round(float(np.mean([p[1] for p in pauses])), 2) if pauses else None,
            "pause_list_first20": [(round(a, 2), round(b, 2)) for a, b, _ in pauses[:20]],
        },
    }
    # per-motion timeline (1 s buckets) for the write-up
    sec = []
    for s in range(int(dur)):
        seg = mot[s * FPS:(s + 1) * FPS]
        if len(seg):
            sec.append((s, round(float(np.mean(seg[:, 0])), 2), round(float(np.mean(seg[:, 1])), 2), round(float(np.mean(seg[:, 3])), 2)))
    res["motion"]["per_second_dx_dy_absdiff"] = sec
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("hist_diff_top",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
