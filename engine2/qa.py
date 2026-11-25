"""QA on a finished video, measured exactly like the reference (research/tools/pb_analyze.py):
camera-compensated residual flow per frame; a second "has motion beyond the camera" when its mean
local-motion fraction > 1 % (this threshold reproduces the reference's 65-97 % across its six reels).
Also: contact sheets, loudness."""
import json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "tools"))
REF_THRESHOLD = 0.01
REFERENCE = {"motion_seconds_pct": [77, 81, 65, 87, 97, 89], "camera_px_per_frame": [0.4, 0.2, 0.23, 0.28, 0.44, 0.26]}


def motion_density(mp4, t_end=None):
    import pb_analyze as PB
    _, v, _ = PB.video_stats(mp4)
    secs = [p for p in v["per_second"] if p["local_frac"] == p["local_frac"] and (t_end is None or p["t"] < t_end)]
    moving = [p for p in secs if p["local_frac"] > REF_THRESHOLD]
    return {"motion_seconds_pct": round(100 * len(moving) / max(1, len(secs)), 1),
            "seconds_measured": len(secs),
            "local_frac_median": round(float(np.median([p["local_frac"] for p in secs])), 4) if secs else None,
            "camera_px_per_frame_at1080_median": v["camera_px_per_frame_at1080_median"],
            "near_static_frac": v["near_static_frac"], "hard_cuts_s": v["hard_cuts_s"], "dissolves_s": v["dissolves_s"],
            "per_second": secs}


def contact_sheet(mp4, out, every=0.5, cols=8, tw=216, t_end=None, t0=0.0):
    """Frames every `every` s with timestamps (what we LOOK at before calling a test done)."""
    import cv2
    cmd = ["ffmpeg", "-v", "error", "-i", mp4, "-vf", f"fps={1 / every},scale={tw}:{tw * 16 // 9}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    th = tw * 16 // 9
    fr = np.frombuffer(raw, np.uint8).reshape(-1, th, tw, 3)
    if t_end:
        fr = fr[: int(t_end / every) + 1]
    rows = int(np.ceil(len(fr) / cols))
    sheet = np.full((rows * (th + 4), cols * (tw + 4), 3), 24, np.uint8)
    for k, f in enumerate(fr):
        r, c = divmod(k, cols)
        im = f.copy()
        cv2.putText(im, f"{t0 + k * every:.2f}s", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        sheet[r * (th + 4):r * (th + 4) + th, c * (tw + 4):c * (tw + 4) + tw] = im
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    return out


def loudness(mp4):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", mp4, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = r.stderr
    out = {}
    for key, tag in (("lufs", "I:"), ("true_peak_db", "Peak:")):
        i = txt.rfind(tag)
        if i >= 0:
            try:
                out[key] = float(txt[i + len(tag):].split()[0])
            except ValueError:
                pass
    return out


if __name__ == "__main__":
    mp4 = sys.argv[1]
    d = motion_density(mp4)
    d.pop("per_second")
    print(json.dumps(d, indent=1))
