"""Adapter: shots4.json + timeline -> living-still jobs for engine4.la.render_la (Kaggle CPU kernels)."""
import os

from .la import render_la

AMOUNT = {"push_in": 0.08, "pull_out": 0.07, "pan_left": 0.07, "pan_right": 0.07, "tilt_up": 0.06, "tilt_down": 0.06, "drift": 0.05}


def jobs_for(shots4, tl, keys_dir, clips_dir=None, only=None, made=None):
    dur = {t["id"]: round(t["t1"] - t["t0"], 3) for t in tl}
    out = []
    for s in shots4["shots"]:
        if only is not None and s["id"] not in only:
            continue
        if only is None and s["class"] == "A":
            continue                               # A shots are rendered after their clips (clip + tail)
        fx = [f for f in (s.get("fx") or []) if f in ("dust", "leaves", "light", "snow", "rain")]   # steam needs a point
        j = {"id": s["id"], "image": os.path.join(keys_dir, s["master"] + ".png"), "dur": dur[s["id"]],
             "crop": s.get("crop_box"), "cam": {"type": s.get("cam", "push_in"), "amount": AMOUNT.get(s.get("cam"), 0.07)},
             "fx": fx, "char_motion": s.get("char_motion") or "breath"}
        if s["class"] == "A" and clips_dir and s["id"] in (made or ()):
            j["tail_of"] = os.path.join(clips_dir, s["id"] + ".mp4")
            j["crop"] = None                       # the clip is already the full frame
        out.append(j)
    return out


def render(jobs, out_dir, tag, n_kernels=3):
    r = render_la(jobs, out_dir, n_kernels=n_kernels, tag=tag)
    missing = [j["id"] for j in jobs if not (r.get(j["id"]) or {}).get("mp4")]
    return {"missing": missing, "kernels": sorted((r.get("_kernels") or {}).keys()), "raw": {k: v for k, v in r.items() if k != "_kernels"}}
