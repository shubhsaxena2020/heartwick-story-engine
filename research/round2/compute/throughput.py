# Weekly capacity model for Heartwick. Inputs are the MEASURED per-clip times in measured.json (written by hand from
# the Kaggle/laptop records). A "clip" = one 5 s Wan2.2-A14B generation (81 frames @16 fps, before RIFE/ESRGAN).
import json, os, sys

H = os.path.dirname(os.path.abspath(__file__))
m = json.load(open(os.path.join(H, "measured.json")))

NEED = {"clips_per_video": (12, 18, 24), "attempts_per_kept": 1.4, "videos_per_week": 7}


def week_need(cpv):
    return cpv * NEED["attempts_per_kept"] * NEED["videos_per_week"]


rows = []
for name, s in m["sources"].items():
    clips = s["hours_per_week"] * 3600 / s["wall_s_per_clip"] * s.get("parallel", 1) * s.get("efficiency", 1.0)
    rows.append((name, s["wall_s_per_clip"], s.get("parallel", 1), s["hours_per_week"], round(clips)))
total = sum(r[4] for r in rows)
print(f"{'source':38s} {'wall s/clip':>11s} {'par':>4s} {'h/week':>7s} {'clips/week':>10s}")
for r in rows:
    print(f"{r[0]:38s} {r[1]:11.0f} {r[2]:4d} {r[3]:7.1f} {r[4]:10d}")
print(f"{'TOTAL':38s} {'':11s} {'':4s} {'':7s} {total:10d}")
for cpv in NEED["clips_per_video"]:
    need = week_need(cpv)
    print(f"need for 7 videos at {cpv} clips/video x{NEED['attempts_per_kept']} attempts = {need:.0f} gens/week "
          f"-> capacity covers {total / need:.1f}x ({total / (cpv * NEED['attempts_per_kept']):.1f} videos/week)")
