"""Render the engine v2 style tests (B vector cel, C paper theatre) at 1080x1920/30 fps on a Kaggle CPU kernel.
Narration: the four cached Kokoro af_heart lines of stories/style_test.md.

    python tools/style_tests.py            -> out/style_tests/style_B.mp4, style_C.mp4 + _sheet.jpg + _report.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine2 import kaggle2, render as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORY = "stories/style_test.md"
TTS = os.path.join(ROOT, "cache", "pip", "assets")


def main():
    cfg = dict(W=1080, H=1920, fps=30, end_card_s=1.8, dissolve_chapters=False, cta="Follow for more Heartwick stories")
    F = R.Film(os.path.join(ROOT, STORY), TTS, dict(cfg, style="B"))
    utts = sorted({a["id"] for a in F.audio})
    job = {"renders": [{"story": STORY, "out": f"style_{s}", "cfg": dict(cfg, style=s), "workers": 4} for s in ("B", "C")]}
    dest = os.path.join(ROOT, "out", "style_tests")
    state = kaggle2.run(job, [STORY], "heartwick-v2-style-tests", dest, tts_dir=TTS, utt_ids=utts)
    print("kaggle:", state)
    for s in ("B", "C"):
        p = os.path.join(dest, f"style_{s}_report.json")
        if os.path.exists(p):
            r = json.load(open(p))
            print(s, {k: r.get(k) for k in ("shots_per_min", "cuts_in_pauses", "character_motion_seconds_pct", "s_per_frame")},
                  r.get("motion", {}).get("motion_seconds_pct"), r.get("audio"))


if __name__ == "__main__":
    main()
