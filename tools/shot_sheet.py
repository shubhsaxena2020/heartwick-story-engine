"""Quick look at every shot of a story without narration or Kaggle: fakes missing narration timing (2.7 words/s,
silent), renders one small frame per shot (mid-shot) and writes a labelled sheet. Light: ~0.1 s per frame.

    python tools/shot_sheet.py stories/pip_short.md B out/v2_preview/pip_short_B_shots.jpg
"""
import json, os, re, shutil, sys, tempfile
import numpy as np
import skia
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine2 import pipeline, render as R  # noqa: E402


def fake_tts(utts, d, have_dirs):
    import soundfile as sf
    for u in utts:
        src = next((h for h in have_dirs if os.path.exists(os.path.join(h, u["id"] + ".json"))), None)
        if src:
            for ext in (".json", ".wav"):
                shutil.copy(os.path.join(src, u["id"] + ext), d)
            continue
        toks = re.findall(r"[\w'’]+|[^\w\s]", u["text"])
        t, words = 0.25, []
        for tok in toks:
            if re.match(r"[\w'’]", tok):
                words.append({"w": tok, "ws": " ", "s": round(t, 3), "e": round(t + 0.3, 3)})
                t += 0.37
            else:
                words.append({"w": tok, "ws": " ", "s": round(t, 3), "e": round(t, 3)})
                if tok in ".!?":
                    t += 0.25
        dur = t + 0.2
        sf.write(os.path.join(d, u["id"] + ".wav"), np.zeros(int(dur * 24000), np.float32), 24000)
        json.dump({"id": u["id"], "text": u["text"], "dur": round(dur, 3), "words": words}, open(os.path.join(d, u["id"] + ".json"), "w"))


def main(story, style, out, W=270):
    meta, utts = pipeline.utterances(story, pipeline.DEFAULT_VOICE, 1.0)
    slug = meta.get("slug")
    d = tempfile.mkdtemp()
    fake_tts(utts, d, pipeline.tts_dirs(slug))
    F = R.Film(story, d, {"style": style, "W": W, "H": W * 16 // 9, "fps": 30, "voice": "af_heart"})
    H = W * 16 // 9
    s = skia.Surface.MakeRasterN32Premul(W, H)
    tiles = []
    for j, sh in enumerate(F.shots):
        t = (sh["s"] + sh["e"]) / 2
        F.frame(s.getCanvas(), t)
        tiles.append((j, t, s.makeImageSnapshot().toarray()[..., [2, 1, 0]].copy()))
    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    import cv2
    sheet = np.full((rows * (H + 4), cols * (W + 4), 3), 24, np.uint8)
    for k, (j, t, im) in enumerate(tiles):
        r, c = divmod(k, cols)
        cv2.putText(im, f"#{j} {t:.1f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        sheet[r * (H + 4):r * (H + 4) + H, c * (W + 4):c * (W + 4) + W] = im[..., :3]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    rep = F.edit_report()
    print(json.dumps({k: v for k, v in rep.items() if k != "camera"}))
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
