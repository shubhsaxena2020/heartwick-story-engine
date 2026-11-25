import json, os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def fake_tts(d, uid, text):
    """Write a Kokoro-shaped utterance (wav + word timings, punctuation as separate tokens) without any model."""
    import soundfile as sf
    words, t = [], 0.05
    for w in text.split():
        core = w.rstrip(".,!?\"")
        punct = w[len(core):]
        dur = 0.12 + 0.05 * len(core.strip("\"'"))
        words.append({"w": core, "ws": "" if punct else " ", "s": round(t, 3), "e": round(t + dur, 3)})
        if punct:
            words.append({"w": punct, "ws": " ", "s": round(t + dur, 3), "e": round(t + dur, 3)})
        t += dur + 0.06
    sr = 24000
    n = int((t + 0.1) * sr)
    tt = np.arange(n) / sr
    audio = np.zeros(n, np.float32)
    for x in words:
        if x["w"].strip(".,!?\"'"):
            i0, i1 = int(x["s"] * sr), int(x["e"] * sr)
            audio[i0:i1] = 0.3 * np.sin(2 * np.pi * 180 * tt[i0:i1])
    sf.write(os.path.join(d, uid + ".wav"), audio, sr)
    json.dump({"id": uid, "text": text, "dur": round(n / sr, 3), "words": words}, open(os.path.join(d, uid + ".json"), "w"))


MINI = '''---
slug: mini
title: "Mini"
series_tag: "Mini"
hook: {kicker: "A HEARTWICK STORY", lines: ["A tiny", "test story"], highlight: "test"}
highlight_words: [Nell, mug]
seo:
  titles: ["A Tiny Test Story | Heartwarming Story"]
  summary: "A test."
  hashtags: ["#a", "#b", "#c", "#d"]
  tags: [one, two]
  pinned_comment: "Hi"
  thumbnail_text: ["TINY", "TEST"]
characters:
  nell: {desc: "elderly woman", seed: 1, poses: {stand: "standing"}}
  pip: {desc: "young gray squirrel", seed: 2, poses: {sit: "sitting"}}
locations:
  kitchen: {prompt: "a kitchen", tint: warm}
  yard: {prompt: "a yard", tall: true}
---
## Opening chapter

::: shot bg=kitchen cam=push_in fx=motes chars=nell:stand@0.5,0.95,0.6
Nell made her coffee. [pause 0.5s] She set out *one* mug. [pause 1s]

::: shot bg=yard cam=tilt_down fx=leaves chars=pip:sit@0.5,0.9,0.1:tremble
Under the maple, something small was moving, and the wind was cold, and the leaves were falling everywhere around the yard. [pause 2s]

## Second chapter

::: shot bg=kitchen cam=pull_back
"Oh, little one," she whispered. "Where's your mama?" [pause 1s]
'''


@pytest.fixture
def mini_story(tmp_path):
    p = tmp_path / "mini.md"
    p.write_text(MINI, encoding="utf-8")
    return str(p)
