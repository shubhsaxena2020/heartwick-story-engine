"""Narration pace from Whisper word timestamps (Parabrahma ASR output).
Writes metrics/narration_pace.json (numbers only; the transcripts themselves stay in the git-ignored asr/ folder).
Usage: python pb_pace.py <research/parabrahma dir>
"""
import glob, json, os, sys
import numpy as np


def main(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "asr", "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        words = [w for s in d["segments"] for w in s["words"]]
        if not words:
            continue
        span = words[-1]["e"] - words[0]["s"]
        gaps = np.array([b["s"] - a["e"] for a, b in zip(words, words[1:])])
        talk = span - gaps[gaps >= 0.25].sum()
        sent_lens = [len(s["words"]) for s in d["segments"]]
        first2 = [w["w"] for w in words if w["s"] < 2.5]
        out[d["id"]] = dict(words=len(words), span_s=round(span, 1), wps_overall=round(len(words) / span, 2),
                            wps_while_talking=round(len(words) / talk, 2), segments=len(d["segments"]),
                            words_per_segment_median=float(np.median(sent_lens)),
                            first_word_at_s=round(words[0]["s"], 2), words_in_first_2_5s=len(first2),
                            low_conf_word_share=round(float(np.mean([w["p"] < 0.5 for w in words])), 3))
        print(d["id"], out[d["id"]])
    json.dump(out, open(os.path.join(root, "metrics", "narration_pace.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
