"""Build the TTS comparison table from a Kaggle bake-off output folder.
Reads <dir>/voices/*.wav, timings.jsonl, asr.json. Writes <dest>/comparison.csv and comparison.json,
loudness-matched (-16 LUFS) MP3 copies for listening, and prints a Markdown table.
Usage: python voices_table.py <kaggle_out_dir> <dest_dir> <tag>
"""
import csv, glob, json, os, subprocess, sys
import numpy as np
import soundfile as sf
import pyloudnorm as pyln


def main(src, dest, tag):
    os.makedirs(dest, exist_ok=True)
    tim = {}
    p = os.path.join(src, "voices", "timings.jsonl")
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line); tim[r["tag"]] = r
    asr = json.load(open(os.path.join(src, "voices", "asr.json"))) if os.path.exists(os.path.join(src, "voices", "asr.json")) else {}
    rows = []
    for f in sorted(glob.glob(os.path.join(src, "voices", "*.wav"))):
        name = os.path.basename(f)[:-4]
        if name.startswith("kokororef_"):
            continue
        x, sr = sf.read(f, dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(x)
        peak = 20 * np.log10(np.abs(x).max() + 1e-9)
        a = asr.get(name, {})
        t = tim.get(name, {})
        rows.append(dict(sample=name, run=tag, sr=sr, dur_s=round(len(x) / sr, 2), lufs=round(float(lufs), 1), peak_dbfs=round(float(peak), 1),
                         wer=a.get("wer"), words_per_s=a.get("wps"), pauses_ge_0_25=len(a.get("pauses_ge_0_25", [])) if a else None,
                         gen_s=t.get("gen_s"), rtf=t.get("rtf"), hyp=a.get("hyp")))
        mp3 = os.path.join(dest, f"voice_{name}.mp3")  # dashboard Research tab picks up "voice_*" audio
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", "-ac", "1",
                        "-b:a", "128k", mp3], check=False)
    with open(os.path.join(dest, f"comparison_{tag}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(os.path.join(dest, f"comparison_{tag}.json"), "w"), indent=1)
    print("| sample | dur s | LUFS | peak dBFS | WER | words/s | pauses>=0.25 s | gen s | RTF |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['sample']} | {r['dur_s']} | {r['lufs']} | {r['peak_dbfs']} | {r['wer']} | {r['words_per_s']} | {r['pauses_ge_0_25']} | {r['gen_s']} | {r['rtf']} |")


if __name__ == "__main__":
    main(*sys.argv[1:4])
