"""Finish a voice-library run: master the chosen takes and write voices.json (the kernel's last step, which crashed
on a missing import after all generation and Whisper scoring had finished). Same chain as
research/kaggle_jobs/voices_lib: 70 Hz high-pass, ffmpeg loudnorm two-pass to -16 LUFS / -1.5 dBTP, verified with
pyloudnorm, mp3 192k. Measurements (WER, pace, pauses, pitch, speaker similarity) come from the kernel's scores.json.
python research/tools/voices_master.py <kaggle_out_dir>"""
import json, math, os, re, shutil, subprocess, sys
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "research", "kaggle_jobs", "voices_lib"))
import voices_lib as VL  # noqa: E402  (VOICES, T)

LIB = os.path.join(REPO, "research", "voices", "library")
SR = 48000
LIC = {"voxcpm2": "VoxCPM2 clone: Apache-2.0 (commercial use OK)", "qwen3clone": "Qwen3-TTS-12Hz-1.7B-Base clone: Apache-2.0 (commercial use OK)"}


def limit(y, thr):
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    env = maximum_filter1d(np.abs(y), size=int(0.005 * SR) * 2 + 1)
    g = np.minimum(1.0, thr / np.maximum(env, 1e-9))
    g = np.minimum(g, uniform_filter1d(g, size=int(0.02 * SR)))  # smooth release, never above the instant gain
    return (y * g).astype(np.float32)


def master(src, res, vid, kp, kl):
    xs = []
    for k, gap in ((kp, 0.75), (kl, 0.0)):
        x, sr = sf.read(os.path.join(src, "takes", k + ".wav"), dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            g = math.gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        a = res[k]
        i0 = max(0, int((a["lead"] - 0.2) * SR))
        i1 = len(x) - max(0, int((a["tail"] - 0.35) * SR))
        x = x[i0:i1].copy()
        n = int(0.03 * SR)
        x[:n] *= np.linspace(0, 1, n)
        x[-n:] *= np.linspace(1, 0, n)
        xs += [x, np.zeros(int(gap * SR), np.float32)]
    x = sosfilt(butter(2, 70, "highpass", fs=SR, output="sos"), np.concatenate(xs)).astype(np.float32)
    raw = os.path.join(LIB, f"_{vid}_raw.wav")
    sf.write(raw, x, SR)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", raw, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    js = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
    af = ("loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}:"
          "measured_thresh={input_thresh}:offset={target_offset}").format(**js)
    wav = os.path.join(LIB, f"_{vid}.wav")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", raw, "-af", af, "-ar", str(SR), "-ac", "1", wav], check=True)
    y, _ = sf.read(wav, dtype="float32")
    for _ in range(4):  # gain to -16 LUFS, then a 5 ms look-ahead limiter at -1.7 dBFS catches the few speech peaks
        lufs = pyln.Meter(SR).integrated_loudness(y)
        y = y * 10 ** ((-16 - lufs) / 20)
        y = limit(y, 10 ** (-1.7 / 20))
    lufs = pyln.Meter(SR).integrated_loudness(y)
    sf.write(wav, y, SR)
    mp3 = os.path.join(LIB, f"voice_{vid}.mp3")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav, "-c:a", "libmp3lame", "-b:a", "192k", mp3], check=True)
    # verify the delivered mp3 itself
    d = subprocess.run(["ffmpeg", "-v", "error", "-i", mp3, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True).stdout
    z = np.frombuffer(d, np.float32)
    for f in (raw, wav):
        try:
            os.remove(f)
        except OSError:
            pass
    return dict(lufs=round(pyln.Meter(SR).integrated_loudness(z), 2), peak_dbfs=round(20 * math.log10(float(np.abs(z).max()) + 1e-9), 2),
                dur_s=round(len(z) / SR, 2))


def main(src):
    res = json.load(open(os.path.join(src, "takes", "scores.json")))
    choice = {}
    for line in open(os.path.join(src, "logs", "score.log")):
        m = re.match(r"CHOICE (\w+) \(([\d.]+), '(\w+)', '([\w_]+)', '([\w_]+)', (True|False)\)", line.strip())
        if m:
            choice[m.group(1)] = (m.group(3), m.group(4), m.group(5), m.group(6) == "True")
    os.makedirs(os.path.join(LIB, "refs"), exist_ok=True)
    lib = []
    for vid, (desc, ref_text, line, f0r, use) in VL.VOICES.items():
        model, kp, kl, clean = choice[vid]
        m = master(src, res, vid, kp, kl)
        shutil.copy(os.path.join(src, "refs", f"{vid}.wav"), os.path.join(LIB, "refs", f"voice_{vid}_ref.wav"))
        ap, al = res[kp], res[kl]
        pauses = ap["pauses"] + al["pauses"]
        words = ap["words"] + al["words"]
        lib.append(dict(id=vid, description=desc, dialogue_line=line, best_use=use,
                        design_model="Qwen3-TTS-12Hz-1.7B-VoiceDesign (Apache-2.0): the voice was designed from the description text, not cloned from a person",
                        model=model, licence=LIC[model] + "; designed reference: Qwen3-TTS VoiceDesign, Apache-2.0",
                        reference_file=f"refs/voice_{vid}_ref.wav", reference_text=ref_text, reference_f0_hz=res[f"{vid}__REF"]["f0"],
                        reference_wer=res[f"{vid}__REF"]["wer"], target_f0_hz=list(f0r), sample_file=f"voice_{vid}.mp3",
                        passage_take=kp, line_take=kl, all_takes_wer0=clean,
                        measured=dict(lufs=m["lufs"], peak_dbfs=m["peak_dbfs"], dur_s=m["dur_s"], wer=round((ap["wer"] * ap["words"] + al["wer"] * al["words"]) / max(1, words), 4),
                                      passage_wer=ap["wer"], line_wer=al["wer"], words_per_s=ap["wps"], pauses_ge_0_25=len(pauses),
                                      mean_pause_s=round(float(np.mean(pauses)), 2) if pauses else 0.0, max_pause_s=max(ap["max_pause"], al["max_pause"]),
                                      median_f0_hz=ap["f0"], speaker_similarity_to_ref=ap["spk_sim"], clip_frac=max(ap["clip_frac"], al["clip_frac"]),
                                      raw_lufs_before_mastering=ap["raw_lufs"]),
                        transcript=ap["hyp"] + " | " + al["hyp"]))
        print(vid, model, m, "wer", ap["wer"], al["wer"], "wps", ap["wps"], "sim", ap["spk_sim"], "f0", ap["f0"])
    json.dump(dict(passage=VL.T, target_lufs=-16, true_peak_dbtp=-1.5, asr="faster-whisper large-v3 (GPU, beam 5)",
                   speaker_model="speechbrain ECAPA (spkrec-ecapa-voxceleb)", voices=lib), open(os.path.join(LIB, "voices.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
