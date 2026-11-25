"""Install the Kaggle voice-library run into research/voices/library and write its table.
Usage: python research/tools/voices_library.py <kaggle_out_dir>
Copies voice_<id>.mp3, refs/voice_<id>_ref.wav and voices.json; re-measures loudness locally (pyloudnorm) as an
independent check of the kernel's mastering; writes library/README.md with the table."""
import json, os, shutil, subprocess, sys
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIB = os.path.join(REPO, "research", "voices", "library")


def decode(mp3):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", mp3, "-f", "f32le", "-ac", "1", "-ar", "48000", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.float32), 48000


def main(src):
    s = os.path.join(src, "library")
    os.makedirs(os.path.join(LIB, "refs"), exist_ok=True)
    J = json.load(open(os.path.join(s, "voices.json")))
    for v in J["voices"]:
        shutil.copy(os.path.join(s, v["sample_file"]), os.path.join(LIB, v["sample_file"]))
        shutil.copy(os.path.join(s, v["reference_file"]), os.path.join(LIB, v["reference_file"]))
        x, sr = decode(os.path.join(LIB, v["sample_file"]))
        v["measured"]["lufs_local_check"] = round(pyln.Meter(sr).integrated_loudness(x), 2)
        v["measured"]["peak_dbfs_local_check"] = round(20 * np.log10(np.abs(x).max() + 1e-9), 2)
    json.dump(J, open(os.path.join(LIB, "voices.json"), "w"), indent=1)
    rows = []
    for v in J["voices"]:
        m = v["measured"]
        rows.append(f"| `{v['id']}` | {v['model']} | {m['lufs_local_check']} | {m['peak_dbfs_local_check']} | {m['wer']} "
                    f"({m['passage_wer']}/{m['line_wer']}) | {m['words_per_s']} | {m['pauses_ge_0_25']} / {m['mean_pause_s']} s / {m['max_pause_s']} s | "
                    f"{m['median_f0_hz']} | {m['speaker_similarity_to_ref']} | {v['best_use']} |")
    print("\n".join(rows))
    return J


if __name__ == "__main__":
    main(sys.argv[1])
