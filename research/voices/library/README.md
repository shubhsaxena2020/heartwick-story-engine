# Heartwick voice library (run 2, 21 Nov 2026)

There are 14 original voices. No real person was cloned:

1. Each voice was **designed from a text description** with Qwen3-TTS-12Hz-1.7B-VoiceDesign (Apache-2.0).
2. That designed clip became the voice's **reference** (`refs/voice_<id>_ref.wav`; its text is in `voices.json`).
3. Two cloning models each rendered the same passage plus one in-character line from that reference, with 2 seeds
   per model: **VoxCPM2** (Apache-2.0) and **Qwen3-TTS-1.7B-Base** (Apache-2.0).
4. **Chatterbox** (MIT) was also scheduled but failed to start on Kaggle (its watermarker module did not load), so it
   produced no takes.
5. **Choosing the takes:** faster-whisper large-v3 (GPU) scored every take. The best pair per voice, with both
   takes at WER 0, was kept.
6. **Mastering:** 70 Hz high-pass, ffmpeg loudnorm, a 5 ms look-ahead limiter, then -16 LUFS. This was verified on
   the delivered MP3s with pyloudnorm.

Re-generate any line in a voice with its reference and reference text, so the voice stays consistent:

```python
voxcpm.generate(text, prompt_wav_path=ref, prompt_text=ref_text, reference_wav_path=ref)
# or: qwen3 Base generate_voice_clone(text, ref_audio=ref, ref_text=ref_text)
```

Use the model the table lists for that voice.

The passage is the Heartwick test passage ("Every autumn, the old maple behind Nell's house…"). It is followed by
0.75 s of silence and the voice's own dialogue line (both in `voices.json`).

## Measurements (all measured, none estimated)

| Voice | File | Model | LUFS | WER | Words/s | Pauses ≥0.25 s: count / mean / max | Median F0 | Best use |
|---|---|---|---|---|---|---|---|---|
| Warm male baritone narrator | `voice_narrator_baritone.mp3` | Qwen3 clone | -16.27 | 0 | 2.96 | 2 / 0.47 / 0.56 s | 99 Hz | Main narrator, as in the reference |
| Warm female narrator | `voice_narrator_warm_f.mp3` | VoxCPM2 | -16.27 | 0 | 3.07 | 2 / 0.80 / 1.12 s | 216 Hz | Cozy Shorts, Nell stories |
| Elderly grandmother | `voice_grandmother.mp3` | VoxCPM2 | -16.26 | 0 | 2.23 | 5 / 0.71 / 1.06 s | 168 Hz | Nell |
| Elderly grandfather | `voice_grandfather.mp3` | VoxCPM2 | -16.28 | 0 | 2.60 | 3 / 0.59 / 0.62 s | 104 Hz | Grandfathers, the wise elder |
| Young woman | `voice_young_woman.mp3` | VoxCPM2 | -16.27 | 0 | 2.91 | 3 / 0.59 / 0.88 s | 246 Hz | Dee, heroines |
| Young man | `voice_young_man.mp3` | VoxCPM2 | -16.30 | 0 | 3.71 | 2 / 0.33 / 0.34 s | 155 Hz | Young heroes, fathers |
| Child-like (performed adult voice) | `voice_child_like.mp3` | Qwen3 clone | -16.26 | 0 | 3.14 | 1 / 0.34 / 0.34 s | 381 Hz | Eli |
| British narrator | `voice_british_narrator.mp3` | Qwen3 clone | -16.27 | 0 | 2.65 | 4 / 0.45 / 0.66 s | 118 Hz | Classic fairy tales, long-form |
| Deep villain / king | `voice_villain_king.mp3` | Qwen3 clone | -16.26 | 0 | 2.66 | 2 / 0.40 / 0.48 s | 121 Hz | Kings, villains |
| Gentle sleep storyteller (female) | `voice_sleep_storyteller.mp3` | Qwen3 clone | -16.26 | 0 | 2.96 | 2 / 0.38 / 0.46 s | 182 Hz | Sleep stories |
| Gentle sleep storyteller (male) | `voice_sleep_storyteller_m.mp3` | Qwen3 clone | -16.27 | 0 | 2.84 | 4 / 0.47 / 0.62 s | 86 Hz | Sleep stories |
| Energetic hook reader | `voice_hook_reader.mp3` | VoxCPM2 | -16.27 | 0 | 2.88 | 1 / 0.46 / 0.46 s | 180 Hz | The 0-2 s hook, CTAs |
| Animal sidekick | `voice_sidekick_critter.mp3` | Qwen3 clone | -16.26 | 0 | 3.12 | 2 / 0.36 / 0.42 s | 239 Hz | Pip "talking", comic relief |
| British female | `voice_british_female.mp3` | Qwen3 clone | -16.27 | 0 | 2.93 | 2 / 0.50 / 0.64 s | 200 Hz | Queens, fairy godmothers |

**Notes on the columns**

- **LUFS** is integrated loudness measured on the MP3 itself. All 14 files sit within 0.04 LU of each other.
- **Peaks** are -1.9 to -2.0 dBFS on every file.
- **Clipping** is zero on every take.
- **Noise floor** in the gaps is -67 dB or lower on every file.
- **Pause columns** count the passage take only.
- **Speaker similarity to the reference** was planned with ECAPA but could not be measured: that model failed to load
  in the scoring environment.

**Top 3 narrators, chosen by the numbers.** Listen before locking one in; I cannot hear audio.

1. **`narrator_baritone`**:
   - a 99 Hz baritone, like the reference channel's narrator (120-135 Hz);
   - 2.96 words/s, which sits in the reference's English-equivalent pace;
   - even 0.47 s pauses (the reference averages 0.58 s).
2. **`narrator_warm_f`**: VoxCPM2 at 48 kHz, with the longest and most natural phrase pauses (0.8 s).
3. **`british_narrator`**: an unhurried 2.65 words/s with 4 phrase pauses. Good for classic tales and long-form.

## Honest gaps

**Where the voices miss their descriptions**

| Voice | Description asked for | What we got |
|---|---|---|
| `villain_king` | very low | 121 Hz: deep-ish, not a true bass |
| `sleep_storyteller` | slow | 2.96 words/s, as fast as the narrator |
| `hook_reader` | fast | 2.88 words/s |

For sleep stories, slow the sleep voices to about 0.85x pace in the engine, or re-design them with a slower
reference. For hooks, re-design the hook reader with a more energetic reference.

**Tooling gaps**

- Pace and emotion can only be steered through the description and the reference clip. Neither cloning model has a
  speed or emotion control.
- Each voice needed 2 seeds to guarantee a clean, WER-0 take.

**Compared with ElevenLabs.** I could not test it, because there is no account. ElevenLabs Creator costs $22/month
for about 100K credits, which is about 100K characters: roughly 1.5-1.7 hours of narration at our pace. On a free
first listen it would still be ahead in:

- emotional range and audio tags (whisper, laugh, sigh);
- consistent prosody over minutes of text;
- precise speed control, and one-take reliability;
- no Kaggle queue: this run waited about 10 minutes for a GPU slot and took 1h45 of T4 time.

The open stack's advantages:

- **Cost:** no per-character fees.
- **Licences:** all Apache-2.0, so the output is ours.
- **Quality:** comparable intelligibility (WER 0 on every voice).
- **Consistency:** fixed reference clips keep each voice identical across episodes.

A fair test is to make one Creator month, render the same passage with 3 stock voices, and compare by ear and on
these metrics.

**Not used: Higgs Audio.** It is allowed only with a visible "Boson AI's Higgs Audio" credit. That choice is
Shubh's.

## Files

| File | Contents |
|---|---|
| `voice_<id>.mp3` | the mastered sample |
| `refs/voice_<id>_ref.wav` | the designed reference |
| `voices.json` | id, description, dialogue line, model, licence, reference file and text, measured LUFS / WER / pace / pauses / pitch, and the chosen takes |

All candidate takes (2 models x 2 seeds x 14 voices) and Whisper scores are in
`research/kaggle_out/voices_lib/takes/`, which is git-ignored.

**Regenerate**

1. Run the kernel `research/kaggle_jobs/voices_lib/voices_lib.py` on Kaggle (T4 x2).
2. Then run locally: `python research/tools/voices_master.py research/kaggle_out/voices_lib`.
