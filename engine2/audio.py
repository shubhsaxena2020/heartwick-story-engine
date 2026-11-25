"""Narration bed + music pad + mastering.

Reference (research/parabrahma/REPORT.md): voice dry and close (no room), a warm low pad with no beat sitting
15-23 dB under the voice, no sound effects on cuts, master at -14.1 LUFS. Heartwick: pad 18-22 dB under the voice
(default 20), master -14 LUFS integrated, true peak <= -1.5 dBTP.
The pad is synthesised here (sines + slow filter + slow chord changes): licence-clean, no samples.
"""
import math
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

SR = 48000


def load_mono(path, sr=SR):
    x, r = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if r != sr:
        g = math.gcd(r, sr)
        x = resample_poly(x, sr // g, r // g).astype(np.float32)
    return x


def voice_track(audio, tts_dir, total_s):
    v = np.zeros(int(total_s * SR) + SR, np.float32)
    for a in audio:
        x = load_mono(f"{tts_dir}/{a['id']}.wav")
        i = int(a["t"] * SR)
        v[i:i + len(x)] += x[: max(0, len(v) - i)]
    sos = butter(2, 70, "highpass", fs=SR, output="sos")  # dry, close: just clear the rumble, no reverb
    return compress(sosfilt(sos, v)).astype(np.float32)


def compress(x, thr_db=-24.0, ratio=3.0, att_ms=4.0, rel_ms=120.0):
    """Gentle narration compressor (feed-forward, peak-ish envelope with fast attack / slow release)."""
    from scipy.signal import lfilter
    a_att, a_rel = math.exp(-1 / (SR * att_ms / 1000)), math.exp(-1 / (SR * rel_ms / 1000))
    rect = np.abs(x)
    fast = lfilter([1 - a_att], [1, -a_att], rect)
    slow = lfilter([1 - a_rel], [1, -a_rel], rect)
    env = np.maximum(fast, slow) + 1e-9
    lvl = 20 * np.log10(env)
    over = np.maximum(0.0, lvl - thr_db)
    g = 10 ** (-over * (1 - 1 / ratio) / 20)
    return x * g


CHORDS = [(57, 64, 69, 72), (53, 60, 65, 69), (48, 55, 60, 64), (55, 62, 67, 71)]  # Am, F, C, G (midi)


def pad(total_s, seed=3, swells=()):
    """Warm, beatless pad: slow chord changes (8 s), soft attack, gentle movement. swells: [(t, dur)] +3 dB."""
    n = int(total_s * SR) + SR
    t = np.arange(n, dtype=np.float64) / SR
    out = np.zeros(n)
    rng = np.random.default_rng(seed)
    seg = 8.0
    for k in range(int(total_s / seg) + 2):
        chord = CHORDS[k % len(CHORDS)]
        a, b = k * seg - 2.0, (k + 1) * seg + 2.0  # 2 s crossfades
        env = np.clip((t - a) / 2.0, 0, 1) * np.clip((b - t) / 2.0, 0, 1)
        sel = env > 0
        if not sel.any():
            continue
        for m in chord:
            f = 440 * 2 ** ((m - 69) / 12) / 2  # an octave down: warm and low
            det = 1 + rng.uniform(-0.002, 0.002)
            ph = rng.uniform(0, 6.28)
            tone = np.sin(2 * np.pi * f * det * t[sel] + ph) + 0.25 * np.sin(2 * np.pi * 2 * f * t[sel] + ph * 1.3)
            out[sel] += env[sel] * tone * (0.5 + 0.12 * np.sin(2 * np.pi * 0.07 * t[sel] + ph))
    sos = butter(2, 1200, "lowpass", fs=SR, output="sos")  # spectral centre ~1 kHz like the reference bed
    out = sosfilt(sos, out)
    g = np.ones(n)
    for (ts, d) in swells:
        g += (10 ** (3 / 20) - 1) * np.clip(np.minimum((t - ts) / 1.0, (ts + d - t) / 1.5), 0, 1)
    out *= g
    fade = np.clip(t / 1.5, 0, 1) * np.clip((total_s - t) / 2.0, 0, 1)
    return (out * fade).astype(np.float32)


def lufs(x):
    m = pyln.Meter(SR)
    return m.integrated_loudness(x.astype(np.float64)) if np.abs(x).max() > 0 else -120.0


def true_peak_db(x):
    up = resample_poly(x.astype(np.float64), 4, 1)
    return 20 * math.log10(max(1e-9, np.abs(up).max()))


def limit(x, ceiling_db=-1.5, look_ms=3.0, release_ms=30.0):
    """Look-ahead peak limiter on the 4x-oversampled signal estimate."""
    thr = 10 ** (ceiling_db / 20) * 0.97
    up = np.abs(resample_poly(x.astype(np.float64), 4, 1))
    up = np.pad(up, (0, (-len(up)) % 4)).reshape(-1, 4).max(1)[: len(x)]  # inter-sample peak per sample
    need = np.minimum(1.0, thr / np.maximum(up, 1e-9))
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    w = max(1, int(SR * look_ms / 1000))
    r = max(w, int(SR * release_ms / 1000))
    # widen each gain dip (min filter), then smooth it (box) narrower than the widening: the smoothed gain is
    # still <= the needed gain at every peak, and recovers over ~release_ms
    g = uniform_filter1d(minimum_filter1d(need, size=2 * r + 1), size=r)
    g = np.minimum(g, minimum_filter1d(need, size=2 * w + 1))
    return (x * g).astype(np.float32)


def master(voice, music=None, sfx=None, target=-14.0, pad_gap_db=20.0, tp=-1.5, sfx_gap_db=24.0):
    """Mix voice + music with the music `pad_gap_db` under the voice (integrated loudness of each stem),
    then hit `target` LUFS with a true-peak ceiling. Returns (mix, report)."""
    lv = lufs(voice)
    rep = {"voice_lufs_raw": round(lv, 2)}
    mix = voice.copy()
    if music is not None:
        music = music[: len(voice)] if len(music) >= len(voice) else np.pad(music, (0, len(voice) - len(music)))
        lm = lufs(music)
        music = music * 10 ** ((lv - pad_gap_db - lm) / 20)
        mix = voice + music
    if sfx is not None and np.abs(sfx).max() > 0:
        sfx = sfx[: len(voice)] if len(sfx) >= len(voice) else np.pad(sfx, (0, len(voice) - len(sfx)))
        sfx = sfx * 10 ** ((lv - sfx_gap_db - lufs(sfx)) / 20)
        mix = mix + sfx
    g = 10 ** ((target - lufs(mix)) / 20)
    mix = mix * g
    for _ in range(8):  # gain to target, limit the peaks it creates, repeat until both hold
        if true_peak_db(mix) > tp - 0.2:
            mix = limit(mix, tp - 0.4)
        d = target - lufs(mix)
        if abs(d) < 0.12 and true_peak_db(mix) <= tp:
            break
        mix = mix * 10 ** (d / 20)
    if true_peak_db(mix) > tp:
        mix = limit(mix, tp - 0.4)
    rep.update(lufs=round(lufs(mix), 2), true_peak_db=round(true_peak_db(mix), 2))
    if music is not None:
        rep["pad_under_voice_db"] = round(lufs(voice * g) - lufs(music * g), 2)
    return mix.astype(np.float32), rep


# ---------------------------------------------------------------- music library + sound effects (round 2)
MUSIC_OF_MOOD = {"storm": "storm", "night": "tender", "home": "tender", "dawn": "tender", "dusk": "tender",
                 "day": "wonder", "winter": "wonder", "morning": "wonder", "golden": "resolution"}


def library(music_dir):
    """{mood: [wav paths]} from assets/music/<mood>_<k>.wav|flac (ACE-Step cues, tools/hero_music.py)."""
    import glob, os
    lib = {}
    for f in sorted(glob.glob(os.path.join(music_dir, "*.wav")) + glob.glob(os.path.join(music_dir, "*.flac"))):
        mood = os.path.basename(f).rsplit("_", 1)[0]
        lib.setdefault(mood, []).append(f)
    return lib


def music_track(segments, total_s, lib, xfade=1.5):
    """segments: [(t0, t1, mood)] in story order. A mood keeps playing across consecutive shots (its cue continues);
    a new mood crossfades in over `xfade` s. Cues loop with a crossfade if a stretch is longer than the cue."""
    n = int(total_s * SR) + SR
    out = np.zeros(n, np.float32)
    runs = []
    for t0, t1, mood in segments:
        if runs and runs[-1][2] == mood:
            runs[-1][1] = t1
        else:
            runs.append([t0, t1, mood])
    cache = {}
    for k, (t0, t1, mood) in enumerate(runs):
        files = lib.get(mood) or next(iter(lib.values()))
        f = files[k % len(files)]
        if f not in cache:
            cache[f] = load_mono(f)
        cue = cache[f]
        a = max(0, int((t0 - xfade / 2) * SR))
        b = min(n, int((t1 + xfade / 2) * SR))
        seg = np.zeros(b - a, np.float32)
        pos, L, fl = 0, len(cue), int(2 * SR)
        while pos < len(seg):  # loop with a 2 s crossfade
            m = min(L, len(seg) - pos)
            piece = cue[:m].copy()
            if pos > 0:
                r = min(fl, m)
                piece[:r] *= np.linspace(0, 1, r)
                seg[pos:pos + r] *= np.linspace(1, 0, r)
            seg[pos:pos + m] += piece
            pos += max(1, m - fl)
        env = np.ones(len(seg), np.float32)
        r = int(xfade * SR)
        env[:r] = np.linspace(0, 1, min(r, len(env)))[: len(env[:r])]
        env[-r:] = np.minimum(env[-r:], np.linspace(1, 0, min(r, len(env)))[-len(env[-r:]):])
        out[a:b] += seg * env
    fade = np.clip(np.arange(n) / SR / 1.5, 0, 1) * np.clip((total_s - np.arange(n) / SR) / 2.5, 0, 1)
    return (out * fade).astype(np.float32)


def swells(x, spans, db=3.0):
    """Raise the music by `db` over reveal spans [(t, dur)], with 1 s ramps."""
    t = np.arange(len(x)) / SR
    g = np.ones(len(x), np.float32)
    for (ts, d) in spans:
        g += (10 ** (db / 20) - 1) * np.clip(np.minimum((t - ts) / 1.0, (ts + d - t) / 1.5), 0, 1)
    return x * g


def _noise(n, seed):
    return np.random.default_rng(seed).normal(0, 1, n).astype(np.float32)


def sfx_wind(dur, seed=1):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = sosfilt(butter(2, [180, 900], "bandpass", fs=SR, output="sos"), _noise(n, seed))
    gust = 0.55 + 0.45 * np.sin(2 * np.pi * 0.13 * t + seed) * np.sin(2 * np.pi * 0.071 * t)
    return (x * gust * 0.25).astype(np.float32)


def sfx_rain(dur, seed=2):
    n = int(dur * SR)
    x = sosfilt(butter(2, 1800, "highpass", fs=SR, output="sos"), _noise(n, seed)) * 0.12
    r = np.random.default_rng(seed)
    for p in r.integers(0, max(1, n - 2000), int(dur * 25)):  # individual drops
        L = 600
        x[p:p + L] += np.exp(-np.arange(L) / 90) * r.uniform(0.05, 0.25) * np.sin(np.arange(L) * r.uniform(0.3, 0.9))
    return x.astype(np.float32)


def sfx_creak(dur=0.9, seed=3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 140 + 90 * np.sin(np.pi * t / dur)  # a slow pitch bend, like a window sash
    phase = 2 * np.pi * np.cumsum(f) / SR
    buzz = np.sign(np.sin(phase)) * (0.5 + 0.5 * np.sin(2 * np.pi * 23 * t))
    x = sosfilt(butter(2, [300, 2200], "bandpass", fs=SR, output="sos"), buzz) * np.hanning(n)
    return (x * 0.2).astype(np.float32)


SFX = {"wind": sfx_wind, "rain": sfx_rain, "creak": sfx_creak}


def sfx_track(events, total_s):
    """events: [(t0, t1, kind)]. Continuous beds (wind, rain) fill their span with fades; one-shots play at t0."""
    n = int(total_s * SR) + SR
    out = np.zeros(n, np.float32)
    for t0, t1, kind in events:
        x = SFX[kind](max(0.5, t1 - t0)) if kind in ("wind", "rain") else SFX[kind]()
        a = int(t0 * SR)
        x = x[: max(0, n - a)]
        r = min(len(x) // 3, int(0.8 * SR))
        if r:
            x[:r] *= np.linspace(0, 1, r)
            x[-r:] *= np.linspace(1, 0, r)
        out[a:a + len(x)] += x
    return out
