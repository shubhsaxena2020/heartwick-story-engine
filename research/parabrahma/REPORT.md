# Parabrahma World: how their reels are made

This is a research copy. We copy the technique and quality bar only, never their footage, characters, logo or text.

## How we got the data

- The Instagram profile page was public: 377 followers, 6 reels posted 15-19 Nov 2026.
- yt-dlp downloaded all 6 reels anonymously, with likes, comments, caption and date. Instagram does not show view counts without a login.
- yt-dlp's profile listing is broken, and Instagram's internal API was blocked by the permission guard, so it wasn't used.
- The YouTube channel @parabrahmaworld (648 subscribers) has the same 6 Shorts with view counts, plus one 117-second long-form video with 51 views.

## Each reel

All files are 1080x1920 at 30 fps except Leela part 2 (720p). Every file is mastered to -14.1 LUFS. "Static" is the share of near-static frames.

| Date | Reel (IG / YT) | Length | IG likes / comments | YT views / likes / comments | Shots per min | Median shot | Music under voice | Static |
|---|---|---|---|---|---|---|---|---|
| 11-15 | Brahma temples (Dfxaq7MJzbS / 8NaNEJuFw-Y) | 117 s | 231 / 6 | 3,572 / 135 / 2 | ~10* | ~6 s* | 15.5 dB | 4% |
| 11-16 | Ganesha abandoned (Dfy2G2eTz-O / j-H_PTun7Yc) | 124 s | 75 / 3 | 2,132 / 75 / 4 | 16.0 | 2.4 s | 20.2 dB | 19% |
| 11-17 | Indra's hair (Df1utU8sgp4 / okUbF5_kw9k) | 166 s | 168 / 5 | 37,479 / 1,718 / 22 | 18.8 | 2.9 s | 20.6 dB | 27% |
| 11-18 | Leela part 1 (Df3wpMpiIIP / QFG9elYEBAY) | 109 s | 41 / 5 | 1,520 / 39 / 4 | 7.7 | 6.3 s | 18.1 dB | 35% |
| 11-18 | Leela part 2 (Df4nj-MBExu / nTr9lAI5sI8) | 96 s | 32 / 4 | 1,763 / 69 / 3 | 8.8 | 5.5 s | 23.0 dB | 35% |
| 11-19 | Other Brahmas (Df6WrJTC3N0 / f1ZGDl2F-gg) | 174 s | 8,557 / 91 | 82,400 / 2,732 / 42 | 12.4 | 3.0 s | 22.4 dB | 20% |

\* An animated fire pillar confuses the shot detector on this reel. Its contact sheet shows about 25 framings.

## Why the winners won

Treat this as a hypothesis: there are only 6 reels. The two winners shared:

- a paradox or cosmic-scale premise;
- 15-20 locations and 12-19 shots per minute;
- the longest runtimes;
- a complete story with the answer given. The two-part story did worse.

## Animation

- **Characters** are jointed puppets: tube-shaped limbs with round joint caps and heads drawn in profile, about 8 heads tall. The same characters are reused across videos as a library.
- **Gestures:** a pose change every 1-1.5 s. Each change eases in over 0.2-0.4 s, then holds for 0.6-1.5 s.
- **Secondary motion:** sashes flutter on a 1.5-2 s cycle. There are walk, run, bow and cry cycles, and a "cosmic form" effect that multiplies a character.
- **Faces:** no lip sync, and no blink seen in 6 s.
- **How much moves:** something moves on its own, beyond the camera, in 65-97% of seconds.
- **Camera:** always moving, at 0.2-0.44 px per frame with 0.7-1.4% zoom per second, plus slow 4-second tilts down tall subjects.
- **Layers:** 5-7, from back to front: sky and stars, far background, middle architecture, characters, foreground, light and particles. A vignette sits on top.
- **Editing:** hard cuts plus 0.7-1.6 s dissolves. 71-100% of the cuts fall in pauses in the narration.

**How it is most likely made:** a code-driven 2D vector puppet pipeline (SVG or canvas, or Rive, Lottie or After Effects templates, driven by code or an LLM), with a reusable library. It is not AI image-to-video and not AI-painted stills. The evidence:

- crisp vectors with gradients and no painterly noise;
- identical repeated sprites and perfect perspective grids;
- joints that rotate rigidly with no morphing;
- characters that look exactly the same across videos;
- 6 videos in 5 days, all at exactly -14.1 LUFS.

## Art

- **Palette:** indigo and violet starry skies; gold architecture with ruby, emerald and sapphire gems; crimson and saffron fabrics; blue deities.
- **Drawing style:** flat vector shapes with airbrushed gradients and glows, dotted fabric patterns, long elegant proportions.

## Voice

- One male baritone narrates all six: median pitch 120-135 Hz, with a range of 12.4-13.3 semitones.
- Recorded close to the mic, with no room sound.
- 56-101 pauses per video, averaging 0.58 s (SD 0.22) and never longer than about 1.25 s.
- Probably AI (medium confidence).
- Words per second will come from the transcription job (`tools/pb_pace.py`).

## Music

A warm, low pad or drone with no beat, 15-23 dB under the voice. Its spectral centre is about 0.8-1.2 kHz. No sound effects on cuts (measured).

## Captions

- Heavy Hindi sans-serif, white, with one key word in gold.
- 2-4 words at a time, each chunk on screen about 1-1.3 s, fading in over 0.2 s.
- Placed at about 71% of the frame height, over a soft dark ellipse.

## Structure

- A two-line gold source tag sits top left the whole time ("X Purana / chapter").
- **Hook (0-2 s):** a small English kicker, a big two-line Hindi title stating the paradox, and the video opens straight on the most dramatic moment.
- **Story beats:**
  1. the hook;
  2. "X Purana says";
  3. the setup;
  4. it escalates through 2-3 jumps between locations;
  5. the big reveal;
  6. the moral or answer;
  7. a question to the viewer;
  8. the end card.
- **End card (7-8 s):** a lotus emblem in a gold ring, then "this story is not made up, it is written in X Purana / follow".

## Posting

- About 1.2 posts a day, posted to Instagram and YouTube on the same day.
- **Instagram caption pattern:**
  - a Hindi hook;
  - a synopsis;
  - "the answer is in the video";
  - a question;
  - an English keyword line;
  - "Source:";
  - a follow CTA;
  - 5 hashtags.
- **YouTube:** bilingual titles with 10-14 tags.

## Files

- Metrics are in `metrics/*.json`.
- Their media and anything derived from it are local only and git-ignored: `media/`, `audio/`, `frames/`, `sheets/`, `motion/`, `meta.json`, `yt_meta.json`.
