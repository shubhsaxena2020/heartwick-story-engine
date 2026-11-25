# Heartwick research report: recommendations for engine v2 (21 Nov 2026)

Results still pending on Kaggle:

| Kaggle kernel | What it produces |
|---|---|
| `heartwick-research-asr` | Hindi transcription, used for pace |
| `heartwick-research-tts-cpu` | Voice comparison, CPU run |
| `heartwick-research-gpu` | Voice comparison including Higgs, art-direction stills A/B/C, and Wan and LTX image-to-video tests |

Fetch any of them with `C:\Users\shubh\.kaggle-venv\Scripts\kaggle.exe kernels output shubhsaxena2008/<slug> -p <dir>`.

## 1. The reference: targets engine v2 must hit

Details are in `parabrahma/REPORT.md`.

- Each character makes a gesture every 1-1.5 s.
- Secondary motion on cloth, hair and props.
- The camera is always moving, with slow tilt reveals.
- 5-7 parallax layers.
- 10-19 shots per minute, with cuts placed in pauses in the narration.
- Light effects.
- Captions of 2-4 words with one highlighted word.
- A source or series tag on screen.
- A paradox hook in the first 2 s.
- An end card.

## 2. The market (Nov 2026)

### What is winning

- **AI "Ghibli cozy village" Shorts** have the biggest audiences:
  - Cozy Village Tales: 369K subscribers, median Short 208K views, top video 175M views, but a long-form median of only 20K.
  - Rainlit Village: 189M views on one video.
  - Shano Ghibli: 86K subscribers, median 107K views.
- **Serialised long-form folk dramas** win on long-form. Viivee Storytime: 92K subscribers, long-form median 213K views, Shorts median only 3.3K.
- **3D fairy-tale retellings:**
  - Magical Studio: 17.4K subscribers, ~12-minute videos with a 57K median, and a 45-minute compilation at 97K.
  - Zgapariko: 44K subscribers, 71K median (aimed at kids).
- **History-for-sleep videos** of 2 hours or more get 2-4M views each.
- **AI-animated moral myths** can explode: Squirrel and Rama got 31M views.
- **Counter-trend:** "[No AI]" now appears in titles as a selling point.

### Money

- **Earnings per 1,000 views:** Shorts make about $0.03-0.07, long-form about $3-8+ for US viewers. Sources: vidIQ, AIR (274 channels), Digiday.
- **YouTube monetisation** needs 1K subscribers plus 4K watch hours, or 10M Shorts views.
- **Facebook** Content Monetization is invite-only, and Instagram has no general payout for creators (only tests).

### Policy

- YouTube's "inauthentic content" rule (renamed 15 Jul 2025) disallows "AI-generated content made with generic or unoriginal templates", "templated storylines" and "image slideshows".
- It allows the same intro and outro on every video, as long as each video's story is distinct.
- Animated or clearly unrealistic content needs the AI disclosure in the description only. YouTube now labels realistic AI automatically.
- A Bible-stories channel with 588K subscribers was reportedly demonetised in early 2026 for templated AI output.

### Recommendation

- Shorts and Reels of 90-170 s, each a complete story, with a paradox or scale hook in the first 2 s.
- One recurring world, and original scripts only.
- 3-5 Shorts a week.
- A weekly 8-15 minute long-form "full story", and a monthly 45-60 minute cozy or sleep compilation for watch hours and higher earnings.
- No cliffhanger splits into parts.

## 3. Voice

### Licences

| Status | Models |
|---|---|
| Usable | Kokoro (Apache-2.0); Chatterbox and Chatterbox-Turbo (MIT); VoxCPM2 (Apache-2.0; 2B, 48 kHz, voice design); Qwen3-TTS VoiceDesign (Apache-2.0); Dia (Apache-2.0); Orpheus (Apache card on a gated Llama 3.2 base); Sesame CSM-1B (Apache, gated); ZONOS3-Mini (Apache, untested, needs its own server runtime) |
| Usable with conditions | IndexTTS2 (bilibili licence: free below 100M monthly users or RMB 1bn revenue); Higgs Audio v2 (commercial below 100K annual users, attribution required); Higgs TTS 3 (non-commercial, but its Creator Use Grant allows monetised channels with a visible "Boson AI's Higgs Audio" credit) |
| Excluded | F5-TTS (CC-BY-NC); XTTS-v2 (Coqui non-commercial); OmniVoice (CC-BY-NC); Fish S2 Pro and Drama-3 (research only); Breeze-TTS-2 (non-commercial); VibeVoice (research only, and it adds an audible AI disclaimer) |

### The test

- **Passage:** one three-sentence Heartwick passage, read as a warm American female storyteller and a warm American male storyteller.
- **House voices:** two original voices are designed with Qwen3-TTS, so no real person is cloned. The cloning models then use those voices as their reference.
- **Scoring:** Whisper word error rate, loudness, words per second, pauses and generation speed.
- **Results:** pending in `voices/`. Build the table with `tools/voices_table.py`.

### Paid options, for comparison only

ElevenLabs:

| Plan | Price | Includes |
|---|---|---|
| Free | $0 | Non-commercial only |
| Starter | $5-6/month | Commercial use |
| Creator | $22/month | About 100-121K credits and professional cloning |
| Pro | $99/month | |

What paid adds: consistently top-quality delivery, and no waiting in the Kaggle queue.

## 4. Music and sound

### Sources

- **YouTube Audio Library:** Google says downloaded tracks are not claimed by Content ID and can be monetised on YouTube. Its terms say nothing about use outside YouTube.
- **Pixabay:** free commercial use with no attribution, but its licence doesn't address Content ID and some tracks get claimed. Avoid it for YouTube.
- **Epidemic Sound Creator:** about $9.99/month billed annually, covering one channel per platform (YouTube and Instagram).

### Open models

| Model | Status |
|---|---|
| ACE-Step 1.5 (MIT) | Use it. |
| Stable Audio Open / Stable Audio 3 (Stability community licence, gated) | Fine for sound effects while revenue stays under the threshold. |
| MusicGen, YuE / YuE2 (CC-BY-NC) | Not usable. |
| MiniMax-Music3 | Licence unclear. |

### Approach (matches the reference)

- A warm, low pad or drone with soft harp, celesta or strings.
- 18-22 dB under the voice, with no beat.
- It swells only at reveals.
- Sparse sound effects, and none on cuts.

## 5. Animation

### What we measured

- **Wan2.2-TI2V-5B on a T4** took 35 minutes for one 2-second clip (an earlier SewaCircle Kaggle job). Image-to-video cannot carry a 2-3 minute story. New Wan and LTX quality tests are pending.
- **LTX-Video 2B** (open weights, commercial use allowed) is faster, but its quality is untested.

### Tools ruled in or out

| Tool | Verdict |
|---|---|
| LivePortrait | Excluded: its face models are non-commercial. |
| Live2D, Spine, Moho | Paid GUI tools that can't be automated. |
| Meta Animated Drawings (MIT) | Archived. |
| Inochi2D (BSD) | A possible format for character rigs. |
| Remotion | Free for individuals and companies of up to 3 people; works as a browser renderer. |

### CPU prototypes (committed in `tools/`)

- **`puppet_test.py`:** builds a puppet from one painting. The joints are found correctly, but the limbs tear when cut out of a single painting.
- **`vector_puppet_test.py`:** the reference technique working in our own art, at 0.84 s per frame at 540x960.

### Recommended pipeline: a layered puppet engine on CPU

1. **Generate rig-ready characters** with Z-Image-Turbo or SDXL: side or three-quarter view, arms clear of the body, plain background.
2. **Rig them automatically:**
   - cut each character out with BiRefNet;
   - find the joints with RTMPose;
   - split it into capsule-shaped parts, with caps where the joints overlap;
   - fill in behind the limbs with LaMa;
   - add variant sprites for eyes and mouth.
3. **Bones and poses:** forward kinematics, keyed pose-to-pose from a pose library: point, reach, hand to heart, nod, look, bow, walk.
4. **Procedural motion:** breathing, sway, cloth springs and blinks.
5. **Scenes:**
   - 5-7 parallax layers;
   - particles and light effects;
   - the camera moves from section 1;
   - cuts placed in narration pauses.
6. **Optional:** one image-to-video hero shot.

**Why this pipeline:** it is the only approach that reaches the reference's amount of motion, across 2-3 minutes per video, on free compute, with characters that stay exactly consistent.

## 6. Art directions

The stills are pending in the GPU job; the prompts are in `kaggle_jobs/gpu/research_gpu.py`.

- **A. Gilded folk tapestry:** lacquer-box and illuminated-manuscript miniature style; emerald, garnet and sapphire with gold filigree.
- **B. Vector cel (recommended):** clean cut-out shapes with soft gradients; ruby, emerald, sapphire, marigold and gold. The easiest to rig.
- **C. Paper theatre:** a layered, gilded cut-paper diorama with jointed paper puppets held by brass split-pins, lit like stained glass. The puppet joints become part of the look.

All three avoid the reference's indigo-and-gold devotional look, and avoid candle and flame imagery.

## 7. Compute

| Option | Details |
|---|---|
| Kaggle | 29.6 GPU hours left this week (resets 28 Nov), at most 2 sessions at a time, and the queue was stuck for 50+ minutes today. CPU kernels don't queue, so use them for anything that can run on CPU: puppet rendering, and TTS as a fallback. |
| Modal | $30/month in credits, T4 about $0.59/hour. Needs an account and a card or verification. |
| Lightning AI | About 15 credits/month, roughly 22 T4 hours. |
| Colab free T4 | Interactive sessions only. |
| Hugging Face ZeroGPU | A few minutes a day on a free account; Pro is $9/month. |

## Decisions needed

1. **Art direction:** A, B or C, once the stills arrive. B is the closest technique.
2. **Character pipeline:** AI raster puppets (a richer look, harder to automate) or SVG/vector puppets (exactly the reference technique, crisp, cheap to render, but the art quality depends on how well the parts are drawn).
3. **Voice:** pick by ear from `voices/`. Also decide whether a visible Higgs credit is acceptable.
4. **Music:** our own ACE-Step cues, the YouTube Audio Library, or Epidemic Sound at $10/month.
5. **Format:** whether to add long-form and compilations alongside the Shorts.

## Sources

- YouTube monetisation policies: <https://support.google.com/youtube/answer/1311392>
- YouTube AI disclosure: <https://support.google.com/youtube/answer/14328491>
- YouTube Audio Library: <https://support.google.com/youtube/answer/3376882>
- Pixabay licence summary: <https://pixabay.com/service/license-summary/>
- AIR, Shorts vs long-form RPM: <https://air.io/en/air-data-findings/youtube-shorts-rpm-vs-long-form-how-much-do-shorts-earn-in-2026>
- Digiday on Shorts revenue: <https://digiday.com/media/how-youtube-shorts-revenue-compares-to-long-form-video-revenue-for-creators>
- vidIQ on Shorts monetisation: <https://vidiq.com/blog/post/youtube-shorts-monetization/>
- TubeBuddy on AI demonetisation risk: <https://www.tubebuddy.com/blog/youtube-ai-demonetization-how-to-spot-risk/>
- Meta Creator Fast Track: <https://about.fb.com/news/2026/03/creator-fast-track-grow-your-audience-earn-money-on-facebook/>
- Epidemic Sound Creator plan: <https://www.epidemicsound.com/our-plans/creator-plan/>
- ElevenLabs API pricing: <https://elevenlabs.io/pricing/api>
- Colab alternatives: <https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025>
- Live2D SDK licence: <https://www.live2d.com/en/sdk/license/>
- Model licences were checked directly on each Hugging Face model card.
- YouTube channel statistics were pulled with yt-dlp on 21 Nov 2026.
