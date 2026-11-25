# Heartwick Stories: Style Bible

What we learned from the reference reel, and the look Heartwick uses instead.

The reference is a Parabrahma World Short (Hindi mythology, 2.5D cut-out animation). We copy its **craft**: pacing, layering, motion and caption mechanics. We do **not** copy its logo, branding, characters, colour scheme, lettering or text. Heartwick has its own palette, type, emblem and end card. There is **no candle or flame imagery** anywhere, because Heartwick Candles is an existing brand.

All numbers below were measured with `tools/analyze_reference.py`, `tools/dissolve_detect.py`, `tools/capband_sheet.py` and `tools/strip.py`. Raw output is in `analysis/`.

---

## 1. Reference: measured facts

### 1.1 Format
| Property | Value |
|---|---|
| Resolution / fps | 1080x1920, 30 fps, VP9 (YouTube copy) |
| Length | 174.3 s (2 min 54 s) |
| Audio | AAC 48 kHz stereo |

### 1.2 Pace (shots)
- **Visual transitions:** about 40 in 174 s, which is **about 14 shots per minute**. The count includes hard cuts, cross-dissolves and big camera reframes that feel like new shots.
- **Shot length:** median **3.2 s**, mean 4.25 s. 10th to 90th percentile is 1.4 s to 8.7 s. The longest hold is about 16 s, a slow push on a hero tableau.
- **Hard cuts** cluster in the middle act: the 16 cuts you listed fall between 50.8 s and 129.5 s. Elsewhere, transitions are mostly **soft cross-dissolves of 0.5 to 1.0 s**, for example at 13.0 s and 142.0 s, where the ghost of the old scene is still visible.
- One scene often gets **several "shots" without a cut**: the camera tilts, pans or pulls back across a tall layered scene and reveals new elements. For example, 13 to 18 s goes from empty sea, to a lotus rising, to the reclining figure. That is how the reel stays fast without needing many backgrounds.

### 1.3 Pace (words and captions)
Caption band read frame by frame from 66 to 81 s (`analysis/capband_66_81.png`):
- **14 caption chunks in 15 s**, so about **1.07 s per chunk**. Typical chunk durations are 1.0 to 1.5 s.
- **3 to 4 words per chunk** (rarely 2). This gives about **3.4 Hindi words per second**. Hindi has many short particles, so the English equivalent is about **2.6 to 2.9 words per second (155 to 175 wpm)**.
- Chunks are **back to back**: there is no empty gap between them, and the old chunk is replaced by the new one. Each new chunk **fades in over about 0.2 to 0.25 s**.
- **One highlighted word in about 70% of chunks**, always the content word that matters (a name, place or key noun). The other words are white. The other 30% of chunks have no highlight.
- Captions are visible **87% of the time**. Most of the rest is the opening title and the end card.
- **Speech pauses:** 93 pauses of 0.3 s or more, mean **0.52 s**. The narration is continuous and brisk, with short breaths between phrases.

### 1.4 Caption look
- **Position:** horizontally centred, with the **text centre at about 71% of frame height** (y ≈ 1365 px). That is the lower-middle of the frame, above the Shorts UI. It is not the exact middle.
- **Type:** heavy sans, cap height about 60 px at 1080 width, white. The highlight colour is a pale gold.
- **Backing:** a **soft, dark, elliptical shadow blob** behind each chunk (blurred, roughly 1.3x the text width and 2.5x its height, about 50% black). There is no box and no hard outline.
- **Top-left tag:** a two-line source/series label at x ≈ 60 px, y ≈ 230 to 300 px, in small gold bold sans, present in every shot after the opening.
- **Opening hook:** for the first **~1.8 s** a large two-line title sits in the upper third (with a small kicker line above it) over the first shot. The tag replaces it at about 2 s.
- **End card:** about 8 s. Logo emblem in a thin ring, wordmark, and native-language subtitle over a starry indigo field. The final CTA captions run over it.

### 1.5 Motion (what makes it "not a slideshow")
- **Something is always moving.** Only 8% of frames are near-static. The median frame-to-frame difference is 4.0 grey levels at 270 px width.
- **Camera:** slow push-in, pull-back, horizontal pan and **vertical tilt reveals** across tall compositions. Moves ease in and out. Pans are typically 0.3 to 1.0 px per frame at 270 px width, which is about **1 to 4 px per frame at 1080 px**.
- **Parallax:** 3 to 5 depth planes per scene: sky or starfield, distant water or land, midground props (trees, pillars, a throne), characters, and foreground elements (lotus, clouds, petals). Planes move at different speeds.
- **Characters** are articulated puppets: limbs rotate at joints (an arm gestures, a reclining figure sits up). They also have idle sway, a breathing bob and small head turns. The same character designs and colours recur in every scene.
- **Particles and atmosphere:** drifting sparkles, falling petals, galaxy swirls, light shafts, haze and cloud banks moving across the frame, and soft glows behind holy figures.
- **Transitions:** cross-dissolves of 0.5 to 1.0 s with a slight zoom continuing through the dissolve. Fade to the end card.

### 1.6 Layer count (typical shot)
1. Sky or space gradient with stars
2. Far background (sea horizon, mountains, palace)
3. Mid props (trees, pillars, a boat)
4. 1 to 4 character puppets
5. Foreground (lotus, clouds, petals, waves)
6. Particle overlay
7. Grade (vignette, glow)
8. Caption + tag overlay

### 1.7 Audio
| Measure | Value |
|---|---|
| Integrated loudness | **-17.2 LUFS** |
| Sample peak | -1.05 dBFS |
| Speech RMS (median, loud segments) | -15.5 dB |
| Floor during speech pauses (music bed) | -41 dB |
| **Music under voice** | **about 25 dB below speech** (a quiet, sustained pad) |
- There is one calm narrator, close-mic, with no sound effects that compete with the voice.

---

## 2. Heartwick: our own look

### 2.1 Mood
Cozy American storybook: New England seasons, kitchen light, maple, snow, porches and windows. It should feel warm and hand-made. The reference is cosmic and devotional; ours is the opposite.

### 2.2 Palette: "Hearth & Maple"
| Token | Hex | Use |
|---|---|---|
| Night Pine | `#10233A` | night skies, end-card base, shadow blob tint |
| Dusk Teal | `#1F4E5C` | dusk and storm scenes, cool shadows |
| Maple | `#B8432F` | maple leaves, warm props |
| **Ember Peach** | `#FFA552` | **caption highlight word**, CTA pill |
| Honey | `#F3C77A` | series tag line 2, sparkles, warm light |
| Cream | `#FFF4E2` | caption base text, wordmark |
| Sage | `#8DA67E` | spring scenes, secondary accents |

The rule is warm light inside and cool light outside. Every scene has one warm light source (a window, a lamp or the sun) that the particles and glow pick up. We **avoid the reference's indigo/purple + gold scheme** as a dominant look.

### 2.3 Illustration style
- A painted storybook with flat shapes and soft gradient shading (gouache / cut paper). It is **not** 3D, not photoreal and not anime.
- Characters are drawn as separate cut-out puppets on clean edges, and backgrounds are painted plates with no characters in them.
- The global style prompt is in `engine/style.py` (`STYLE_POSITIVE` / `STYLE_NEGATIVE`). The negative prompt always includes candles, flames, text and watermarks.

### 2.4 Lettering
| Element | Font | Spec |
|---|---|---|
| Captions | **Nunito Black (900)** (OFL) | 66 px at 1080 w, Cream `#FFF4E2`, 1 highlight word in Ember Peach, warm dark blob behind (Night Pine at 55%, blur 28 px) |
| Hook title | **Fraunces SemiBold, soft** (OFL) | 84 px, two lines, upper third, first 2.2 s; kicker line in Nunito 800, 30 px, letter-spaced |
| Series tag | Nunito ExtraBold 26 px, letter-spaced caps `HEARTWICK STORIES` (Cream at 85%), then Fraunces italic 34 px story label in Honey | top-left, x = 64, y = 210 |
| End card wordmark | Fraunces SemiBold 92 px | Cream |

Captions are centred at **70% of frame height**, with chunks of 2 to 4 words, a maximum of 18 characters per line and one line. A chunk breaks early at punctuation. The highlighted word is the longest content word, or a name, or a word from the story's `highlight` list.

### 2.5 Emblem and end card
- **Emblem:** an open storybook whose two top page-curls form a heart, with a small maple leaf as a bookmark ribbon. It is drawn in code (`engine/brand.py`), in Cream and Ember on Night Pine, inside a thin dotted circle. There is no flame, wick or candle.
- **End card (3.5 to 4 s):** a Night Pine to Dusk Teal gradient, drifting honey motes, the emblem scaling up gently, the wordmark, the tagline "Stories that stay with you", and an Ember pill with the CTA ("Follow for Part 2", or "Full story on Heartwick Stories" on a long video).

### 2.6 Pace targets for Heartwick
| | Shorts / pilot | Long-form (8 min) |
|---|---|---|
| Narration speed | Kokoro speed 1.0 (about 150 to 165 wpm) | 0.92 to 0.95 (about 135 to 145 wpm, per file 12) |
| Pause scale ([pause Xs] markers) | x0.45 | x0.8 |
| Shot length | 2.5 to 5 s (auto-split long lines into sub-shots of the same scene) | 4 to 8 s |
| Shots per minute | 13 to 16 | 8 to 11 |
| Caption chunk | 2 to 4 words, about 1.1 s | same |
| Dissolve | 0.6 s (0.3 s within a scene) | 0.8 s |

### 2.7 Motion recipe (engine v1, superseded by section 4)
- Every shot gets a camera move: `push_in`, `pull_back`, `pan_left`/`pan_right`, `tilt_down`/`tilt_up` (on tall plates) or `drift`. Each uses a cubic ease of 4 to 9% scale or 60 to 300 px travel.
- **Background plates** are split into depth with Depth-Anything-V2. The far plate moves at 0.55x and the near plate at 1.0x of camera travel, and the plate is displacement-warped for real 2.5D parallax.
- **Character puppets** have idle breathing (1.2% vertical scale at about 0.25 Hz), a sway from the feet (shear ±1.5°), a bend deformation (the top of the body leans more than the feet), and per-pose extras (`hop`, `nod`, `tremble`).
- **Foreground props** (leaves, branches, window frames) move at 1.3x for depth.
- **Particles** are chosen per scene: `motes`, `leaves`, `rain`, `snow`, `fireflies` or `mist`.
- **Grade:** a warm or cool tint per scene, a vignette, and a soft bloom on highlights.

### 2.8 Audio
- One warm American female narrator (Kokoro-82M, Apache-2.0). The default is `af_heart`, and `af_bella`/`af_nicole` are the alternatives (see `out/voice_samples/`).
- The **final mix is -15 LUFS integrated**, with a true peak of -1.5 dBTP or lower. YouTube normalises to about -14, so we sit just under that.
- **Music bed slot:** a WAV/MP3 supplied by the channel owner from the YouTube Audio Library, auto-ducked to **22 dB under the voice**. This is a little louder than the reference's 25 dB, because English narration has more pauses. See `brand/MUSIC_PICKS.md`.

---

## 3. What we deliberately do NOT copy
- Their logo, emblem, wordmark, the "Parabrahma World" name and their source-tag text.
- Their characters (deities), their props and their indigo/gold devotional palette.
- Their lettering (we use Nunito and Fraunces) and their caption colours (our highlight is Ember Peach, not gold).
- Their scripts or text. Every Heartwick script is original (see `05_AUTOMATION_PLAN` guardrails).

---

## 4. Engine v2: puppet animation (21 Nov 2026)

v1's 2.5D slideshows were rejected: characters stood still for 5-6 s. v2 matches the reference's *technique*, measured
across six Parabrahma World reels (`research/parabrahma/REPORT.md`): code-driven jointed puppets in layered scenes.
Every number below is enforced by code and checked by `tests/test_engine2.py` and each render's `_report.json`.

### 4.1 Characters
- **Jointed puppets** (`engine2/characters.py`): tube limbs with round joint caps, a profile head about 1/8 of the
  height, one shared skeleton for every human (pelvis, spine, chest, neck, head, two arms of three bones, two legs of
  three bones) plus spring chains (scarf, hair wisps, squirrel tails). Outfits are data: a new character is a dict.
- **Two part sources, one rig:** vector part libraries (the reference's own technique, crisp, cheap), or raster part
  sheets cut automatically from an AI image (`engine2/autorig.py`: BiRefNet cut-out, RTMPose joints, capsule parts with
  joint caps, the torso filled behind the limbs). Both use the same bone names, so every pose drives either.
- **Pose library** (`engine2/poses.py`): stand, point, point_up, reach, reach_down, heart, both_heart, arms_open,
  think, hold, hold_up, phone, wave, look_up, look_down, bow, shrug, hands_clasped, talk; kneel, kneel_reach,
  kneel_heart, kneel_look_up, kneel_cup; sit, sit_hold, sit_sad, sit_look_up, sit_read, sit_heart; a procedural walk
  cycle with planted feet; squirrel sit, sit_look, alert, nibble, crouch, look_back, curl and a run cycle.
- **Timing (reference):** a new pose every 1-1.5 s, eased over 0.2-0.4 s, held 0.6-1.5 s. The scheduler draws from the
  current pose family (once she kneels, she gestures from kneeling), and story beats pin poses to words
  (`do=nell:kneel@knelt`).
- **Life on top:** breathing (0.27 Hz, chest and shoulders), weight sway and head micro-motion (smooth noise), blinks
  every 2.2-4.8 s, short overlays (nod, sigh, tremble, waving hand), damped springs on scarves, hair and tails that
  lag and overshoot, a puppet "turn" (squash through zero), soft contact shadows. No lip sync, like the reference.

### 4.2 Camera, layers, light
- **Always moving:** pans 6-13 px/s (0.2-0.44 px/frame at 1080), zoom 0.7-1.4 %/s, 4 s smoothstep tilt reveals on tall
  scenes that then keep creeping. Framings cycle wide / medium (1.32x) / close (1.7x) inside a scene; heads are never cropped.
- **5-7 parallax layers** per scene: sky (0.08), far hills (0.3), mid houses/trees (0.6), ground (0.92), actors (1.0),
  foreground (1.25), light (rays, glows, mist, particles), then the vignette. Static layer art is rasterised once per
  scene and cached; only moving parts (swaying canopy, twinkles, particles) are drawn per frame.
- **Scenes are data** (`engine2/scenery.py` `LOCATIONS`): an exterior builder (time of day, season, house, maple,
  church, weather) and an interior builder (window with the outside, curtains, wainscot, rug, furniture, a sill stage).
  Every location of `stories/pip.md` is mapped.

### 4.3 Editing, captions, brand, sound
- **Cuts land in narration pauses** (word timestamps): story shots split at the pause nearest the ideal length, giving
  10-19 shots/min; dissolves of 0.7-1.6 s at chapter changes or `trans=dissolve`, sized to the pause they sit in.
- **Captions:** 2-4 words, one highlighted word (Ember), fade in over 0.2 s, centred at 71 % of the height on the
  soft dark ellipse; Nunito Black. Series tag top-left after the 0-2 s hook title; end card (emblem, wordmark, CTA).
- **Sound:** narration dry and close (70 Hz high-pass, gentle 3:1 compression, no room), a synthesised warm pad with no
  beat (slow Am-F-C-G, low-passed at 1.2 kHz, +3 dB swells on tilt reveals) set 20 dB under the voice by integrated
  loudness, master -14 LUFS, true peak <= -1.5 dBTP. No sound effects on cuts.

### 4.4 The two art directions under test
| | B "vector cel" | C "paper theatre" |
|---|---|---|
| Surfaces | clean shapes, soft linear-gradient shading | flat cut paper with fibre grain |
| Edges | crisp, no outline | thin gold-leaf edge on every piece |
| Depth | parallax + vignette | parallax + a paper drop shadow under every piece (diorama) |
| Joints | round caps with a small gold dot | brass split-pins (domed head, slot) |
| Frame | full bleed | gilded ruby proscenium with curtain swags, warm footlight glow |
| Render cost (1080x1920, per frame, one core) | about 0.5 s | about 0.8 s |

Both keep Heartwick's palette (warm inside / cool outside, emerald-ruby-sapphire-marigold-gold accents), avoid the
reference's indigo-and-gold devotional look, and never show candles or flames.
