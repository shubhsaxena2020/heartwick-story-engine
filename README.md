# Heartwick engine

> **This is a sanitized public snapshot.** Personal data, credentials, the channel owner's unpublished stories and their rendered outputs (`out/`) were removed. `stories/sample.md` is a placeholder story; numbers quoted below were measured on the original (private) stories.

One command turns a Heartwick story script into a finished vertical story video (1080x1920, 30 fps), with an `.srt`, a contact sheet and a QA report. **Engine v2** (default since 21 Nov 2026) is a code-driven **jointed-puppet animation engine**: characters change pose every 1-1.5 s, breathe, sway, blink and have springy scarves and tails, in 5-7 layer parallax scenes with an always-moving camera and cuts on narration pauses. It replaces v1's 2.5D slideshows (rejected: characters stood still for 5-6 s). The look is in [STYLE_BIBLE.md](STYLE_BIBLE.md) section 4; the targets come from `research/parabrahma/REPORT.md`.

**Guardrails (from the channel owner's planning docs, `05_AUTOMATION_PLAN`):** original scripts only, a distinct angle per video, no mass-templated uploads, and **nothing is ever uploaded by the engine**. The channel owner watches every video and approves every upload.

## Engine v2: one command
```powershell
cd C:\Users\shubh\Projects\heartwick
python heartwick.py pick-style B            # once: B vector cel or C paper theatre becomes the default
python heartwick.py make stories/sample.md                          # -> out/pip_short_B/
python heartwick.py make stories/sample.md --style C --voice voxcpm2:female --name pip_short_C_vox
python heartwick.py make stories/sample.md --dry                    # just count lines and missing narration
python tools/shot_sheet.py stories/sample.md B out/v2_preview/sheet.jpg   # one frame per shot, no Kaggle, seconds
python tools/style_tests.py                                            # re-render the B/C style tests
```
`make` (engine v2) does:
1. Parses the story (v1 format plus optional v2 beats, below) and lists every narration line for the voice. Lines are content-addressed and cached in `cache/<slug>/tts/` (v1's Kokoro caches are reused).
2. Missing lines go to **one Kaggle narration job** (`kaggle/tts_v2.py`): Kokoro on a **CPU kernel** (no GPU queue, native word timestamps), or a cloning voice (`voxcpm2:female|male`, `chatterbox:female|male`) on a GPU kernel, cloned from the designed house voices (no real person) with faster-whisper word times aligned onto the script text.
3. Renders on a **Kaggle CPU kernel** (`kaggle/worker_v2.py`, 4 worker processes, skia). Narration rides inside the script as FLAC, or through the private `heartwick-cache-<slug>-tts` dataset when bigger.
4. Writes `out/<name>/`: `<name>.mp4`, `<name>_sheet.jpg`, `<name>_report.json` (shots/min, cuts in pauses, pose-change intervals, camera speeds, motion density measured exactly like the reference, loudness), `<name>.srt`, `<name>_timeline.json`.

`--engine v1` runs the old pipeline (documented below).

### Story format additions (v2 only; v1 stories render unchanged)
```
::: shot bg=yard_morning cam=push_in chars=nell:stand@0.34,0.95,0.6;pip_kit:shiver@0.6,0.95,0.07:tremble do=nell:kneel@knelt,kneel_reach@complaining
```
- `do=<char>:<pose>@<word>,...;<char>:...` pins a pose to the first matching word (gestures in between are automatic, every 1-1.5 s, from the current pose family).
- `walk=<char>:<x0>><x1>@<t0>-<t1>` walks across the frame; `turn=<char>@<word>` puppet turn; `trans=dissolve`; `frame=wide|medium|close`.
- `bg=` names a location in `engine2/scenery.py` `LOCATIONS` (all of the original pilot story is mapped); v1 pose names map to v2 poses plus props (`engine2/story2.py` `POSE_ALIAS`: `stand_mug` -> hold + mug, `shiver` -> curl + tremble, ...).

### Two puppet sources
- **Vector** (default): `engine2/characters.py` part library. Crisp, cheap, exactly the reference's technique.
- **Raster (AI image)**: generate a rig-ready figure (side or three-quarter view, arms clear of the body, plain background; prompts `rigB_nell` / `rigC_nell` in `research/kaggle_jobs/gpu/research_gpu.py`), then
  `python tools/autorig.py research/art/rigB_nell_s11.png --name nell` runs BiRefNet + RTMPose + `engine2/autorig.py` (+ LaMa) on a Kaggle CPU kernel and installs `characters/nell/rig/`. `python heartwick.py make <story> --raster` then uses it. Same bone names, so every pose, beat and camera works unchanged (`tests/test_engine2.py::test_autorig_partsheet_reproduces_source_and_poses`).

### v2 layout
```
engine2/rig.py        bones, forward kinematics, vector + raster parts, grounding, part-sheet loader
engine2/characters.py vector puppet cast (Nell, Dee, Eli, Pip, the newborn kit), props
engine2/autorig.py    AI image -> raster part sheet (cut-out + COCO keypoints -> capsule parts, joint caps, torso fill)
engine2/poses.py      pose library (semantic angles), walk/run cycles
engine2/motion.py     pose tracks, 1-1.5 s gesture scheduler, breathing/sway/blinks, springs, Actor
engine2/camera.py     always-moving camera within the reference's speed ranges, 4 s tilt reveals
engine2/scene.py      layered parallax scenes with cached static layers, vignette
engine2/scenery.py    props + exterior/interior builders + LOCATIONS     engine2/fx.py  particles, rays, glows, mist
engine2/style.py      painters: B vector cel, C paper theatre             engine2/direct.py  placement, beats, cameras
engine2/edit.py       cuts on pauses, sub-shots, dissolves, captions     engine2/audio.py   voice chain, pad, mastering
engine2/render.py     frames -> mp4 (parallel), QA facts                  engine2/qa.py      motion density, contact sheets
engine2/pipeline.py   the `make` pipeline     engine2/kaggle2.py  CPU-kernel push      kaggle/worker_v2.py, kaggle/tts_v2.py
```

### Measured on the style tests (1080x1920/30 fps, Kaggle CPU, `out/style_tests/`)
| Target (reference) | B vector cel | C paper theatre |
|---|---|---|
| Seconds with motion beyond the camera, reference method (65-97 %) | 100 % | 100 % |
| Seconds where a character's joints move > 6 deg (stricter, no particles) | 100 % | 100 % |
| Pose change interval (1-1.5 s) | median 1.36 s | median 1.36 s |
| Camera, median px/frame at 1080 (0.2-0.44) | 0.29 | 0.24 |
| Near-static frames (reference 4-35 %) | 3.9 % | 7.6 % |
| Shots per minute (10-19) / cuts in pauses (71-100 %) | 16.7 / 100 % | 16.7 / 100 % |
| Dissolve (0.7-1.6 s) | 1.52 s | 1.52 s |
| Loudness / true peak / pad under voice | -14.0 LUFS / -1.8 dBTP / 20 dB | same |
| Render cost per frame, one core | 0.55 s | 0.81 s |

**Pip Short, style B** (`out/pip_short_B/`, one `make` command, Kokoro `af_heart` provisional voice): 106 s, 30 shots (17.6/min), motion beyond the camera in 92.2 % of seconds, characters' joints moving in 80.4 % (the rest are character-less establishing shots), pose change median 1.27 s, camera median 0.24 px/frame, 79 % of cuts in measured silences (the rest after a punctuated word: Kokoro's word timestamps have no gaps inside a line), dissolves 1.28-1.6 s, -14.0 LUFS / -1.7 dBTP, pad 20 dB under. Kaggle CPU: narration 158 s, render 486 s (0.49 s/frame/core).

**Research inputs:** voice comparison run 1 in `research/voices/comparison_gpu1.csv` plus `voice_*.mp3` (Kokoro, Qwen3-TTS VoiceDesign, VoxCPM2 design and clone of the designed house voices; Whisper large-v3 WER 0.0 for all, 2.1-3.0 words/s). Reference narration pace from the Hindi ASR: 1.7-2.2 words/s, first word at 0.0 s (`research/parabrahma/metrics/narration_pace.json`).

**AI-art puppets:** put them in `assets/characters/<name>/` (contract in `engine2/assets.py`: a ready part sheet `rig.json` + PNGs with the engine's bone names, or one rig-ready figure `<name>.png` + COCO-17 `keypoints.json`, auto-rigged once). `make` then uses them automatically; `--vector` forces the vector cast.

The camera is always moving (pans 6-13 px/s, zoom 0.7-1.4 %/s), captions are 2-4 words with one highlight, fading in over 0.2 s at 71 % height. Tests: `python -m pytest -q tests/test_engine2.py` (rig/FK, both part sources, poses, scheduler, springs, camera limits, cuts on pauses, captions, mastering, layer caching; ~9 s).

## Engine v1 (legacy 2.5D slideshow, `--engine v1`)

## Where the compute runs
| Step | Where | Model / tool | Licence |
|---|---|---|---|
| Narration + word timestamps | Kaggle GPU (T4) | Kokoro-82M, voice `af_heart` | Apache-2.0 |
| Background plates, characters | Kaggle GPU (2x T4 in parallel) | SDXL 1.0 base + `madebyollin/sdxl-vae-fp16-fix` | CreativeML OpenRAIL++-M (commercial use allowed) / MIT |
| Character consistency | Kaggle GPU | IP-Adapter Plus SDXL (ViT-H) from a fixed master image + fixed seeds | Apache-2.0 |
| Cut-outs (character layers) | Kaggle GPU | BiRefNet | MIT |
| Depth for 2.5D parallax | Kaggle GPU | Depth-Anything-V2-Small | Apache-2.0 |
| Compositing + encode | Kaggle CPU (4 workers inside the same job) | this repo (`engine/render.py`), ffmpeg/x264 | - |
| Orchestration, SEO, thumbnail, QA | Laptop (light, BelowNormal priority) | Python + ffmpeg `-threads 4` | - |

Fonts are Nunito and Fraunces (SIL Open Font License, in `brand/fonts/`). No music is generated: see [brand/MUSIC_PICKS.md](brand/MUSIC_PICKS.md).

## Weekly run
```powershell
cd C:\Users\shubh\Projects\heartwick
# 1. write the story: stories/<slug>.md (copy stories/sample.md as a template; see "Story format")
# 2. first time a new character appears: make master candidates, look, pick
python heartwick.py make stories/<slug>.md --all-masters --no-render
#    -> out/<slug>/master_candidates.jpg ; then for each character:
python heartwick.py pick <character> <k>
# 3. make the video (long form, ~8 min)
python heartwick.py make stories/<slug>.md
#    or a Short from chapters 1-2 at Shorts pace
python heartwick.py make stories/<slug>.md --sections 1-2 --name <slug>_short --short
#    optional music bed from the YouTube Audio Library
python heartwick.py make stories/<slug>.md --music music/<track>.mp3
# 4. check Kaggle GPU usage
python heartwick.py status
```
Each `make` call does the following:
1. Parses the story and builds a plan.
2. Works out which assets are missing. Assets are content-addressed, so unchanged plates, poses and narration lines are reused.
3. Pushes **one** Kaggle job (`kaggle/worker.py` plus the engine code inside it), polls it, and downloads the results.
4. Writes `out/<name>/`:

| File | What |
|---|---|
| `<name>.mp4` | final video, 1080x1920, 30 fps, H.264, AAC 48 kHz, -15 LUFS |
| `<name>.srt` | English captions for YouTube (cue-synced to the narration) |
| `SEO_PACK.md` / `.json` | titles, file name, description with chapters, tags, hashtags, pinned comment, settings checklist |
| `thumbnail.jpg` | 1280x720 thumbnail with the story's thumbnail text |
| `contact_sheet.jpg`, `vs_reference.jpg` | visual QA |
| `qa_report.json` | shots/min, loudness, true peak, caption-sync check |
| `timeline.json`, `plan.json` | exact shot/caption timing (useful for edits) |

If assets are all cached and only the render changes, the job runs as a **CPU** kernel (`heartwick-render`, no GPU quota). The cached assets go up through the private Kaggle dataset `heartwick-cache`. `--local-preview` renders a 540x960 / 15 fps preview on the laptop at BelowNormal priority instead.

### One command, start to finish
For a story with **new characters**, `--auto-pick` makes one master per new character inside the same Kaggle job. That job then makes the poses, plates, narration, render and Whisper check, so the whole thing is a single command:
```powershell
python heartwick.py make stories/sample.md --short --auto-pick --name sample_short
```
The masters are saved to `characters/<name>/` and reused by later stories. Use `--all-masters --no-render` plus `pick` instead when you want to choose between 4 candidates.

## Dashboard (Heartwick Studio)
**https://heartwick-studio.vercel.app** (password login, then a 30-day cookie). Works from any laptop or phone. Same design system as the clip-factory ops console, but its own code, repo folder (`studio/`), Vercel project (`heartwick-studio`) and Blob store (`heartwick-studio-data`).

```
main laptop                                                   Vercel (heartwick-studio)
dashboard/exporter.py --snapshot on change (gzip)-->  /api/ingest --> Blob snapshot.json  (offline fallback)
                      --new/changed media--------->  Blob m/<sha>/<file>  (thumbs, contact sheets, 480p previews)
                      <--queued writes, at start + hourly ------  /api/inbox <-- Blob inbox.json <-- /api/actions
dashboard/live.py  127.0.0.1:8788 <- cloudflared quick tunnel <- browser polls /state every 3 s while visible
                   (URL posted to /api/live-url when it changes)   + POST /action (instant writes), /media
```
- **Tabs:** Overview (KPIs, now doing, pipeline board script → narration → images → render → review → approved → published, per-video lines/images, needs attention, activity), Videos (480p player, thumbnail, duration, shots/min, LUFS, true peak, caption sync and Whisper WER from `qa_report.json`, SRT / SEO pack / contact sheet, approve / needs changes + note / mark published / re-render, queue a story), Kaggle (GPU hours vs the 30 h quota from `kaggle quota`, every job with wait and run time, minutes per day and per video, queue-runner log), Charts (videos per stage over time, render time, GPU minutes, QA trends), Errors, Research (`research/RESEARCH_REPORT.md` + voice samples, read only), Channel (placeholder until the channel exists).
- **Accuracy:** everything comes from the real files (`studio_state.py`). The header's self-check badge recomputes key numbers a second way (ffprobe duration, ffmpeg loudness of the mp4, file counts, Kaggle's own quota vs the local job log) and lists any mismatch.
- **Writes** (approve, needs changes, queue story, re-render) are queued on Vercel first, so none is lost while the laptop is off; when the laptop is online the browser also hands them to `live.py`, which applies them at once. Each write has an id and is applied once (`dashboard/data/actions_applied.json`). Approvals land in `dashboard/data/approvals.json`; queue / re-render requests become pending jobs in `kaggle/queue.json`, run by `tools/run_queue.py`.
- **Laptop offline:** the site shows the last snapshot with a "laptop offline since X" banner and reconnects by itself.
- **Unattended:** the scheduled task "Heartwick Studio" (`tools/register_heartwick_task.ps1`) runs `tools/start_heartwick.ps1` hidden at logon + 2 min and every 15 min; it starts only what is not running, and starts the queue runner only when a job is pending.
- **Load:** the exporter stats ~100 files a minute and does real work only when something changed; `live.py` sleeps in `accept()` until someone watches. Both run BelowNormal on CPUs 12-13.
- **Blob budget** (Hobby: 2,000 uploads a month for the whole account): the snapshot is posted at once only for changes that matter (stage, job state, approval, new media, new error) and otherwise with a 2-hourly heartbeat; media is uploaded only when its bytes change.
- **Secrets:** `dashboard/data/studio.env` (git-ignored): `DASHBOARD_URL`, `INGEST_TOKEN`, `DASHBOARD_PASSWORD`, `BLOB_READ_WRITE_TOKEN`. The same `INGEST_TOKEN` and `DASHBOARD_PASSWORD` are set in the Vercel project (production).
- **Site development:** `cd studio; npm install; python scripts/make_fixture.py; npm run dev` (uses `fixtures/sample-snapshot.json` and local media through `LOCAL_MEDIA_INDEX`). Checks: `npm run lint`, `npm run typecheck`, `npm test`. Deploy: `vercel deploy --prod --scope shubh-saxena-s-projects` from `studio/`.
- The older local page (`python dashboard/app.py`, 127.0.0.1:8766) still works as a local fallback.

## Tests
```powershell
python -m pytest -q tests
```
The tests cover story parsing (pauses, emphasis, character specs, plain-text fallback), content-addressed planning and prompt length, caption chunking and timing, shot continuity, SRT validity, the SEO pack (hashtags, tags ≤500 characters, chapter rules), the Whisper WER/offset maths, and a tiny **real** render. That render goes from synthetic plates to an mp4 with audio, and the test checks its duration, 48 kHz audio, -15 LUFS, true peak, caption/speech sync and that frames are not static. The dashboard tests cover state, approvals, path traversal and password enforcement. No GPU is needed, and the suite takes about 35 s on the laptop at low priority.

## Kaggle notes (verified on real runs)
- Kaggle's image is now **Python 3.13**, and Kokoro/misaki pin `<3.13`. The worker installs them with `--ignore-requires-python`, which works.
- On T4: an SDXL plate (768x1344, 28 steps, plus a 1.5x hi-res pass) takes ~53 s, and a character takes ~30 s. Both GPUs run in parallel.
- An embedded job above a few MB is rejected (HTTP 400), so reference images are sent as small JPEGs and cached assets go through a private dataset.
- Only one GPU session runs at a time on this account; a second push waits in "queued".
- **CLIP reads only 77 tokens**, so the style prefix goes first and prompts stay short. `heartwick.py` warns when a prompt is too long.

## Story format
A Markdown file with YAML frontmatter. See `stories/sample.md` for a complete example.
```markdown
---
slug: pip
title: "..."
series_tag: "Pip"            # shown top-left under HEARTWICK STORIES
hook: {kicker: "A HEARTWICK STORY", lines: ["Something stirred", "in the leaves"], highlight: "leaves"}
highlight_words: [Nell, Pip, maple, ...]   # words that get the Ember caption highlight
seo: {titles: [...], summary: "...", hashtags: [...], tags: [...], pinned_comment: "...", thumbnail_text: [...]}
characters:
  nell: {desc: "...", seed: 1101, poses: {kneel: "kneeling ...", stand_mug: "..."}}
locations:
  yard_morning: {prompt: "...", tint: cool}     # tall: true for tilt-reveal plates
---
## Chapter title                               (becomes a YouTube chapter)
::: shot bg=yard_morning cam=push_in fx=mist,motes chars=nell:kneel@0.38,0.93,0.5;pip_kit:shiver@0.72,0.92,0.07:tremble
Narration line with [pause 1s] markers and *emphasis*.
```
- `cam`: `push_in`, `pull_back`, `pan_left`, `pan_right`, `tilt_down`, `tilt_up` or `drift`. Long shots are split automatically into sub-shots of the same scene with new framings (wide, medium, close) to keep the pace.
- `fx`: `motes`, `leaves`, `rain`, `snow`, `mist`, `fireflies` or `lightning`.
- `chars`: `name:pose@x,y,h[:anim]`. Here x is the centre and y the feet position (fractions of the frame), and h is the height as a fraction of frame height. `anim` is `idle` (breathing and sway), `tremble`, `hop` or `climb`.
- A story with no `::: shot` lines still renders: each paragraph becomes a shot, and locations are cycled. A storyboarded script looks much better, though.

## Cost and time per video (measured 2026-11-21)
**Money: $0.** Everything runs on Kaggle's free tier with open, commercially usable models. Music, if added, comes from the free YouTube Audio Library.

| Step | Where | Measured | Per video |
|---|---|---|---|
| SDXL plate (768x1344 + 1.5x hi-res) | Kaggle T4 | ~53 s each, 2 GPUs in parallel | |
| SDXL character master/pose (832x1216) | Kaggle T4 | ~30 s each | |
| Kokoro narration | Kaggle T4 | 61 s of speech in ~29 s | |
| Install + model loads | Kaggle | ~3-4 min per GPU job | |
| Render 72 s, 1080x1920, 30 fps | Kaggle CPU (4 workers) | 344-420 s | free, no GPU quota |
| Whisper check | Kaggle CPU / GPU | ~260 s CPU for 72 s (~20 s on GPU) | |
| **72 s Short, new characters** (2 masters, 5 poses, 6 plates, 17 lines) | | | **~12 GPU-min + ~12 CPU-min, ~30 min wall** |
| **8 min long video** (19 plates, 25 poses, 80 lines) | | | **~21 GPU-min + ~80 CPU-min (estimate)** |

The weekly quota is 30 GPU-hours (checked with `kaggle quota`), which is far more than a weekly upload needs. **The real limit is Kaggle's T4 queue.** On 2026-11-21, batch GPU jobs sat in QUEUED for more than 2 hours with 29.6 h of quota left, and even a 1-line probe kernel queued. Queued jobs count toward the **2-session limit**. So:
- `make` waits for a free slot by itself (it retries when the limit is reached).
- `make --no-gpu` renders on a CPU kernel from cached assets (no queue). Missing poses fall back to the character master cut-outs.
- `tools/run_queue.py` runs a list of jobs one at a time, detached, and logs to `kaggle/queue_runner.log` (shown in the dashboard).
- `python heartwick.py fetch <kernel> --story <slug>` collects a job that finished while nothing was waiting for it.

## Repo layout
```
heartwick.py          CLI (make / pick / status)
engine/story.py       story parser          engine/plan.py      assets + content-addressed IDs
engine/timeline.py    shots, sub-shots, caption chunks, chapters
engine/render.py      2.5D renderer (depth-parallax plates, puppets, particles, grade, captions, end card)
engine/brand.py       emblem, caption/tag/hook sprites, end card     engine/post.py  SRT, SEO, thumbnail, QA
engine/kaggle_run.py  push / poll / download / usage log             kaggle/worker.py  the Kaggle GPU job
characters/<name>/    chosen master images (series bible for recurring characters)
stories/              story scripts          tools/  reference-analysis scripts
```
