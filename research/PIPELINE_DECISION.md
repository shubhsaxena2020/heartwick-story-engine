# Heartwick pipeline decision: the best $0 way to make Parabrahma-class story Shorts (21 Nov 2026)

**Bottom line.** Stop hand-building puppets. Open-weight models are now good enough, and their licences allow commercial use:

- **Z-Image-Turbo** (Apache-2.0) paints each keyframe.
- **Wan2.2 image-to-video** (Apache-2.0) animates it.
- **RIFE and Real-ESRGAN** (MIT / BSD) finish it to 1080x1920 at 30 fps.

Together they reached the Gemini reference on the exact Nell beat. In **one 5-second clip**, Nell walks, crouches, lifts the squirrel and stands up holding it, with falling leaves and a push-in, and she stays on-model throughout. There is no watermark, no paid credits and no account login.

Our existing code engine stays, but as the *director and finisher*, not the animator. It handles the script, shot list, camera, captions, cuts on narration pauses, audio, QA and assembly.

The real constraint is **GPU minutes, not quality**. The throughput section below sizes them.

All prototypes live under `research/pipeline_proto/`. Side-by-side boards:

- `research/pipeline_proto/board_ours.jpg`: v2, A and B.
- `research/pipeline_proto/_local/board_with_refs.jpg`: also Parabrahma and Gemini. It is local only and not committed, because it contains their frames.

---

## 1. What was tested: the same beat in each candidate

**The beat:** Nell, an older woman in a cardigan, walks to a maple, kneels and lifts a small grey squirrel, with falling leaves and an always-moving camera.

| | **B. Open AI keyframe + open image-to-video** | **A. Blender 3D + CC0 animation library** | **C. A's blocking → pose → Wan-VACE repaint** |
|---|---|---|---|
| Where it ran | Public Hugging Face ZeroGPU Spaces, anonymously (no login): keyframe 31 s, video 66 s. Finished on the laptop iGPU through Vulkan: RIFE 18 s and ESRGAN 147 s for 152 frames. A Kaggle T4 run measures the free-compute speed (§4). | Laptop iGPU EEVEE: **224 s for 270 frames at 540x960** (0.83 s/frame). Laptop CPU stayed at about 30%. **Kaggle CPU kernel** (4 vCPU, no queue): EEVEE through software EGL **15.9 s/frame** at 540x960, so a 9 s shot takes about 72 min. Cycles CPU ran at 8.9 s/frame, but Cycles can't do the Shader-to-RGB toon ramp, so the character renders flat (`kaggle_cpu/eevee_vs_cycles_f30.jpg`). | Kaggle T4. v1 OOMed in the VAE encode, the same text-encoder memory issue. v2 is fixed (`vace_t4.py`) but **not pushed**: both GPU slots were taken (painted-assets and i2v-t4 v2 queued), and we never exceed 2. Push it with `kaggle kernels push -p research/pipeline_proto/C_pose_repaint/push` once a slot frees. |
| Output | `B_open_i2v/zerogpu/wan22_i2v.mp4` (480x832, 16 fps, 5.06 s), `wan22_i2v_1080p30.mp4` (finished), `wan22_sheet.jpg`, `upscale_compare.jpg` | `A_blender3d/out/nell_beat_blender_540x960.mp4`, `out/sheet.jpg`, `nell_beat.py` (fully code-built scene) | `C_pose_repaint/`: 81 OpenPose control frames exported from A's animation (`pose_sheet.jpg`, `pose_frames.zip`), the reference image `vace_ref.png`, and the kernel |
| What I saw | The whole beat in one take: a natural weight shift into a crouch, the squirrel lifted to her face, a smile, leaves drifting, a slow push-in. The finished frame has clean anime line art at 1080p, the same level as the Gemini frame. Faults: she crouches rather than kneels on one knee, the squirrel grows about 1.5x once lifted, the clip is 16 fps before RIFE, and the motion is not timed to a voice-over. | The walk and kneel are real mocap-quality motion, and the 3D camera parallax against the painted plate works. The look is a faceless mannequin with a hair cap, and the skirt balloons when she kneels. With a proper character model this would look like a "3D toon" show, not the reference's 2D style. | Not run yet. The control signal looks right: walk, then kneel and gather, in profile. |
| Quality vs reference (0-10) | **8.5** (≈ the Gemini bar, higher than the v2 puppet) | 3 (look) / 7 (motion) | — (not run) |
| Motion naturalness | **9** | 7 | — (not run) |
| Character consistency | 8 within a shot. Across shots it depends on the keyframes (§3.3). | **10** (same rig every time) | — (not run) |
| Automation by Claude | 9: prompts + JSON, Python and gradio/diffusers, no GUI | 8: pure bpy, but every new action or character needs assets | 8 |
| $ cost | 0 | 0 | 0 |
| Throughput (videos/week) | GPU-bound (§4) | Effectively unlimited: CPU or iGPU, about 4 min per 9 s at 540p | GPU-bound |
| Legal / commercial safety | 9: Apache-2.0 weights, outputs ours. YouTube's AI disclosure is description-only for animation. | 10: CC0 + GPL tool | 9 |
| Robustness | 6: model drift, prompt failures and queue risk, so it needs QA and retries | 9 | 6 |

The rejected v2 skia puppet, for comparison: quality about 4.6-5/10 (`out/walk_test_B/passes/scores.csv`), robustness 9, consistency 10.

---

## 2. The landscape: every option, ranked

### 2.1 Hosted AI video generators: free tiers, Nov 2026

**Strict rules:** a visible watermark, non-commercial free output, a needed login to Shubh's accounts, or an 18+ age gate each disqualify an option for *publishing*.

| Service | Free quota (reported) | Watermark on free | Commercial use on free | API / automation | Verdict |
|---|---|---|---|---|---|
| Kling | about 66 credits/day | yes | no (Standard plan, about $10/month, needed) | paid API only; scraping the web UI breaks its terms | ✗ |
| Hailuo / MiniMax | welcome credits plus a small daily amount | yes | no (paid only) | paid API | ✗ |
| Vidu | trial and daily login credits | yes, visible lower-right | no ("personal, non-commercial") | paid API | ✗ |
| PixVerse | 90 sign-up credits + 60/day | yes | no | paid API | ✗ |
| Pika | has changed recently; now reportedly no free credits or licence | yes | no | — | ✗ |
| Luma (Ray) | small | may be watermarked | **no**: Luma's own licence page says Free and Lite are "personal, non-commercial use only" | paid API | ✗ |
| Runway | 125 one-time credits | yes | — | paid API | ✗ |
| Wan official (create.wan.video) | small sign-up amount + 10/day check-in | reportedly yes on free | paid tiers only | Google login | ✗. **The same model family is open-weight (Apache-2.0)**, so self-host it instead. |
| Hunyuan / Tencent | web app quotas | — | — | open weights under the Tencent Hunyuan Community licence (excludes EU/UK/KR) | Self-host only. Wan2.2's Apache-2.0 licence is cleaner. |
| Meta AI / Vibes | free with rate limits | Meta branding | no clear grant | none | ✗ |
| Grok Imagine | video paywalled since Mar 2026 (Premium, about $30) | yes, xAI says it cannot be removed | — | — | ✗ |
| Sora app | **shut down 26 Apr 2026**; the API ends 24 Sep 2026 | — | — | — | ✗ |
| Bing Video Creator (Sora 2) | 10 fast generations, then unlimited slow | **visible watermark + C2PA**; removing provenance marks is forbidden | the terms are silent; "bot/scraping behaviours" are banned | none | ✗ |
| Adobe Firefly free | a few premium (video) generations a month | Content Credentials (metadata) | Adobe says outputs are commercially safe | needs an Adobe ID; free video allowance is tiny | ✗ for volume (fine for a one-off, but it is Shubh's account and a tiny quota) |
| Canva free | Veo is on paid plans only; a few Magic Media clips for free | conditional | design-use rules | none | ✗ |
| Google Flow / Gemini free | Flow: about 50 credits/day (≈5 Veo 3.1 Lite clips at 720p) | visible Veo watermark (reported to apply to all plans below Ultra; we saw it on Gemini Pro) | Google gen-AI terms; Labs tools are age-gated (18+ in most regions, unverified) | none | ✗ (watermark, account risk, age) |
| **Hugging Face ZeroGPU Spaces** | **2 GPU-min/day anonymous; 5 min/day with a free account** (HF docs). GPU: half of an RTX Pro 6000 Blackwell (48 GB). A free account can also host 2 ZeroGPU Spaces of its own, but the daily usage quota still applies. | **none**: you run the open model | follows the model licence: **Wan2.2, Z-Image and Qwen-Image-Edit are all Apache-2.0** | **official `gradio_client` API**; HF account minimum age is 13 | ✓ as a **supplement**: about 1-2 clips/day anonymously, 3-4 with a free account. We used it for prototype B. |

**Sources (one per row):**

- eesel.ai, Kling pricing: <https://eesel.ai/blog/kling-ai-pricing>
- wavespeed.ai, Hailuo pricing 2026: <https://wavespeed.ai/blog/cost-and-billing/hailuo-ai-pricing-2026/>
- wavespeed.ai, Vidu pricing 2026: <https://wavespeed.ai/blog/cost-and-billing/vidu-ai-pricing-2026/>
- atlascloud.ai, PixVerse free plan: <https://www.atlascloud.ai/nl/blog/guides/pixverse-ai-free>
- Luma licensing: <https://lumalabs.ai/learning-hub/licensing>
- Filmora free-tier comparison: <https://filmora.wondershare.com/answers/pika-vs-runway-vs-luma-dream-machine-free-tiers-compared.html>
- Wan pricing: <https://www.imagine.art/blogs/wan-ai-pricing-guide>
- TechCrunch on Vibes: <https://techcrunch.com/2025/09/25/meta-launches-vibes-a-short-form-video-feed-of-ai-slop>
- Grok Imagine free status: <https://aliteq.com/is-grok-imagine-free>
- Sora shutdown: <https://zsky.ai/blog/is-sora-free> (the free-cut history is in the Tom's Guide piece below)
- Tom's Guide on free-usage cuts: <https://www.tomsguide.com/ai/google-and-openai-are-cutting-back-free-usage-for-gemini-3-pro-and-sora-2-here-are-the-new-limits>
- Neowin, Bing Video Creator with Sora 2: <https://www.neowin.net/news/bing-video-creator-is-now-powered-by-openais-sora-2/>
- Bing Image and Video Creator terms: <https://www.bing.com/new/termsofuseimagecreator>
- Adobe Firefly plans FAQ: <https://www.adobe.com/cc-shared/fragments/products/firefly/plans/faq.plain.html>
- Canva AI access: <https://www.canva.com/help/ai-access/>
- Google Flow pricing: <https://wavespeed.ai/blog/cost-and-billing/google-flow-pricing-2026/>
- ZeroGPU docs: <https://huggingface.co/docs/hub/spaces-zerogpu>
- Hugging Face terms of service: <https://huggingface.co/terms-of-service>

Many quota figures come from third-party pricing blogs and conflict with each other. The disqualifying facts (watermark, non-commercial) are consistent across sources, and Luma's comes from Luma itself.

**Conclusion:** no hosted free tier is publishable. The only $0 route to AI video is **open weights on free compute** (Kaggle, ZeroGPU, or later Modal or Lightning credits on the father's account; see §5).

### 2.2 Open models we can self-host, all checked on their Hugging Face cards

| Model | Licence | Role | Notes |
|---|---|---|---|
| **Z-Image-Turbo** (Tongyi-MAI) | Apache-2.0 | keyframes, plates | 8-step; gave the B keyframe and the A background plate |
| **Wan2.2-I2V-A14B** + lightx2v **Lightning 4-step LoRA** | Apache-2.0 / Apache-2.0 | hero image-to-video | best motion; GGUF Q4 builds (QuantStack) for 16 GB cards |
| **Wan2.2-TI2V-5B** / **FastWan2.2-TI2V-5B** (3-step DMD) | Apache-2.0 | cheap image-to-video | fits a T4 in fp16 |
| **Wan2.1-VACE-1.3B** | Apache-2.0 | pose- and depth-controlled repaint with a reference character | prototype C |
| **Qwen-Image-Edit-2509** | Apache-2.0 | "same character, new shot" keyframes | the consistency tool (§3.3) |
| ToonCrafter | Apache-2.0 per README; the weights card has no licence field | in-betweening of cartoon keyframes | 512x320 only, landscape; superseded by Wan first-last-frame |
| LTX-Video | LTXV open-weights licence (free below a revenue cap) | fast image-to-video | weaker motion; our earlier Kaggle run failed |
| HunyuanVideo-1.5 | Tencent Hunyuan Community | image-to-video | territory exclusions; prefer Wan |
| RIFE (ncnn-vulkan) | MIT | 16 → 30/60 fps | 18 s per 5 s clip on the iGPU |
| Real-ESRGAN animevideov3 | BSD-3 | 2x anime upscale | about 1 s/frame on the iGPU |

### 2.3 Open-source animation toolchains Claude can drive by code

| Tool | Licence | Scripting | Can it reach Parabrahma-quality jointed puppets automatically? |
|---|---|---|---|
| **Blender** (bones, Grease Pencil, EEVEE / Cycles / Workbench) | GPL; output is ours | full Python (`bpy`), headless | **Yes for motion.** Tested: CC0 clips sequenced and blended by code. EEVEE needs a GPU / OpenGL (the iGPU works, 0.83 s/frame at 540p); Cycles runs on CPU anywhere. Weak spot: **character art**. Every character needs modelling (or image-to-3D + auto-rig), and skirts need cloth sim or careful weights. |
| Synfig | GPL | XML (.sif) + plugins; skeleton deformation | possible but archaic; no good automation story |
| OpenToonz | BSD | limited scripting | GUI-centric; no |
| Krita animation | GPL | Python (frames) | raster frame-by-frame; no rigs |
| Motion Canvas | MIT | TypeScript | yes for vector puppets and camera; it is the same architecture as our skia v2, so the same art bottleneck |
| Remotion | free for individuals and companies of 3 people or fewer | React | a good **compositor/finisher** (captions, overlays); not an animator |
| Manim | MIT | Python | diagrams, not characters |
| Lottie / bodymovin | MIT | JSON | playback format; authoring needs After Effects |
| Rive | editor free, runtimes MIT; **exporting `.riv` for use needs a paid plan (Cadet, about $9/month)** | state machines | ✗ at $0 |
| DragonBones | MIT | JSON skeleton + mesh | an abandoned editor; runtime usable |
| Spine | runtimes need a paid Spine Editor licence | JSON | ✗ |
| Inochi2D | BSD-2 | Live2D-style mesh deformation | GUI rigging; no headless renderer |
| Godot (Skeleton2D + Polygon2D weights, `--write-movie`) | MIT | GDScript / C# | yes: **mesh-deformed 2D cutout**, which fixes v2's rigid-part "trouser split"; still needs well-drawn layered parts |

**Takeaway.** Code animators (v2 skia, Motion Canvas, Godot, Blender 2D) are limited by **art and rigging**, not by the renderer. Parabrahma likely has an illustrator plus a template rig library, built once in After Effects / Moho / Cartoon Animator and then driven by code. That illustrator is the piece we lack and AI keyframes now replace.

### 2.4 Assets

- **Quaternius Universal Animation Library 1 + 2** (CC0, `License.txt` verified):
  - 45 + 43 clips on one shared UE-named skeleton, including Walk_Formal, Farm_PlantSeed (kneel), Fixing_Kneeling, PickUp, Sitting, Idle_Talking and Walk_Carry, plus a Female Mannequin;
  - downloadable without login from OpenGameArt (`universal_animation_library_*standard.zip`);
  - used in prototype A.
- **Mixamo:**
  - royalty-free for films and games, but you may not redistribute the raw files or use them to train ML;
  - **needs an Adobe ID login**, so it is excluded from automation under our rules. Shubh could pull a one-time pack manually.
- **CMU mocap** (free, commercial use allowed, no reselling of the raw data): a fallback motion source.
- **Open Peeps / OpenGameArt 2D rigs:** fine for flat vector looks. They don't match the "cozy anime" look the market rewards (`RESEARCH_REPORT.md` §2).
- **AI to vector** (StarVector, or vtracer on Z-Image output): workable for props. Character rigs traced this way still need a human part split.
- **How Parabrahma built its library (inferred):**
  - one illustrator-made set of jointed characters and architecture kits, reused across all videos;
  - identical sprites, rigid joint rotation, and the same -14.1 LUFS master on every video point to a template pipeline, not per-video AI.

### 2.5 The four hybrid designs from the brief

| Design | What we learned | Verdict |
|---|---|---|
| (a) Claude script → JSON timeline → deterministic renderer (Blender / Motion Canvas / Remotion) with a rig library, rendered on CPU or CI | It works mechanically (v2 skia, and prototype A in Blender). Quality is capped by the hand-built art and rigs: about 5/10 for v2, and a mannequin look for A. CPU cost: Kaggle EEVEE at 16 s/frame (540p); a private GitHub Actions runner has only 2 vCPU and 2,000 min/month. | Keep as the **ambient / camera / caption layer**, not as the character animator |
| (b) Free AI video for some shots + code renderer for the rest | Prototype B shows AI shots reach the bar. The code layer fills plates, parallax, particles and captions for free. | **Recommended** |
| (c) 3D Blender + Mixamo-style mocap + toon shader on free GPU | Mixamo needs an Adobe login, so we used the CC0 Quaternius library instead. Real walking and kneeling work. A good-looking stylised old woman needs a modeller or image-to-3D + auto-rig (Hunyuan3D / TRELLIS + UniRig): several days of work per character, with uncertain results. | Not now. Reuse its **blocking** for prototype C. |
| (d) AI keyframes + interpolation (ToonCrafter / RIFE / FILM) | ToonCrafter is 512x320 landscape only. RIFE between distant poses morphs. Wan2.2 image-to-video (and its first-last-frame mode) does this job better. | RIFE only as a **frame-rate doubler** after Wan |

Qwen/Tongyi's hosted video is "Wanxiang", i.e. the Wan models, covered in §2.1. Every hosted service in §2.1 supports image-to-video. Their length and resolution limits (5-10 s, 720p-1080p on paid tiers) don't matter once the watermark or non-commercial rules disqualify them.

---

## 3. Recommended permutation ("Heartwick engine v3")

### 3.1 Stage by stage

| Stage | Tool | Runs on | Notes |
|---|---|---|---|
| Script | Claude | here | the 90-170 s complete-story format from `RESEARCH_REPORT.md` |
| Shot list + timeline | Claude writes `shots.json`: per shot, the beat text, keyframe prompt, motion prompt, camera move, duration from the TTS word timings, and the cut points snapped to narration pauses | here | the engine's single source of truth |
| Character bible | a one-time **reference sheet** per character: front, three-quarter and side, plus 3-4 expressions, from Z-Image + Qwen-Image-Edit; stored in `assets/characters/<name>/` | Kaggle / ZeroGPU | locks the look across a whole series |
| Keyframes | **Z-Image-Turbo** for plates and establishing shots; **Qwen-Image-Edit-2509** "put <ref sheet> character into <shot>" for every character shot | Kaggle T4 (GGUF) / ZeroGPU | consistency control; reject and retry with a VLM check |
| Animation (characters) | **Wan2.2-I2V** (A14B Q4 + Lightning 4-step for hero shots; FastWan-5B for fill shots); 5 s per clip, prompt from `shots.json` | Kaggle T4 x2 (2 sessions max) | about 8-12 animated shots per Short |
| Animation (ambient) | our engine: parallax Ken Burns on plates, particles (leaves, snow, light rays), sway | Kaggle CPU / GitHub Actions | keeps the reference's "something moves in 65-97% of seconds" for free |
| Camera | engine push-ins, drifts and tilts on every shot, including over Wan clips (crop-zoom after the 2x upscale, so it stays sharp) | CPU | always moving, as in the reference |
| Frame rate + resolution | RIFE v4.6 to 30 fps (60 optional) and Real-ESRGAN animevideov3 x2, then Lanczos to 1080x1920 | laptop iGPU (Vulkan) or Kaggle | measured 165 s per 5 s clip on the iGPU, with the CPU free |
| Voice | the other agent's TTS library | — | its word timings drive the cut points |
| Music | ACE-Step 1.5 (MIT) pad at -18 to -22 dB, swells on reveals | Kaggle | per `RESEARCH_REPORT.md` §4 |
| Captions | engine: 2-4 word chunks, one highlighted word, at 71% of the frame height | CPU | existing v2 code |
| QA | a vision critic per clip (identity match to the ref sheet, hands, morphing, squirrel size), auto-retry with a new seed, plus a contact sheet for a human pass | Claude + free VLM | |
| Assembly + master | ffmpeg; -14 LUFS; AI disclosure in the description | CPU | |

### 3.2 Engine architecture

```
story.md ──Claude──> shots.json ──┬─> keyframes job (Kaggle GPU): Z-Image / Qwen-Edit + ref sheets ─> keys/*.png
                                  │        └─ VLM identity check ─ retry
                                  ├─> i2v job (Kaggle GPU, ≤2 sessions, resumable queue): Wan2.2 ─> clips/*.mp4
                                  ├─> TTS job (other agent) ─> vo.wav + word timings ─> cut points
                                  └─> finish (iGPU/CPU): RIFE + ESRGAN + engine camera/particles/captions/music ─> master.mp4
```

- **Job queue:** a tiny `jobs.jsonl` with a state per shot (`todo/keyed/animated/finished/rejected`). Each Kaggle kernel picks up the shots that aren't done yet, so a preempted or timed-out session loses at most one clip. Kaggle's 12-hour session cap and the 2-session limit are respected by a local scheduler that checks `kernels list --mine` first.
- **Hybrid rule (option b):** every shot is either an **AI-animated character shot** or an **engine-animated ambient shot** (plate + parallax + particles). Roughly 50/50, which halves GPU need while keeping motion everywhere.
- **Prototype C path (optional, later):** when a shot needs exact choreography that Wan won't follow from text (for example "she walks left to right, then kneels at the exact moment the narrator says 'knelt'"), use Blender + the CC0 animation library to block it and export pose (already working: `A_blender3d/nell_beat.py ... POSE`). Then Wan-VACE repaints it with the character reference.

### 3.3 Character consistency, the main remaining risk of B

- **Within a shot:** it was excellent in the test.
- **Across shots:** the risk is drift in face, glasses and cardigan pattern. Mitigations, in order:
  1. Qwen-Image-Edit from a fixed reference sheet. Tested only up to the ZeroGPU quota: the anonymous quota ran out before the edit test, so this needs a Kaggle run next.
  2. A character LoRA for Z-Image or SDXL trained once on Kaggle, about 1 GPU-hour per character.
  3. The VLM identity gate.
  4. Wan first-last-frame mode, so consecutive clips share frames.

---

## 4. Throughput and quality estimate

**Measured:**

- ZeroGPU (half an RTX Pro 6000): **66 s wall for a 5 s Wan2.2-A14B clip** (6 steps, Lightning, fp8 AoT) and 31 s for a Z-Image keyframe, including queueing.
- Laptop iGPU finishing: **165 s per 5 s clip**.
- Blender A: 0.83 s/frame on the iGPU, 15.9 s/frame on Kaggle CPU (540p).

**Kaggle T4, first attempt (v1):**

- It failed fast on memory, not speed. The 11 GB umt5 text encoder next to the 5B transformer OOMed the 15 GB T4. Loading it beside the two A14B GGUF experts was killed at 31 GB RAM. Logs: `B_open_i2v/kaggle_t4/t1_v1_oom.log`, `status_v1_failed.json`.
- v2 precomputes the text embeddings, frees the encoder and tiles the VAE. It was pushed at 22:13 IST and was **still in Kaggle's GPU queue at 23:35** (the other agent's `heartwick-painted-assets` had been queued since 21:26). So the T4 timings below are calibrated estimates, not measurements. To fetch the results once it runs: `kaggle kernels output shubhsaxena2008/heartwick-proto-i2v-t4 -p research/pipeline_proto/B_open_i2v/kaggle_t4/out`. Then replace the table with `t1.json` and `t2.json`.

**Estimate for the T4,** calibrated on our earlier real measurement (Wan2.2-TI2V-5B, 50 steps with CFG, 2 s at 704p = 35 min). That implies about 8-10 effective TFLOPS on a T4 with offload.

| Clip type (5 s, 480x832) | Transformer FLOPs per step | Steps | Estimated T4 time per clip | Clips per T4-hour |
|---|---|---|---|---|
| FastWan2.2-TI2V-5B (DMD 3-step), ~12k tokens | ~1.8e14 | 3 | **~2-4 min** (VAE decode dominates) | 15-30 |
| Wan2.2-I2V-A14B Q4 + Lightning, ~33k tokens | ~1.8e15 | 4 | **~12-18 min** | 3-5 |

**Per Short** (120 s, about 25 shots, about half of them AI character shots):

- 12 kept clips at about 1.5 attempts each = 18 generations;
- plus 18 keyframes at about 1 min each on a T4.

**Weekly capacity:**

- Kaggle gives about 28 session-hours/week of **T4 x2**. Two clips run in parallel, one per GPU, so that is ≈ **56 T4-hours**.
- About half that if the voice agent keeps one of the two sessions busy.

| Mix per Short | T4-minutes per Short | Shorts/week (theoretical, full quota) | Shorts/week (realistic: half quota, queue losses) |
|---|---|---|---|
| All FastWan-5B | ~90 | ~35 | **10-15** |
| 6 hero A14B + 12 FastWan | ~150 | ~22 | **6-10** |
| All A14B | ~290 | ~11 | **3-5** |

ZeroGPU adds about 1-2 A14B hero clips a day anonymously, or 3-4 with a free account (about 10-25 a week).

The `RESEARCH_REPORT.md` target is 3-5 Shorts plus 1 long-form a week. That is met with margin in the mixed configuration. A long-form 8-15 minute "full story" would reuse the Shorts' clips plus extra ambient shots.

**Expected quality:**

- Character shots at the B prototype level: ≈ the Gemini bar, about 8-8.5/10 on our rubric. The v2 puppet scored 4.6-5/10.
- Ambient shots at the v1 painted-parallax level.
- With pause-snapped cuts, captions and a moving camera on everything, the overall Short should be clearly above Parabrahma on art and motion. Parabrahma's advantage stays its exact consistency, which the reference-sheet workflow (§3.3) has to earn.

---

## 5. Free compute and student programmes (Shubh is 17)

| Option | What you get | Age / requirements | Use |
|---|---|---|---|
| Kaggle | about 30 T4x2 GPU-hours/week, 2 GPU sessions, CPU kernels don't queue | the terms tie eligibility to legal contracting age, with a parental-consent route for minors | **primary** |
| HF ZeroGPU | 2 min/day anonymous, 5 min/day free account, on half an RTX Pro 6000 Blackwell (48 GB) | account 13+ | supplement: hero shots, keyframes |
| GitHub Actions | public repos: free, 4 vCPU runners. Private: 2,000 min/month on Free (3,000 on Pro via the Student Pack), 2 vCPU. | GitHub 13+ | CPU finishing / assembly (the repo is private, so 2 vCPU and a cap apply) |
| GitHub Student Developer Pack | Copilot, Pro, partner offers | **13+**; some partner offers have their own age limits | worth claiming |
| Azure for Students | $100 credit | **18+** (Microsoft says applicants must affirm 18+) | ✗ until Dec 2026 |
| Google Cloud / AWS education credits | course- or institution-gated | generally 18+ | ✗ for now |
| Oracle Cloud Always Free | 4 Arm OCPU + 24 GB RAM; no GPU | credit card + phone; one account per person | CPU rendering/finishing box. Must be the father's (adult) account. Arm is fine for Blender, ffmpeg and RIFE-ncnn on CPU. |
| Modal | about $30/month free credits (some sources say a card is needed) | adult account | ≈ 50 T4-hours or 15+ A10G-hours a month: a big boost **if the father opens it** |
| Lightning AI | free credits (amount unpublished; about 15/month reported) | adult, phone verification | a backup GPU |
| Laptop iGPU (Arc 130T) | Vulkan: RIFE, ESRGAN and EEVEE run at no CPU cost | — | finishing (CPU stays ≤ 75%; it was 30% here) |

**Sources:**

- Kaggle terms: <https://kaggle.com/terms>
- Hugging Face age (via eSafety): <https://esafety.gov.au/key-topics/esafety-guide/hugging-face>
- GitHub Actions billing: <https://docs.github.com/en/billing/concepts/product-billing/github-actions>
- GitHub Student Developer Pack terms: <https://docs.github.com/en/education/about-github-education/github-education-for-students/github-terms-and-conditions-for-the-student-developer-pack>
- Azure for Students: <https://azure.microsoft.com/en-us/free/students/>
- Oracle Free Tier FAQ: <https://www.oracle.com/cloud/free/faq/>
- Modal free tier: <https://agentdeals.dev/vendor/modal>
- Lightning AI billing FAQ: <https://lightning.ai/docs/platform/overview/faq/billing.md>

---

## 6. Risks

1. **GPU supply.** Kaggle queues of 2 hours or more, the 2-session cap shared with the voice agent, and weekly hours. Mitigations: a resumable job queue, 480p generation followed by iGPU finishing, ambient shots on CPU, and ZeroGPU as overflow.
2. **Character drift across shots.** See §3.3. Not yet proven at series scale.
3. **YouTube "inauthentic / templated" policy** (`RESEARCH_REPORT.md` §2). Mitigations: original scripts, a varied shot grammar, the voice-led story. Animated AI needs only a description disclosure. AI-only output may also not be copyrightable, so others could reupload it. That is low impact.
4. **Model artefacts:** scale drift (the squirrel grew), hands, occasional morphing. Mitigations: the VLM QA gate plus seed retries. Budget about 1.5 attempts per kept clip.
5. **Third-party Spaces** can change their API or vanish. They are a supplement only; the pipeline runs on Kaggle with pinned model revisions.
6. **Licence drift.** Pin model revisions and keep a `LICENSES.md` per asset. Everything recommended is Apache-2.0, MIT, BSD or CC0.

---

## 7. Decisions for Shubh

1. **Adopt B (open AI keyframe + Wan2.2 image-to-video) as the core**, with the engine as director and finisher? (Recommended.) Or keep investing in code puppets (A or v2)? Recommendation: no, except Blender blocking for C-style choreography later.
2. **Art direction:** the "cozy hand-drawn anime / Ghibli-like watercolor" look from the B keyframe (the biggest market niche in `RESEARCH_REPORT.md`) or a flatter vector look closer to Parabrahma's. B's quality comes from the painterly look.
3. **Extra free GPU:** should the father (an adult) open a **Modal** ($30/month free) and/or **Oracle Always Free** account? That roughly doubles throughput at $0. A card or phone is needed for verification only. Shubh can't open these himself until he turns 18.
4. **A free HF account** (father's, 13+ is fine) to raise ZeroGPU from 2 to 5 min/day?
5. **Squirrel and pet scale, and hero beats:** accept about 1.5 tries per kept clip, or reserve A14B for hero beats only and use FastWan-5B elsewhere?

---

## 8. Build order if B is approved (about 2 working sessions)

1. **The Kaggle "studio" kernel** (`heartwick-v3-gen`). One GPU session per batch. It:
   - loads Z-Image-Turbo (keyframes), then Wan2.2 (A14B Q4 + Lightning, or FastWan-5B);
   - reads `shots.json` and writes `keys/` and `clips/`;
   - resumes from `jobs.jsonl`;
   - always uses precomputed umt5 text embeddings. The v1 T4 run proved that loading the 11 GB text encoder next to the video transformer OOMs a 15 GB T4, and that loading it beside both A14B experts gets the job killed for exceeding 31 GB RAM.
2. **Character bible job:** reference sheets for Nell, Pip and each new character, plus a Qwen-Image-Edit "same character, new shot" test. This settles §3.3.
3. **Finisher** (laptop iGPU or Kaggle): `B_open_i2v/finish.sh`, extended with engine2's existing captions, camera, particles, audio bed and pause-snapped cuts (`engine2/edit.py`).
4. **QA critic:** reuse the v2 refine-loop scorer, pointed at identity, hands and scale, with retries on a new seed.
5. **Pilot:** re-make the Pip Short end to end; compare it against v2 `out/pip_short_B` and the Gemini clip; ship if Shubh approves.
