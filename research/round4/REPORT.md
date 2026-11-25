# Heartwick round 4: an engine for complete 7-8 minute episodes (24 Nov 2026)

Measured data is in `research/round4/results/MEASURED.json`. **M** means measured today on Kaggle; **E** means estimated, with the basis given. The round-4 research agent wrote this report; the main session saved it here.

## 0. Why v3 cannot scale

- **M** v3 takes 863 T4-s per 5.06 s clip. From `rec_g0.json`: VAE encode 84 s + 4 steps × 139 s + VAE decode 143 s.
- 7.5 min = 450 s = 89 clips:
  - 76.8k T4-s = **21.3 T4-h**;
  - about 30 T4-h with the 1.4x retry rate, which is the whole weekly quota for one episode.
- 60 fps finishing:
  - **M** 3,144 T4-s for 88 s of output;
  - so about 4.5 T4-h more per episode.
- CPU audio:
  - **M** 29 min for 84 s of narration;
  - so about 2.6 h for 450 s.
- So motion must come mostly from cheap limited animation on high-quality stills, as real anime does.

## 1. Kaggle facts measured today

| Fact | Result |
|---|---|
| P100 | Retired. The CLI maps `NvidiaTeslaP100` to T4, so T4 x2 is the only GPU. |
| TPU v5e-8 via an API push | No TPU in 2 of 2 tries (`heartwick-r4-tpu`, `heartwick-r4-tpuprobe`): `COLAB_IMAGE_TYPE=cpu`, `jax.devices()=[CpuDevice]`, no `/dev/accel*`, 0.00 h of the 20 h TPU quota charged. Batch kernels can't use the TPU. The web UI is untested and would need a human. MaxDiffusion does support Wan2.2-I2V-A14B with the lightx2v 4-step LoRA on TPU, so a human-triggered TPU lane stays possible. |
| CPU kernels | 4 vCPU Xeon 2.2 GHz, 31 GB RAM, 2.5 TB scratch disk; `/kaggle/working` is a 20 GB loop device. All 6 started within about 1 min; 3 ran concurrently while a GPU kernel was queued. No GPU quota used. |
| Hugging Face download inside Kaggle | 66 GB in 341 s (~195 MB/s); 12.8 GB in 34 s (~380 MB/s) |
| Weights saved as a kernel output (mounted via `kernel_sources`) | Saving the 12.8 GB output took ~20 min (one time). Loading from the mount: VAE 2.8 GB in 79 s; transformer 10 GB in **2,883 s** (~35 MB/s). One run, with a FLUX process holding ~16 GB of RAM at the same time. Use Hugging Face downloads per kernel (~1 min per 10 GB); keep kernel outputs for small artefacts. Untested: copying the mount to local disk first. |
| GPU queue | `heartwick-r4-bench`: 170 min queued with nothing else running. `heartwick-r4-fw2`: 0 min. `heartwick-r4-a14b`: still queued from 18:26 to past 20:10. Round 3 waited 4.7 h. |
| Quota | At the start: GPU 14.2/30 h, TPU 0/20 (refresh 28 Nov 00:00 UTC). This round used ~1.4 GPU session-hours. |

## 2. Recommended design: "Engine v4", storybook limited animation plus a few AI acting shots

### Episode shape (7.5 min)

About 1,000 words, ~80 shots averaging 5.6 s, 10 scenes (location × time). Acts:

| Act | Ends at | Shots |
|---|---|---|
| Hook | 0:25 | 4 |
| Setup | 2:00 | 15 |
| Inciting event | 2:45 | 8 |
| Bonding / montage | 4:30 | 18 |
| Midpoint turn | 5:00 | — |
| Release | 6:15 | 14 |
| Climax | 7:00 | 8 |
| Moral + end card | 7:30 | 5 |

About 36 unique master keys; medium and close shots are free crops of the masters.

### Shot classes

| Class | Share | Shots | Motion |
|---|---|---|---|
| L "living still" | ~60% (~270 s) | ~45 | <ul><li>FLUX key at 1088x1920, layered 2.5D</li><li>Camera push/pan with depth parallax, on ones</li><li>Character on twos: breathing and sway warp</li><li>Blink sprite every 3–5 s</li><li>FX layer: steam, snow behind glass, rain, leaves, dust motes, light breathing</li></ul> |
| E establishing / insert | ~15% (~70 s) | ~12 | Plates and cut-ins with parallax and weather FX |
| A acting | ~20% (~90 s) | 8–12 | <ul><li>Wan2.2-I2V-A14B lightx2v 4-step Q5 at 480x832 (the v3 model; identity holds)</li><li>Only for actions the story needs: walk, kneel, lift, jump, open window</li><li>Generated at 49 frames (3 s) or 81 frames (5 s), never stretched</li></ul> |
| D dialogue | inside L | — | 3 mouth sprites (closed/mid/open), only inside a mouth mask, driven by that character's own TTS energy |

### Stages

1. **Shot list:** Claude, local, text only. Class, duration from TTS word times, camera move, FX.
2. **Audio:** Qwen3-TTS plus Whisper on one T4 (**E** 3–5 min) instead of CPU. Music: one ACE-Step cue.
3. **Keys:** FLUX.2-klein-4B (Apache-2.0, 4-step, multi-reference) with the Nell and Pip sheets as references.
   - **M** 48–62 s per 1088x1920 key.
   - Blink and mouth variants are edits (**M** 34 s at 768x1344), pasted back only inside an eye or mouth mask.
4. **Layers:**
   - Real-ESRGAN animevideov3 (**M** 0.4 s on GPU)
   - Depth-Anything-V2-Small, Apache-2.0 (**M** 0.5 s on GPU)
   - isnet-anime matte (**M** 3.7 s on CPU)
   - Telea clean plate (**M** 4.9 s on CPU)
5. **Acting clips:**
   - Phase-split per GPU, as in v3.
   - The prompt contains no narration or dialogue words, plus "mouth stays closed, lips together, silent".
   - Then the `mouth_lock` pass.
   - FLF2V when the end pose matters.
   - Faster decode via fp16 VAE or LightTAE (**E**; the fp16 test is in the queued a14b kernel).
6. **Render:**
   - L and E shots at 24 fps, 1080x1920: **M** 20 ms/frame on GPU with torch, or 145 ms/frame on a CPU kernel.
   - Acting clips: RIFE 16→24 fps at constant speed, plus ESRGAN.
7. **QA (no models):** `motion_audit`, `mouth_audit`, plus the existing `qa.py` identity check on acting clips.
8. **Assembly:** v3 `assemble_main.py` at 24 fps, on a CPU kernel.

### Mouths stay closed

1. **L and E shots (~75% of runtime):** there is no mouth animation at all. All 7 FLUX keys came out closed-mouthed, apart from the deliberate mouth edit.
2. **Acting shots:**
   - With CFG off in the distilled model, negative prompts do nothing. NAG is the only few-step option, and it isn't built.
   - `tools/mouth_lock.py` tracks the head with Lucas-Kanade corners outside the lips and pastes frame 0's lower face back in with a feathered ellipse.
   - **M** on v3 clip S02: residual mouth flow (median) fell from 0.468 to 0.056 px, and lip spikes over 1 px fell from 21 to 7 out of 80 frames. The lips are visibly closed in calm frames (`analysis/v3_S02_mouthlock_before_after.jpg`).
   - It fails when the head turns more than ~20°, as in S02's last ~10 frames.
   - So `mouth_audit` gates the clip: more than 5 spikes, or fewer than 50% of tracked points, means a new seed or the shot is demoted to L.
3. **Dialogue:** sprite lip-flap only while that character speaks.

### No freezes

- **No retiming.** L and E shots render at any exact length. Acting clips are 49 or 81 frames, padded by a 0.5–1.5 s L-hold instead of slow motion. RIFE only does a constant-speed 16→24 fps conversion.
- **At least 2 motion layers per shot.**
- **`motion_audit` gate:** the longest static run must be ≤ 0.5 s, and either subject motion ≥ 1 px/frame or a camera move ≥ 3%.
  - **M** LA renders: 1.2–1.3 px/frame, longest static run 0.12–0.29 s.
  - v3 S02 and S03: subject motion was only 2.1 px at 12 fps.
- **Caveat:** LA motion is subtle. Push the camera, blinks and FX harder, and review this first in the pilot.

## 3. Cost and time for a 7.5 min episode (2 sessions × 2 T4, plus CPU kernels)

| Item | Count | T4-s each | T4-min total | Status |
|---|---|---|---|---|
| Setup: pip + Hugging Face download + model loads, per session | 2 | ~240 | 8 | **M** (61–76 / 64–82 / 53–79 s) |
| FLUX master keys, 1088x1920 | 36 | 55 | 33 | **M** |
| Blink/mouth variants | 20 | 34 (~10 as face crops) | 11 | **M** at 768x1344 |
| TTS on GPU + music | 1 | ~420 | 7 | **E** |
| L/E layers + torch render | 57 shots, ~8,000 frames | 1 + 0.02/frame | 4 | **M** unit costs |
| A14B acting clip, 81 frames | n | 783 (fp32 VAE); ~640 with fp16 VAE (**E**) | — | **M** v3 |
| A14B acting clip, 49 frames | n | ~450 (**E**) | — | unmeasured (queued) |
| RIFE + ESRGAN, acting frames only | ~2,400 frames | ~0.15/frame | 6 | **E** |
| CPU kernels (matte, plates, assembly) | — | — | 0 GPU, ~15 min CPU in parallel | **M** / **E** |

| Tier | AI acting | T4-min | Wall clock on 4 T4s (excl. queue) | Session-h per episode | 6 episodes/week |
|---|---|---|---|---|---|
| 30-min | 6 × 49 frames (~18 s) | ~114 | ~29–32 min | ~1.0 | ~6 h of 30 |
| Balanced | 10 (6 × 49 + 4 × 81 frames) | ~150 | ~40–45 min | ~1.5 | ~9 h |
| Rich | 20 | ~250 | ~70 min | ~2.4 | ~14 h |

**Verdict:**
- Under 30 min of compute is reachable only in the 30-min tier, with about 18 s of real AI acting per episode and everything else as limited animation.
- Under 30 min end to end can't be guaranteed on free Kaggle, because GPU queue waits today ran 0–170+ min.
- The weekly quota is no longer the constraint.

## 4. Measured results and honest assessment

Paths are relative to `research/round4/results/`.

- **FastWan2.2-TI2V-5B** (DMD 3-step, fp16, own sampler), output in `fw2/out/clips/*.mp4` and `*_sheet.jpg`:
  - **M** steps take 10.8–13.4 s, so ~33 s to denoise at 480x832 × 121 frames.
  - The fp32 VAE decode takes **313 s**, so ~347 s per clip, with a 14.9 GB peak. At 704x1280 it ran out of memory.
  - **Rejected:** after frame 0 Nell becomes a young girl with a ponytail and oversaturated colours. The frame 0→1 difference is 16–21 (A14B: 1.0), and first-vs-last is 43 (A14B: 7.3).
  - The sampler may contribute, but the decode cost kills the speed benefit anyway.
- **FLUX.2-klein-4B** (fp16, CPU offload), output in `fw2/out/keys/*.png`, `fw2/out/keys_board.jpg`, `fw2/out/face_variants.jpg`:
  - **M** timings: load 53 s; 61.7 s per new 1088x1920 key with 2 references; 39 s at 768x1344; 34 s per edit.
  - Assessment: good.
    - Nell stays on-model (glasses, bun, green cardigan, rust skirt) with clean anime line art.
    - The blink edit works.
    - The mouth edit also changed her eyes and head, hence the region-only compositing.
    - One 768 key drifted to a watercolour paper border.
    - It added an unrequested squirrel, because the K0 reference contains one.
  - About 4× faster than Qwen-Edit at about 5× the pixels. Identity across a full episode still needs Shubh's review.
- **Limited-animation renderer `kaggle/bench/la.py`**, output in `lacpu/out/la/la_k2.mp4` and `la_k10.mp4` (6 s, 1080x1920, 24 fps) with sheets, mattes, depth and plates; GPU run in `bench/out/la/`:
  - **M** prep 17 s per shot on CPU; render 145 ms/frame on CPU or 20 ms/frame with torch on GPU.
  - Frames are crisp anime, better than 480p Wan upscaled.
  - Snow behind the glass, steam and dust read well.
  - The matte catches some chair and table legs, with a faint seam near the mug.
  - Motion is subtle.
- **Mouth tools:** `tools/mouth_audit.py` and `tools/mouth_lock.py` (results in §2). `tools/motion_audit.py` gives camera-compensated subject motion; v3 results are in `analysis/v3_motion.json`.
- TPU, weight cache and CPU concurrency are in §1, with outputs in `tpu/`, `tpuprobe/`, `wcache/`.

## 5. Open for the next session

- **`shubhsaxena2008/heartwick-r4-a14b` was QUEUED from 18:26; deleted unrun at 20:50 at Shubh's request (laptops closing). Re-push it from `research/round4/kaggle/a14b/` to run this test.**
  - It measures A14B Q5 4-step on K2 with the mouth-closed prompt: 81 frames on GPU0 (plus an fp16-VAE decode test with time and pixel error), and 49 frames on GPU1.
  - Fetch: `python research/round4/tools/r4push.py x heartwick-r4-a14b research/round4/results/a14b --no-push`
  - Then:
    - run `mouth_audit` on `a14b_k2_mouth_81f.mp4` against v3 `S02_s7` (same key and seed, no mouth prompt);
    - read the 49-frame step times and `vae_decode_fp16_s` from `rec_g0.json` / `rec_g1.json`;
    - update §3.
- Still unmeasured:
  - TTS on GPU;
  - LightTAE decode;
  - FLUX without CPU offload (text encoder on GPU1);
  - mouth-sprite lip-flap;
  - a full 7-minute pilot.

## 6. Risks

1. **GPU queue of 0–3 h** is the dominant term. Push both sessions early, put all non-GPU work on CPU kernels, and keep a backlog lane.
2. **Limited animation may read as too static.** Pilot it and tune the amplitudes.
3. **FLUX character drift across 36 keys.** Keep the sheet references on every key and use the identity QA gate; a character LoRA is the fallback.
4. **A14B mouths and head turns that beat `mouth_lock`.** Regenerate or demote the shot.
5. **A14B cost.** Use fp16 or LightTAE decode and 49-frame clips.
6. **Matte and plate artefacts.** Keep parallax small and use better inpainting for wide reveals.

## 7. Build plan

1. Fetch the a14b output, then fix the acting-clip unit cost and the mouth evidence.
2. Build `engine4/la_render.py`: a torch GPU port of `la.py`, plus blink and mouth sprites and per-shot FX presets.
3. Build the FLUX.2-klein keys stage: 2-GPU split, region-only variant compositing, identity QA.
4. Port acting clips from v3 `gen_flf.py`: fp16 or TAE decode, 49/81 frames, mouth prompt + `mouth_lock` + `mouth_audit`.
5. Move TTS to GPU and measure it.
6. Write the scheduler:
   - Session A: TTS, music, keys, layers, LA renders.
   - Session B: acting keys, then clips.
   - CPU kernels: mattes and assembly.
7. Write the story template (§2) and Claude shot-list prompts for 80 shots in 10 scenes.
8. Pilot one 7.5 min Nell & Pip episode, review it with Shubh, then tune the amplitudes and the clip-count tier.
