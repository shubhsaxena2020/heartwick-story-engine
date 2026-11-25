# Heartwick round 2: compute and throughput (22 Nov 2026)

**Question.** Can we make at least 7 finished videos a week (60-120 s, vertical), using the big Wan2.2-I2V-A14B model for every shot, for $0, without any card?

**Answer.** Yes. Kaggle alone just covers it if its GPU hours go almost entirely to video. With the laptop's night batch and ZeroGPU added, there is about 1.8x headroom. Measured today:

- **Kaggle T4 x2** with the lightx2v 4-step distilled A14B (GGUF, one expert at a time, one process per T4) makes **one 5 s clip every ~435 s of session time**. That is about **8 clips per session-hour**, so **~200 clips a week** from the 30-hour quota once overheads are taken out.
  - Seven videos at 18 clips each, with 1.4 attempts per kept clip, need **~176 generations a week**.
  - Quality matches round 1: see `research/round2/compute/finished/q5_480x832_1080p30.mp4`, finished exactly like round 1's clip; set it against `research/pipeline_proto/B_open_i2v/zerogpu/wan22_i2v_1080p30.mp4`.
- **The laptop iGPU** can run the same A14B model. On stable-diffusion.cpp Vulkan, a small test clip came out coherent and on-model. Measured per-block speed puts a 480p clip at **~28 min**, so **~17 clips a night**. A full 480p clip could not finish tonight: the shared laptop ran out of Windows commit memory (§5b).
- **Hugging Face ZeroGPU** on the father's free account adds about 5 clips a day.
- **Nothing else is free, card-less and usable.** Kaggle TPU never attached through the API. Colab, Lightning, Studio Lab, Paperspace, Saturn, Intel Tiber and Deepnote are all ruled out (§4).

**Two things need Shubh or his father before this goes live (§6):**

1. **Kaggle's Terms of Use say Kaggle may be used only for "your own internal, personal, non-commercial use, and not on behalf of or for the benefit of any third party".** A monetised channel run for his father is arguably commercial use. This is a real risk to the whole plan, which round 1 did not flag.
2. **The laptop's page file** (16 GB, fixed, 100% used) caps how much the iGPU can hold, because iGPU memory is system RAM. Raising it is a one-time admin setting.

All numbers, logs and clips are in `research/round2/compute/`.

---

## 1. Measured: seconds of GPU per second of finished video

**One clip:**

- Wan2.2-I2V-A14B, lightx2v 4-step distill `260412` (Apache-2.0; GGUF from `jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF`).
- 81 frames at 16 fps = **5.06 s of motion**.
- The same keyframe, prompt and seed as round 1.
- **"GPU s"** = VAE encode + 4 denoise steps + VAE decode. Each batch also pays a one-time model load of 42 s per expert.

| Where | Setting | GPU s per clip | GPU s per video-second | Session s per video-second | Quality vs round 1 (looked at every clip) |
|---|---|---|---|---|---|
| Kaggle T4 (both GPUs busy at once) | **480x832, Q5_K_M** | **863** (steps 156-162 s each, encode 83, decode 147) | **171** | **85** (2 T4s per session) | **Same or better.** She *kneels*, which round 1 missed. Squirrel still grows ~1.5x. |
| Kaggle T4 | 480x832, Q4_K_M | 888 | 176 | 88 | Same as round 1: walk → crouch → lift → stand, on-model |
| Kaggle T4 | 480x832, last-frame continuation of the clip above | 865 | 171 | 85 | Identity and colour hold. The background house changes (scene drift), so prompt the background too. |
| Kaggle T4 | 384x672 (then ESRGAN x2) | 443 | 88 | 44 | Good motion, slightly softer. One squirrel-morph frame. **Not "full quality"; see §3.** |
| Kaggle T4 | 480x832, **41 frames** (then RIFE x2) | 337 | 67 (if stretched to 5 s) | 33 | **Rejected.** The motion is crammed into 2.5 s and the squirrel turns into a blob mid-lift. |
| Kaggle T4 | 544x960 | 1,553 | 307 | 153 | Fits only just (15.16 GB peak). Sharper, but 1.8x the cost. |
| Laptop Arc 130T, PyTorch XPU | 480x832, Q4_K_M | **~1,665 (estimate):** measured 8.32 s per block × 40 blocks + 0.15 s dequant per block → 339 s per step; VAE encode measured at 113 s; decode estimated ~195 s | ~329 | — (runs unattended at night) | Same weights as Kaggle, so same output. The full clip died 3 times on commit memory, not on compute. |
| Laptop Arc 130T, PyTorch XPU | 384x672 | **~880 (estimate):** 4.18 s per block; encode measured at 70 s | ~174 | — | — |
| Laptop Arc 130T, stable-diffusion.cpp Vulkan | 256x448x17 smoke test | 38 s/step (weights streamed from disk) | | | Coherent: walk → kneel, on-model (`laptop/out/smoke3_sheet.jpg`) |
| HF ZeroGPU (round 1) | 480x832, Lightning 6-step fp8 | ~66 s wall incl. queue | ~13 | — | the round-1 reference clip |
| Laptop finishing (re-measured today on the Q5 clip) | RIFE → 30 fps + Real-ESRGAN x2 + 1080x1920 encode | **95 iGPU s** (RIFE 8.7, ESRGAN 77.2, x264 8.9); round 1 measured 165 s | 19 | — | `round2/compute/finished/q5_480x832_1080p30.mp4` |

**Raw speeds:**

- **T4:** 21.2 TFLOPS fp16 GEMM. SDPA attention 12.4 TFLOPS, which is 1.77 s per 40-head pass at 32,760 tokens.
- **So per 480p step on a T4:**
  - 40 layers × 1.77 s ≈ **71 s is attention**;
  - ≈ 45 s is linear layers;
  - the rest is GGUF dequantisation and overhead.
- **Arc 130T through XPU:** 11.8-17.5 TFLOPS fp16 GEMM (XMX works), but **SDPA only 4.5 TFLOPS** (4.87 s per pass). Attention is the iGPU's bottleneck.

**Files:**

- `kaggle_t4/out_v1/out/rec_*.json`: per-step timings, peak VRAM.
- `bench.json`.
- `measured.json`.
- The 8 Kaggle clips and their contact sheets.

## 2. What each speed-up is worth (on the hardware we actually have)

| Speed-up | Result | Verdict |
|---|---|---|
| **Step distillation** (lightx2v 4-step, CFG off) | Already in. Merged distill weights (`260412`) avoid LoRA-on-GGUF problems: no LoRA loading, no peft. The sampler is the exact lightx2v `WanStepDistillScheduler`: shift 5, steps [1000,750,500,250], Euler. | **Use.** About 20x fewer transformer passes than 40 steps with CFG. |
| **CausVid** | Superseded by lightx2v for Wan2.2. It is a Wan2.1 self-forcing LoRA and is not needed. | Skip |
| **GGUF quantisation** | Q4_K_M and Q5_K_M run at the **same speed** on a T4. Dequantisation costs the same; the step time is compute-bound. Q5 peaks at 15.14 GB of the T4's 15.36, and it fits. Q8_0 (15.4 GB of weights) cannot fit. | **Use Q5_K_M** on Kaggle (closest to fp8, free). Q4_K_M on the laptop (RAM). |
| **fp8 / int8** | The T4 (sm75) has no fp8 tensor cores. int8 weights are 15 GB per expert, too big. | Not on T4. (ZeroGPU already uses fp8.) |
| **TeaCache / MagCache / FBCache** | These skip steps whose output barely changes. With 4 steps (2 per expert) there is nothing to skip. MagCache's own Wan2.2 table is for 40-step runs. | **No gain.** Skip. |
| **SageAttention** | `sageattention==1.0.6` **fails to compile on T4**: Triton int8 dot on sm75, `PassManager::run failed`, logged in `bench.log`. SageAttention 2 has no official sm75 kernels. | Not available. Attention stays at ~45% of the step. |
| **Offload / block swap** | Replaced by the **expert-phase split** in `kaggle_t4/gen.py`. It encodes every clip, then loads the high-noise expert once for all clips (steps 0-1), then the low-noise expert once (steps 2-3), then decodes. Only one 14B expert is ever resident, so a 15 GB T4 holds it whole: no per-step PCIe swapping, and the host RAM stays small. | **Use.** Loads cost 42 s per expert per batch, about 1% amortised. |
| **Two T4s in parallel** | Two independent `gen.py` processes (`CUDA_VISIBLE_DEVICES=0/1`) ran side by side with no slowdown: GPU0 158 s/step and GPU1 157 s/step at the same time. Loading is serialised by a file lock to cap host RAM. | **Use: doubles throughput.** |
| **Lower resolution + Real-ESRGAN** | 384x672 is 2.0x faster (443 s vs 863 s), with visibly softer detail and one morph frame. | Only if Shubh accepts it for distant or ambient shots. Not default. |
| **Fewer frames + RIFE** | 41 frames is 2.6x faster, but the model plans the action for 2.5 s. Stretched to 5 s it is slow-motion, and it morphed. | **Rejected.** |
| **Last-frame continuation** | Same cost as any clip. Identity holds across the chain. The background can drift. | Use for long actions. Restate the background in the prompt. |
| **fp16 VAE** (not measured) | VAE encode + decode is **230 s of the 863 s (27%)**, run in fp32. Running it in fp16 on T4 tensor cores should cut that by about half. **Estimate: −100 to −120 s per clip.** | **Next thing to test.** It is a one-line change in `gen.py` (`torch_dtype=torch.float16`). Check for NaNs and colour shift first. |

## 3. Kaggle specifics

- **Queue.**
  - My kernel, pushed with `--accelerator NvidiaTeslaT4`, started **immediately**, at 00:21 and 00:33 IST.
  - Meanwhile the round-1 `heartwick-proto-i2v-t4` v2 (same T4 machine shape) has sat **QUEUED since 22:13 IST**, and it holds one of the 2 GPU slots.
  - So long waits are not a hard capacity wall. A stuck queued version should be deleted and re-pushed. The quality agent was blocked behind that slot (`round2/quality/kaggle/push_wait.log`).
  - **Recommendation:** whoever owns that kernel deletes or re-pushes it. Its test (A14B Q4_K_S + LoRA in diffusers) is superseded by this round's measurements.
- **Job sizing.** One long session (3-4 h, about 25-30 clips) is better than many short ones. Fixed cost per session ≈ 5 min:
  - pip install 26 s;
  - umt5 embedding 100 s;
  - GGUF download 3 min, in parallel;
  - expert loads 85 s per phase.
  - Kaggle sessions can run up to 12 h.
- **Prompt embeddings** (umt5, 11 GB) cost 100 s of GPU per session. Precompute them on a **Kaggle CPU kernel** (no queue) and attach them as a dataset. This also frees the GPU job from the text encoder.
- **CPU kernels for everything non-GPU:**
  - TTS (Kokoro on CPU already works);
  - captions;
  - music mixing;
  - assembly.
  - Today the voice agent also uses GPU sessions. Moving all voice work to CPU kernels gives Heartwick's video jobs the whole ~30 h.
- **TPU.**
  - Google's MaxDiffusion does support Wan2.2-I2V-A14B with the lightx2v distill LoRA on TPU (v5e and up).
  - On paper a v5e-8 would make a 4-step 480p clip in well under a minute.
  - But both API pushes ran on a **4-vCPU / 31 GB CPU VM with `jax.devices() == [CpuDevice]`**: `machine_shape` `TpuV5E8` and `TpuV6E8`, kernel `heartwick-r2-tpu-wan`, logs in `kaggle_tpu/out_v1`, `out_v2`.
  - **As tested, TPU is not viable by automation.** One thing is worth a 2-minute check by Shubh in the browser: Settings → Accelerator → is TPU offered, and what is the TPU quota? If it works interactively, the MaxDiffusion route is worth a dedicated test.

## 4. Other free GPU sources (no card, Shubh 17, father adult)

Checked 22 Nov 2026. Nothing was signed up for or logged into. Details are in `research/round2/compute/providers.json`.

| Source | GPU / quota | Card? | Age / other | Automation / terms | Verdict |
|---|---|---|---|---|---|
| **Kaggle** | T4 x2, ~30 h/week, 2 GPU sessions | no (phone for GPU) | under legal age → parental-consent process | official CLI. **Non-commercial, personal use only** (see §6). **One account per person**: "may not have, control, or operate under more than one active Kaggle account". | primary, with the terms risk |
| **Kaggle second account** | — | — | — | **Forbidden** for the same person. The father's own account run by Shubh's scripts is "control[ling]" a second account in practice. | **don't** |
| **HF ZeroGPU** | half RTX Pro 6000 (48 GB). 2 min/day anonymous; **5 min/day free account** | no | 13+ | official `gradio_client`. **Quota is charged to the *caller***, so duplicating a Space or hosting 2 Spaces adds no quota. Hosting needs an account more than 30 days old with a verified email. No non-commercial clause in the HF ToS. | **supplement:** father's free account → ~4-5 clips/day |
| Google Colab free | T4 when available, unpublished limits | no | 18+ only for the AI features | Free tier bans "bypassing the notebook UI to interact primarily via a web UI", remote control and distributed workers. All tiers ban multiple accounts to dodge limits. | manual top-up at most; not schedulable |
| Lightning AI | Sep 2026: one-time 5 credits on sign-up, 25 more **only after adding a card**. Nov 2026: starter hours cut. | card for anything useful | adult | — | ✗ |
| SageMaker Studio Lab | 4 h GPU / 24 h | no | — | **closed to new customers** (reported Sep 2026) | ✗ |
| Paperspace Gradient free | M4000 8 GB, 6 h | unclear | — | cannot hold a 14B expert | ✗ |
| Saturn Cloud | no current free GPU tier found | — | — | — | ✗ |
| Intel Tiber AI Cloud | free JupyterLab learning env; free Max-GPU instances reported removed | — | — | — | ✗ |
| Deepnote free | CPU only | — | — | — | ✗ |
| Modal / Oracle / HF Pro | — | **card** | — | — | ✗ (excluded) |

**Sources:**

- [Kaggle Terms](https://www.kaggle.com/terms), read in a browser on 22 Nov: the multiple-account, non-commercial and parental-consent clauses.
- [HF ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu).
- [HF ToS](https://huggingface.co/terms-of-service).
- [Colab FAQ](https://research.google.com/colaboratory/faq.html).
- [Lightning billing FAQ](https://lightning.ai/docs/platform/overview/faq/billing.md).
- [gpuperhour "Free Cloud GPUs in 2026"](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), Sep 2026.
- [Paperspace free tier (aimultiple)](https://aimultiple.com/free-cloud-gpu).
- [Intel Tiber forum: free tier](https://community.intel.com/t5/Intel-Developer-Cloud/What-happened-to-the-free-tier/m-p/1614678).
- [Deepnote free tier](https://agentdeals.dev/vendor/deepnote).
- [MaxDiffusion README](https://github.com/AI-Hypercomputer/maxdiffusion).
- [lightx2v Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models) (Apache-2.0).
- [SageAttention](https://github.com/thu-ml/sageattention).

## 5. The plan: daily schedule for 7+ videos a week

**What one video needs:** 18 animated 5 s clips (12-24 depending on length), with ~1.4 generations per kept clip. That is **~25 A14B generations per video**, so **~176 a week** for 7 videos.

**Weekly capacity from measured speeds** (`research/round2/compute/throughput.py`, output in `throughput.txt`):

| Source | Speed | Hours/week used | Clips/week |
|---|---|---|---|
| Kaggle T4 x2, Q5_K_M 480p, one process per GPU | 870 s per clip per T4 (measured) | 26 of ~30 (the rest: keyframes, embeddings, setup) | **~200** |
| Laptop Arc 130T at night, Q4_K_M 480p | ~1,665 s per clip (block-measured) | 8 h × 7 nights, 80% success | **~97** (needs the page-file fix, §6) |
| HF ZeroGPU, father's free account | ~75 s per clip, 5 min/day | — | **~28** |
| **Total** | | | **~325, which covers 1.8x of the 176 needed** |

**Videos per week, by clips per video:**

| Setup | 12 clips/video | 18 clips/video | 24 clips/video |
|---|---|---|---|
| All three sources | 19 | 13 | 10 |
| Kaggle alone | 12 | 8 | 6 |
| Laptop + ZeroGPU only (no Kaggle) | 7 | 5 | 4 |

**Daily schedule (IST).** It runs one day ahead: day D's video is generated on D-1 and finished on the morning of D.

| Time | Machine | Job |
|---|---|---|
| 08:00-08:30 | here (Claude) | Story D+1 → `shots.json`: beats, keyframe and motion prompts, durations from the TTS word timings |
| 08:30 | **Kaggle CPU kernel** (no queue) | TTS for D+1 (voice library); umt5 prompt embeddings for all D+1 shots, saved as a private dataset |
| 09:00-12:30 | **Kaggle GPU session (one slot, T4 x2)** | Keyframes for D+1 first (Z-Image / Qwen-Edit; time per the quality agent). Then ~26 Wan clips, 13 per T4, Q5_K_M 480x832x81. Expert-phase batches of ~6 clips per GPU, with a resumable `jobs.jsonl` so a lost session costs one batch. Uses ~3.5-4 h of quota (7 × ~3.7 h ≈ 26 h/week). **The second GPU slot stays free** for the voice and quality agents. |
| through the day | HF ZeroGPU (father's account) | 4-5 rescue clips for shots the QA critic rejected (`gradio_client`, 5 min/day) |
| 12:30-13:30 | laptop iGPU (Vulkan ncnn) | Download D+1 clips. RIFE → 30 fps, Real-ESRGAN x2, Lanczos to 1080x1920: 95 s × 18 ≈ 30 min (measured today). CPU stays around 30%. |
| 13:30-14:00 | laptop CPU (capped ≤ 75%) | Assembly: camera moves, captions, music bed, -14 LUFS master, QA contact sheet → **video D+1 ready**. A human glance, then upload YouTube Short + Instagram Reel. |
| 23:00-07:00 | **laptop iGPU, unattended** | Night batch: ~17 A14B 480p clips for D+2 (the buffer, retries, hero shots, long-form stock). Same `kaggle_t4/gen.py --device xpu`. Docker and the clipping jobs are paused so commit memory is free. Stop at 75 °C (temperature peaked at 71 °C in today's runs), plus a CPU guard. |
| — | laptop 2 (i5-2430M) | **No AI work** (no usable GPU). At most uploads and fetching outputs over SSH. |

**Ranked plan:**

1. **Kaggle T4 x2 with the expert-phase split, both GPUs, Q5_K_M distill, 480x832x81.** About 200 clips/week, which carries 7-8 videos/week on its own at 18 clips per video. Measured, high confidence. It depends on the Kaggle terms question (§6).
2. **fp16 VAE** in `gen.py`. Estimated −12% per clip, so ~+25 clips/week. Test next.
3. **The laptop night batch** on the Arc 130T. About +97 clips/week once commit memory is fixed. It also makes the plan survive Kaggle being taken away. Medium confidence: speed is block-measured, and no full 480p clip has run end to end yet.
4. **ZeroGPU on the father's free account.** About +28/week; for rescues.
5. **Optional, only if Shubh accepts it for wide or ambient shots:** 384x672 + ESRGAN, 2x cheaper.

**Not used:**

- TeaCache/MagCache: nothing to skip at 4 steps.
- SageAttention: does not compile on a T4.
- Fewer frames + RIFE: motion is crammed into half the time, and it morphs.
- Kaggle TPU: no TPU via the API.
- All other free clouds (§4).

**Estimate.** **7 videos/week: likely** (Kaggle measured at ~200 clips against ~176 needed). **10+/week: possible** once the laptop night batch works.

**Confidence:**

- **High** on the per-clip speeds: 8 Kaggle clips, very consistent; step times varied only 154-181 s.
- **Medium** on the weekly Kaggle hours: the quota varies, and other agents share it.
- **Medium-low** on the laptop until one full night batch runs.

### 5b. Why the laptop clip did not finish tonight

- Windows charges iGPU memory as system commit. The limit is 63.4 GB: 47.4 GB RAM + a 16 GB page file, and the **page file was 100% used**.
- Other workloads were running at the same time: the clipping renders, Docker/WSL, 50+ idle Python processes and browsers. Together they held ~50 GB.
- So the three full-size attempts each failed on allocation, not on speed:
  - XPU OOM with "13.9 GiB free";
  - `bad allocation`;
  - `MemoryError`.
- stable-diffusion.cpp hit the same limit: it could not allocate a 2.5 GB `Vulkan_Host` buffer.
- What did work:
  - umt5 + VAE encode on XPU;
  - the per-block A14B timings (`laptop/layer_bench.json`);
  - an end-to-end sd.cpp clip at 256x448 (`laptop/out/sdcpp_smoke3_256.webm`).
- Vulkan gotcha: the VAE needs `--conv-direct auto,vae=off`, or the device is lost. With it off, the 480p VAE encode is slow (319 s), so prefer the **PyTorch XPU path** (`gen.py --device xpu`) for the night batch.


## 6. Needs Shubh or his father (no cards anywhere)

1. **Kaggle terms vs. a monetised channel (a decision; no money needed).**
   - Kaggle's Terms of Use say: *"You will only use the Services for your own internal, personal, non-commercial use, and not on behalf of or for the benefit of any third party."*
   - Making videos for the father's monetised channel on Shubh's Kaggle account is arguably both commercial use and use for a third party.
   - Choices:
     - accept the risk (an account ban would remove ~60% of the capacity overnight);
     - keep Kaggle for R&D only and rely on the laptop + ZeroGPU (~5 videos/week at 18 clips);
     - or ask Kaggle support.
   - **Do not open a second Kaggle account for quota.** The Terms say you "may not have, control, or operate under more than one active Kaggle account", and a ban would hit the main account too. The father's own account, driven by Shubh's scripts, is effectively a second account controlled by Shubh.
2. **The laptop page file (one admin setting, then a reboot).**
   - The iGPU's memory is system RAM. Windows' commit limit is 47 GB RAM + a **16 GB fixed page file that is 100% used**.
   - Set it to "System managed", or 48-64 GB on C: (868 GB free), and reboot.
   - During the 23:00-07:00 Heartwick batch, also pause Docker Desktop and the clipping renders.
   - Optional: raise the shared-GPU-memory limit in Intel Graphics Software. Today the iGPU shows "24 GB", half of RAM.
3. **A free Hugging Face account for the father** (adult; 13+ is enough; email only, **no card**).
   - It gives 5 GPU-min/day on ZeroGPU, against 2 min/day anonymous.
   - He creates it himself and gives Shubh a read token. No agent logs in for him.
   - Hosting his own Space adds no quota: ZeroGPU charges the caller.
4. **A 2-minute browser check of Kaggle TPU access** (notebook Settings → Accelerator).
   - API-pushed TPU jobs ran on a CPU VM.
   - If a TPU v5e/v6e and a TPU quota show up interactively, it is worth one MaxDiffusion test. On paper it is far faster than the T4s.
5. **The old queued kernel.** `heartwick-proto-i2v-t4` v2 has been queued since 21 Nov 22:13 IST and holds one of the 2 GPU slots, which blocked the quality agent. Its test is superseded by this round. Delete it, or let its owner re-push it. I did not touch it.
6. **Voice/TTS onto Kaggle CPU kernels** (agree with the voice agent), so all ~26-30 GPU-hours go to video.

Nothing in this plan needs a card or any payment details.

