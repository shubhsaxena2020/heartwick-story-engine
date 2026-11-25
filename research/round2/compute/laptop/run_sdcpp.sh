#!/usr/bin/env bash
# stable-diffusion.cpp (Vulkan, commit 532d2c1) Wan2.2-I2V-A14B lightx2v-distill 260412 Q4_K_M on the Arc 130T.
# Weights stream from disk (mmap + graph cut) because Windows commit headroom is small on this shared laptop.
# usage: run_sdcpp.sh tag W H frames [extra sd-cli args]
cd "$(dirname "$0")"; TAG=$1; W=$2; H=$3; F=$4; shift 4
M=/c/Users/shubh/tools/r2models
P="2D hand-drawn anime animation. The older woman in the green cardigan takes two slow steps to the maple tree, kneels down on the leaves, and gently lifts the small grey squirrel in both hands, smiling. Golden maple leaves keep falling. The camera slowly pushes in and drifts right. Smooth natural motion, consistent character, no morphing."
powershell -NoProfile -ExecutionPolicy Bypass -File monitor.ps1 -Out "mon_$TAG.csv" -Every 15 -Count 600 &
MON=$!
start=$(date +%s)
/c/Users/shubh/tools/sdcpp/bin/sd-cli.exe -M vid_gen --diffusion-model $M/lo_260412_Q4_K_M.gguf --high-noise-diffusion-model $M/hi_260412_Q4_K_M.gguf \
  --vae $M/wan_2.1_vae.safetensors --t5xxl $M/umt5-xxl-encoder-Q8_0.gguf -i key.jpg -p "$P" -n "static, morphing, extra limbs, blurry, text, watermark, low quality" \
  -W $W -H $H --video-frames $F --fps 16 --steps 4 --cfg-scale 1 --high-noise-cfg-scale 1 --sampling-method euler --high-noise-sampling-method euler \
  --flow-shift 5 --moe-boundary 0.9 --seed 7 -t 4 --vae-tiling "$@" -o "out/sdcpp_$TAG.webm" -v > "sdcpp_$TAG.log" 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - start ))" >> "sdcpp_$TAG.log"
kill $MON 2>/dev/null
