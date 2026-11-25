#!/usr/bin/env bash
# Finishing pass for an open-model I2V clip (480x832 @16 fps) -> 1080x1920 @30 fps, on the laptop iGPU via Vulkan.
#   RIFE v4.6 (MIT) frame interpolation to exact 30 fps frame count
#   Real-ESRGAN realesr-animevideov3 x2 (BSD-3) anime upscaler, then Lanczos to 1080x1920
#   slow push-in (1.00 -> 1.05) so the camera never stops, like the reference
# usage: finish.sh in.mp4 out.mp4 [workdir]
set -euo pipefail
IN=$1; OUT=$2; WD=${3:-$(mktemp -d)}
T=/c/Users/shubh/tools/ncnn
mkdir -p "$WD/src" "$WD/rife" "$WD/esr"
ffmpeg -v error -y -i "$IN" "$WD/src/%05d.png"
N=$(ls "$WD/src" | wc -l)
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")
TARGET=$(python -c "print(round($DUR*30))")
t0=$(date +%s.%N)
"$T/rife-ncnn-vulkan-20221029-windows/rife-ncnn-vulkan.exe" -i "$WD/src" -o "$WD/rife" -n "$TARGET" -m "$T/rife-ncnn-vulkan-20221029-windows/rife-v4.6" -f %05d.png > "$WD/rife.log" 2>&1
t1=$(date +%s.%N)
"$T/esr/realesrgan-ncnn-vulkan.exe" -i "$WD/rife" -o "$WD/esr" -n realesr-animevideov3-x2 -s 2 -m "$T/esr/models" -f png > "$WD/esr.log" 2>&1
t2=$(date +%s.%N)
ffmpeg -v error -y -framerate 30 -i "$WD/esr/%05d.png" \
  -vf "scale=1188:2112:flags=lanczos,zoompan=z='1+0.05*on/$TARGET':x='(iw-iw/zoom)/2+12*on/$TARGET':y='(ih-ih/zoom)/2':d=1:s=1080x1920:fps=30,format=yuv420p" \
  -c:v libx264 -crf 17 -preset slow "$OUT"
t3=$(date +%s.%N)
python -c "print({'in_frames':$N,'out_frames':$TARGET,'rife_s':round($t1-$t0,1),'esrgan_s':round($t2-$t1,1),'encode_s':round($t3-$t2,1)})"
