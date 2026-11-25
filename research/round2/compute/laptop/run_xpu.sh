#!/usr/bin/env bash
# laptop Arc 130T run of gen.py (XPU) with CPU/temperature/GPU sampling alongside. usage: run_xpu.sh plan.json tag
cd "$(dirname "$0")"
powershell -NoProfile -ExecutionPolicy Bypass -File monitor.ps1 -Out "mon_$2.csv" -Every 15 -Count 400 &
MON=$!
start=$(date +%s)
/c/Users/shubh/tools/xpuvenv/Scripts/python.exe ../kaggle_t4/gen.py --plan "$1" --out out --device xpu > "gen_$2.log" 2>&1
echo "rc=$? wall_s=$(( $(date +%s) - start ))" >> "gen_$2.log"
kill $MON 2>/dev/null
