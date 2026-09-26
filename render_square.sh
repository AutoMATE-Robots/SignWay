#!/usr/bin/env bash
# render_square.sh — PHASE 2 (jmem venv): calibrated odom + clip C render.
# Run as a FILE:  source $SCRATCH/venvs/jmem/bin/activate && bash render_square.sh 2>&1 | tee render_square.log
set -euo pipefail
export SCRATCH=/scratch.global/$USER
BAG=rosbag2-square
BAGDIR=~/SignWay/ros2_bags/$BAG
JSONL=$SCRATCH/ar_replay/$BAG.jsonl
ODOM=junction_memory/odom_${BAG}_cal.csv
OUT=$SCRATCH/video/clipC_square.mp4
cd ~/SignWay
source ~/.secrets
python -c "import PIL" 2>/dev/null || pip install -q pillow

# odometry: raw dump -> yaw calibration (k=0.83 is the platform constant)
if [ ! -f "$ODOM" ]; then
  ( cd junction_memory
    [ -f "odom_${BAG}.csv" ] || python odom_memory_probe.py --bag "$BAGDIR" --dump-csv "odom_${BAG}.csv"
    python calibrate_yaw.py --csv "odom_${BAG}.csv" --scale 0.83 --out "odom_${BAG}_cal.csv" )
fi

mkdir -p "$(dirname "$OUT")"
python -m adaptive_reasoning.replay.render_submission_memory \
  --bag "$BAGDIR" --topic /image_raw --jsonl "$JSONL" --odom-csv "$ODOM" \
  --goals square_goals.yaml --tau 0.55 \
  --vlm real --model gemini-3.6-flash \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --rpm 15 --reasoning low \
  --latency 1 --fixed-latency --h264 \
  --out "$OUT"
