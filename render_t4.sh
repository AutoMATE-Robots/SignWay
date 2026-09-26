#!/usr/bin/env bash
# render_t4.sh — Clip B (complex sign): direct read should refuse, the VLM earns its call.
# Run as a FILE: bash render_t4.sh 2>&1 | tee render_t4.log
set -euo pipefail
export SCRATCH=/scratch.global/$USER
BAG=rosbag2-keller-t4
GOAL="5-150"          # goal string from t4's ar_extra.csv row
JUNCTION=0              # junction_frame (raw idx) from that row
BAGDIR=~/SignWay/ros2_bags/$BAG
JSONL=$SCRATCH/ar_replay/$BAG.jsonl
OUT=$SCRATCH/video/clipB_t4.mp4
cd ~/SignWay
source ~/.secrets
[ "$GOAL" != "FILL_ME" ] || { echo "set GOAL and JUNCTION at the top of this script"; exit 1; }

command -v rclone >/dev/null || module load rclone/1.74.4
if [ ! -f "$BAGDIR/metadata.yaml" ]; then
  rclone copy "r2:rosbags/$BAG/" "$BAGDIR/" -P
fi
compgen -G "$BAGDIR/*.db3" >/dev/null || compgen -G "$BAGDIR/*.mcap" >/dev/null \
  || { echo "no .db3/.mcap — truncated transfer?"; exit 1; }

if [ ! -f "$JSONL" ]; then
  # --stride auto = fps//10, so a 20 fps bag gets 2 and a 30 fps bag gets 3
  python -m adaptive_reasoning.replay.dump_features --csv ~/SignWay/ar_extra.csv \
    --bags-root ~/SignWay/ros2_bags --only "$BAG" --stride auto --out "$SCRATCH/ar_replay"
fi

mkdir -p "$(dirname "$OUT")"
python -m adaptive_reasoning.replay.render_submission \
  --bag "$BAGDIR" --jsonl "$JSONL" \
  --tau 0.65 --goal "$GOAL" --junction "$JUNCTION" \
  --min-plate-h 0 \
  --vlm real --model gemini-3.1-flash-lite \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --rpm 15 --reasoning low --latency 4.0 --fixed-latency \
  --hide-irrelevant --h264 \
  --out "$OUT"
