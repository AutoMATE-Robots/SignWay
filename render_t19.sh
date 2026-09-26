#!/usr/bin/env bash
# render_t19.sh — Clip A (easy sign): direct read, Gemini only as fallback.
# Run as a FILE: bash render_t19.sh 2>&1 | tee render_t19.log
set -euo pipefail
export SCRATCH=/scratch.global/$USER
BAG=rosbag2-keller-t19
BAGDIR=~/SignWay/ros2_bags/$BAG
JSONL=$SCRATCH/ar_replay/$BAG.jsonl
OUT=$SCRATCH/video/clipA_t19.mp4
cd ~/SignWay
source ~/.secrets

[ -f "$BAGDIR/metadata.yaml" ] || { echo "bag missing: $BAGDIR"; exit 1; }
if [ ! -f "$JSONL" ]; then
  python -m adaptive_reasoning.replay.dump_features --csv ~/SignWay/ar_extra.csv \
    --bags-root ~/SignWay/ros2_bags --only "$BAG" --stride 3 --out "$SCRATCH/ar_replay"
fi

mkdir -p "$(dirname "$OUT")"
python -m adaptive_reasoning.replay.render_submission \
  --bag "$BAGDIR" --jsonl "$JSONL" \
  --tau 0.55 --goal "Elevators" --junction 312 \
  --direct --min-plate-h 80 --direct-min-conf 0.5 \
  --vlm real --model gemini-3.1-flash-lite \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --rpm 15 --reasoning low --latency 3.0 --fixed-latency \
  --hide-irrelevant --h264 \
  --out "$OUT"