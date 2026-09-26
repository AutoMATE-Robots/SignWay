#!/usr/bin/env bash
# prep_square.sh — PHASE 1 (ar env): evidence dump for the square bag.
# Run as a FILE on a GPU node:  bash prep_square.sh 2>&1 | tee prep_square.log
set -euo pipefail
export SCRATCH=/scratch.global/$USER
BAG=rosbag2-square
cd ~/SignWay
[ -e ros2_bags/$BAG/metadata.yaml ] || { echo "bag not pulled yet (pull_square.sh)"; exit 1; }
grep -q "$BAG" ar_extra.csv || { echo "add a row for $BAG to ar_extra.csv first (goal 6-220, fps from metadata.yaml)"; exit 1; }
[ -f "$SCRATCH/ar_replay/$BAG.jsonl" ] || \
  python -m adaptive_reasoning.replay.dump_features --csv ar_extra.csv --bags-root ros2_bags \
    --only "$BAG" --topic /image_raw --stride auto --out "$SCRATCH/ar_replay"
ls -la "$SCRATCH/ar_replay/" | grep "$BAG"
