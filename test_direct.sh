#!/usr/bin/env bash
# test_direct.sh — try the direct-read path on existing dumps. CPU only, no GPU or VLM needed.
# Run as a FILE:  bash test_direct.sh 2>&1 | tee test_direct.log   (never paste these lines)
set -euo pipefail
export SCRATCH=/scratch.global/$USER
cd ~/SignWay

# 1. one crop from t19, closest sighting (last file)
CROP=$(ls "$SCRATCH"/ar_replay/rosbag2-keller-t19_crops/*.jpg | tail -1)
echo "== single crop: $CROP"
python -m adaptive_reasoning.evidence.direct one "$CROP"

# 2. coverage + precision over every bag that has a *_crops dir
echo "== evaluate"
python -m adaptive_reasoning.evidence.direct evaluate \
  --crops-root "$SCRATCH/ar_replay" --csv ar_extra.csv \
  --out "$SCRATCH/ar_replay/direct_eval.json"
