#!/usr/bin/env bash
# render_t19.sh — Clip A (OCR-easy, no VLM): pull rosbag2-keller-t19, dump evidence, render.
# Run on an MSI compute node inside tmux, with the `ar` env active (full-path conda activate).
set -euo pipefail
export SCRATCH=/scratch.global/$USER
BAG=rosbag2-keller-t19
BAGDIR=~/SignWay/ros2_bags/$BAG
JSONL=$SCRATCH/ar_replay/$BAG.jsonl
OUT=$SCRATCH/video/clipA_t19.mp4

# --- 1. pull from R2 (name on R2 may be rosbag2-... or rosbags2-...; resolve it) --------
module load rclone/1.74.4
if [ ! -f "$BAGDIR/metadata.yaml" ]; then
  REMOTE_NAME=$(rclone lsd r2:rosbags/ | awk '{print $NF}' | grep -i "keller-t19" | head -1)
  [ -z "$REMOTE_NAME" ] && { echo "t19 not found under r2:rosbags/"; rclone lsd r2:rosbags/ | head; exit 1; }
  rclone copy "r2:rosbags/$REMOTE_NAME/" "$BAGDIR/" -P
fi
ls -la "$BAGDIR"
ls "$BAGDIR"/*.db3 "$BAGDIR"/*.mcap 2>/dev/null || { echo "no .db3/.mcap — truncated transfer?"; exit 1; }

# --- 2. evidence dump (30 fps bag -> stride 3 -> 10 Hz) ---------------------------------
if [ ! -f "$JSONL" ]; then
  # TODO(Akul): your usual dump_features invocation for one bag, e.g.
  # python -m adaptive_reasoning.replay.dump_features --bag "$BAGDIR" --stride 3 --out "$JSONL"
  echo "jsonl missing: $JSONL — run dump_features for $BAG first"; exit 1
fi

# --- 3. render: fast path on, SIM VLM (no server needed) — expect fires=0 in the summary --
mkdir -p "$(dirname "$OUT")"
python -m adaptive_reasoning.replay.render_submission \
  --bag "$BAGDIR" --jsonl "$JSONL" \
  --tau 0.45 --goal "Elevators" --junction 312 \
  --fast-path --vlm sim --decision turn_left --latency 3.0 \
  --captions clipA_t19_captions.json --hide-irrelevant --h264 \
  --out "$OUT"
echo "done -> $OUT   (check the printed 'fires=' — it must be 0 for this clip)"
