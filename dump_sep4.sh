#!/bin/bash
export SCRATCH=/scratch.global/$USER
module load rclone/1.74.4
cd ~/SignWay
python -c "import doctr" 2>/dev/null || { echo "wrong env: activate ar"; exit 1; }
CSV=adaptive_reasoning/annotations/ar_extra-sep-4.csv
R2=r2:rosbags/4sep-new
BAGS=$(python -c "import csv;print(' '.join(r['bag'] for r in csv.DictReader(open('$CSV'))))")

for b in $BAGS; do
  echo "==== $b ===="
  ls ros2_bags/$b/*.db3 ros2_bags/$b/*.mcap >/dev/null 2>&1 || rclone copy "$R2/$b" "ros2_bags/$b" -P || exit 1
  TOPIC=$(grep -o '/[A-Za-z0-9_/]*image_raw[A-Za-z0-9_/]*' ros2_bags/$b/metadata.yaml | head -1)
  [ -n "$TOPIC" ] || { echo "no image topic in metadata.yaml for $b"; exit 1; }
  echo "topic: $TOPIC"
  rm -rf "$SCRATCH/ar_replay/${b}_crops"
  python -m adaptive_reasoning.replay.dump_features --csv "$CSV" \
    --bags-root ros2_bags --out $SCRATCH/ar_replay --only "$b" \
    --topic "$TOPIC" --stride auto || exit 1
  [ -f "$SCRATCH/ar_replay/$b.npz" ] || { echo "NPZ MISSING $b - raw kept"; exit 1; }
  rclone check "ros2_bags/$b" "$R2/$b" --one-way || { echo "R2 CHECK FAILED - raw kept"; exit 1; }
  rm -rf "ros2_bags/$b"
  echo "DONE $b"
done
rclone copy $SCRATCH/ar_replay r2:rosbags/ar_replay_full -P
echo "ALL 30 DUMPED"
