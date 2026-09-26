#!/bin/bash
export SCRATCH=/scratch.global/$USER
module load rclone/1.74.4
cd ~/SignWay
python -c "import doctr" 2>/dev/null || { echo "wrong env: activate ar"; exit 1; }
CSV=adaptive_reasoning/annotations/ar_extra-aug-20.csv
R2=r2:rosbags/20aug-rosbags-tate-keller

for t in 18_45_34 18_47_20 18_53_30; do
  b="rosbag2_2026_08_20-${t}"
  echo "==== $b ===="
  ls ros2_bags/$b/*.db3 ros2_bags/$b/*.mcap >/dev/null 2>&1 || rclone copy "$R2/$b" "ros2_bags/$b" -P || exit 1
  rm -rf "$SCRATCH/ar_replay/${b}_crops"
  python -m adaptive_reasoning.replay.dump_features --csv "$CSV" \
    --bags-root ros2_bags --out $SCRATCH/ar_replay --only "$b" \
    --topic /image_raw --stride auto || exit 1
  [ -f "$SCRATCH/ar_replay/$b.npz" ] || { echo "NPZ MISSING $b - raw kept"; exit 1; }
  rclone check "ros2_bags/$b" "$R2/$b" --one-way || { echo "R2 CHECK FAILED - raw kept"; exit 1; }
  rm -rf "ros2_bags/$b"
  echo "DONE $b"
done
rclone copy $SCRATCH/ar_replay r2:rosbags/ar_replay_full -P
echo "REMAINING 12 REDUMPED"
