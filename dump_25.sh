#!/bin/bash
export SCRATCH=/scratch.global/$USER
module load rclone/1.74.4
cd ~/SignWay
CSV=adaptive_reasoning/annotations/ar_extra-aug-20.csv
R2=r2:rosbags/20aug-rosbags-tate-keller

for t in 17_12_19 17_13_22 17_14_52 17_18_32 17_20_55 17_22_43 17_28_28 17_30_56 \
         17_38_08 17_39_11 17_40_34 17_41_42 17_42_38 17_43_32 17_44_27 17_45_35 \
         18_08_58 18_11_09 18_11_54 18_16_36 18_21_01 18_34_52 18_45_34 18_47_20 18_53_30; do
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
echo "ALL 25 REDUMPED"
