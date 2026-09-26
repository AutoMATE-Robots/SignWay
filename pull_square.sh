#!/usr/bin/env bash
# pull_square.sh — 35 GB square-loop bag -> $SCRATCH (delete after the final render)
set -euo pipefail
export SCRATCH=/scratch.global/$USER
DEST=$SCRATCH/ros2_bags/rosbag2-square          # scripts refer to it by this name
command -v rclone >/dev/null || module load rclone/1.74.4
mkdir -p "$DEST"
rclone copy "r2:rosbags/22aug/rosbag2_2026_08_22-20_29_37/" "$DEST/" -P --transfers 4
ls -la "$DEST"
compgen -G "$DEST/*.db3" >/dev/null || compgen -G "$DEST/*.mcap" >/dev/null || { echo "data file missing"; exit 1; }
ln -sfn "$DEST" ~/SignWay/ros2_bags/rosbag2-square
echo "done; delete with: rm -rf $DEST  (only after clipC renders)"