#!/bin/bash
#SBATCH -J tateredump
#SBATCH -N 1 --ntasks-per-node=16 --mem=60gb --gres=gpu:a100:1 -t 2:00:00 -p msigpu
#SBATCH -o /users/1/munda057/SignWay/tate_redump_%j.log
export SCRATCH=/scratch.global/$USER PYTHONUNBUFFERED=1
eval "$(conda shell.bash hook)"; conda activate $SCRATCH/conda_envs/ar
module load rclone/1.74.4
cd ~/SignWay
R2=r2:rosbags/20aug-rosbags-tate-keller
for t in 17_12_19 17_13_22 17_22_43 17_30_56 17_38_08 17_39_11; do
  b="rosbag2_2026_08_20-${t}"
  echo "==== $b ===="
  ls ros2_bags/$b/*.db3 >/dev/null 2>&1 || rclone copy "$R2/$b" "ros2_bags/$b" -P || exit 1
  rm -rf "$SCRATCH/ar_replay/${b}_crops"
  python -m adaptive_reasoning.replay.dump_features --csv adaptive_reasoning/annotations/ar_extra-aug-20.csv \
    --bags-root ros2_bags --out $SCRATCH/ar_replay --only "$b" --topic /image_raw --stride auto || exit 1
  python -c "import numpy as np,sys; sys.exit(0 if 'scene_grid' in np.load('$SCRATCH/ar_replay/$b.npz') else 1)" \
    || { echo "no scene_grid for $b"; exit 1; }
  rclone check "ros2_bags/$b" "$R2/$b" --one-way && rm -rf "ros2_bags/$b"
  echo "DONE $b"
done
echo "TATE REDUMP DONE"
