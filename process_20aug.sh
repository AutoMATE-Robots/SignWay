#!/bin/bash
# process_20aug.sh — one bag at a time: pull → mp4 → VLA cache → AR harvest → verify → DELETE raw.
# Run on a GPU node (AR dump needs docTR/CUDA):  bash process_20aug.sh 2>&1 | tee figs/ingest_20aug.log
# NOTE: no `set -e` on purpose (srun --pty + set -e kills the allocation); every step checks explicitly.

export SCRATCH=/scratch.global/$USER
module load rclone/1.74.4
cd ~/SignWay

OFT=$SCRATCH/conda_envs/oft
AR=$SCRATCH/conda_envs/ar
R2DIR="r2:rosbags/20aug-rosbags-tate-keller"

TIMES="17_12_19 17_13_22 17_14_52 17_18_32 17_20_55 17_22_43 17_28_28 17_30_56 \
       17_38_08 17_39_11 17_40_34 17_41_42 17_42_38 17_43_32 17_44_27 17_45_35 \
       18_08_58 18_11_09 18_11_54 18_16_36 18_21_01 18_34_52 18_45_34 18_47_20 18_53_30"

fail() { echo "FAIL [$bag]: $1 — raw KEPT, stopping so nothing is lost"; exit 1; }

for t in $TIMES; do
  short="rosbag2_2026_08_20-${t}"
  bag="${short}_image_raw"
  echo "================ $bag ================"

  # 1. pull (skip if already present from the earlier partial loop)
  if [ ! -d "ros2_bags/$bag" ]; then
    rclone copy "$R2DIR/$short" "ros2_bags/$bag" -P || fail "rclone pull"
  fi

  # 2. mp4 for annotation (oft env) — ADJUST to your exact day-5 invocation:
  conda run -p $OFT python bag_to_mp4_low_memory.py "ros2_bags/$bag" \
    || fail "mp4 conversion"

  # 3. VLA 224x224 npz cache (oft env) — ADJUST: paste your day-5 cache command,
  #    it must read ros2_bags/$bag and write into $SCRATCH/bag_cache/
  # conda run -p $OFT python <your_cache_script>.py "ros2_bags/$bag" ... \
  #   || fail "VLA cache"
  echo "REMINDER: step 3 (VLA cache) is a placeholder — paste your day-5 command"

  # 4. AR harvest dump (ar env): crops + contact sheet + jsonl + φ, dummy goal
  conda run -p $AR python -m adaptive_reasoning.replay.dump_features \
    --harvest --only "$bag" --fps 30 \
    --bags-root ros2_bags --out $SCRATCH/ar_replay_20aug --stride auto \
    || fail "AR harvest dump"

  # 5. verify all artifacts exist
  ls "ros2_bags/${bag}"*.mp4 >/dev/null 2>&1 || ls "mp4s/${bag}"*.mp4 >/dev/null 2>&1 \
    || echo "WARN [$bag]: mp4 not found where expected — check your mp4 output dir"
  [ -f "$SCRATCH/ar_replay_20aug/$bag.npz" ]            || fail "AR npz missing"
  [ -f "$SCRATCH/ar_replay_20aug/${bag}_contactsheet.jpg" ] || fail "contact sheet missing"
  # [ -f "$SCRATCH/bag_cache/$bag.npz" ] || fail "VLA cache missing"   # enable once step 3 is real

  # 6. confirm R2 still holds the original, byte-for-byte, then DELETE the raw
  rclone check "ros2_bags/$bag" "$R2DIR/$short" --one-way || fail "rclone check"
  rm -rf "ros2_bags/$bag"
  echo "DONE [$bag] — raw deleted, artifacts kept"
done

echo "ALL BAGS PROCESSED"
echo "Now mirror the harvest to R2 (scratch purge insurance):"
echo "  rclone copy $SCRATCH/ar_replay_20aug r2:rosbags/ar_replay_20aug -P"
