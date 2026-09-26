#!/usr/bin/env bash
# ingest_20aug.sh -- the 25 Aug-20 Tate/Keller bags: R2 -> .npz cache -> delete bag.
#
#   r2:rosbags/20aug-rosbags-tate-keller/<stamp>   ->   rosbag2-keller-t94 .. t118
#
# NOTE ON NAMES: the annotation list writes each bag as "<stamp>_image_raw", but the
# R2 folders are plain "<stamp>". This script uses the R2 (plain) form; the t-names
# are what the builder sees, so the suffix never matters downstream.
#
# t94+ is free: the Aug-12 day-5 batch that was going to take those numbers never
# cached (no /odom), so nothing was written to the map or the cache for them.
#
# Resumable (existing caches skipped). No `set -e` -- a bad bag logs and the loop
# continues, and it cannot kill an interactive srun shell.
#
#   cp ingest_20aug.sh ~/SignWay/scripts/ && chmod +x ~/SignWay/scripts/ingest_20aug.sh
#   cd ~/SignWay && ./scripts/ingest_20aug.sh 2>&1 | tee $SCRATCH/ingest_20aug.log
#
# COMPUTE node only:
#   srun -N 1 --ntasks-per-node=8 --mem=48gb -t 6:00:00 -p amdsmall --tmp 200gb --pty bash
#   export SCRATCH=/scratch.global/$USER && conda activate $SCRATCH/conda_envs/oft
#   module load rclone/1.74.4 && cd ~/SignWay

set -uo pipefail          # deliberately NOT -e

: "${SCRATCH:=/scratch.global/$USER}"
SRC="rosbags/20aug-rosbags-tate-keller"
TMP="$SCRATCH/tmp_20aug"
CACHE="$SCRATCH/bag_cache"
MAP="$HOME/SignWay/data/bag_rename_map.csv"
mkdir -p "$TMP" "$CACHE"

# stamp -> t-name, in annotation (chronological) order
PAIRS="
rosbag2_2026_08_20-17_12_19 rosbag2-keller-t94
rosbag2_2026_08_20-17_13_22 rosbag2-keller-t95
rosbag2_2026_08_20-17_14_52 rosbag2-keller-t96
rosbag2_2026_08_20-17_18_32 rosbag2-keller-t97
rosbag2_2026_08_20-17_20_55 rosbag2-keller-t98
rosbag2_2026_08_20-17_22_43 rosbag2-keller-t99
rosbag2_2026_08_20-17_28_28 rosbag2-keller-t100
rosbag2_2026_08_20-17_30_56 rosbag2-keller-t101
rosbag2_2026_08_20-17_38_08 rosbag2-keller-t102
rosbag2_2026_08_20-17_39_11 rosbag2-keller-t103
rosbag2_2026_08_20-17_40_34 rosbag2-keller-t104
rosbag2_2026_08_20-17_41_42 rosbag2-keller-t105
rosbag2_2026_08_20-17_42_38 rosbag2-keller-t106
rosbag2_2026_08_20-17_43_32 rosbag2-keller-t107
rosbag2_2026_08_20-17_44_27 rosbag2-keller-t108
rosbag2_2026_08_20-17_45_35 rosbag2-keller-t109
rosbag2_2026_08_20-18_08_58 rosbag2-keller-t110
rosbag2_2026_08_20-18_11_09 rosbag2-keller-t111
rosbag2_2026_08_20-18_11_54 rosbag2-keller-t112
rosbag2_2026_08_20-18_16_36 rosbag2-keller-t113
rosbag2_2026_08_20-18_21_01 rosbag2-keller-t114
rosbag2_2026_08_20-18_34_52 rosbag2-keller-t115
rosbag2_2026_08_20-18_45_34 rosbag2-keller-t116
rosbag2_2026_08_20-18_47_20 rosbag2-keller-t117
rosbag2_2026_08_20-18_53_30 rosbag2-keller-t118
"

total=$(echo "$PAIRS" | grep -c '[^[:space:]]')
echo "ingesting $total bags -> rosbag2-keller-t94 .. t118"
echo "cache before: $(ls "$CACHE" | wc -l) files"
ok=0; skip=0; fail=0; failed_list=""
i=0

while read -r stamp tname; do
  [ -z "$stamp" ] && continue
  i=$((i+1))
  if [ -f "$CACHE/$tname.npz" ]; then
    echo "[skip $i/$total] $tname"
    skip=$((skip+1)); continue
  fi
  echo "=== [$i/$total] $stamp -> $tname ==="
  if ! rclone copy "r2:$SRC/$stamp" "$TMP/$stamp" --transfers 8 -P; then
    echo "[FAIL rclone] $stamp"
    fail=$((fail+1)); failed_list="$failed_list $tname(rclone)"; continue
  fi
  if ! python scripts/extract_cache.py --bag "$TMP/$stamp" --out "$CACHE/$tname.npz"; then
    echo "[FAIL extract] $stamp -> $tname   (missing /odom? check metadata.yaml)"
    rm -rf "${TMP:?}/$stamp"
    fail=$((fail+1)); failed_list="$failed_list $tname(extract)"; continue
  fi
  # 5 columns, matching the existing rows: r2_path, t_name, old_npz, new_npz, renamed
  echo "20aug-tate-keller/$stamp,$tname,20aug_${stamp}.npz,$tname.npz,True" >> "$MAP"
  rm -rf "${TMP:?}/$stamp"
  ok=$((ok+1))
done <<< "$PAIRS"

echo
echo "=========================================================="
echo "ingest done: $ok cached, $skip skipped, $fail failed"
echo "cache now:   $(ls "$CACHE" | wc -l) files"
[ "$fail" -gt 0 ] && echo "FAILED:$failed_list  (rerun to retry -- completed bags are skipped)"
echo "=========================================================="
exit 0
