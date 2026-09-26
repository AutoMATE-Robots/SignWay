#!/usr/bin/env bash
# ingest_21aug.sh -- the 50 Aug-21 Tate/Keller bags: R2 -> .npz cache -> delete bag.
#   r2:rosbags/20aug-rosbags-tate-keller/<stamp>  ->  rosbag2-keller-t119 .. t168
# (Aug-21 stamps live in the "20aug" folder; annotation names carry an
#  _image_raw suffix the R2 folders do not -- this script uses the R2 form.)
# Resumable; no `set -e`; a bad bag logs and the loop continues.
#   cd ~/SignWay && ./scripts/ingest_21aug.sh 2>&1 | tee $SCRATCH/ingest_21aug.log

set -uo pipefail

: "${SCRATCH:=/scratch.global/$USER}"
SRC="rosbags/20aug-rosbags-tate-keller"
TMP="$SCRATCH/tmp_21aug"
CACHE="$SCRATCH/bag_cache"
MAP="$HOME/SignWay/data/bag_rename_map.csv"
mkdir -p "$TMP" "$CACHE"

PAIRS="
rosbag2_2026_08_21-16_01_39 rosbag2-keller-t119
rosbag2_2026_08_21-16_02_41 rosbag2-keller-t120
rosbag2_2026_08_21-16_02_58 rosbag2-keller-t121
rosbag2_2026_08_21-16_03_17 rosbag2-keller-t122
rosbag2_2026_08_21-16_03_48 rosbag2-keller-t123
rosbag2_2026_08_21-16_04_17 rosbag2-keller-t124
rosbag2_2026_08_21-16_04_39 rosbag2-keller-t125
rosbag2_2026_08_21-16_04_59 rosbag2-keller-t126
rosbag2_2026_08_21-16_06_16 rosbag2-keller-t127
rosbag2_2026_08_21-16_06_56 rosbag2-keller-t128
rosbag2_2026_08_21-16_07_54 rosbag2-keller-t129
rosbag2_2026_08_21-16_08_12 rosbag2-keller-t130
rosbag2_2026_08_21-16_17_22 rosbag2-keller-t131
rosbag2_2026_08_21-16_17_45 rosbag2-keller-t132
rosbag2_2026_08_21-16_18_52 rosbag2-keller-t133
rosbag2_2026_08_21-16_19_12 rosbag2-keller-t134
rosbag2_2026_08_21-16_20_26 rosbag2-keller-t135
rosbag2_2026_08_21-16_21_11 rosbag2-keller-t136
rosbag2_2026_08_21-16_21_31 rosbag2-keller-t137
rosbag2_2026_08_21-16_21_49 rosbag2-keller-t138
rosbag2_2026_08_21-16_26_43 rosbag2-keller-t139
rosbag2_2026_08_21-16_27_40 rosbag2-keller-t140
rosbag2_2026_08_21-16_28_39 rosbag2-keller-t141
rosbag2_2026_08_21-16_30_19 rosbag2-keller-t142
rosbag2_2026_08_21-16_30_45 rosbag2-keller-t143
rosbag2_2026_08_21-16_31_10 rosbag2-keller-t144
rosbag2_2026_08_21-16_31_46 rosbag2-keller-t145
rosbag2_2026_08_21-16_33_03 rosbag2-keller-t146
rosbag2_2026_08_21-16_33_28 rosbag2-keller-t147
rosbag2_2026_08_21-16_34_02 rosbag2-keller-t148
rosbag2_2026_08_21-16_34_33 rosbag2-keller-t149
rosbag2_2026_08_21-16_34_59 rosbag2-keller-t150
rosbag2_2026_08_21-16_35_32 rosbag2-keller-t151
rosbag2_2026_08_21-16_36_29 rosbag2-keller-t152
rosbag2_2026_08_21-16_36_56 rosbag2-keller-t153
rosbag2_2026_08_21-16_39_24 rosbag2-keller-t154
rosbag2_2026_08_21-16_39_50 rosbag2-keller-t155
rosbag2_2026_08_21-16_40_13 rosbag2-keller-t156
rosbag2_2026_08_21-16_40_58 rosbag2-keller-t157
rosbag2_2026_08_21-16_41_34 rosbag2-keller-t158
rosbag2_2026_08_21-16_42_01 rosbag2-keller-t159
rosbag2_2026_08_21-16_42_24 rosbag2-keller-t160
rosbag2_2026_08_21-16_42_49 rosbag2-keller-t161
rosbag2_2026_08_21-16_43_07 rosbag2-keller-t162
rosbag2_2026_08_21-16_44_52 rosbag2-keller-t163
rosbag2_2026_08_21-16_45_21 rosbag2-keller-t164
rosbag2_2026_08_21-16_45_55 rosbag2-keller-t165
rosbag2_2026_08_21-16_51_18 rosbag2-keller-t166
rosbag2_2026_08_21-16_51_48 rosbag2-keller-t167
rosbag2_2026_08_21-16_53_10 rosbag2-keller-t168
"

total=$(echo "$PAIRS" | grep -c '[^[:space:]]')
echo "ingesting $total bags -> rosbag2-keller-t119 .. t168"
echo "cache before: $(ls "$CACHE" | wc -l) files"
ok=0; skip=0; fail=0; failed_list=""
i=0
while read -r stamp tname; do
  [ -z "$stamp" ] && continue
  i=$((i+1))
  if [ -f "$CACHE/$tname.npz" ]; then
    echo "[skip $i/$total] $tname"; skip=$((skip+1)); continue
  fi
  echo "=== [$i/$total] $stamp -> $tname ==="
  if ! rclone copy "r2:$SRC/$stamp" "$TMP/$stamp" --transfers 8 -P; then
    echo "[FAIL rclone] $stamp"; fail=$((fail+1)); failed_list="$failed_list $tname"; continue
  fi
  if ! python scripts/extract_cache.py --bag "$TMP/$stamp" --out "$CACHE/$tname.npz"; then
    echo "[FAIL extract] $stamp -> $tname (missing /odom?)"
    rm -rf "${TMP:?}/$stamp"; fail=$((fail+1)); failed_list="$failed_list $tname"; continue
  fi
  echo "20aug-tate-keller/$stamp,$tname,21aug_${stamp}.npz,$tname.npz,True" >> "$MAP"
  rm -rf "${TMP:?}/$stamp"
  ok=$((ok+1))
done <<< "$PAIRS"

echo
echo "ingest done: $ok cached, $skip skipped, $fail failed"
echo "cache now:   $(ls "$CACHE" | wc -l) files   (144 + 50 = 194 when complete)"
[ "$fail" -gt 0 ] && echo "FAILED:$failed_list  (rerun to retry)"
exit 0
