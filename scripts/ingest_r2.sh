#!/usr/bin/env bash
# one-off: pull one bag from R2 -> extract cache -> delete bag, so peak disk stays at one bag, used 2026-08-08, kept for provenance
# ingest_r2.sh -- pull each bag from Cloudflare R2, extract its cache, delete the bag.
#
# Peak disk on MSI = ONE bag (~3 GB) + the growing cache dir (~2-4 GB for 100 bags).
# The 300 GB of raw bags never lands here.
#
# Bags are discovered RECURSIVELY by looking for metadata.yaml, so any nesting works:
#     r2:rosbags/training_day_4/rosbag2_2026_08_08-13_39_01/
#     r2:rosbags/training_day_5/rosbag2_.../
# Cache filenames flatten the path:  training_day_4__rosbag2_2026_08_08-13_39_01.npz
#
# Resumable: bags already cached are skipped, so re-running after an interruption
# picks up where it left off.
#
# USAGE
#   module load rclone/1.74.4
#   conda activate $SCRATCH/conda_envs/oft
#   export R2_REMOTE=r2:rosbags
#   export CACHE_DIR=$SCRATCH/bag_cache
#   bash ingest_r2.sh 2>&1 | tee $SCRATCH/ingest.log
#
# Run on a COMPUTE NODE (extraction loads full-res frames into RAM), inside tmux.

set -uo pipefail

R2_REMOTE="${R2_REMOTE:?set R2_REMOTE, e.g. r2:rosbags}"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/bag_cache}"
TMP_BAG="${TMP_BAG:-$SCRATCH/_tmp_bag}"
EXTRACT="${EXTRACT:-$HOME/SignWay/extract_cache.py}"

mkdir -p "$CACHE_DIR" "$TMP_BAG"

echo "=== discovering bags under $R2_REMOTE (looking for metadata.yaml) ==="
BAGS=$(rclone lsf --recursive --files-only "$R2_REMOTE" \
       | grep '/metadata\.yaml$' | sed 's:/metadata\.yaml$::' | sort)

if [ -z "$BAGS" ]; then
  echo "No bags found. Check: rclone lsf --recursive --files-only $R2_REMOTE | head"
  exit 1
fi

TOTAL=$(echo "$BAGS" | wc -l)
echo "found $TOTAL bags"
echo

i=0; ok=0; fail=0
FAILED_LIST=""

while IFS= read -r BAG; do
  i=$((i+1))
  FLAT=$(echo "$BAG" | tr '/' '_')          # training_day_4_rosbag2_...
  CACHE="$CACHE_DIR/${FLAT}.npz"
  LOCAL="$TMP_BAG/$FLAT"

  if [ -f "$CACHE" ]; then
    echo "[$i/$TOTAL] $BAG -- cached, skip"
    ok=$((ok+1)); continue
  fi

  echo "[$i/$TOTAL] $BAG -- downloading..."
  rm -rf "${LOCAL:?}"
  if ! rclone copy "$R2_REMOTE/$BAG" "$LOCAL" --transfers=8 --checkers=8; then
    echo "   DOWNLOAD FAILED"
    fail=$((fail+1)); FAILED_LIST="$FAILED_LIST\n  download: $BAG"; continue
  fi

  # integrity: metadata.yaml AND a non-trivial data file
  if [ ! -f "$LOCAL/metadata.yaml" ]; then
    echo "   MISSING metadata.yaml"
    fail=$((fail+1)); FAILED_LIST="$FAILED_LIST\n  no-metadata: $BAG"
    rm -rf "${LOCAL:?}"; continue
  fi
  DB=$(ls "$LOCAL"/*.db3 "$LOCAL"/*.mcap 2>/dev/null | head -1)
  if [ -z "$DB" ] || [ "$(stat -c%s "$DB")" -lt 1000000 ]; then
    echo "   MISSING or TRUNCATED data file"
    fail=$((fail+1)); FAILED_LIST="$FAILED_LIST\n  truncated: $BAG"
    rm -rf "${LOCAL:?}"; continue
  fi

  echo "   extracting..."
  if python "$EXTRACT" --bag "$LOCAL" --out "$CACHE"; then
    ok=$((ok+1))
  else
    echo "   EXTRACT FAILED"
    fail=$((fail+1)); FAILED_LIST="$FAILED_LIST\n  extract: $BAG"
  fi

  rm -rf "${LOCAL:?}"                        # free the ~3 GB immediately
done <<< "$BAGS"

echo
echo "=== done: $ok cached, $fail failed, out of $TOTAL ==="
[ -n "$FAILED_LIST" ] && echo -e "failures:$FAILED_LIST"
du -sh "$CACHE_DIR"
