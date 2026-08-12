#!/usr/bin/env bash
# one-off: copy-verify-delete rename of the bags on Cloudflare R2, used 2026-08-08, kept for provenance
# rename_r2.sh -- rename the timestamp-named bags on R2 to rosbag2-keller-t30, t31, ...
#
# SAFETY DESIGN (this is the important part):
#   For each bag:  copy -> VERIFY size and object count match -> only then delete source.
#   `rclone move` was NOT used, because it deletes as it goes: an interruption mid-bag
#   leaves files split across the old and new names, which is how a metadata.yaml got
#   stranded and then lost. With copy-verify-delete, an interruption at ANY point leaves
#   the source complete and the destination merely partial (which the next run redoes).
#
#   Peak extra storage = one bag (~5 GB), not a full duplicate of the bucket.
#   Resumable: bags already renamed are skipped.
#
# USAGE
#   module load rclone/1.74.4
#   bash rename_r2.sh                 # DRY RUN, prints what it would do
#   bash rename_r2.sh --apply         # actually does it
#
# Reads bag_rename_map.csv (written by rename_bags.py) so the numbering matches the
# cache filenames exactly.

set -uo pipefail

REMOTE="${REMOTE:-r2:rosbags}"
MAP="${MAP:-$HOME/SignWay/data/bag_rename_map.csv}"
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

[ -f "$MAP" ] || { echo "missing $MAP -- run rename_bags.py first (dry run is fine)"; exit 1; }

echo "remote : $REMOTE"
echo "map    : $MAP"
[ $APPLY -eq 0 ] && echo "MODE   : DRY RUN (pass --apply to execute)" || echo "MODE   : APPLY"
echo

total=0; done_already=0; renamed=0; failed=0; skipped=0
FAILED_LIST=""

# skip csv header; fields: old_path,new_name,old_cache,new_cache,cache_present
while IFS=, read -r OLD NEW _ _ _; do
  [ "$OLD" = "old_path" ] && continue
  [ -z "$OLD" ] && continue
  total=$((total+1))

  SRC="$REMOTE/$OLD"
  DST="$REMOTE/$NEW"

  src_n=$(rclone lsf "$SRC" 2>/dev/null | wc -l)
  dst_n=$(rclone lsf "$DST" 2>/dev/null | wc -l)

  if [ "$src_n" -eq 0 ] && [ "$dst_n" -gt 0 ]; then
    echo "[$total] $NEW -- already renamed, skip"
    done_already=$((done_already+1)); continue
  fi
  if [ "$src_n" -eq 0 ] && [ "$dst_n" -eq 0 ]; then
    echo "[$total] $OLD -- NOT FOUND at either name, skip"
    skipped=$((skipped+1)); FAILED_LIST="$FAILED_LIST\n  missing: $OLD"; continue
  fi
  if [ "$dst_n" -gt 0 ]; then
    echo "[$total] $NEW -- destination already has $dst_n file(s) while source has $src_n;"
    echo "        leaving BOTH alone, inspect manually"
    skipped=$((skipped+1)); FAILED_LIST="$FAILED_LIST\n  split: $OLD"; continue
  fi

  if [ $APPLY -eq 0 ]; then
    echo "[$total] would rename $OLD ($src_n files) -> $NEW"
    continue
  fi

  echo "[$total] $OLD -> $NEW  (copying $src_n files...)"
  if ! rclone copy "$SRC" "$DST" --transfers=2 --s3-chunk-size=128M \
        --s3-upload-concurrency=4 --retries=5 --low-level-retries=20 --progress; then
    echo "        COPY FAILED -- source untouched"
    failed=$((failed+1)); FAILED_LIST="$FAILED_LIST\n  copy: $OLD"; continue
  fi

  # VERIFY before deleting anything
  s_bytes=$(rclone size "$SRC" --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)
  d_bytes=$(rclone size "$DST" --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)
  d_n=$(rclone lsf "$DST" | wc -l)

  if [ "$s_bytes" != "$d_bytes" ] || [ "$src_n" -ne "$d_n" ]; then
    echo "        VERIFY FAILED: src ${src_n} files/${s_bytes}B vs dst ${d_n} files/${d_bytes}B"
    echo "        source left intact; destination left for inspection"
    failed=$((failed+1)); FAILED_LIST="$FAILED_LIST\n  verify: $OLD"; continue
  fi

  echo "        verified ${d_n} files / ${d_bytes} bytes -- removing source"
  rclone purge "$SRC" && renamed=$((renamed+1))
done < "$MAP"

echo
echo "=== $total in map | $renamed renamed | $done_already already done | $failed failed | $skipped skipped ==="
[ -n "$FAILED_LIST" ] && echo -e "issues:$FAILED_LIST"
if [ $APPLY -eq 1 ]; then
  echo
  echo "clean empty day folders with:  rclone rmdirs $REMOTE --leave-root"
  echo "then re-check completeness with the 96-bag audit."
fi
