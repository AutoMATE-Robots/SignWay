#!/usr/bin/env bash
# fetch_fig_bag.sh — pull one ROS2 bag from Cloudflare R2 to MSI for figure work.
#
#   ./fetch_fig_bag.sh                          # defaults to 4sep-new/rosbag2-pwb-20
#   ./fetch_fig_bag.sh 4sep-new/rosbag2-pwb-20
#
# Save it, don't paste it. Run on a compute node (login nodes are fine for the
# copy itself, but you'll want the node for the replay anyway).

set -euo pipefail

BAG_REL="${1:-4sep-new/rosbag2-pwb-20}"
REMOTE="${R2_REMOTE:-r2:rosbags}"
DEST_ROOT="${SCRATCH:-/scratch.global/$USER}/figbags"
BAG_NAME="$(basename "$BAG_REL")"
DEST="$DEST_ROOT/$BAG_NAME"

echo "[1/4] loading rclone"
module load rclone/1.74.4

echo "[2/4] copying $REMOTE/$BAG_REL -> $DEST"
mkdir -p "$DEST"
rclone copy "$REMOTE/$BAG_REL/" "$DEST/" \
  --progress --transfers 4 --checkers 8 --checksum --retries 5 --low-level-retries 10

echo "[3/4] verifying against remote (hash check)"
rclone check "$REMOTE/$BAG_REL/" "$DEST/" --checksum --one-way

echo "[4/4] local sanity check"
if [[ ! -f "$DEST/metadata.yaml" ]]; then
  echo "FAIL: metadata.yaml missing in $DEST" >&2; exit 1
fi
DATAFILE="$(find "$DEST" -maxdepth 1 \( -name '*.db3' -o -name '*.mcap' \) | head -n1 || true)"
if [[ -z "$DATAFILE" ]]; then
  echo "FAIL: no .db3/.mcap in $DEST" >&2; exit 1
fi
SIZE=$(stat -c %s "$DATAFILE")
if (( SIZE < 1048576 )); then
  echo "FAIL: $DATAFILE is only $SIZE bytes — truncated transfer" >&2; exit 1
fi

echo
echo "OK  $DEST"
ls -la "$DEST"
echo
echo "topics:"
python - "$DEST" <<'PY' 2>/dev/null || echo "  (run 'ros2 bag info' in a ROS env to list topics)"
import sys, yaml, pathlib
m = yaml.safe_load((pathlib.Path(sys.argv[1]) / "metadata.yaml").read_text())
info = m["rosbag2_bagfile_information"]
print("  duration:", round(info["duration"]["nanoseconds"] / 1e9, 1), "s")
print("  messages:", info["message_count"])
for t in info["topics_with_message_count"]:
    md = t["topic_metadata"]
    print(f"  {md['name']:<24} {t['message_count']:>7}  {md['type']}")
PY
