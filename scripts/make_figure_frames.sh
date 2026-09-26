#!/usr/bin/env bash
# make_figure_frames.sh — env -> replay -> 5 frames + fact sheet for Ajay.
#
# Run INSIDE tmux on a compute node:
#   srun -N 1 --ntasks-per-node=16 --mem=60gb --gres=gpu:a40:1 -t 4:00:00 \
#        -p interactive-gpu --tmp 100gb --pty bash
#   tmux new -s fig
#   bash ~/SignWay/scripts/make_figure_frames.sh
#
# Skip the replay and go straight to picking (trace already exists):
#   bash ~/SignWay/scripts/make_figure_frames.sh --pick-only

set -euo pipefail

PICK_ONLY=0
[[ "${1:-}" == "--pick-only" ]] && PICK_ONLY=1

# ---------------------------------------------------------------- 1. env ----
export SCRATCH=/scratch.global/$USER
export HF_HOME=$SCRATCH/hf

REPO=$HOME/SignWay
AR_ENV=$SCRATCH/conda_envs/ar          # full path: ar and jmem shadow each other
BAG=$REPO/ros2_bags/figure
OUT=$SCRATCH/figtrace/figure
FOR_AJAY=$REPO/adaptive_reasoning/for_ajay

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$AR_ENV"
export PYTHONPATH=$REPO/adaptive_reasoning:$REPO/tools:${PYTHONPATH:-}

echo "python : $(which python)"
python -c "import numpy; print('numpy  :', numpy.__version__)"
[[ "$(which python)" == "$AR_ENV"/* ]] || { echo "WRONG ENV — jmem is shadowing ar" >&2; exit 1; }

# ------------------------------------------------------------- 2. checks ----
mkdir -p "$OUT"
[[ -f "$BAG/metadata.yaml" ]] || { echo "no metadata.yaml in $BAG" >&2; exit 1; }
find "$BAG" -maxdepth 1 \( -name '*.db3' -o -name '*.mcap' \) | grep -q . \
  || { echo "no .db3/.mcap in $BAG" >&2; exit 1; }

# native fps straight off the bag — do not guess it, mismatched rates have
# bitten this pipeline before
FPS=$(python - "$BAG" <<'PY'
import sys, yaml, pathlib
m = yaml.safe_load((pathlib.Path(sys.argv[1]) / "metadata.yaml").read_text())
i = m["rosbag2_bagfile_information"]
dur = i["duration"]["nanoseconds"] / 1e9
img = [t for t in i["topics_with_message_count"]
       if "image" in t["topic_metadata"]["name"]]
n = img[0]["message_count"] if img else i["message_count"]
print(f"{n/dur:.2f}")
PY
)
echo "bag    : $BAG"
echo "fps    : $FPS   (native, from metadata.yaml)"

# ------------------------------------------------------------- 3. replay ----
if [[ $PICK_ONLY -eq 0 ]]; then
  # FILL THIS IN. Find the real flags first:
  #   python $REPO/adaptive_reasoning/replay_reasoning.py --help
  #   grep -n add_argument $REPO/adaptive_reasoning/replay_reasoning.py
  #
  # It needs to end up writing $OUT/trace.jsonl + $OUT/frames/ via the
  # TraceWriter patch. Replace the line below with your actual invocation:

  python "$REPO/adaptive_reasoning/replay_reasoning.py" \
      --bag "$BAG" \
      --out "$OUT" \
      --goal "Room 304"

  [[ -f "$OUT/trace.jsonl" ]] || {
    echo "no trace.jsonl in $OUT — the TraceWriter patch did not fire" >&2; exit 1; }
  echo "trace  : $(wc -l < "$OUT/trace.jsonl") frames"
fi

# --------------------------------------------------------------- 4. pick ----
python "$REPO/adaptive_reasoning/pick_frames.py" \
    --trace "$OUT/trace.jsonl" \
    --frames "$OUT/frames" \
    --fps "$FPS" \
    --out "$FOR_AJAY"

echo
echo "done -> $FOR_AJAY"
ls -la "$FOR_AJAY"
echo
echo "if a pick is wrong, rerun with your own frames:"
echo "  python $REPO/adaptive_reasoning/pick_frames.py --trace $OUT/trace.jsonl \\"
echo "      --frames $OUT/frames --fps $FPS --out $FOR_AJAY --panels 0,150,220,268,394"
