#!/usr/bin/env bash
# MSI-side launcher for the occupancy server. Run INSIDE an A40 srun session:
#   srun -N 1 --ntasks-per-node=16 --mem=60gb --gres=gpu:a40:1 -t 8:00:00 \
#        -p interactive-gpu --tmp 100gb --pty bash
# then: bash run_msi_server.sh [--model-size small|base|large] [extra args...]
set -euo pipefail

export SCRATCH="${SCRATCH:-/scratch.global/$USER}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"

# Separate env — do NOT install into the pinned oft env during training.
OCC_ENV="${OCC_ENV:-$SCRATCH/conda_envs/occ}"
if [[ ! -d "$OCC_ENV" ]]; then
  echo "ERROR: occupancy env not found at $OCC_ENV" >&2
  echo "Create it once (see README_MSI.md), then re-run." >&2
  exit 1
fi
if [[ "${CONDA_PREFIX:-}" != "$OCC_ENV" ]]; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$OCC_ENV"
fi
echo "[run] env: $CONDA_PREFIX"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA torch not available — are you on an A40 node (agc*)?"
print(f"torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
PY

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$HERE/msi_occupancy_server.py" \
  --depth-repo "$SCRATCH/Depth-Anything-V2/metric_depth" \
  "$@"
