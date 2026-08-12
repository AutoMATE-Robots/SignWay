#!/usr/bin/env bash
# isaac_container.sh — pull and run Isaac Sim under Singularity on MSI.
#
# WHY A CONTAINER: MSI is RHEL8 with GLIBC 2.28. Isaac's pip wheels need GLIBC 2.35+, and the
# system graphics libraries are old. A container brings its own GLIBC + graphics userspace;
# --nv injects the host NVIDIA driver. This is the standard way Isaac runs on HPC.
#
# WHERE: an A40 or L40S node only.
#   srun -N 1 --ntasks-per-node=16 --mem=60gb --gres=gpu:a40:1 -t 8:00:00 -p interactive-gpu --tmp 100gb --pty bash
# NOT msigpu — A100/H100 have no RT cores and NVIDIA does not support Isaac on them.
#
# USAGE:
#   export NGC_API_KEY=<your key from ngc.nvidia.com>
#   bash isaac_container.sh pull          # once, ~20GB into scratch
#   bash isaac_container.sh shell         # interactive shell inside the container
#   bash isaac_container.sh run script.py # run a python script inside the container

set -euo pipefail

VERSION="${ISAAC_VERSION:-5.1.0}"
SCRATCH="${ISAAC_SCRATCH:-/scratch.global/$USER/isaac}"
SIF="$SCRATCH/isaac-sim_${VERSION}.sif"
CACHE="$SCRATCH/cache"

mkdir -p "$SCRATCH" "$CACHE"/{kit,ov,pip,glcache,computecache,logs,data,documents}

# Apptainer's build cache + tmp default to $HOME, which has a quota on MSI — a 20GB pull will
# blow it up with "disk quota exceeded" halfway through. Keep both in scratch. The cache needs
# roughly 2x the image size (raw layers + assembled SIF).
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/apptainer_cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH/apptainer_tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

# The container must be able to WRITE its shader cache. On HPC the image is read-only, which is
# the #1 cause of "Read-only file system: .../nv_shadercache" crashes. We bind writable scratch
# over every cache path Isaac uses.
BINDS=(
  --bind "$CACHE/kit:/isaac-sim/kit/cache:rw"
  --bind "$CACHE/ov:/root/.cache/ov:rw"
  --bind "$CACHE/pip:/root/.cache/pip:rw"
  --bind "$CACHE/glcache:/root/.cache/nvidia/GLCache:rw"
  --bind "$CACHE/computecache:/root/.nv/ComputeCache:rw"
  --bind "$CACHE/logs:/root/.nvidia-omniverse/logs:rw"
  --bind "$CACHE/data:/root/.local/share/ov/data:rw"
  --bind "$CACHE/documents:/root/Documents:rw"
  --bind "$HOME/SignWay:/workspace/SignWay:rw"
)

# Vulkan needs to find the host NVIDIA driver from inside the container.
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
  BINDS+=( --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro )
fi

ENVS=(
  --env ACCEPT_EULA=Y
  --env PRIVACY_CONSENT=Y
  --env OMNI_KIT_ACCEPT_EULA=YES
)

case "${1:-}" in
  pull)
    : "${NGC_API_KEY:?set NGC_API_KEY first — get one at ngc.nvidia.com (Setup -> Generate API Key)}"
    export APPTAINER_DOCKER_USERNAME='$oauthtoken'
    export APPTAINER_DOCKER_PASSWORD="$NGC_API_KEY"
    export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
    export SINGULARITY_DOCKER_PASSWORD="$NGC_API_KEY"
    echo "[isaac] pulling isaac-sim:$VERSION -> $SIF  (~20GB, be patient)"
    singularity pull "$SIF" "docker://nvcr.io/nvidia/isaac-sim:$VERSION"
    ls -lh "$SIF"
    ;;
  shell)
    [ -f "$SIF" ] || { echo "no image at $SIF — run 'pull' first"; exit 1; }
    singularity shell --nv "${ENVS[@]}" "${BINDS[@]}" "$SIF"
    ;;
  run)
    [ -f "$SIF" ] || { echo "no image at $SIF — run 'pull' first"; exit 1; }
    shift
    # /isaac-sim/python.sh is Isaac's own python — it has all the extensions on its path.
    singularity exec --nv "${ENVS[@]}" "${BINDS[@]}" "$SIF" /isaac-sim/python.sh "$@"
    ;;
  *)
    echo "usage: NGC_API_KEY=... bash isaac_container.sh {pull|shell|run <script.py>}"
    echo "  image : $SIF"
    echo "  cache : $CACHE"
    exit 1
    ;;
esac