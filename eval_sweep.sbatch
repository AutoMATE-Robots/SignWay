#!/bin/bash
#SBATCH --job-name=pepper_eval_v7v8
#SBATCH --account=gini
#SBATCH --partition=interactive-gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60gb
#SBATCH --tmp=100gb
#SBATCH --time=2:00:00             # 12 runs x (7B load + ~500 frames) — generous on purpose
#SBATCH --output=/scratch.global/%u/eval_sweep_%j.out

# ============================================================================
# Pepper-VLA eval sweep: v7@9k, v7@18k, v8@9k  x  frozen anchors (e1/e2/e3)
#                        + training-reference bag (t1)
# One GPU, fully sequential, unattended. Everything lands in one results dir.
# Submit:  sbatch eval_sweep.sbatch
# ============================================================================

set -uo pipefail   # NOT -e: one failed bag must not kill the whole sweep

# ---- environment (fresh shells inherit nothing) ----------------------------
export SCRATCH=/scratch.global/$USER
export HF_HOME=$SCRATCH/hf
export WANDB_MODE=offline

# conda: sbatch shells are non-interactive AND the user's login shell is zsh,
# so ~/.bashrc has no conda init. The conda BINARY is on PATH (Slurm propagates
# the submit environment) — initialize the bash hook directly from it.
if ! command -v conda >/dev/null 2>&1; then
    source ~/.bashrc 2>/dev/null || true   # long-shot fallback
fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || \
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
else
    # last resort: common conda install locations
    for c in ~/miniconda3 ~/anaconda3 /common/software/install/migrated/anaconda; do
        [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
    done
fi
conda activate $SCRATCH/conda_envs/oft || { echo "FATAL: conda activate failed"; exit 1; }
echo "conda env: $CONDA_PREFIX"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    || { echo "FATAL: env sanity check failed"; exit 1; }

echo "=== node: $(hostname)  gpu: ==="
nvidia-smi || true
echo "==============================="

# ---- checkpoints under test -------------------------------------------------
# !!! VERIFY THESE with `ls $SCRATCH/oft_ckpts_v7 $SCRATCH/oft_ckpts_v8` !!!
# Dir names encode the recipe (b?, lr, lora-r) — if v7/v8 changed batch size
# or anything else, the names below are wrong. Paste the real ones.
CKPT_NAMES=( "v7_9000" "v7_18000" "v8_9000" )
CKPT_PATHS=(
  "$SCRATCH/oft_ckpts_v7/openvla-7b+signway+b4+lr-0.0001+lora-r32+dropout-0.0--image_aug--9000_chkpt"
  "$SCRATCH/oft_ckpts_v7/openvla-7b+signway+b4+lr-0.0001+lora-r32+dropout-0.0--image_aug--18000_chkpt"
  "$SCRATCH/oft_ckpts_v8/openvla-7b+signway+b4+lr-0.0001+lora-r32+dropout-0.0--image_aug--9000_chkpt"
)

# ---- bags: "name flip decision turn_done stride" ---------------------------
# e1/e2/e3 = frozen anchors (the ruler across versions, same args as always).
# t1       = training-reference bag: eval-vs-train gap in the same sweep.
# Add more lines here if you want extra bags graded (30fps bags -> stride 3).
BAGS=(
  "rosbag2-keller-e1 190 turn_right 522 2"
  "rosbag2-keller-e2 190 turn_left 443 2"
  "rosbag2-keller-e3 0 straight 9999 2"
  "rosbag2-keller-t1 159 turn_right 380 2"
)
BAG_ROOT=/users/1/munda057/SignWay/ros2_bags

# ---- optional noise-probe script (mandatory gate, if you have it scripted) --
# Point this at your probe script to auto-run it per checkpoint before evals;
# leave empty to skip (then probe manually before trusting results).
PROBE_SCRIPT=""

# ---- results layout ---------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M)
RESULTS_ROOT=$SCRATCH/eval_sweep_$STAMP
LOG_DIR=$RESULTS_ROOT/logs
mkdir -p "$LOG_DIR"
echo "Results root: $RESULTS_ROOT"

export PYTHONPATH=~/SignWay/tools:~/SignWay/signway_dataset:${PYTHONPATH:-}
EVAL=~/SignWay/signway_dataset/eval_openloop.py

# ---- sanity-check every checkpoint BEFORE burning GPU hours -----------------
# (saves are staged: require 4 safetensors + config.json)
for i in "${!CKPT_NAMES[@]}"; do
    N=${CKPT_NAMES[$i]}; P=${CKPT_PATHS[$i]}
    if [ ! -d "$P" ]; then
        echo "WARNING: checkpoint dir MISSING for $N -> $P (will skip)"
        continue
    fi
    NSAFE=$(ls "$P"/*.safetensors 2>/dev/null | wc -l)
    if [ "$NSAFE" -lt 4 ] || [ ! -f "$P/config.json" ]; then
        echo "WARNING: $N looks incomplete ($NSAFE safetensors) — staged save not flushed? (will skip)"
    else
        echo "OK: $N ($NSAFE safetensors)"
    fi
done

# ---- the sweep --------------------------------------------------------------
TOTAL=0; FAILED=0
for i in "${!CKPT_NAMES[@]}"; do
    NAME=${CKPT_NAMES[$i]}
    CKPT=${CKPT_PATHS[$i]}

    if [ ! -d "$CKPT" ] || [ "$(ls "$CKPT"/*.safetensors 2>/dev/null | wc -l)" -lt 4 ]; then
        echo ">>> SKIPPING $NAME (missing/incomplete checkpoint)"
        continue
    fi

    if [ -n "$PROBE_SCRIPT" ] && [ -f "$PROBE_SCRIPT" ]; then
        echo ">>> [$NAME] noise probe"
        python "$PROBE_SCRIPT" --checkpoint "$CKPT" \
            2>&1 | tee "$LOG_DIR/${NAME}_noise_probe.log"
    fi

    for B in "${BAGS[@]}"; do
        set -- $B
        BAG=$1; FLIP=$2; DEC=$3; TDONE=$4; STRIDE=$5
        OUT="$RESULTS_ROOT/eval_${NAME}_${BAG}"
        RUNLOG="$LOG_DIR/${NAME}_${BAG}.log"
        TOTAL=$((TOTAL+1))

        echo ""
        echo ">>> [$NAME] $BAG  ($(date +%H:%M:%S))"
        T0=$SECONDS
        python "$EVAL" \
            --bag "$BAG_ROOT/$BAG" --flip-frame "$FLIP" --decision "$DEC" \
            --turn-done-frame "$TDONE" --stride "$STRIDE" \
            --checkpoint "$CKPT" --out "$OUT" \
            2>&1 | tee "$RUNLOG"
        RC=${PIPESTATUS[0]}
        echo ">>> [$NAME] $BAG done in $((SECONDS-T0))s (rc=$RC)"
        [ "$RC" -ne 0 ] && FAILED=$((FAILED+1))

        # loud check: raw series saved? (the "never read pixels off a PNG again" patch)
        if [ ! -f "$OUT/series.npy" ]; then
            echo "!!! $OUT has no series.npy — the np.save patch is NOT applied to"
            echo "!!! eval_openloop.py. PNGs only; report falls back to a montage."
        fi
    done
done

echo ""
echo "=== sweep complete: $TOTAL runs, $FAILED failed ==="

# ---- build the Ajay-ready report (CPU only, seconds) ------------------------
python ~/SignWay/signway_dataset/make_eval_report.py --root "$RESULTS_ROOT" \
    2>&1 | tee "$LOG_DIR/report.log"

echo ""
echo "Done. Pull figures + summary from: $RESULTS_ROOT/report/"
