#!/bin/bash -l
#SBATCH --job-name=fig_traj
#SBATCH --account=gini
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=48gb
#SBATCH --tmp=100gb
#SBATCH --time=1:00:00
#SBATCH --output=/scratch.global/%u/slurm_fig_traj_%j.out
#SBATCH --error=/scratch.global/%u/slurm_fig_traj_%j.err
#
# Run the VLA over a bag frame range and save RGB frames + chained trajectory,
# for the trajectory-overlay figure. Use when no interactive GPU is available.
#
#   sbatch --export=ALL,BAG=ros2_bags/figure_lidar_image,RANGE=1700:2100:3,PROMPT=turn_left \
#          jobs/fig_traj.sbatch
#
# Get RANGE from:  python signway_dataset/find_turns.py --bag <bag>
# Defaults below are used for anything not exported.

set -uo pipefail

export SCRATCH=/scratch.global/$USER
export HF_HOME=$SCRATCH/hf
export TORCH_HOME=$SCRATCH/torch_hub
export WANDB_MODE=offline

: "${BAG:=ros2_bags/figure_lidar_image}"
: "${RANGE:?set RANGE, e.g. --export=ALL,RANGE=1700:2100:3}"
: "${PROMPT:=turn_left}"
: "${ENVNAME:=oft2}"
: "${CKPT:=$SCRATCH/oft_ckpts_v10/openvla-7b+signway+b4+lr-0.0001+lora-r32+dropout-0.0--image_aug--6000_chkpt}"
: "${OUTDIR:=$SCRATCH/fig_traj}"

eval "$(conda shell.bash hook)"
conda activate $SCRATCH/conda_envs/$ENVNAME || {
    echo "FATAL: conda env $ENVNAME not usable"; exit 1; }

echo "=== $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import numpy, torch; print('numpy', numpy.__version__,
 '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" || exit 1
echo "bag=$BAG  range=$RANGE  prompt=$PROMPT"
echo "ckpt=$(basename $CKPT)"
echo "out=$OUTDIR"
echo "==============================="

cd $HOME/SignWay || exit 1
export PYTHONPATH=$HOME/SignWay/tools:$HOME/SignWay/signway_dataset:${PYTHONPATH:-}

python signway_dataset/run_vla_on_bag.py \
    --bag "$BAG" \
    --checkpoint "$CKPT" \
    --prompt "$PROMPT" \
    --chain --chain-range "$RANGE" \
    --also-straight \
    --out "$OUTDIR"

echo
echo "=== done $(date) ==="
ls -la "$OUTDIR" | head -12
echo
echo "Copy to your laptop and overlay there (no GPU needed):"
echo "  scp -r munda057@agate.msi.umn.edu:$OUTDIR ."
