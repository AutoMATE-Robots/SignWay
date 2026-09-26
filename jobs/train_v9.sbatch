#!/bin/bash -l
#SBATCH --job-name=signway_v9
#SBATCH --account=gini
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=60gb
#SBATCH --tmp=100gb
#SBATCH --time=24:00:00
#SBATCH --output=/scratch.global/%u/slurm_signway_v9_%j.out
#SBATCH --error=/scratch.global/%u/slurm_signway_v9_%j.err
#
# Pepper-VLA v9 -- 119 bags (96 keller + 23 amundson, the 2nd building).
#
# A100 (80GB) instead of the A40. The recipe is UNCHANGED -- do not raise
# batch_size to "use" the extra memory: batch 4 / lr 1e-4 is the validated
# pairing, and changing batch without re-tuning lr is how a working run stops
# working. The A100 buys wall-clock speed, nothing else.
# If a100-4 queues badly, try: -p a100-8, or fall back to -p a40 --gres=gpu:a40:1.
#
# CANONICAL RECIPE (do not change without a reason; see SIGNWAY_LESSONS.md):
#   --use_film False   FiLM makes the whole ~485M vision backbone trainable and
#                      destroyed the pretrained features on small homogeneous
#                      data (failure #2). NON-NEGOTIABLE.
#   --learning_rate 1e-4   protects the frozen backbone. Higher is the dangerous
#                      direction.
#   NUM_ACTIONS_CHUNK=1, ACTION_DIM=16   cumulative waypoints (failure #1 fix).
#   "Next Actions L1 Loss = nan" in wandb is EXPECTED at chunk=1.
#
# Steps: v8 shipped at 9k on 96 bags. 119 bags -> 12000 keeps roughly the same
# passes-per-sample. save_freq 3000 gives intermediate checkpoints for the
# overfitting check (if 12k evals worse than 9k on the frozen anchors, ship 9k).

set -uo pipefail

export SCRATCH=/scratch.global/$USER
export HF_HOME=$SCRATCH/hf
export TORCH_HOME=$SCRATCH/torch_hub
export WANDB_MODE=offline
export SIGNWAY_CACHE_DIR=$SCRATCH/bag_cache
export SIGNWAY_FLIP_ZERO=1          # matches the dataset build; logged for provenance

# conda in a non-interactive shell: bashrc sourcing does NOT work here
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/conda_envs/oft

cd $SCRATCH/openvla-oft || { echo "openvla-oft not found in \$SCRATCH"; exit 1; }

echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import sys; sys.argv.append('--signway'); \
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM; \
print('constants: chunk', NUM_ACTIONS_CHUNK, 'dim', ACTION_DIM)" || exit 1

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir $SCRATCH/rlds_v9 \
  --dataset_name signway \
  --run_root_dir $SCRATCH/oft_ckpts_v9 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --lora_rank 32 \
  --max_steps 12000 \
  --save_freq 3000 \
  2>&1 | tee $SCRATCH/oft_train_v9.log

echo "=== done $(date) ==="
ls -la $SCRATCH/oft_ckpts_v9/ 2>/dev/null | tail -5
