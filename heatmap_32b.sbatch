#!/bin/bash
#SBATCH -J heatmap32b
#SBATCH -N 1 --ntasks-per-node=16 --mem=60gb --gres=gpu:a100:1 -t 2:00:00 -p msigpu
#SBATCH -o /users/1/munda057/SignWay/heatmap_32b_%j.log
export SCRATCH=/scratch.global/$USER
export HF_HOME=$SCRATCH/hf VLLM_USE_FLASHINFER_SAMPLER=0 OPENAI_API_KEY=dummy PYTHONUNBUFFERED=1
eval "$(conda shell.bash hook)"
cd ~/SignWay
conda activate $SCRATCH/conda_envs/vllm
vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ --port 8000 --max-model-len 4096 \
  --limit-mm-per-prompt '{"image": 2}' --gpu-memory-utilization 0.75 > $SCRATCH/ar_eval/vllm_heatmap.log 2>&1 &
SERVER=$!
for i in $(seq 1 120); do curl -s localhost:8000/v1/models >/dev/null && break; sleep 10; done
curl -s localhost:8000/v1/models | head -c 120 || { echo "server never came up"; kill $SERVER; exit 1; }
conda activate $SCRATCH/conda_envs/ar
python -m adaptive_reasoning.replay.fig_heatmap --csv-dir heatmap --images-root figs \
  --folder-map numeric_sign=numeric unclear_sign=unclear goal_irrelevant=goal_irrelevant \
  --out $SCRATCH/ar_eval/heatmap --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --base-url http://localhost:8000/v1 --reasoning none --rpm 0
cp $SCRATCH/ar_eval/heatmap/ar_heatmap.png $SCRATCH/ar_eval/heatmap/ar_heatmap.pdf ~/SignWay/figs/
kill $SERVER
echo "HEATMAP DONE"
