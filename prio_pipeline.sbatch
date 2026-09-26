#!/bin/bash
#SBATCH -J signway_prio
#SBATCH -N 1 --ntasks-per-node=16 --mem=90gb --gres=gpu:a100:1 -t 6:00:00 -p msigpu
#SBATCH --tmp 100gb
#SBATCH -o /users/1/munda057/SignWay/prio_%j.log

set -uo pipefail
export SCRATCH=/scratch.global/$USER
export HF_HOME=$SCRATCH/hf VLLM_USE_FLASHINFER_SAMPLER=0 OPENAI_API_KEY=dummy PYTHONUNBUFFERED=1
eval "$(conda shell.bash hook)"
cd ~/SignWay

CSV=adaptive_reasoning/annotations/ar_extra-sep14.csv
LED=$SCRATCH/ar_eval/ledger.jsonl
FIT="rosbag2_2026_08_20-17_12_19 rosbag2_2026_08_20-17_13_22 rosbag2_2026_08_20-17_22_43 rosbag2_2026_08_20-17_30_56 rosbag2_2026_08_20-17_38_08 rosbag2_2026_08_20-17_39_11"
REMOTE=r2:rosbags/4sep-new
STAGE=$SCRATCH/bag_stage

conda activate $SCRATCH/conda_envs/ar
BAGS36=$(python -c "import csv;print(' '.join(r['bag'] for r in csv.DictReader(open('$CSV'))))")
echo "=== $(date) | bags ($(echo $BAGS36 | wc -w)) ==="
echo "$BAGS36"

echo "=== dumps already present: $(ls $SCRATCH/ar_replay/*.npz | wc -l) npz ==="

# ---------------------------------------------------------------- 2. 32B SERVER
echo "=== STEP 2: $(date +%H:%M:%S) vLLM server ==="
conda activate $SCRATCH/conda_envs/vllm
vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ --port 8000 --max-model-len 4096 \
  --limit-mm-per-prompt '{"image": 2}' --gpu-memory-utilization 0.75 \
  > $SCRATCH/ar_eval/vllm_night.log 2>&1 &
SERVER=$!
for i in $(seq 1 120); do curl -s localhost:8000/v1/models >/dev/null && break; sleep 10; done
curl -s localhost:8000/v1/models | head -c 150 || { echo "SERVER NEVER CAME UP"; kill $SERVER 2>/dev/null; exit 1; }

# ---------------------------------------------------------------- 3. CROP PASS
echo ""
echo "=== STEP 3: $(date +%H:%M:%S) crop pass RESUME (cached answers are skipped) ==="
conda activate $SCRATCH/conda_envs/ar
python -m adaptive_reasoning.replay.label_crops --dumps $SCRATCH/ar_replay \
  --csv adaptive_reasoning/annotations/ar_priority.csv \
  --out $SCRATCH/ar_eval --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --base-url http://localhost:8000/v1 --rpm 0 --foreign-goals 0 2>&1 | tail -40
kill $SERVER 2>/dev/null; sleep 5
echo "ledger lines: $(wc -l < $LED)"

# ---------------------------------------------------------------- 4. THETA SWEEP (in SignScene's favour)
echo ""
echo "=== STEP 4: $(date +%H:%M:%S) signscene theta_parse sweep ==="
for tp in 0.3 0.4 0.5 0.6 0.7; do
  python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay --csv $CSV --ledger $LED \
    --out $SCRATCH/ar_eval/g36_tp$tp --tau 0.55 --model 32B --gates signscene --theta-parse $tp >/dev/null 2>&1
  python -c "import json;s=json.load(open('$SCRATCH/ar_eval/g36_tp$tp/summary.json'))['gates']['signscene'];print(f'  theta_parse=$tp calls={s[\"calls\"]:.2f} acc={s[\"acc\"]:.0f}%')" 2>/dev/null
done
BEST_TP=$(for tp in 0.3 0.4 0.5 0.6 0.7; do
  python -c "import json;s=json.load(open('$SCRATCH/ar_eval/g36_tp$tp/summary.json'))['gates']['signscene'];print($tp, s['acc'])" 2>/dev/null
done | sort -k2 -nr | head -1 | cut -d' ' -f1)
BEST_TP=${BEST_TP:-0.5}
echo "signscene best theta_parse = $BEST_TP"

# ---------------------------------------------------------------- 5. MAIN EVAL (all gates incl. signnav)
echo ""
echo "=== STEP 5: $(date +%H:%M:%S) eval_gates (36 approaches) ==="
python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay --csv $CSV --ledger $LED \
  --out $SCRATCH/ar_eval/gates36 --tau 0.55 --model 32B --fit-kfc $FIT --theta-parse $BEST_TP \
  --gates always periodic5s iros signscene relevance ours signnav 2>&1 | tee $SCRATCH/ar_eval/gates36_log.txt
# fallback: if coverage blocked the full set, evaluate the covered subset and say so
if [ ! -f $SCRATCH/ar_eval/gates36/per_approach.csv ]; then
  echo "=== full-set eval blocked by coverage; evaluating COVERED SUBSET ==="
  python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay --csv $CSV --ledger $LED \
    --out $SCRATCH/ar_eval/gates36 --tau 0.55 --model 32B --theta-kfc 6.374612450599671e-05 \
    --theta-parse $BEST_TP --min-coverage 0.9 --drop-uncovered \
    --gates always periodic5s iros signscene relevance ours signnav 2>&1 | tee -a $SCRATCH/ar_eval/gates36_log.txt
fi

# ---------------------------------------------------------------- 6. FIGURES + MEMORY
echo ""
echo "=== STEP 6: $(date +%H:%M:%S) figures ==="
PA=$SCRATCH/ar_eval/gates36/per_approach.csv
python -m adaptive_reasoning.replay.fig_metric_panels --per-approach $PA \
  --gates always iros signscene ours --out figs/ar_metric_panels || true
python -m adaptive_reasoning.replay.fig_score_shades --per-approach $PA --annotations $CSV \
  --datasets temporary numeric compound --out figs/ar_score_shades || true
python -m adaptive_reasoning.replay.fig_bars --summary $SCRATCH/ar_eval/gates36/summary.json \
  --gates iros signscene always ours --out figs/ar_bars || true
python -m adaptive_reasoning.replay.fig_tradeoff --per-approach $PA --out figs/ar_tradeoff --clean || true
python -m adaptive_reasoning.replay.fig_memory --dumps $SCRATCH/ar_replay --per-approach $PA \
  --csv $CSV --tau 0.55 --out figs/ar_memory || true
python -m adaptive_reasoning.replay.gen_memory_pairs --dumps $SCRATCH/ar_replay --out heatmap/memory_pairs.csv || true
python -m adaptive_reasoning.replay.memory_recognition --csv heatmap/memory_pairs.csv --images-root / \
  --out $SCRATCH/ar_eval/memory || true
python -m adaptive_reasoning.replay.stats --per-approach $PA --out $SCRATCH/ar_eval/gates36/stats || true

# ---------------------------------------------------------------- 7. RAW NUMBERS
echo ""
echo "==================== RAW NUMBERS ===================="
echo "--- main table ---"
cat $SCRATCH/ar_eval/gates36/table_main.md
echo ""
echo "--- per-approach: temporary-sign bags ---"
python - <<'PY'
import csv, os
S = os.environ['SCRATCH']
rows = list(csv.DictReader(open(f"{S}/ar_eval/gates36/per_approach.csv")))
for r in sorted((r for r in rows if "2026_09_14" in r["bag"]), key=lambda r: (r["bag"], r["gate"])):
    print(f"{r['bag'][-8:]:10s} {r['gate']:10s} calls={r['calls']:>5s} correct={r['correct']} "
          f"decided={r.get('decided','?')} overhead={r['overhead_s']:>6s}s dur={r.get('duration_s','?')}s")
PY
echo ""
echo "--- per-situation metrics (SR / T / ST' / WT') ---"
python -m adaptive_reasoning.replay.fig_score_shades --per-approach $PA --annotations $CSV \
  --datasets temporary numeric compound --out /tmp/_shades 2>/dev/null | grep -v "^wrote"
echo ""
echo "--- overall metric panels ---"
python -m adaptive_reasoning.replay.fig_metric_panels --per-approach $PA \
  --gates always iros signscene ours signnav --out /tmp/_panels 2>/dev/null | grep -v "^wrote"
echo ""
echo "--- memory: cumulative calls per building ---"
python -m adaptive_reasoning.replay.fig_memory --dumps $SCRATCH/ar_replay --per-approach $PA \
  --csv $CSV --tau 0.55 --out /tmp/_mem 2>/dev/null | grep -v "^wrote"
echo ""
echo "--- memory recognition (recall / precision) ---"
cat $SCRATCH/ar_eval/memory/summary.json 2>/dev/null || echo "(no pairs)"
echo ""
echo "--- paired statistics vs ours ---"
cat $SCRATCH/ar_eval/gates36/stats/paired.md 2>/dev/null || true
echo ""
echo "==================== DONE $(date) ===================="
