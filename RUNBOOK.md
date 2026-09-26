# SignWay AR experiments — RUNBOOK (Sept 14, 2026)

Everything below runs from `~/SignWay` on MSI. Replace the whole
`adaptive_reasoning/` package with the copy in this bundle (48 files, 114 tests).

```
cd ~/SignWay
rm -rf adaptive_reasoning && cp -r <bundle>/adaptive_reasoning .
cp <bundle>/figs/scope.json figs/
cp <bundle>/figs/capability_table.tex figs/
conda activate $SCRATCH/conda_envs/ar
python -m pytest adaptive_reasoning/tests -q          # expect 114 passed
export SCRATCH=/scratch.global/$USER
```

--------------------------------------------------------------------------------
## 1. Directory structure

```
~/SignWay/
  adaptive_reasoning/                  the package (from this bundle)
    annotations/
      ar_extra-sep-4.csv               the 18 held-out approaches (exists)
      ar_extra-aug-20.csv              Tate/Keller fit set (exists)
      ar_routes.csv                    NEW — chained routes, per junction (see ROUTE_COLLECTION_PLAN.md)
  heatmap/                             per-category CSVs written by the annotator
    numeric_sign.csv                   image,goal,expected
    named_goal.csv                     image,goal,expected
    goal_irrelevant.csv                image,goal,expected      (expected = not_applicable)
    unclear_sign.csv                   image,goal,expected
    temporary_sign.csv                 image,goal,expected      (optional; annotator decides correct answer)
    memory_pairs.csv                   first_image,image,same_sign,goal   (memory recall/precision)
    annotate_state.json                annotator progress (auto)
  figs/
    heatmap_pool/                      ONE folder with all frames; csv image paths are relative to it
    scope.json                         by-design ✗ cells with citations (from bundle)
    capability_table.tex               Table A (from bundle)
    *.pdf / *.png                      figures land here

$SCRATCH/
  ar_replay/                           dumps: <bag>.npz + <bag>.jsonl + <bag>_crops/   (exists)
  ar_eval/
    ledger.jsonl                       every VLM ask (7B + 32B crop passes)  (exists)
    vlm_cache/                         cached answers                        (exists)
    gates32/                           eval_gates output (rerun — see §3)
    trigger/                           trigger-mode output
    heatmap_final/                     reading-mode output (Gemini)
    memory/                            memory recognition output
```

CSV formats (header row required, no blank rows):
- category csvs: `image,goal,expected`; `expected` ∈ turn_left|turn_right|straight|stop|not_applicable
- memory_pairs.csv: `first_image,image,same_sign,goal`; `same_sign` 1 = second view of the SAME sign,
  0 = a DIFFERENT sign (precision test); `goal` optional
- ar_extra-sep-4.csv must keep columns `bag,goal,decision,junction_frame,semantic_only,temporary_present`
  (they drive the category split in the nav heatmap and the stages figure)

--------------------------------------------------------------------------------
## 2. Annotate the frame pool (no GPU)

```
python -m adaptive_reasoning.replay.annotate_signs --images figs/heatmap_pool --out heatmap --port 8765
# open http://localhost:8765 (VS Code forwards the port). Save & next; "add another goal" for a 2nd row.
wc -l heatmap/*.csv                                   # aim ~40 rows per category
```

--------------------------------------------------------------------------------
## 3. Main decision benchmark (from cache, ~1 min, any node)

```
FIT="rosbag2_2026_08_20-17_12_19 rosbag2_2026_08_20-17_13_22 rosbag2_2026_08_20-17_22_43 rosbag2_2026_08_20-17_30_56 rosbag2_2026_08_20-17_38_08 rosbag2_2026_08_20-17_39_11"
CSV=adaptive_reasoning/annotations/ar_extra-sep-4.csv; LED=$SCRATCH/ar_eval/ledger.jsonl

# SignScene-style viewpoint threshold: unreported -> set in THEIR favour (pick the best acc)
for tp in 0.5 0.6 0.7 0.8; do
  python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay --csv $CSV --ledger $LED \
    --out $SCRATCH/ar_eval/g_tp$tp --tau 0.55 --model 32B --gates signscene --theta-parse $tp >/dev/null
  python -c "import json;s=json.load(open('$SCRATCH/ar_eval/g_tp$tp/summary.json'))['gates']['signscene'];print('theta_parse=$tp',f'calls={s[\"calls\"]:.2f} acc={s[\"acc\"]:.0f}%')"
done
TP=<best>

python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay --csv $CSV --ledger $LED \
  --out $SCRATCH/ar_eval/gates32 --tau 0.55 --model 32B --fit-kfc $FIT --theta-parse $TP \
  2>&1 | tee $SCRATCH/ar_eval/gates32_log.txt
# check: "answer coverage: ~100%" and misses ~0

python -m adaptive_reasoning.replay.stats --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv --out $SCRATCH/ar_eval/gates32/stats
python -m adaptive_reasoning.replay.fig_bars --summary $SCRATCH/ar_eval/gates32/summary.json \
  --gates iros signscene periodic5s always ours --out figs/ar_bars                      # Fig: accuracy vs calls
python -m adaptive_reasoning.replay.fig_tradeoff --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv --out figs/ar_tradeoff --clean
python -m adaptive_reasoning.replay.fig_overhead --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv --out figs/ar_overhead

# ablations (each ~1 min, from cache)
python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay --csv $CSV --ledger $LED --out $SCRATCH/ar_eval/abl_nomem  --tau 0.55 --model 32B --gates ours --no-memory    | grep SignWay
python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay --csv $CSV --ledger $LED --out $SCRATCH/ar_eval/abl_nofast --tau 0.55 --model 32B --gates ours --no-fast-path | grep SignWay
python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay --csv $CSV --ledger $LED --out $SCRATCH/ar_eval/gates7b   --tau 0.55 --model 7B  --gates always periodic5s reactive signscene relevance ours
```

--------------------------------------------------------------------------------
## 4. Trigger correctness by situation (GPU for docTR, NO model calls, minutes)

```
srun -N 1 --ntasks-per-node=8 --mem=32gb --gres=gpu:a40:1 -t 2:00:00 -p interactive-gpu --pty bash
export SCRATCH=/scratch.global/$USER; eval "$(conda shell.bash hook)"; conda activate $SCRATCH/conda_envs/ar; cd ~/SignWay

python -m adaptive_reasoning.replay.fig_heatmap --mode trigger --csv-dir heatmap --images-root figs/heatmap_pool \
  --out $SCRATCH/ar_eval/trigger --tau 0.55
cp $SCRATCH/ar_eval/trigger/ar_trigger_heatmap.* figs/
```

--------------------------------------------------------------------------------
## 5. Reading accuracy by situation (same GPU session; Gemini; ~45 min for 160 rows)

```
source ~/.secrets
PYTHONUNBUFFERED=1 python -m adaptive_reasoning.replay.fig_heatmap --csv-dir heatmap --images-root figs/heatmap_pool \
  --scope figs/scope.json --out $SCRATCH/ar_eval/heatmap_final --model gemini-3.1-flash-lite \
  --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --reasoning low --rpm 15 \
  2>&1 | tee $SCRATCH/ar_eval/heatmap_final.log
# regenerate later without calls:
python -m adaptive_reasoning.replay.fig_heatmap --csv-dir heatmap --images-root figs/heatmap_pool \
  --scope figs/scope.json --out figs/heatmap_final --from-ledger $SCRATCH/ar_eval/heatmap_final/ledger.jsonl
```

--------------------------------------------------------------------------------
## 6. Navigation-performance figures (from §3–§5 outputs, no calls)

```
# metrics heatmap: SR / SCT / SPL(n/a) / ST / WT  ×  overall/ambiguous/numeric/compound/revisited
python -m adaptive_reasoning.replay.fig_nav_heatmap --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv \
  --annotations adaptive_reasoning/annotations/ar_extra-sep-4.csv --out figs/ar_nav_heatmap

# stages figure: triggered / understood / correct action, per situation
python -m adaptive_reasoning.replay.fig_stages --trigger $SCRATCH/ar_eval/trigger/trigger_results.csv \
  --reading $SCRATCH/ar_eval/heatmap_final/results.csv --approach $SCRATCH/ar_eval/gates32/per_approach.csv \
  --annotations adaptive_reasoning/annotations/ar_extra-sep-4.csv --out figs/ar_stages
```

--------------------------------------------------------------------------------
## 7. Semantic memory (same GPU session; docTR only; minutes)

```
python -m adaptive_reasoning.replay.memory_recognition --csv heatmap/memory_pairs.csv \
  --images-root figs/heatmap_pool --out $SCRATCH/ar_eval/memory
# prints recall (same-sign pairs recognised) and precision (different-sign false matches), lists misses
```
Cumulative-calls-vs-junction plot: needs the chained route bags (ROUTE_COLLECTION_PLAN.md).
Script to follow once `ar_routes.csv` exists.

--------------------------------------------------------------------------------
## 8. Caption facts to carry (do not lose these)
- Rows labelled "-style†" are re-implemented invocation policies on our stack: same reasoner
  (Qwen2.5-VL-32B-AWQ, 3.9 s median), same payload, same data. IROS scene descriptor = patch-grid
  proxy (SigLIP unavailable); SignScene viewpoint threshold set to maximise its accuracy.
- Baselines may re-query a sign after "not applicable"; SignWay consumes it (memory).
- SCT assumes the robot stops during reasoning; driving time equal across policies.
- Reading heatmap uses Gemini as the constant reasoner; trigger heatmap uses no VLM.
- unclear: abstaining scored correct; 12/20 frames yield no detection for any strategy.
- τ = 0.55 fixed before held-out data existed; τ sensitivity curve in appendix.
- Coverage guard: eval_gates refuses to run below 90% cached-answer coverage.

## 9. Known open items
- SPL, SignNav row, revisited column: need end-to-end / route runs (post-deadline).
- Temporary-sign category: collect ~30 frames if time; else drop the row from Table A.
- Placeholder legibility ℓ; Ajay's calibrated threshold replaces --tau when available.
