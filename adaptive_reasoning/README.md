# `adaptive_reasoning/` — Deadline-Aware, Evidence-Accumulating Sign Reasoning

The adaptive-reasoning system for **SignWay**: a robot navigating mapless,
sign-guided indoor routes should invoke its expensive VLM reasoner
**anticipatorily, on accumulated evidence, against a hard deadline** — never
reactively, and never by stopping to think.

```
                         ┌────────────────────────────────────────────────┐
 camera frame ──────────►│  evidence/   detect sign text, score crops,    │
 (already streaming      │              keep best-K diverse buffer        │
  to the policy server)  └───────┬────────────────────────────────────────┘
                                 │ EvidenceItems
        ┌────────────────────────▼─────────────────────────┐
        │  gate/      NO_SIGN → ARMED → DECIDED            │
        │   fire when  sufficiency ≥ τ                     │
        │          OR  d_q10 / v  ≤  L_p90 + margin        │◄── deadline/  d_q10
        │   speed-for-evidence when close & insufficient   │    (frozen DINOv2-S
        └───────┬──────────────────────────▲───────────────┘     + quantile MLP)
                │ fire (L3)                │ preempt (L1/L2)
        ┌───────▼───────────┐      ┌───────┴───────────┐
        │ reasoning/  VLM   │─────►│ memory/  cache the │
        │ multi-crop prompt │write │ READING, not the   │
        │ structured output │      │ decision           │
        └───────────────────┘      └───────────────────┘
```

## The cost ladder

| level | path | cost |
|---|---|---|
| L0 | VLA acts (decision-vs-timing policy) | 0 reasoning calls |
| L1 | memory hit → cached content → local goal match | ~10 ms, 0 calls |
| L2 | memory hit → cached content → text-only LLM | cheap, no vision |
| L3 | evidence buffer → vision-VLM read (**written to memory**) | expensive |

Every L3 becomes an L1/L2 for every future goal at that sign: the *reading* is
goal-agnostic; matching a goal against cached content is nearly free.

## Models

| component | model | trained? |
|---|---|---|
| policy (Pepper-VLA) | OpenVLA-7B (DINOv2+SigLIP vision, Llama-2-7B), LoRA-finetuned | yes (separate; see repo root) |
| deadline estimator | **frozen** DINOv2-ViT-S/14 backbone + 2-layer MLP (2 heads: p(junction), d quantiles q10/q50/q90) | head only (~0.5 M params) |
| evidence scoring | docTR (det+reco) or PaddleOCR — off the shelf | no |
| memory embeddings | SigLIP/CLIP ViT-B crops — off the shelf | no |
| context objects | open-vocab detector (YOLO-World / OWLv2), fire-time only | no |
| reasoner (L3) | **Gemini (flash tier for the curve, pro tier atop the M-ladder)** via its OpenAI-compat endpoint; local Qwen2.5-VL (vLLM) as the no-API fallback | no |

Design rule across the project: **every trained component is a small head on a
frozen backbone** (the vision-collapse lesson: protect pretrained features).

## Quickstart (offline pipeline — no robot required)

```bash
# 0. environment (once): into the oft conda env
pip install --break-system-packages -r adaptive_reasoning/requirements.txt
export TORCH_HOME=$SCRATCH/torch_hub        # DINOv2 hub cache off home quota
export GEMINI_API_KEY=...                    # L3 default reasoner

# annotations: regenerate the CSV from the BAGS single-source-of-truth any time
python -m adaptive_reasoning.deadline.bags_to_csv \
    --builder ~/SignWay/signway_dataset/tfds_builder.py \
    --out data/annotations_v8.csv

# 1. FREE deadline labels from odometry (compute node; loads all frames)
python -m adaptive_reasoning.deadline.make_deadline_labels \
    --annotations data/annotations_v8.csv \
    --bag-root /users/1/munda057/SignWay/ros2_bags \
    --out $SCRATCH/deadline_labels
# review any [FAIL onset] bags before proceeding

# 2. sanity-check the evidence score on one bag (should RISE toward the junction)
python -m adaptive_reasoning.evidence.run_scorer \
    --bag .../rosbag2-keller-t1 --backend doctr --out $SCRATCH/score_check_t1 \
    --labels $SCRATCH/deadline_labels/deadline_labels_all.csv

# 3. THE headline figure: accuracy vs invoke distance (dry run first, then real VLM)
python -m adaptive_reasoning.reasoning.offline_curve \
    --labels $SCRATCH/deadline_labels/deadline_labels_all.csv \
    --bag-root .../ros2_bags --out $SCRATCH/offline_curve \
    --vlm dry --backend doctr   # then: --vlm gemini:gemini-2.5-flash
#   local fallback: --vlm openai:Qwen/Qwen2.5-VL-7B-Instruct@http://127.0.0.1:8001/v1

# 4. train the deadline estimator (extract once, train in seconds)
python -m adaptive_reasoning.deadline.train_deadline \
    --labels $SCRATCH/deadline_labels/deadline_labels_all.csv \
    --bag-root .../ros2_bags --out $SCRATCH/deadline_model --extract --train
# metrics.json includes q10_shortfall_p90_m -> sets GateConfig.margin_s

# 5. replay the gate on a frozen eval bag: when would it have fired?
python -m adaptive_reasoning.gate.replay_gate \
    --bag .../rosbag2-keller-e1 \
    --labels $SCRATCH/deadline_labels/deadline_labels_all.csv \
    --out $SCRATCH/gate_replay_e1 --backend doctr --vlm dry:turn_right \
    --deadline-ckpt $SCRATCH/deadline_model/deadline_head.pt

# tests (numpy-only; heavy deps are lazy)
pytest adaptive_reasoning/tests -q
```

## Calibration (all knobs in `config.py::GateConfig`, logged per run)

- `margin_s` ≥ the estimator's `q10_shortfall_p90_m / typical_speed`
  (printed by `train_deadline.py`). The margin is **measured**, not guessed.
- `tau_sufficiency`: v0 = threshold at the knee of the offline curve;
  v1 = logistic regression (score components → P(correct)) trained on
  `offline_curve` results.
- `latency_p90_init_s`: seed with a measured VLM latency; the gate keeps a
  running p90 thereafter (`Gate.report_vlm_latency`).
- The gate plans against **d_q10** (pessimistic quantile): "junction is 10–15 m
  away" ⇒ act as if 10.

## Robot integration (after the offline pipeline validates)

The gate runs **server-side** next to the policy (the frame already arrives
there): `policy_server.py` imports `Gate`, returns gate state + any decision in
the `/predict` response; `pepper_vla_node.py` consumes `gate.decision` and
publishes to its existing `/pepper_vla/decision` topic (re-arm on turn-done
already exists). `slow_factor` multiplies the node's `max_speed` cap —
speed-for-evidence, the continuous alternative to stopping to think.

## Repo hygiene

Generated artifacts (`deadline_labels/`, `features/`, `offline_curve/`,
`gate_replay_*/`, `memory/`) live under `$SCRATCH`, not in git. `GateConfig`
should be serialized alongside every experiment's outputs.

## Related work this design is positioned against

Reactive dual-process gating (change/ambiguity-triggered) and single-snapshot
prompt augmentation exist (IROS, arXiv:2601.21506); uncertainty-triggered
reasoning exists (AdaNav); learned when-to-reason exists (OneTwoVLA). This
package's claims are the **anticipatory deadline** grounded in external sign
events, **temporal evidence accumulation** (best-K over the approach), the
**speed-for-evidence** mechanism, and **reading-level memoization** with
context-bound aliasing defense. See `docs/` for the full positioning.
