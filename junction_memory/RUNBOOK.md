# Junction-memory replay — runbook for `rosbag2_2026_08_22-20_29_37`

Everything is standalone (no ROS install): `pip install rosbags numpy opencv-python matplotlib pyyaml`.
Keep these files in one directory next to the four probe scripts — the replay
imports `odom_memory_probe` for the bag reader and its §7.3 typestore fallback.

```
memory_runtime.py      the ONLINE memory (beam + undirected edge match + recall)
ar_gate_shim.py        annotation-driven AR gate (necessity = pending ∧ ¬memory)
replay_ar_memory.py    orchestrator + HUD mp4 renderer
test_memory_runtime.py 12 tests: README cases 1/2/3, parallel corridor, rival,
                       recall bins, yaw parity, gate accounting
test_bag_io.py         2 tests: defless-bag fallback round-trip (README §7.3)
make_synthetic_demo.py end-to-end rehearsal on a generated square bag
```

## Order of operations on the real bag

```bash
BAG=/path/to/rosbag2_2026_08_22-20_29_37

# 1. odom to CSV, once (~2 min for 35.81 GB; image payload never enters python)
python odom_memory_probe.py --bag $BAG --dump-csv odom.csv

# 2. fix the ~20% rotational scale error BEFORE anything else (README §7.1)
python calibrate_yaw.py --csv odom.csv --expected-turn 360
#    -> odom_calibrated.csv  (expect k ≈ 0.83; adjust --expected-turn if the
#       robot did extra manoeuvring beyond the four corners)

# 3. FAST loop — seconds, no images. Iterate here until events look right.
python replay_ar_memory.py --csv odom_calibrated.csv \
    --annotations square_bag.yaml --no-video --out out_fast
cat out_fast/summary.txt          # expect: 1 VLM call, 1 saved, reuse 0.50
less out_fast/events.jsonl        # the append-only observation log (§8)

# 4. the video (reads /c1/image_raw; odom stays on the calibrated CSV)
python replay_ar_memory.py --bag $BAG --csv odom_calibrated.csv \
    --annotations square_bag.yaml --out out_video
# useful while iterating:  --stride 2   --start-s 120 --end-s 260
#                          --max-frames 500
```

Outputs per run: `replay.mp4`, `events.jsonl` (observation log), `graph.json`
(schema v1, README §8), `summary.txt`, `final_map.png` (paper-figure draft),
`stills/` (one PNG at the first occurrence of each major event).

## Annotations (`square_bag.yaml`)

Windows can be keyed by **image frame index** (`arm_frame`/`decide_frame`,
the usual SignWay convention — frame N = Nth message on `/c1/image_raw`) or by
**bag-relative seconds** (`arm_s`/`decide_s`) so the fast loop needs no image
scan. Fill the four TODOs from the bag:

```yaml
bag: rosbag2_2026_08_22-20_29_37
goals:
  - {from_s: 0.0,   goal: goal_A}
  - {from_s: TODO,  goal: goal_E}      # after the first turn at A
signs:
  # first (north-bound) approach to A — the one real VLM read
  - arm_s:    TODO       # OCR/evidence starts accumulating
    decide_s: TODO       # sufficiency reached -> call dispatched
    decision: left
    conf: 0.91
    directory:           # the WHOLE plate (§2.1) — this gates everything.
      goal_A: left       # if the plate doesn't list goal_E, recall correctly
      goal_E: straight   # returns MEMORY_MISS and the test degenerates.
  # second approach — exists only for counterfactual accounting; with memory
  # working, this window is SUPPRESSED and the decide never fires
  - arm_s:    TODO
    decide_s: TODO
    decision: straight
    conf: 0.90
    directory: {goal_A: left, goal_E: straight}
```

Directions accept synonyms (`turn_left`, `forward`, ...). Goals are matched
case-insensitively. Confidences feed `--verify-tau` if you want the
calls-vs-accuracy curve later.

## What the mp4 shows

AR replay continues throughout; the reasoning source switches:

- `LATERAL BEAM lock on n_001 — 22.4 m out` (universal trigger, cases 1–3)
- `corridor e_001 recognised (same direction) => approaching n_001` /
  `(REVERSE traversal) — index drift - => approaching n_001` (confirming
  channel, undirected + perpendicular, §4.2)
- `MEMORY RECALL at n_001: goal_E -> STRAIGHT ... VLM call not needed`
- `AR gate arming point reached but necessity is ZERO (memory)` — or
  `VLM call cancelled before dispatch` if memory resolves mid-window
- `[baseline] 2 m proximity gate fires only now` — the §4.3 comparison
- straight-through visits, crossing rejections, rival suppression, goal
  switches, post-commit verify, legibility prior all get captions too.

## Wiring the real gate later (replaces `ar_gate_shim.py` only)

```python
rt = MemoryRuntime(bag_name=...)
rt.current_goal = goal                      # keep updated on goal switches
# every odom tick:
rt.step(t, x, y, yaw)
# gate necessity term:
necessity = pending and (rt.resolves(goal) is None)
# sufficiency prior (skip futile calls):
rt.legibility_prior()                       # True/False/None
# when a first-visit VLM read completes:
rt.provide_directory(full_plate_rel, goal=goal, decision=dec, conf=c, frame=f)
# standing prompt for the VLA:
hit = rt.resolves(goal)                     # {'action', 'source', 'lead_m', ...}
```

Thresholds sit at the top of `memory_runtime.py`, values straight from
README §10. Re-derive `LATERAL_M` after the parallel-corridor bag.

## Validated here (synthetic, 14 tests + full demo)

- square rehearsal end-to-end **through the real toolchain** (probe →
  calibrate → replay, on a generated .db3 with the +20% yaw error injected):
  k recovered as 0.8333, recall at **22.2 m** lead vs **2.0 m** proximity,
  flip margin 45°, **1 call instead of 2**, straight-through visit splits the
  corridor at A, crossing corridors rejected at ~90°.
- case 2 (reverse): perpendicular match survives the 2 m U-turn offset, tight
  1.5 m tier correctly fails (the cliff), direction inferred from index drift,
  beam recovers after recentering.
- case 3 (undriven corridor): beam-only lock >15 m; stored `left` read facing
  north reprojects to `right` arriving westbound.
- parallel corridor at 3 m rejected everywhere a 10° bearing cone fires;
  rival ⇒ miss, never a wrong turn.
- defless-bag fallback: a bag stripped to the real bag's condition fails
  AnyReader with the same error and is recovered via the db3 `schema` table.

## Open seams (deliberate)

- **Occlusion veto** (§11.1) not implemented: a node through a wall passes the
  beam. `_beam_step` is the hook for the Depth-Anything free-space check.
- **Parallel corridors untested on real data** — record that bag before
  trusting `LATERAL_M = 1.5` in Keller.
- **Cross-session is out of scope** (case 4): memory is within-session; say so
  in the paper rather than leaving it ambiguous.
- The shim *replays* annotated reads; it does not call Gemini. Live wiring is
  the snippet above plus the §2.1 prompt change (full-plate directory), which
  remains order-of-work #1.
