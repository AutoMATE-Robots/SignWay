# Live Occupancy-Aware Trajectory Refinement — Runbook

Implements the flowchart: VLA path -> lookahead extension -> footprint-aware
grid check -> SAFE passthrough / candidate contest -> execute. Refinement runs
IN-PROCESS in `pepper_vla_node.py` (~5 ms, no extra server).

```
policy_server (MSI) --16-dim--> pepper_vla_node --process()--> executor
                                     ^
                                     | /local_occupancy_grid (+ stamp = image time)
jetson_occupancy_client (v2, npz) ---+     /odom (pose ring buffer)
                                     ^
msi_occupancy_server (?fmt=npz) -----+
```

## Files

| file | runs where | tested |
|---|---|---|
| `trajectory_refiner.py` | robot, in-process (pure numpy/cv2) | 14 unit tests pass |
| `tests/test_trajectory_refiner.py` | anywhere | — |
| `jetson_occupancy_client.py` (v2) | Jetson, replaces v1 | py_compile only (needs ROS) |
| `refinement_adapter.py` | imported by pepper_vla_node | py_compile only (needs ROS) |

`trajectory_refiner.py` must sit on the PYTHONPATH of both the client and
pepper_vla_node (it owns the OccupancyGrid packing convention — one
implementation on both ends, same rule as the occupancy math).

## How a cycle works

1. Occupancy client posts the newest camera frame, gets the raw int16 grid
   back (npz), publishes it stamped with the SOURCE IMAGE time.
2. Adapter caches the grid, precomputes the distance transform once, and
   records the odometry pose nearest the grid's stamp.
3. When the policy server returns 16 numbers, `process(action16)`:
   - reshapes to 8 waypoints, densifies, extends along the last-segment
     heading to a FIXED METRIC lookahead (2.0 m; 0.8 m while turning, so the
     straight extrapolation never fights a junction turn);
   - transforms those points into the grid-capture frame using the odometry
     delta (kills the latency + inter-frame-gap position error);
   - reads clearance = distance-transform minus robot radius (the
     "robot-width area" check, done as one subtraction);
   - SAFE (min clearance >= 0.5 m over the whole lookahead) -> raw executes;
   - else contest: raw + ~90 deterministic variants (9 rotations x 5 ramped
     laterals x 2 speed scales), hard-reject anything whose footprint touches
     an obstacle, score obstacle^2 + fidelity + smoothness + tiny slow
     penalty, lowest wins;
   - nothing safe at all -> emit raw scaled hard toward stop, `blocked=True`;
   - grid missing/stale (>1.5 s) or ANY internal error -> raw passes through.
     The robot can never behave worse than it does today.
4. Every cycle appends one JSON line: raw + chosen action, clearances, first
   obstacle distance, unknown fraction, grid age, candidate name, compute ms.
   This log IS the D2 table data (raw-baseline row included, thanks to
   min_clearance_raw being logged even when passthrough happens).

## Deployment order — do not skip shadow

1. **Bench:** run `python -m pytest tests/` on the Jetson (no ROS needed).
2. **Grid check:** start server + client v2; `ros2 topic hz
   /local_occupancy_grid`; open rviz, add the OccupancyGrid — it must render
   in front of the robot, obstacles on the correct side. This single look
   catches any packing/sign mistake.
3. **SHADOW RUNS (first live sessions):** `RefinementAdapter(self,
   shadow=True)`. Robot behavior is bit-identical to today; the log records
   what refinement WOULD have done. Drive the standard corridors + a cart.
   Review the log: does `reason` go `safe` in open corridor, `refined` near
   the cart, never `refined` mid-turn at junctions? Shadow logs double as the
   raw-VLA baseline for the ablation.
4. **Active, cart trials:** `shadow=False`, walking-pace collection speed,
   hand on the kill switch. Compare intervention counts vs the shadow
   sessions.

## Tuning knobs (RefinerConfig)

- `d_safe` 0.50 / `robot_radius` 0.30 — verify Pepper's real half-width.
- `ext_len` 2.0 — the scoring horizon. The 8/16/32-style horizon experiments
  are ext_len sweeps here, no retraining.
- `max_grid_age_s` 1.5 — with CPU occupancy (~seconds/frame) everything will
  be stale -> permanent passthrough. Live refinement effectively REQUIRES the
  GPU occupancy server; shadow mode on CPU still logs `stale_grid` cycles,
  which is itself a latency datapoint.
- `w_fid` up if the layer fights turns; `w_obs` up if corrections are timid.
  Tune on replayed bags, not live.

## Known gaps (deliberate, v1)

- Single-frame grid: no temporal fusion yet (that is Teja's rolling-costmap
  item; when it lands, the adapter's snapshot just gets built from the fused
  grid instead — refiner unchanged).
- Camera frame is assumed == base_link (no TF). If the camera is offset or
  tilted, add the static transform in `compensate()`'s pose_grid or in the
  client before packing.
- `pepper_vla_node.py` integration is a 3-line snippet, untested here since
  that file lives on the robot. The seam is one function call with the same
  16-dim in/out — wire it where the policy response currently reaches the
  executor.
