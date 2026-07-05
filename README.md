# SignWay

Mapless, reasoning-aware indoor **sign-guided navigation** that orchestrates a VLA (OmniVLA) for
action and a VLM for reasoning. The contribution is the *orchestration* — when to reason, how the
action and reasoning loops synchronize, and selective on-demand reasoning to cut latency — not the
policy itself, which is a swappable backend.

Full design: [`SignWay_Orchestration_Spec.md`](SignWay_Orchestration_Spec.md).

## Design rule

The **logic is separate from the transport.** All the orchestration (state machine, triggers,
blackboard, decision mapping) is pure Python in `signway_core/`, with **no ROS2 and no Habitat
imports**. Sim and the real robot are thin shells that feed the same core. This means the state
machine is written once, unit-tested in plain Python, and reused unchanged everywhere.

## Layout

```
signway_core/       PURE PYTHON — the contribution (no ros2/habitat/model deps)
  types.py            shared data types + enums (the contract)
  interfaces.py       Policy / Safety / Detector / Reasoner (abstract)
  geometry.py         SE(2) helpers (world<->robot, look-ahead pose)
  blackboard.py       shared runtime state
  triggers.py         the trigger predicates (spec §5.1)
  fsm.py              the state machine (spec §5)
  config.py           tunable params (spec §12)
signway_backends/   concrete implementations of the interfaces
  mocks.py            mock policy + safety (tests, Phase 1 smoke)
  policy_omnivla.py   client to the Omni server           (Phase 1)
  safety_occupancy.py wraps occupancy_replan.py           (Phase 1)
  detector_gdino.py   GroundingDINO                        (Phase 2)
  reasoner_vlm.py     the VLM                              (Phase 2)
signway_sim/        Habitat harness (plain-Python loop)
  omni_server.py      OmniVLA inference server (runs in the omnivla env)
  habitat_bridge.py   Habitat -> (rgb, depth, pose); step(action)
  run_sim.py          Phase 1 drive loop; Phase 2 adds the reasoner
signway_tools/      offline viz + logging (odom_replay.py, trip viewer)
assets/signs/       the A4 printed sign benchmark
tests/              test_fsm.py, test_triggers.py — no GPU, run in CI
config/params.yaml  all thresholds, one place
# signway_ros2/     added later for the real robot; thin node wrappers only
```

## Run contexts (the env split)

| Runs where | Env | What |
|---|---|---|
| anywhere (laptop, CI) | any Python | `signway_core` + `tests` — no heavy deps |
| GPU node | `omnivla` (3.10) | `omni_server.py` — loads the model, serves waypoints over a socket |
| GPU node | `habitat` (3.9) | `run_sim.py` — Habitat + `signway_core`, calls the Omni server as a client |

`policy_omnivla.py` in the sim is just a network client to `omni_server.py` — that is how the
3.9 / 3.10 split stays clean without cramming both stacks into one env.

## Run the tests (Phase 1, no GPU)

```bash
pip install pytest numpy pyyaml
cd SignWay
python -m pytest -q
```

## Status

- **Phase 0** — orchestration spec: done.
- **Phase 1** — core scaffold (`types`, `interfaces`, `blackboard`, `geometry`, `triggers`,
  `fsm`) + mocks + tests: **done, passing**. Next: `omni_server` + `policy_omnivla`, then the
  Habitat bridge and the drive loop.
- Phases 2–4: reasoner + stop-and-reason, then async, then the benchmark.
```
