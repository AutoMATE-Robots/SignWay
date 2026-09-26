# SignWay — What This Is, In Plain Language

A robot drives through a building. There are signs on the walls ("Cafeteria →", "Staff Only",
"Elevator Closed"). The robot has to read the signs when they matter, ignore them when they
don't, avoid walls, and reach its destination.

That's it. Everything below is a piece of that.

---

## The one idea that explains the whole design

We use **two AI models**, and they are good at different things:

- A **fast model** that decides *where to move next*, ten times a second. It's dumb but quick.
  It can't read. It just goes toward wherever you point it.
- A **slow model** that can *read a sign and think*. It's smart but slow (2–3 seconds per look).

If you ran the slow model constantly, the robot would crawl. If you never ran it, the robot
couldn't read signs. **So the whole trick is: run the slow model only when it's actually
needed.** Most of the time, just drive.

That's the contribution. Not the models — the *decision of when to think*.

**And the way they talk to each other is deliberately tiny:** the slow model never drives the
robot. It only says "the goal is now over there." The fast model is always driving toward
whatever the current goal is. The slow model just moves the goalpost.

---

## Repository layout

```
adaptive_reasoning/   THE reasoning system — the paper: evidence → gate → reasoning → memory
deploy/               policy_server.py, pepper_vla_node.py        robot runtime
jobs/                 eval_sweep.sbatch                            slurm
data/                 annotations_v8.csv, bag_rename_map.csv       metadata, not bags
docs/                 orchestration_spec.md + design docs
signway_dataset/      tfds_builder, eval_openloop, reports
tools/                bag_to_episode, viz, detectors
scripts/              one-off pipeline utilities, headed with provenance
tests/                the legacy suite (65 tests)
config/               params.yaml — every threshold, one place
c1..c5/, common/      legacy orchestration stack — import-coupled to tools/, left in place
_attic/, ros2_bags/   gitignored
```

**`adaptive_reasoning/` is the one to read first.** It is independent of the `c*` stack: the
`c1..c5/` + `common/` tree is the earlier orchestration system this grew out of — the OmniVLA
baseline and its control law in `c2_action/`, the occupancy envelope and A\* replanner in
`c4_safety/`, the FSM in `c5_orchestrator/` — kept because `tools/` still imports it and
`tests/` still covers it.

---

## How they combine

The loop, in plain English. This repeats forever, about twice a second:

1. **Simulator** gives us: a camera picture, a depth picture, and where the robot is.
2. **Safety layer** turns the depth picture into the chess board of what's solid.
3. **Orchestrator** checks its triggers: *Is there a sign I should read? Am I stuck? Did I
   arrive?*
   - If a sign matters → switch to REASON → ask the **reasoning model** → it says "turn
     left" → **the goal moves to the left** → back to DRIVE.
   - Otherwise → stay in DRIVE.
4. **Action model** looks at the picture and the current goal → gives 8 waypoints.
5. **Replanner** checks those waypoints against the chess board → passes them through, or
   routes around a wall.
6. **Simulator** moves the robot along those waypoints.
7. Repeat.

Step 3 is the whole research contribution. Everything else is plumbing that already exists in
the world.

---

## Glossary — every jargon word, defined

| Word | Plain meaning |
|---|---|
| **FSM** | Finite State Machine. The robot is in one named situation at a time (DRIVE, REASON…), with rules for switching. Like a traffic light. |
| **State** | One of those named situations. |
| **Trigger** | A rule that switches states. "Sign in view → REASON." |
| **VLA** | Vision-Language-Action model. Takes a picture, gives movement. Our fast model (OmniVLA). |
| **VLM** | Vision-Language Model. Takes a picture, gives words. Our slow model (the reasoner). |
| **Pose** | Where something is: x, y, and which way it's facing. Three numbers. |
| **Subgoal** | The spot the robot is currently driving toward. Not the final destination — the next waypoint on the way. |
| **Waypoint chunk** | The 8 dots the action model outputs. "Chunk" just means "a batch of them at once." |
| **Occupancy grid** | The chess board of what's solid. Each square: empty or blocked. |
| **A\*** | A standard algorithm for finding the shortest route on a grid. Pronounced "A-star." |
| **Blackboard** | One place where all the current facts live (where am I, what's the goal, what did the reasoner last say) so every component reads the same truth. |
| **Bridge** | The adapter between our code and a simulator. Isaac gets one, the fake test corridor gets one. Same shape, so our code can't tell them apart. |
| **Backend** | A swappable implementation. "The action model" is a slot; OmniVLA is one backend; a fake one is another. Lets us test without the real model. |
| **Headless** | Running with no monitor attached. Everything on a cluster is headless. |
| **EGL / Vulkan** | Two different systems for talking to a graphics card. Habitat uses EGL (broken here). Isaac uses Vulkan (works here). |
| **Mock** | A fake stand-in for a real component, used for testing. Our mock action model outputs a simple curve instead of running a 7B model. |
| **Closed loop** | The robot's own decisions determine what it sees next. (Opposite: replaying a recording, where nothing you do changes the video.) |

---

## What is actually confirmed working, today

| | |
|---|---|
| Real OmniVLA producing sensible waypoints | ✅ verified |
| The full loop driving to a goal (fake corridor, real model) | ✅ **ARRIVED in 14 steps** |
| Occupancy + replanner around obstacles | ✅ tested, but only on fake data |
| Orchestrator: DRIVE / HALT / ARRIVED / FAILED | ✅ 15 tests passing |
| Vulkan works on MSI A40 → Isaac is possible | ✅ confirmed today |
| Isaac installed | ⬜ next |
| Reasoning model wired in | ⬜ the big one |
| Real depth → occupancy | ⬜ after Isaac |

**You already have an end-to-end pipeline that works.** It drives to a goal using the real
action model with safety checking. What it lacks is (a) pretty pictures from a real simulator,
and (b) the ability to read signs. Those are the two remaining components.

---

## Where everything lives

```
signway/
  README.md          ← this file
  common/            ← the shared vocabulary: Pose, Detection, Decision, the interfaces
  c1_simulator/      ← the video game        (isaac_bridge = TO FILL, fake_bridge works now)
  c2_action/         ← the fast model        (OmniVLA — WORKING)
  c3_reasoning/      ← the slow model        (mock works; real VLM = TO FILL)
  c4_safety/         ← occupancy + replanner (works on fake data)
  c5_orchestrator/   ← the FSM: states, triggers, blackboard, decision mapping
  tools/             ← the log viewer
  tests/             ← 26 tests, no GPU needed
  config/params.yaml ← every threshold, one place
  docs/              ← the full orchestration spec
```

Folders are `c1_`..`c5_` rather than `1_`..`5_` because Python can't import a name that starts
with a digit. `common/` holds the words every component uses — it's not a sixth component, it's
the dictionary.

**Two files are the whole remaining job:**
- `c3_reasoning/reasoner_vlm.py` — port the SignNav prompt + VLM call. The contract is in the
  file. Everything else waits on this.
- `c1_simulator/isaac_bridge.py` — four methods. Copy the shape from `fake_bridge.py`.

## Run the tests (no GPU, 3 seconds)

```bash
pip install pytest numpy pyyaml pillow
cd signway
python -m pytest -q          # 26 passed
```

## Run the pipeline right now (no simulator, no model)

```bash
python -m c1_simulator.run_sim --bridge fake --policy mock --goal 5 0 0 \
    --out sim_out --log sim_out/run.jsonl
```

That drives a fake corridor to the goal and writes a log you can open in `tools/`.
