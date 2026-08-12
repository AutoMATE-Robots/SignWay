# SignWay Cleanup Plan — 2026-08-12

Read-only inventory (Phases 0–2 of `claude_code_cleanup_prompt.md`).
**Nothing has been moved, deleted, staged, or committed.** No branch created yet.
`ros2_bags/` was never entered beyond a `du -sh`.

---

## Phase 0 — Git forensics

### The prompt's diagnosis is wrong, and the difference is dangerous

The prompt says: *"~84 staged deletions (`D`) … the signature of `git rm -r --cached .`"*.

Actual state:

| | claimed | measured |
|---|---|---|
| deletion type | staged (`D` in col 1) | **unstaged worktree deletions** (` D`, col 2) |
| `git diff --cached --diff-filter=D` | 84 | **0** |
| `git ls-files -d` | — | **84** |
| index | emptied | **intact — 92 files tracked at HEAD** |

The index was never touched. The 84 files are tracked at HEAD and **physically gone from
disk**. This is not a `--cached` accident: it is a **package restructure that was never
committed**. `signway_core/` → `common/` + `c5_orchestrator/`, `signway_backends/` →
`c2_action/` + `c4_safety/`, `signway_sim/` → `c1_simulator/`, `signway_tools/` → `tools/`,
`SignWay_Orchestration_Spec.md` → `docs/orchestration_spec.md`. The new paths are the 49
untracked entries.

**Why this matters:** the prompt's proposed recovery is `git add -A`. Under the claimed
diagnosis that is harmless. Under the real one it **commits 84 deletions** in the same
breath as it adds the new tree. That is still the right end state — but only because the
content survives, which had to be verified first (below), not assumed.

### Content-survival audit (every deleted tracked file, by content hash)

| deleted path | survives as | delta |
|---|---|---|
| `SignWay_Orchestration_Spec.md` | `docs/orchestration_spec.md` | byte-identical |
| `signway_backends/_wire.py` | `c2_action/_wire.py` | byte-identical |
| `signway_core/geometry.py` | `common/geometry.py` | byte-identical |
| `signway_tools/odom_replay.py` | `tools/odom_replay.py` | byte-identical |
| `signway_tools/signnav_trip_viewer.html` | `tools/signnav_trip_viewer.html` | byte-identical |
| `signway_core/interfaces.py` | `common/interfaces.py` | 2 lines |
| `signway_core/types.py` | `common/types.py` | 10 lines |
| `signway_core/config.py` | `common/config.py` | 7 lines |
| `signway_core/blackboard.py` | `c5_orchestrator/blackboard.py` | 6 lines |
| `signway_core/fsm.py` | `c5_orchestrator/fsm.py` | 153 lines (grew) |
| `signway_core/triggers.py` | `c5_orchestrator/triggers.py` | 67 lines (grew) |
| `signway_backends/occupancy_replan.py` | `c4_safety/occupancy_replan.py` | 13 lines |
| `signway_backends/safety_occupancy.py` | `c4_safety/safety_occupancy.py` | 35 lines |
| `signway_backends/omni_backend.py` | `c2_action/omni_backend.py` | 11 lines |
| `signway_backends/omni_server.py` | `c2_action/omni_server.py` | 27 lines |
| `signway_backends/policy_omnivla.py` | `c2_action/policy_omnivla.py` | 16 lines |
| `signway_sim/bridge.py` | `c1_simulator/bridge.py` | 2 lines |
| `signway_sim/fake_bridge.py` | `c1_simulator/fake_bridge.py` | 4 lines |
| `signway_sim/habitat_bridge.py` | `c1_simulator/habitat_bridge.py` | 6 lines |
| `signway_sim/run_sim.py` | `c1_simulator/run_sim.py` | 244 lines (grew) |
| `signway_backends/mocks.py` | **split** into `c2_action/mock_policy.py` + `c4_safety/mock_safety.py` | — |
| `sim_out_l2/*` (16) | **no on-disk copy** — generated sim frames, still in git history | — |
| 43 × `__pycache__/*.pyc` | n/a — should never have been tracked | — |

**Verdict: no source is at risk.** Every `.py` has a live descendant. The only
content that exists solely in git history is `sim_out_l2/` (15 PNG frames + `run.jsonl`,
July sim output) — recoverable with `git checkout HEAD -- sim_out_l2` if ever wanted.

### Proposed recovery (NOT executed — needs approval)

1. `git bundle create ~/signway_backup_2026-08-12.bundle --all` + verify (~181 MB).
2. `git checkout -b cleanup/2026-08`.
3. Write the real `.gitignore` **first** (current one is a single line: `.env`) so 16
   `__pycache__/` dirs and the 103 GB `ros2_bags/` can never be staged.
4. `git rm -r --cached '*__pycache__*'` to drop the 43 tracked `.pyc` files.
5. `git add -A` → **one** commit: `restructure: signway_* packages -> c1..c5/common/tools`.

Step 3 before step 5 is not optional. With today's `.gitignore`, `git add -A` in this
directory would attempt to stage `ros2_bags/`.

---

## Phase 1 — Inventory

Repo excluding `ros2_bags/` and `.git/`: **5.7 MB**. `.git`: 181 MB.
`ros2_bags/`: **103 GB** (the prompt says ~300 GB — measured value is 103 G).

### Top-level table

| entry | size | what it is | evidence | verdict |
|---|---|---|---|---|
| `adaptive_reasoning/` | 108K | new core pkg, 20 py, README, own tests | **21/21 pass**; self-contained; only external dep is `tools/bag_to_episode.py` | **KEEP** |
| `signway_dataset/` | 709K | TFDS builder + eval/report/video | `eval_sweep.sbatch:154` calls `signway_dataset/make_eval_report.py` | **KEEP** (2 backups → ATTIC, see below) |
| `tools/` | 162K | bag→episode, video, viz, detector tools | imported by `adaptive_reasoning`; 12 py files | **KEEP** |
| `policy_server.py` | 2.0K | deployment | on KEEP list | **KEEP** |
| `pepper_vla_node.py` | 8.0K | ROS2 deployment node | on KEEP list | **KEEP** |
| `eval_sweep.sbatch` | 6.5K | live eval sweep | mtime 2026-08-10 | **KEEP** |
| `annotations_v8.csv` | 4.5K | labels | mtime 2026-08-12 | **KEEP** |
| `bag_rename_map.csv` | 9.0K | bag rename ledger | mtime 2026-08-08 | **KEEP** |
| `README.md` | 11K | — | modified, uncommitted | **KEEP** |
| `docs/` | 20K | `orchestration_spec.md` | supersedes deleted root spec | **KEEP** |
| `.gitignore` | 12B | only `.env` | — | **KEEP — must be rewritten** |
| `config/` | 1.0K | `params.yaml` | `README.md:209` — "every threshold, one place" | **KEEP** |
| `conftest.py` | 512B | pytest rootdir/pythonpath | legacy suite fails without it | **KEEP** |
| `ros2_bags/` | **103 G** | robot data | — | **DO NOT TOUCH** |
| `c1_simulator/` | 73K | Habitat + Isaac bridges, run_sim | 13 py; tests depend | **SALVAGE-FIRST** |
| `c2_action/` | 22K | **OmniVLA integration** | 7 py; paper baseline | **SALVAGE-FIRST → KEEP** |
| `c3_reasoning/` | 84K | legibility, triggers, VLM, detectors | 9 py; **`tools/` imports it** | **SALVAGE-FIRST → KEEP** |
| `c4_safety/` | 17K | occupancy + replanner | 4 py; paper paragraph | **SALVAGE-FIRST → KEEP** |
| `c5_orchestrator/` | 22K | FSM, blackboard, decision, triggers | 5 py | **SALVAGE-FIRST** |
| `common/` | 11K | types/interfaces/geometry/config | imported by **all** `c*` + all tests | **SALVAGE-FIRST — moves only with `c*`** |
| `tests/` | 36K | 9 files, **65 tests, all passing** | tests the `c*` stack | **SALVAGE-FIRST — moves with `c*` or not at all** |
| `extract_cache.py` … `evaltable.py` (12) | ~50K | live data-pipeline one-offs | mtimes 2026-08-08…08-10 | **SCRIPTS** |
| `make_eval_report.py` (root) | 11K | **byte-identical duplicate** | `diff` clean; sbatch calls the `signway_dataset/` path | **ATTIC** (drop root copy) |
| `probe_out/`, `probe_out2/` | 490K+494K | Isaac probe renders | generated by `c1_simulator/isaac_probe.py --out probe_out` | **ATTIC** |
| `episode_check/` | 99K | 2 files | generated by `tools/bag_to_episode.py --out episode_check` | **ATTIC** |
| `compare/`, `dino_videos/`, `videos/` | 0 | **empty dirs** | 0 files each | **ATTIC** (git ignores empty dirs anyway) |
| `.pytest_cache/`, 16 × `__pycache__/` | — | caches | — | **ATTIC + gitignore** |
| `compare_run.log` | 25K | stale log, 2026-07-24 | unreferenced | **ATTIC** |
| `image.png` | 410K | stray | unreferenced | **ATTIC** |
| `plan.png`, `plan_full_warehouse.png`, `aisle_map.png`, `west_map.png`, `route_map.png` | 366K | July simulator-era maps | **no references in README/docs**; only `usd_floorplan.py` uses `plan.png` as a *default output name* | **ATTIC** |
| `isaac_container.sh` | 4.0K | Singularity/Isaac launcher on MSI | self-documenting; simulator-era | **ASK** — attic or `scripts/`? |
| `assets/` | 1.6M | `warehouse.usd`, one file | referenced by `c1_simulator/isaac_scene.py:17` | **ASK** — simulator-only; likely ATTIC with `c1_simulator` |
| `probe/` | 711K | 2 debug PNGs from bag frames | unreferenced | **ATTIC** |
| `import habitat_sim; …cuda_enabled)` | 512B | **a curl cookie jar** (`# Netscape HTTP Cookie File`), not habitat output | — | **ATTIC** |
| `claude_code_cleanup_prompt.md` | 9.0K | this task's brief | — | **ASK** — probably not public |

### The dependency fact that changes the plan

```
tools/detector_compare.py  ─┐
tools/dino_video.py        ─┼─► c3_reasoning.{backends, detector_dino}
tools/dino_probe.py        ─┘
c3_reasoning/* ─► common/*
```

`tools/` is on the **KEEP** list and imports `c3_reasoning`, which imports `common`.
**`c3_reasoning/` and `common/` cannot go to the attic without breaking KEEP code.**
The `c*` stack is not dead legacy — it is a live dependency of the kept tools.

Meanwhile `adaptive_reasoning/` imports **nothing** from `common/` or any `c*`
(verified by grep) — its only repo dependency is `tools/bag_to_episode.py`. The two
stacks are cleanly separable.

---

## SALVAGE-FIRST summaries

### `c2_action/` — OmniVLA integration ✅ *keep, it is the paper's baseline*

- **`policy_omnivla.py`** (2.2K) — `OmniVLAClient(Policy)`: TCP client so the FSM cannot
  tell the model is remote. Exists to keep the habitat (3.9) and omnivla (3.10) conda envs
  separate. Standalone check: `python -m c2_action.policy_omnivla --ping`.
- **`omni_server.py`** (4.1K) — inference server, loads OmniVLA once on a GPU node, serves
  waypoint chunks over TCP, one request per connection (~0.5 Hz so reconnect is free).
  Runs standalone; has a `--backend mock` mode needing no model.
- **`omni_backend.py`** (9.0K) — the model seam: `MockBackend` (no GPU, forward-biased arc)
  and `OmniVLABackend` (wraps the real `run_omnivla.py`). One `predict_waypoints` interface.
- **`controller.py`** (3.5K) — **not on the prompt's list but the most valuable file here.**
  OmniVLA's real control law, ported from SignNav's `run_omnivla.py`. Its docstring records
  a solved bug: earlier code used only `[dx, dy]` from the model's `[dx, dy, hx, hy]`,
  invented a heading, and spiralled the robot. Constants are the reference implementation's.
  **Rebuilding this in September means rediscovering that bug.**
- **`_wire.py`** (762B) — length-prefixed pickle framing, stdlib only, protocol 4 for 3.9/3.10.
- **Overlap with `adaptive_reasoning/`: none.** Nothing there does VLA action.

### `c3_reasoning/` — overlap is partial, not total

| file | `adaptive_reasoning/` counterpart | real overlap |
|---|---|---|
| `readability.py` (5.9K) — 3 cheap features (text height px, Laplacian blur, detector conf); docstring argues *against* the earlier 3×-Qwen self-consistency gate | `evidence/scorer.py` — `ocr_conf · √area · sharpness` + best-K buffer | **Same idea, different formulation.** New one scores+buffers evidence over time; old one is a per-frame boolean. New supersedes it, but the old docstring is the *written rationale* for the design and reads like paper prose. Keep as reference. |
| `triggers.py` (4.8K) | `gate/gate.py` — NO_SIGN→ARMED→DECIDED with deadline | **Superseded.** The new gate is anticipatory (fires on `d_q10/v ≤ L_p90 + margin`); the old is reactive predicates. |
| `reasoner_vlm.py` (12.6K) — `GeminiReasoner`, chain-of-thought → `Decision` | `reasoning/vlm_client.py` — Anthropic / OpenAI-compat / DryRun, returns a **goal-agnostic reading** | **Overlaps but does not supersede.** Old returns a *decision*; new deliberately returns a *reading* so memory can amortize it. The old prompt engineering (which arrow governs which line, room-range reasoning) is not reproduced in the new one. |
| `detector.py` (9.6K) — `TextSignDetector`, Paddle/stub text backends, depth→distance, bearing from intrinsics | none | **No counterpart. Keep.** |
| `detector_dino.py` (10.4K) — Grounding DINO open-vocab | none (README lists YOLO-World/OWLv2 as *future* context-object detectors) | **No counterpart. Keep — this is the memory context-object detector.** |
| `detector_omdet.py` (6.3K) — OmDet-Turbo, same protocol | none | **Keep.** It exists to A/B the DINO query-dilution hypothesis — that comparison is a paper result. |
| `backends.py` (2.1K) — `build_backend(name)` registry | none | **Keep**, `tools/detector_compare.py` imports it. |

**Duplicate found:** `c3_reasoning/triggers.py` and `c5_orchestrator/triggers.py` are the
same module, diverged only in comments (13 lines). The tests import the **`c5_orchestrator`**
one. `c3_reasoning/triggers.py` is the stale copy → propose ATTIC.

### `c4_safety/` — keep

- **`occupancy_replan.py`** (11.4K) — `Grid`, A* on an inflated occupancy grid, `refine()`
  replans OmniVLA's chunk toward its intended bearing when the chunk would collide, plus a
  `render()` debug visualiser. Runs standalone (`__main__` demo, no GPU/depth needed).
  Design line worth quoting in the paper: *"an occupancy grid alone can only veto; the A\*
  planner on top is what actually reroutes."*
- **`safety_occupancy.py`** (3.9K) — `SafetyOccupancy(Safety)` wrapper + `grid_from_depth`.
- **Overlap: none** with `adaptive_reasoning/`. Covered by `tests/test_occupancy.py` (passing).

### `c5_orchestrator/` + `c1_simulator/` — lower relevance, still live

- **`fsm.py`** (11.2K) — DRIVE/HALT/ARRIVED/FAILED + REASON/MANEUVER, priority-ordered
  trigger dispatch. Pure Python, runs identically in sim and on the robot.
- **`blackboard.py`** (2.1K) — the single shared state object; **`decision.py`** (2.8K) —
  maps a VLM `Decision` to a pose subgoal ("the VLM moves the goalpost, the VLA plays the
  game"); **`triggers.py`** (4.1K) — the canonical trigger predicates.
- **`c1_simulator/`** (13 files) — `bridge.py` seam + `fake_bridge` (synthetic corridor, no
  deps), `habitat_bridge` (habitat-sim), `isaac_bridge`/`isaac_probe`/`isaac_scene`/
  `isaac_scout`/`isaac_smoke`/`signs.py`/`usd_floorplan.py` (Isaac Sim era, July),
  `sign_oracle.py`, `run_sim.py` (16.8K, the Phase-1 closed loop).
- The **Isaac** half is the strongest attic candidate in the whole repo: July-only, needs a
  20 GB container in scratch, and the project has since moved to real bags. The **Habitat +
  fake** half still backs `tests/test_sim_loop.py` and `test_bridge.py`.

### `common/` — cannot move

`types.py` (11 dataclasses/enums), `interfaces.py` (Policy/Safety/Detector/Reasoner ABCs),
`geometry.py` (SE(2)), `config.py` (`Params`, every threshold). Imported by **23 files**
across every `c*` module and every legacy test — **and transitively by `tools/`, which is
KEEP.** Verdict: **KEEP**, not salvage.

### `tests/` — 65 passing, not 9

```
$ python -m pytest tests -q          →  65 passed in 3.81s
$ python -m pytest adaptive_reasoning/tests -q  →  21 passed in 0.53s
```

The legacy suite is not rotting — it is green. That is direct evidence the `c*` code still
runs. It must move with the `c*` modules or not at all; `conftest.py` goes with it.

---

## Specific checks

1. **`make_eval_report.py` duplicate** — `diff` is **empty: byte-identical**.
   `eval_sweep.sbatch:154` calls `~/SignWay/signway_dataset/make_eval_report.py`.
   → propose ATTIC for the root copy, keep `signway_dataset/`.
2. **`viz_ar.py` / `demo_hero.py`** — **not found** anywhere under `$HOME` (depth-6 search,
   `ros2_bags` excluded). One live reference survives:
   `signway_dataset/eval_openloop.py:128` mentions "viz_ar.py AR overlays". Nothing created
   or deleted on their account. Likely on another machine or never written.
3. **`signway_dataset/` file count** — 22 files total, 12 top-level entries:
   - 5 real `.py` (the KEEP list) ✅
   - **2 stale backups**: `tfds_builder.py.v6` (17 lines behind) and `tfds_builder.py.prebag`
     (262 lines behind) → **ATTIC**
   - `dataset_inspect/` — 10 generated `episode_*.png` → **gitignore**
   - `__pycache__/` → **gitignore**

---

## Phase 2 — Secret scan

**Clean.** No literal key in the working tree and none in git history.

| hit | file | assessment |
|---|---|---|
| `api_key="EMPTY"` | `adaptive_reasoning/reasoning/vlm_client.py:132` | docstring example, local vLLM sentinel |
| `api_key=key` / `api_key=os.environ.get(...)` | `vlm_client.py:138,178,185,186` | parameter passing |
| `os.environ.get("GEMINI_API_KEY", "")` | `c3_reasoning/reasoner_vlm.py:148` | **correct** — env only |
| `api_key="fake"` ×5 | `tests/test_reasoner_vlm.py` | test literal |
| `export GEMINI_API_KEY=...` | `adaptive_reasoning/README.md:60` | documentation placeholder |

History check — `git log --all -S'AIza…'` and `-S'sk-…|hf_…'` over all refs: **no matches.**
**No history rewrite is needed.** Every key is read from the environment.

⚠️ One gap: `.gitignore` covers `.env` and nothing else. That is the only thing standing
between a local `.env` and a public repo. Keep it at the top of the new file.

---

## Corrections to the brief (facts, not quibbles)

| the prompt says | measured |
|---|---|
| 84 **staged** deletions, `git rm -r --cached .` | 84 **unstaged** deletions; index intact; it was an uncommitted restructure |
| `ros2_bags/` ≈ 300 GB | **103 GB** |
| `signway_dataset/` has 22 files, 5 `.py` | 22 files total, but 12 top-level; 5 `.py` + **2 stale `.py` backups** + 10 generated PNGs |
| junk root file is "a shell-redirect accident" | it is a **curl cookie jar** (`# Netscape HTTP Cookie File`) |
| `tests/` = 9 files (implied small) | 9 files, **65 tests, all passing** |
| `c*` modules are legacy the human said "aren't required" | **`tools/` (KEEP) imports `c3_reasoning`** — they are a live dependency |
| `make_eval_report.py` may differ | byte-identical |

---

## Phase 3 — Proposed moves (⛔ NOTHING MOVED — awaiting explicit approval)

### A. `scripts/` (12 items, ~50K) — live pipeline, not attic
`extract_cache.py`, `validate_caches.py`, `verify_cache_parity.py`,
`patch_builder_for_cache.py`, `patch_flip_zero.py`, `rename_bags.py`, `rename_r2.sh`,
`ingest_r2.sh`, `make_annotation_sheet.py`, `make_mp4.py`, `probe.py`, `evaltable.py`
— each gets a one-line header: what it did, when it was used.

### B. `_attic/2026-08/` (~4.1 MB)

| item | size | why |
|---|---|---|
| `probe_out/`, `probe_out2/`, `episode_check/` | 1.1M | regenerable from `isaac_probe.py` / `bag_to_episode.py` |
| `compare/`, `dino_videos/`, `videos/` | 0 | empty |
| `.pytest_cache/`, 16 × `__pycache__/` | — | caches; also gitignored |
| `image.png`, `compare_run.log` | 435K | stale, unreferenced |
| `plan.png`, `plan_full_warehouse.png`, `aisle_map.png`, `west_map.png`, `route_map.png` | 366K | July simulator maps, zero references in README/docs |
| `probe/` (2 PNGs) | 711K | debug frames |
| `import habitat_sim; …` (cookie jar) | 512B | accident |
| `make_eval_report.py` (root) | 11K | exact duplicate of the `signway_dataset/` copy the sbatch calls |
| `signway_dataset/tfds_builder.py.{v6,prebag}` | 24K | stale backups; git holds history |
| `c3_reasoning/triggers.py` | 4.8K | stale duplicate of `c5_orchestrator/triggers.py` (tests use the latter) |

### C. ASK — no verdict without you

1. **`c1_simulator/isaac_*.py` + `signs.py` + `usd_floorplan.py` + `assets/warehouse.usd`
   (1.6M) + `isaac_container.sh`** — the whole Isaac Sim era. Attic together, or is Isaac
   coming back for a video figure? Habitat/fake bridges stay either way (tests need them).
2. **`claude_code_cleanup_prompt.md`** — attic before publishing?
3. **`sim_out_l2/`** — 16 files that exist only in git history. Restore, or let the
   restructure commit record their deletion?
4. **The `c*` stack for the public repo** — it is green (65 tests) and `tools/` needs
   `c3_reasoning`. Recommend: **keep all of it**, and add a README section saying
   `adaptive_reasoning/` is the paper's system and `c1..c5/` the orchestration stack it
   sits in. That costs 230K and loses nothing.

### D. `.gitignore` to write before any `git add`

```gitignore
.env
ros2_bags/
_attic/
__pycache__/
*.py[cod]
.pytest_cache/
*_out*/
sim_out*/
wandb/
eval_*/
deadline_labels/
features/
memory/
*.bundle
probe_out*/
episode_check/
dino_videos/
videos/
compare/
signway_dataset/dataset_inspect/
*.log
*.py.v[0-9]
*.py.prebag
```

---

## Phase 4/5 — not started

No branch, no moves, no commits, no `.gitignore` edit. `du -sh ros2_bags` = **103 G**
(baseline for the post-cleanup re-check).
