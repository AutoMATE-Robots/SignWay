# SignWay Repo Cleanup — Instructions for Claude Code

Launch Claude Code from `~/SignWay` on the MSI login node, then paste everything
below the line.

---

You are cleaning up the SignWay research repo (`~/SignWay`) for public release
alongside an ICRA 2027 paper (deadline Sept 15). This system is LIVE and
load-bearing. **Safety beats tidiness every time.** Inventory → propose → get
approval → archive. Never blind-delete.

## ABSOLUTE RULES (violating any of these is catastrophic)

1. **NEVER touch `ros2_bags/`** or anything containing `metadata.yaml`, `*.db3`,
   or `*.mcap`. That is ~300 GB of irreplaceable robot data sitting INSIDE the
   repo directory. Do not move it, do not `git add` it, do not recurse into it.
2. **NEVER run** `rm -rf`, `git clean` (any flags), `git reset --hard`,
   `git checkout .`, `git push --force`, or anything touching `/scratch.global`
   or paths outside `~/SignWay`.
3. **Nothing is deleted this session.** "Removing" = `mv` (or `git mv` if
   tracked) into `_attic/2026-08/<original-path>/`, preserving structure. Real
   deletion happens after the paper, by the human.
4. **Do not move anything until the human approves the plan** (Phase 3).
5. When uncertain: **ASK, don't act.**

---

## Phase 0 — Git forensics FIRST (the repo is in an odd state)

`git status --short` reports ~84 staged deletions (`D`), ~48 untracked entries
(`??`), 7 modified (`M`) — and files that plainly exist on disk (`.gitignore`,
`c1_simulator/`, `assets/`) show as UNTRACKED. That is the signature of
`git rm -r --cached .` having been run: the index dropped everything, the files
stayed on disk.

Do this before anything else, and **do not commit until the human has seen your
diagnosis**:

```bash
# 1. full history backup (cheap insurance; .git is ~181 MB)
git bundle create ~/signway_backup_$(date +%F).bundle --all
git bundle verify ~/signway_backup_$(date +%F).bundle

# 2. are the "deleted" files actually still on disk?
git status --short | grep '^D ' | awk '{print $2}' | while read f; do
  [ -e "$f" ] && echo "ON-DISK" || echo "GONE"; done | sort | uniq -c

# 3. what does HEAD think exists?
git log --oneline -8
git ls-files | wc -l
```

Report: index artifact (files on disk → yes) or genuine deletions (files gone →
a different problem)? Propose a recovery — most likely `git add -A` once
`.gitignore` is correct, producing ONE clean "re-track working tree" commit —
but **wait for approval before committing.**

Create branch `cleanup/2026-08` and work only there.

---

## Phase 1 — Inventory with evidence (read-only)

Produce `CLEANUP_PLAN.md`: a table over every top-level entry —
**name · size · what it is · evidence · verdict**.

Evidence means actual checks, never vibes:
- imported anywhere? `grep -rn "import <name>\|from <name>" --include=*.py .`
- referenced by any KEEP-list script, sbatch, or README?
- last commit: `git log -1 --format=%cd --date=short -- <path>`
- generated output? (plots, `*_out*`, caches, `__pycache__`)

### Verdict categories
`KEEP` · `SCRIPTS` (move to `scripts/`) · `SALVAGE-FIRST` (read & report before
any decision) · `ATTIC` · `ASK`

### KEEP — never propose moving these
- `adaptive_reasoning/` — the new core package (20 py files, 21 passing tests)
- `signway_dataset/` — `tfds_builder.py`, `eval_openloop.py`,
  `inspect_dataset.py`, `make_eval_report.py`, `make_trajectory_video.py`
- `tools/` — `bag_to_episode.py`, `bag_to_mp4.py`, `screen_bags.py`,
  `odom_replay.py`, `plot_loss.py`, `viz.py`, and siblings
- `policy_server.py`, `pepper_vla_node.py` (deployment)
- `eval_sweep.sbatch`, `annotations_v8.csv`, `bag_rename_map.csv`
- `README.md`, `docs/`, `.gitignore`

### SALVAGE-FIRST — DO NOT MOVE. Read, summarize, report.
The human previously said these "aren't required," but they contain work
directly relevant to the current paper. For **each** file below report: a 3-line
summary of what it does, whether it runs standalone, and whether it duplicates
or predates something in `adaptive_reasoning/`.

- `c2_action/policy_omnivla.py`, `omni_server.py`, `omni_backend.py`,
  `controller.py` — an **OmniVLA integration**. OmniVLA zero-shot is a named
  BASELINE in the paper plan. Losing this means rebuilding it in September.
- `c3_reasoning/readability.py`, `triggers.py`, `reasoner_vlm.py`,
  `backends.py`, `detector_dino.py`, `detector_omdet.py`, `detector.py` —
  prior implementations of legibility scoring, gating triggers, VLM reasoning,
  and open-vocab detection. `adaptive_reasoning/` reimplements the first three;
  the detectors are needed for memory context-objects. Report overlap
  explicitly, file by file.
- `c4_safety/occupancy_replan.py`, `safety_occupancy.py` — occupancy / safety
  envelope, a planned paper paragraph.
- `c5_orchestrator/fsm.py`, `blackboard.py`, `decision.py`, `triggers.py` and
  `c1_simulator/*` — orchestration FSM and Habitat/Isaac simulator bridges.
  Lower relevance, but summarize before proposing anything.
- `common/` (`interfaces.py`, `types.py`, `geometry.py`, `config.py`) — shared
  by the `c*` modules. Check whether anything in the KEEP list imports it.
- `tests/` (9 files: `test_readability`, `test_triggers`, `test_occupancy`,
  `test_fsm`, `test_reasoner_vlm`, …) + root `conftest.py` — these test the
  `c*` modules. **They move together with those modules or not at all.**
  Run `pytest tests -q` and report how many pass; a passing legacy suite is
  evidence the code still works.

### SCRIPTS — propose moving into a new `scripts/` dir (NOT the attic)
One-off but this is the live data pipeline: `extract_cache.py`,
`validate_caches.py`, `verify_cache_parity.py`, `patch_builder_for_cache.py`,
`patch_flip_zero.py`, `rename_bags.py`, `rename_r2.sh`, `ingest_r2.sh`,
`make_annotation_sheet.py`, `make_mp4.py`, `probe.py`, `evaltable.py`.
Add a one-line header comment to each saying what it was for and when it was used.

### Likely ATTIC (verify each before proposing)
- Generated output dirs: `probe_out/`, `probe_out2/`, `episode_check/`,
  `compare/`, `dino_videos/`, `videos/`, `.pytest_cache/`, `__pycache__/`
- Stale artifacts: `compare_run.log`, `image.png`
- Simulator-era maps (July): `plan.png`, `plan_full_warehouse.png`,
  `aisle_map.png`, `west_map.png`, `route_map.png`, `isaac_container.sh` —
  check `docs/` and `README.md` for references first.
- The junk-named root file
  `import habitat_sim; from habitat_sim._ext.habitat_sim_bindings import ...`
  (a shell-redirect accident — safe, but show it to the human).
- `probe/`, `assets/` (1.6 MB), `config/` — inspect contents first, then propose.

### Specific checks to run
1. **Duplicate:** `diff make_eval_report.py signway_dataset/make_eval_report.py`.
   If identical, propose keeping ONLY the `signway_dataset/` copy (the sbatch
   calls that path). If they differ, show the diff — the newer is the patched one.
2. `viz_ar.py` and `demo_hero.py` are referenced in project notes but absent
   here. Search (`find ~ -name "viz_ar.py" 2>/dev/null`) and report where they
   live. Do not create or delete anything on their account.
3. `signway_dataset/` has 22 files but only 5 `.py` — list the other 17 and say
   whether they are generated outputs that should be gitignored.

---

## Phase 2 — Secret scan (this repo is going PUBLIC)

```bash
grep -rInE "AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|(api|secret)_key\s*=" \
  --include="*.py" --include="*.sh" --include="*.md" --include="*.json" \
  --include="*.yaml" --include="*.ipynb" . | grep -v "_attic/\|ros2_bags/"
```

A Gemini API key (`AIza…`) is in use on this machine. Report every hit. If a
real key sits in a **tracked** file, flag it LOUDLY: git history will also need
scrubbing before publish. **Do not rewrite history yourself.**

---

## Phase 3 — STOP and present

Show `CLEANUP_PLAN.md`. List every proposed move with size and one-line
justification, and separately list the SALVAGE-FIRST summaries. Ask for approval
as a batch or item-by-item. **No moves before an explicit yes.**

---

## Phase 4 — Execute (approved items only)

- `mkdir -p _attic/2026-08 scripts` and move approved items, preserving paths.
- Update `.gitignore` to cover: `_attic/`, `ros2_bags/`, `__pycache__/`,
  `.pytest_cache/`, `*_out*/`, `wandb/`, `eval_*/`, `deadline_labels/`,
  `features/`, `memory/`, `*.bundle`, plus patterns found in Phase 1.
- Commit on `cleanup/2026-08` with a clear message. Do NOT push.

## Phase 5 — Verify and report

- `python -m pytest adaptive_reasoning/tests -q` — must pass (21 tests).
- `python -m py_compile` every `.py` in KEEP + SCRIPTS; report failures.
- Confirm `du -sh ros2_bags` is unchanged from Phase 1.
- Write `CLEANUP_REPORT.md`: before/after sizes (excluding `ros2_bags`), every
  moved item, `.gitignore` diff, secret-scan result, legacy-test status, and a
  proposed final public layout — including a flagged recommendation that
  `ros2_bags/` should eventually live OUTSIDE the repo (flag only; moving
  300 GB is not this session's job).

Work in small steps and narrate what you find.
