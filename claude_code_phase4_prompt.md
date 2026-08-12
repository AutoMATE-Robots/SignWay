# SignWay Cleanup — Phase 4 Execution (approval granted)

Paste into the same Claude Code session (or a new one from `~/SignWay`; CLEANUP_PLAN.md
has the full context).

---

Your CLEANUP_PLAN.md is approved, including your corrections to the original brief.
Your Phase 0 diagnosis (uncommitted restructure, not a --cached accident) is accepted
and your proposed recovery sequence is the one to run. Decisions on the four ASK items:

1. **Isaac era → ATTIC.** `c1_simulator/isaac_*.py`, `usd_floorplan.py`,
   `assets/warehouse.usd`, `isaac_container.sh`, and `signs.py` IF nothing that
   currently passes imports it. Rule for edge cases: **anything imported by a
   currently-passing test stays.** Habitat + fake bridges stay.
2. **Cleanup artifacts:** move `claude_code_cleanup_prompt.md` and `CLEANUP_PLAN.md`
   to the attic as your LAST action. `CLEANUP_REPORT.md` stays in the repo root but
   add `CLEANUP_REPORT.md` to `.gitignore` (local record, not public).
3. **`sim_out_l2/`:** do NOT restore. Let the restructure commit record the deletion;
   git history retains it.
4. **`c*` stack: KEEP, public.** Add a short section to `README.md` (before any
   existing architecture section) titled "Repository layout" explaining the two
   stacks: `adaptive_reasoning/` = the paper's deadline-aware reasoning system;
   `c1..c5/` + `common/` = the orchestration stack and simulator scaffolding it grew
   out of (OmniVLA baseline in `c2_action/`, occupancy safety in `c4_safety/`);
   `signway_dataset/` + `tools/` = data pipeline; `scripts/` = one-off utilities.
   Keep it under 15 lines. Do not rename any directory.

## Standing rules (unchanged)
No `rm -rf`, no `git clean`, no `git reset --hard`, no `git push`, never touch
`ros2_bags/` or anything outside `~/SignWay`. Moves = `git mv` (tracked) or `mv`
(untracked) into `_attic/2026-08/<original-path>/`. If anything unexpected appears,
STOP and ask.

## Execution order (exactly this)

1. `git bundle create ~/signway_backup_2026-08-12.bundle --all` and
   `git bundle verify` it. Do not proceed on verify failure.
2. `git checkout -b cleanup/2026-08`.
3. Write the new `.gitignore` — your draft from CLEANUP_PLAN.md §D, plus:
   `.claude/`, `CLEANUP_REPORT.md`, `.secrets`. THIS COMMITS FIRST, alone:
   `chore: real .gitignore before any staging`.
4. `git rm -r --cached` every tracked `__pycache__` path (the 43 .pyc files).
5. `git add -A` (verify with `git status --short | grep ros2_bags` → MUST be empty;
   if not empty, STOP). One commit:
   `restructure: signway_* packages -> c1..c5/common/tools (content survival verified in CLEANUP_PLAN.md)`.
6. `mkdir -p _attic/2026-08 scripts`. Execute the approved moves:
   - Plan §A: the 12 pipeline files -> `scripts/`, adding a one-line header comment
     to each (`# one-off: <what it did>, used <when>, kept for provenance`).
   - Plan §B: everything listed -> `_attic/2026-08/` preserving paths.
   - ASK #1 as decided above (Isaac era -> attic, test-imported files stay).
7. Commit: `cleanup: scripts/ consolidation + attic experiment debris (see CLEANUP_REPORT.md)`.
8. README "Repository layout" section (ASK #4). Commit separately.
9. **Verification gates — all must pass before the report:**
   - `python -m pytest tests -q` → 65 passed (if the attic broke a test, restore the
     imported file from the attic and re-run; report which).
   - `python -m pytest adaptive_reasoning/tests -q` → 21 passed.
   - `python -m py_compile` every `.py` under KEEP dirs + `scripts/`.
   - `python -c "import sys; sys.path.insert(0,'tools'); import bag_to_episode"` OK.
   - `du -sh ros2_bags` → identical to the 103 G baseline.
   - `git log --oneline -6` shows the expected commit sequence.
10. Write `CLEANUP_REPORT.md`: before/after sizes, every move (old -> new), the
    `.gitignore` diff, verification results, the two-stack README text, and one
    flagged recommendation: `ros2_bags/` should eventually live outside the repo
    directory (flag only).
11. Last action: `claude_code_cleanup_prompt.md` + `CLEANUP_PLAN.md` -> attic.
    Final commit. **Do not push. Do not merge to main.** End by printing:
    the branch name, the bundle path, and
    `git diff --stat main..cleanup/2026-08 | tail -3`.

The human reviews the branch before it merges. Your job ends at the report.
