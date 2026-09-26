# Chained-Route Collection Plan — semantic memory experiment

## What the plot needs
One figure: **cumulative VLM calls vs. junctions traversed**, memory on vs. off.
The memory line flattens where the robot re-enters corridors whose signs it has
already interpreted. For that curve to exist, one recording must contain
**several goals in sequence through the same corridors**, so that later goals
pass signs that were read during earlier goals.

A single approach to a single junction — what we have 18 of — cannot show this.

## What to record

**8 routes.** Each route = ONE continuous bag, one building, 3–5 junctions,
2–3 goals pursued back to back without stopping the recording.

Per route:
- Drive to goal A (2–3 junctions, reading signs along the way).
- On arrival, continue **without stopping the bag** and drive to goal B.
- Choose goal B so that **at least half of its junctions reuse corridors already
  traversed for goal A** — this is the whole point. The natural version is an
  out-and-back: A is deep in a wing, B is on the way back or in a wing that
  shares the same spine corridor.
- If the building allows, add goal C the same way.

**Route mix across the 8:**
| # | Shape | Why |
|---|---|---|
| 4 | out-and-back (B reuses A's corridor) | the flattening curve |
| 2 | disjoint goals (B shares no corridor with A) | the null case: memory must not hurt |
| 2 | three goals, A→B→C | shows the effect compounding |

**Buildings:** 3+ different buildings across the 8 routes. Reuse Keller/Tate if
needed, but at least one building not in the 18-approach test set.

**Length:** aim 3–6 minutes per route. Longer is fine; drive at the usual speed,
do not pause at signs.

## Recording specifics (same as before)
- Topics: `/image_raw` (or `/c1/image_raw`) and `/odom`, nothing else.
- Note the camera fps per bag (20 or 30) — it sets the stride.
- One bag per route. Name them `rosbag2-<building>-route<N>` (e.g.
  `rosbag2-tate-route1`).
- Transfer via rclone to R2 (`r2:rosbags/routes-sep/`), not Google Drive.

## Annotation (one CSV row per junction, not per bag)
File `adaptive_reasoning/annotations/ar_routes.csv`:

```
bag,route,goal_index,goal,junction_frame,decision,sign_text,revisit
rosbag2-tate-route1,1,1,B50,1180,turn_left,"S50-S74 | Classroom B50 <",0
rosbag2-tate-route1,1,1,B50,2310,straight,"Elevator ^ | B50 <",0
rosbag2-tate-route1,1,2,2-120,3050,turn_right,"2-101 to 2-140 >",0
rosbag2-tate-route1,1,2,2-120,3820,turn_left,"S50-S74 | Classroom B50 <",1
```

- **goal_index** — 1 for the first goal, 2 for the second, etc. (this is what
  segments the x-axis of the plot).
- **junction_frame** — RAW frame index where the robot commits to the turn.
- **decision** — `turn_left` / `turn_right` / `straight`.
- **sign_text** — what the sign at that junction says (a few words is enough).
- **revisit** — `1` if this junction's sign was ALREADY passed earlier in this
  same bag (that is the row the memory should make free), else `0`.

The `revisit` column is the ground truth for the experiment: it is annotated
from the video, not from our system's own memory, so the plot is not circular.

## How many junctions do we need?
8 routes × 3–5 junctions ≈ **30–40 junctions**, of which ~12–15 should be
revisits. That is enough for a clear curve and for a stated revisit rate.

## What NOT to do
- Don't stop and restart the bag between goals (the memory spans the route).
- Don't pick goals whose signs are in the 18-approach test set if avoidable —
  keeps the memory experiment independent of the trigger results.
- Don't stage or add signage; use the building as it is.
- Don't drive the same route twice for two goals — it must be one continuous run.

## Checks before transfer
- `ls -la <bag>/` shows `metadata.yaml` + a `.db3` of plausible size.
- Play back the mp4 and confirm every annotated junction is visible.
- Confirm each route really does revisit: at least one `revisit=1` row in the
  out-and-back routes, none in the disjoint ones.
