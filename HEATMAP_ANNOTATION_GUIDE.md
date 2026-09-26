# Heatmap Annotation Guide (read this first — 5 minutes)

## What we are testing
We are checking how well a robot can read signs and pick a direction.
For every picture, we tell the robot a **goal** (a room number or a place name),
and then we check whether it picks the right answer. Your job is to write
down, for each picture, **what the goal is** and **what the right answer is**.

One picture + one goal + one correct answer = one test. The same picture can
be used for several tests with different goals.

## What you produce
One CSV file per folder, saved in `heatmap/`:

| folder in heatmap_images/ | csv file to create        |
|---|---|
| Numeric sign              | `heatmap/numeric_sign.csv`   |
| Temporary sign            | `heatmap/temporary_sign.csv` |
| Unclear sign              | `heatmap/unclear_sign.csv`   |
| Goal Irrevelent           | `heatmap/goal_irrelevant.csv` |

(The csv file name becomes the column label on the figure, so keep them clean.)

Every csv has exactly three columns:

```
image,goal,expected
Numeric sign/IMG_0012.jpg,215,turn_right
Numeric sign/IMG_0012.jpg,203,turn_left
Numeric sign/IMG_0015.jpg,5-182,straight
```

- **image** — the path starting with the folder name, exactly as it is on disk
  (spaces are fine). Copy-paste the filename; do not retype it.
- **goal** — what the robot is trying to reach. Write it the way it is printed
  on signs (see rules below).
- **expected** — the correct answer. Only these five words are allowed:
  `turn_left`  `turn_right`  `straight`  `stop`  `not_applicable`

## How to pick the goal (the important part)

**Room numbers:** write the number exactly as printed, e.g. `215`, `5-182`,
`2-344`, `B50`. If the sign shows a range like `Rooms 209–230`, pick a number
**inside** the range that is not one of the two endpoints (e.g. `215`, not
`209` or `230`).

**Named places:** write the name as printed on the sign, e.g.
`Health Sciences Education Center`, `Elevator`, `Lind Hall`. Don't add words
the sign doesn't have.

**Goal Irrelevant folder — special rule:** here the goal must be something the
sign does **NOT** talk about. Pick a room number that is not in any range on
the sign, or a place name that is not on the sign. The correct answer for
every row in this folder is `not_applicable`. This tests whether the robot
admits "this sign doesn't help me" instead of guessing.

**Unclear sign folder:** the sign is blurry / far / at an angle. Still write the
goal and the true answer (look at a clearer photo of the same sign if you
need to). We *want* the robot to struggle here — that's the test.

**Temporary sign folder:** there is a caution cone, a paper notice, etc. in the
picture. The goal and answer refer to the **real directional sign**, not the
temporary one. The test is whether the robot gets distracted.

## How to pick the expected answer
Find the line on the sign that matches your goal and read its arrow:
- `←` → `turn_left`     - `→` → `turn_right`     - `↑` → `straight`
- `↗` / `↖` (ahead then turn) → `straight`
- goal not on the sign → `not_applicable`

Look at the arrow **on the sign**, not where you think the room actually is.

## Please do
- Reuse pictures with 2–3 different goals when the sign has several lines —
  it's free extra tests, and it directly tests "same sign, different goal".
- Aim for **at least 20 rows per csv**. Fewer than that and the percentages
  on the figure jump around by 5% per picture.
- Keep one header row, no blank rows, no quotes needed unless a goal has a comma.

## Please don't
- Don't invent goals that need outside knowledge ("the dean's office").
- Don't use both endpoints of a range as goals.
- Don't put a goal that IS on the sign into the Goal Irrelevant csv.
- Don't type `left` / `right` / `forward` — only the five allowed words.

## Quick self-check before you send it
- Every `image` path opens.
- Every `expected` is one of the five allowed words.
- Goal Irrelevant csv: every row is `not_applicable`.
- Each csv has ≥ 20 rows.
