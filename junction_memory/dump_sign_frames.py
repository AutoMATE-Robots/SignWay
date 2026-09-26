#!/usr/bin/env python3
"""
dump_sign_frames.py — look at the sign, then write the goal schedule.

The goal schedule is the MISSION (which destination the robot was chasing
when), not a record of what happened. You choose it retrospectively so it
matches the driven trajectory: one destination whose arrow points the way the
robot turned at A, and one whose arrow points straight through A on the second
pass. To choose, you have to see the plate -- so this decodes ONLY the frames
where the sign is within reading range on each approach (a few dozen frames out
of the whole bag, seconds of work), writes them as PNGs plus a contact sheet,
and prints a ready-to-paste goals YAML with the switch time already computed.

    python dump_sign_frames.py --bag $BAG --odom-csv odom_calibrated.csv

Then open sign_frames/contact_sheet.png, read the plate, and paste the printed
YAML into annotations/square_goals.yaml with the two labels filled in.

WHAT TO CHECK while looking (this decides whether scenario 1 works at all):
does the plate list at least TWO destinations with DIFFERENT arrows -- one
pointing the way you turned, one pointing straight ahead? If yes, one call
amortises across both goals and the second pass needs no VLM call. If the
plate only lists destinations in one direction, the bag can still demonstrate
the geometry (recognition tens of metres out), but the "1 call instead of 2"
claim needs a plate with a real directory.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from junction_memory.replay_ar_memory import (open_bag, decode_image,
                                                  pick_image_connection)
    from junction_memory import odom_memory_probe as P
except ImportError:
    from replay_ar_memory import open_bag, decode_image, pick_image_connection
    import odom_memory_probe as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--odom-csv", required=True,
                    help="CALIBRATED odom (calibrate_yaw.py output)")
    ap.add_argument("--topic", default="/image_raw")
    ap.add_argument("--turn-index", type=int, default=0,
                    help="which detected turn is junction A")
    ap.add_argument("--turn-curv-thresh", type=float, default=0.10)
    ap.add_argument("--node-at-s", type=float, default=None,
                    help="place A manually at this arc length, skipping detection")
    ap.add_argument("--from-m", type=float, default=22.0, help="far edge of the window")
    ap.add_argument("--to-m", type=float, default=1.5, help="near edge")
    ap.add_argument("--every", type=int, default=4, help="keep every Nth frame")
    ap.add_argument("--switch-after-m", type=float, default=8.0,
                    help="goal switches this far past A on the departure")
    ap.add_argument("--flip-at-m", type=float, default=10.0,
                    help="distance at which the text is assumed readable; the "
                         "draft flip_frame. Adjust after looking at the PNGs.")
    ap.add_argument("--building", default="keller")
    ap.add_argument("--floor", default="6")
    ap.add_argument("--goal-left", default="<room in the LEFT range>")
    ap.add_argument("--goal-straight", default="<room in the STRAIGHT range>")
    ap.add_argument("--sign-text", default="<label ^ | label < | label <>")
    ap.add_argument("--decision-first", default="turn_left")
    ap.add_argument("--decision-second", default="straight")
    ap.add_argument("--out", default="sign_frames")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- odom: arc length, node A, and the two approach windows -------------
    od = np.loadtxt(a.odom_csv, delimiter=",", skiprows=1)
    ot, ox, oy, oyaw = od[:, 0], od[:, 1], od[:, 2], od[:, 3]
    s, yaw_u, rate, kappa, hz = P.derive(ot, ox, oy, oyaw)

    if a.node_at_s is not None:
        ai = int(np.argmin(np.abs(s - a.node_at_s)))
    else:
        turns = P.detect_turns(ot, ox, oy, yaw_u, rate, kappa,
                              curv_thresh=a.turn_curv_thresh)
        if not turns:
            raise SystemExit("no turns detected — lower --turn-curv-thresh")
        print(f"[info] {len(turns)} turns detected:")
        for i, e in enumerate(turns):
            mark = "   <-- junction A" if i == a.turn_index else ""
            print(f"       turn {i}: s={s[e.anchor_i]:7.1f} m  "
                  f"({e.x:7.2f}, {e.y:7.2f})  {e.turn_deg:+6.1f} deg{mark}")
        ai = turns[a.turn_index].anchor_i
    ax, ay = ox[ai], oy[ai]
    s_a1 = float(s[ai])

    # second pass: closest approach to A after leaving it by the lockout
    d = np.hypot(ox - ax, oy - ay)
    left = np.where(d[ai:] > P.LOCKOUT_M)[0]
    if len(left) == 0:
        raise SystemExit(f"robot never got {P.LOCKOUT_M} m from A — no revisit")
    after = ai + int(left[0])
    k2 = after + int(np.argmin(d[after:]))
    s_a2 = float(s[k2])
    print(f"\n[info] junction A at ({ax:.2f}, {ay:.2f})")
    print(f"       first arrival  s={s_a1:7.1f} m  t+{ot[ai]-ot[0]:6.1f} s")
    print(f"       revisit        s={s_a2:7.1f} m  t+{ot[k2]-ot[0]:6.1f} s  "
          f"(closest {d[k2]:.2f} m)")

    # goal switch: once clear of A on the departure
    dep = np.where(s[ai:] > s_a1 + a.switch_after_m)[0]
    t_switch = float(ot[ai + int(dep[0])] - ot[0]) if len(dep) else None

    # ---- decode ONLY the frames inside the two windows ---------------------
    reader = open_bag(a.bag)
    conns = pick_image_connection(reader, a.topic)
    kept = {1: [], 2: []}
    seen = {1: [], 2: []}          # every in-window frame, for the CSV row
    arrival_frame = {1: None, 2: None}
    t_a = {1: float(ot[ai] - ot[0]), 2: float(ot[k2] - ot[0])}
    t0 = None
    try:
        for i, (conn, tns, raw) in enumerate(reader.messages(connections=conns)):
            ti = tns * 1e-9
            if t0 is None:
                t0 = ti
            s_now = float(np.interp(ti, ot, s))
            for tag, s_node in ((1, s_a1), (2, s_a2)):
                dist = s_node - s_now
                if not (a.to_m < dist < a.from_m):
                    continue
                seen[tag].append((i, ti - t0, dist))
                if len(kept[tag]) and (i - kept[tag][-1][0]) < a.every:
                    continue
                img = decode_image(reader.deserialize(raw, conn.msgtype),
                                   conn.msgtype)
                kept[tag].append((i, ti - t0, dist, img))
            for tag in (1, 2):       # image frame nearest the arrival at A
                if arrival_frame[tag] is None or \
                        abs((ti - t0) - t_a[tag]) < abs(arrival_frame[tag][1] - t_a[tag]):
                    arrival_frame[tag] = (i, ti - t0)
    finally:
        reader.close()

    import cv2
    for tag in (1, 2):
        for (i, trel, dist, img) in kept[tag]:
            cv2.imwrite(str(out / f"approach{tag}_{dist:05.1f}m_f{i:05d}.png"), img)
        print(f"[info] approach {tag}: {len(kept[tag])} frames "
              f"({a.from_m:.0f} m -> {a.to_m:.0f} m out)")

    # contact sheet: rows = approach, columns = closing distance
    sheets = []
    for tag in (1, 2):
        rows = kept[tag]
        if not rows:
            continue
        cols = min(len(rows), 6)
        pick = [rows[int(round(j * (len(rows) - 1) / max(cols - 1, 1)))]
                for j in range(cols)]
        tiles = []
        for (i, trel, dist, img) in pick:
            t = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))
            cv2.rectangle(t, (0, 0), (t.shape[1] - 1, 22), (20, 20, 20), -1)
            cv2.putText(t, f"appr{tag} f{i} {dist:.1f}m t+{trel:.0f}s",
                        (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(t)
        h = min(t.shape[0] for t in tiles)
        sheets.append(np.hstack([t[:h] for t in tiles]))
    if sheets:
        w = max(sh.shape[1] for sh in sheets)
        sheets = [np.pad(sh, ((0, 0), (0, w - sh.shape[1]), (0, 0)))
                  for sh in sheets]
        cv2.imwrite(str(out / "contact_sheet.png"), np.vstack(sheets))
        print(f"[info] wrote {out/'contact_sheet.png'}  <-- READ THE PLATE HERE")

    # ---- draft ar_extra.csv rows ------------------------------------------
    fps_native = None
    try:
        import yaml
        md = yaml.safe_load((Path(a.bag) / "metadata.yaml").read_text())
        info = md["rosbag2_bagfile_information"]
        dur = info["duration"]["nanoseconds"] / 1e9
        for t in info["topics_with_message_count"]:
            if t["topic_metadata"]["name"] == a.topic:
                fps_native = t["message_count"] / dur
                print(f"\n[info] {a.topic}: {t['message_count']} msgs over "
                      f"{dur:.1f} s -> {fps_native:.1f} fps "
                      f"(stride auto = {max(int(fps_native) // 10, 1)})")
    except Exception as e:
        print(f"[warn] could not read fps from metadata.yaml: {e}")

    def _row(tag, approach_idx, goal, decision):
        sv = seen[tag][0][0] if seen[tag] else 0
        ex = seen[tag][-1][0] if seen[tag] else 0
        flip = min(seen[tag], key=lambda r: abs(r[2] - a.flip_at_m))[0] \
            if seen[tag] else 0
        junc = arrival_frame[tag][0] if arrival_frame[tag] else 0
        return (f'{Path(a.bag).name},{approach_idx},{a.building},{a.floor},'
                f'{goal},directional,room_range,"{a.sign_text}",'
                f'{sv},{flip},{junc},{ex},{decision},1,0,0,0,'
                f'junction memory revisit,NN,{int(round(fps_native or 30))}')

    print("\n" + "=" * 72)
    print("APPEND TO adaptive_reasoning/annotations/ar_extra.csv")
    print("=" * 72)
    print(_row(1, 0, a.goal_left, a.decision_first))
    print(_row(2, 1, a.goal_straight, a.decision_second))
    print("=" * 72)
    print("flip_frame is a DRAFT (frame nearest "
          f"{a.flip_at_m:.0f} m out) — check it against the PNGs: it is the")
    print("frame where the text first becomes readable. sign_exit_frame is the")
    print("last frame with the plate still in view.")

    # ---- the draft schedule -----------------------------------------------
    print("\n" + "=" * 72)
    print("PASTE INTO annotations/square_goals.yaml (fill in the two labels)")
    print("=" * 72)
    print("goals:")
    print(f'  - {{from_s: 0.0, goal: "{a.goal_left}"}}')
    sw = f"{t_switch:.1f}" if t_switch is not None else "<after the turn>"
    print(f'  - {{from_s: {sw}, goal: "{a.goal_straight}"}}')
    print("=" * 72)
    print("These MUST be the same goal strings as the ar_extra.csv rows above,")
    print("so relevance and memory are scored against the same mission. Room")
    print("numbers do not equal printed plate labels, so wire the matcher:")
    print("    from junction_memory.goal_matcher import match")
    print("    rt = MemoryRuntime(..., goal_matcher=match)")
    print("(prefer your gate's own goal-affinity scorer over that fallback).")


if __name__ == "__main__":
    main()