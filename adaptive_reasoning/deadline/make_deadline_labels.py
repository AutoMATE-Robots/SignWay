#!/usr/bin/env python3
"""
make_deadline_labels.py -- FREE deadline supervision from odometry.

For every frame of every annotated bag, compute the distance-to-junction label
d(frame) = arc length along the driven odom path from this frame's pose to the
turn-onset pose. The junction anchor is detected from odometry (sustained yaw-rate
crossing in the annotated turn window), NOT annotated by hand -- every bag ever
collected supervises the deadline estimator automatically.

Labels written per frame:
    bag, split, frame_index, t, d_m, junction_ahead, past_junction,
    decision, speed_mps, onset_frame

Semantics:
  junction_ahead = 1  iff a junction lies ahead within --d-max metres
                       (frames further out, and ALL frames of straight bags, get 0:
                        the classifier head learns "no junction I should plan for").
  d_m            = arc-length distance to onset (only meaningful when junction_ahead
                       or past_junction; d_m = -1.0 otherwise).
  past_junction  = 1  for frames at/after onset (excluded from d regression).

Input annotations CSV (same fields as the BAGS table / annotation sheet):
    bag,split,flip_frame,decision,turn_done_frame,stride
    rosbag2-keller-t1,train,159,turn_right,380,2
    rosbag2-keller-t4,train,240,straight,,2
(`turn_done_frame` empty for straight bags; stride is carried through but labels are
per RAW frame -- the estimator trains on every frame.)

Usage (compute node; the reader loads all frames):
    python -m adaptive_reasoning.deadline.make_deadline_labels \
        --annotations $SCRATCH/annotation_sheet.csv \
        --bag-root /users/1/munda057/SignWay/ros2_bags \
        --out $SCRATCH/deadline_labels [--d-max 15] [--bags t1,t2]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_reasoning.config import bootstrap  # noqa: E402

# --------------------------------------------------------------------------- #
# pure geometry (no ROS imports -> unit-testable anywhere)
# --------------------------------------------------------------------------- #
def unwrap(yaws: np.ndarray) -> np.ndarray:
    return np.unwrap(yaws)


def smooth(x: np.ndarray, t: np.ndarray, win_s: float = 0.3) -> np.ndarray:
    """Centered moving average over ~win_s seconds (odom rate inferred from t)."""
    if len(t) < 3:
        return x
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(win_s / max(dt, 1e-4))))
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def detect_turn_onset(
    odom_t: np.ndarray,
    odom_yaw: np.ndarray,
    t_lo: float,
    t_hi: float,
    decision: str,
    rate_thr: float = 0.25,
    sustain_s: float = 0.3,
    fallback_net_deg: float = 15.0,
):
    """Time of turn onset within [t_lo, t_hi].

    Primary: first time the (smoothed) yaw rate exceeds rate_thr rad/s, with the
    correct sign for the decision (left -> +, right -> -), sustained for sustain_s.
    Fallback: first time |net yaw since t_lo| crosses fallback_net_deg.
    Returns (t_onset, method) or (None, reason).
    """
    yaw = unwrap(odom_yaw)
    yaw_s = smooth(yaw, odom_t)
    rate = np.gradient(yaw_s, odom_t, edge_order=1)
    sgn = +1.0 if decision == "turn_left" else -1.0

    m = (odom_t >= t_lo) & (odom_t <= t_hi)
    idx = np.nonzero(m)[0]
    if len(idx) < 3:
        return None, "empty-window"
    dt = float(np.median(np.diff(odom_t[idx])))
    need = max(1, int(round(sustain_s / max(dt, 1e-4))))

    r = sgn * rate[idx]
    above = r > rate_thr
    run = 0
    for j, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= need:
            return float(odom_t[idx[j - need + 1]]), "yaw-rate"

    net = sgn * (yaw_s[idx] - yaw_s[idx[0]])
    cross = np.nonzero(net >= math.radians(fallback_net_deg))[0]
    if len(cross):
        return float(odom_t[idx[cross[0]]]), "net-yaw-fallback"
    return None, "no-onset-found"


def arc_length_to(odom_t, odom_x, odom_y, t_target):
    """Cumulative arc length s(t) along odom; returns (s_of_t interpolator array, s_target)."""
    ds = np.hypot(np.diff(odom_x), np.diff(odom_y))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_target = float(np.interp(t_target, odom_t, s))
    return s, s_target


def speed_at(odom_t, odom_x, odom_y, t_query, win_s: float = 0.4):
    """Mean speed in a small window around t_query (for the deadline's v)."""
    lo, hi = t_query - win_s / 2, t_query + win_s / 2
    m = (odom_t >= lo) & (odom_t <= hi)
    if m.sum() < 2:
        return 0.0
    x, y, t = odom_x[m], odom_y[m], odom_t[m]
    dist = float(np.hypot(np.diff(x), np.diff(y)).sum())
    dur = float(t[-1] - t[0])
    return dist / dur if dur > 1e-4 else 0.0


def label_frames(frame_times, odom, flip_frame, decision, turn_done_frame, d_max):
    """Core labeling for one bag. `odom` = (t, x, y, yaw) arrays. Returns rows + meta."""
    ot, ox, oy, oyaw = odom
    n = len(frame_times)
    rows = []

    if decision == "straight" or decision == "stop":
        for i, t in enumerate(frame_times):
            rows.append(dict(frame_index=i, t=t, d_m=-1.0, junction_ahead=0,
                             past_junction=0, speed_mps=speed_at(ot, ox, oy, t)))
        return rows, dict(onset_frame=-1, onset_method="n/a-straight")

    # turn bag: onset search window = [flip .. turn_done or end]
    t_lo = frame_times[min(max(flip_frame, 0), n - 1)]
    t_hi = frame_times[min(turn_done_frame, n - 1)] if turn_done_frame else frame_times[-1]
    t_onset, method = detect_turn_onset(ot, oyaw, t_lo, t_hi, decision)
    if t_onset is None:
        return None, dict(onset_frame=-1, onset_method=method)

    s, s_onset = arc_length_to(ot, ox, oy, t_onset)
    onset_frame = int(np.searchsorted(frame_times, t_onset))

    for i, t in enumerate(frame_times):
        s_i = float(np.interp(t, ot, s))
        d = s_onset - s_i
        past = int(t >= t_onset)
        ahead = int((not past) and (0.0 < d <= d_max))
        rows.append(dict(frame_index=i, t=t,
                         d_m=round(d, 3) if (ahead or past) else -1.0,
                         junction_ahead=ahead, past_junction=past,
                         speed_mps=round(speed_at(ot, ox, oy, t), 3)))
    return rows, dict(onset_frame=onset_frame, onset_method=method)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def read_annotations(path: Path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            td = r.get("turn_done_frame", "").strip()
            out.append(dict(
                bag=r["bag"].strip(), split=r.get("split", "train").strip(),
                flip_frame=int(r["flip_frame"]), decision=r["decision"].strip(),
                turn_done_frame=int(td) if td and td != "-" else None,
                stride=int(r.get("stride", 1) or 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="CSV: bag,split,flip_frame,decision,turn_done_frame,stride")
    ap.add_argument("--bag-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--bags", default=None, help="comma list to restrict, e.g. t1,t2 (substring match)")
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    args = ap.parse_args()

    bootstrap()
    from bag_to_episode import read_bag  # heavy import (rosbags) only in CLI path

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    anns = read_annotations(Path(args.annotations))
    if args.bags:
        keys = [k.strip() for k in args.bags.split(",")]
        anns = [a for a in anns if any(k in a["bag"] for k in keys)]

    combined_path = out_dir / "deadline_labels_all.csv"
    fields = ["bag", "split", "decision", "frame_index", "t", "d_m",
              "junction_ahead", "past_junction", "speed_mps", "onset_frame"]
    n_ok = n_fail = 0
    with open(combined_path, "w", newline="") as fall:
        wall = csv.DictWriter(fall, fieldnames=fields)
        wall.writeheader()
        for a in anns:
            bag_path = Path(args.bag_root) / a["bag"]
            try:
                frames, odom = read_bag(bag_path, args.image_topic,
                                        args.odom_topic, args.ros_distro)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL read] {a['bag']}: {type(e).__name__}: {e}")
                n_fail += 1
                continue
            ft = np.array([t for t, _ in frames], dtype=float)
            ot = np.array([o[0] for o in odom]); ox = np.array([o[1] for o in odom])
            oy = np.array([o[2] for o in odom]); oyw = np.array([o[3] for o in odom])

            rows, meta = label_frames(ft, (ot, ox, oy, oyw), a["flip_frame"],
                                      a["decision"], a["turn_done_frame"], args.d_max)
            if rows is None:
                print(f"[FAIL onset] {a['bag']}: {meta['onset_method']} "
                      f"(window flip={a['flip_frame']}..{a['turn_done_frame']}) -- INSPECT")
                n_fail += 1
                continue
            for r in rows:
                r.update(bag=a["bag"], split=a["split"], decision=a["decision"],
                         onset_frame=meta["onset_frame"])
                wall.writerow(r)
            na = sum(r["junction_ahead"] for r in rows)
            print(f"[ok] {a['bag']:<28} onset={meta['onset_frame']:>4} "
                  f"({meta['onset_method']}) frames={len(rows)} ahead={na}")
            n_ok += 1
    print(f"\n{n_ok} bags labeled, {n_fail} failed -> {combined_path}")
    print("Review any [FAIL onset] bags before training: noisy odom or wrong window.")


if __name__ == "__main__":
    main()
