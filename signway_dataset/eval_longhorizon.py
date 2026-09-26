#!/usr/bin/env python3
"""
eval_longhorizon.py -- evaluate a MULTI-JUNCTION bag (e.g. a square loop).

e1/e2/e3 each contain one junction, so they cannot test what the paper claims:
that the policy holds straight between junctions and turns at each one, over a
long episode. This runs a segment-annotated bag and reports metrics PER JUNCTION
as well as overall, so "does patience degrade over four corners" becomes a
number.

MEMORY: a 33 GB bag is ~4100 frames at 1920x1080 = ~25 GB decoded, which will
OOM a 60 GB node once torch has the 7B model loaded. So this NEVER holds the
episode in RAM:
  pass 1  read timestamps + odom only (no image decode) -> ground-truth actions
  pass 2  stream, decode one frame, predict, discard

Prompt schedule: "straight" everywhere except inside a segment, where it is that
segment's decision. Frame indices are RAW bag indices, matching the annotator.

    python signway_dataset/eval_longhorizon.py \
        --bag /users/1/munda057/SignWay/ros2_bags/rosbag2-square \
        --segments square_segments.json --stride 3 \
        --checkpoint "$CKPT" --out $SCRATCH/eval_square_v10_6000

Writes series.npy in the SAME schema as eval_openloop, so traj_metrics.py and
make_figures.py work on it unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

YUV = {"yuv422", "uyvy", "yuv422_yuy2", "yuyv"}
CH = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}
HORIZON = 8
FLAT_TOL = 0.01          # m: below this the robot is "still going straight"


# --------------------------------------------------------------------------- #
def _accept(msg) -> bool:
    """The SAME accept rule as bag_to_episode.read_bag, so indices align."""
    enc = (getattr(msg, "encoding", "") or "").lower()
    if enc not in YUV and enc not in CH:
        return False
    return len(msg.data) >= msg.width * (2 if enc in YUV else CH[enc])


def read_stamps_and_odom(bag, image_topic, odom_topic, distro):
    """Frame timestamps + odom, WITHOUT decoding any image."""
    from bag_to_episode import _make_typestore, _open_reader

    ts = _make_typestore(distro)
    stamps, odom = [], []
    with _open_reader(Path(bag), ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        itopic = image_topic if image_topic in conns else next(
            (t for t in ("/c1/image_raw", "/image_raw") if t in conns), None)
        if itopic is None:
            raise SystemExit(f"no image topic; bag has {sorted(conns)}")
        if odom_topic not in conns:
            raise SystemExit(f"no {odom_topic}; bag has {sorted(conns)}")
        want = [conns[itopic], conns[odom_topic]]
        for conn, t, raw in reader.messages(connections=want):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == itopic:
                if _accept(msg):
                    stamps.append(t / 1e9)
            else:
                p = msg.pose.pose
                q = p.orientation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y ** 2 + q.z ** 2))
                odom.append((t / 1e9, float(p.position.x), float(p.position.y),
                             float(yaw)))
    return stamps, odom, itopic


def stream_frames(bag, image_topic, distro, stride):
    """Yield (raw_index, rgb) for every stride-th accepted frame. One at a time."""
    from bag_to_episode import _decode_image, _make_typestore, _open_reader

    ts = _make_typestore(distro)
    idx = 0
    with _open_reader(Path(bag), ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        for conn, _t, raw in reader.messages(connections=[conns[image_topic]]):
            msg = reader.deserialize(raw, conn.msgtype)
            if not _accept(msg):
                continue
            if idx % stride == 0:
                rgb = _decode_image(msg)
                if rgb is not None:
                    yield idx, rgb
            idx += 1


def prompt_at(raw_idx, segments):
    for s in segments:
        if s["legible"] <= raw_idx < s["turn_done"]:
            return s["decision"]
    return "straight"


def build_actions(stamps, odom, stride):
    """Ground-truth cumulative waypoints via build_steps, with dummy images so
    nothing large is allocated (build_steps only needs frame TIMES for odom
    interpolation; the image is carried through untouched)."""
    from bag_to_episode import build_steps

    dummy = np.zeros((2, 2, 3), np.uint8)
    frames = [(t, dummy) for t in stamps]
    steps = build_steps(frames, odom, 0, "straight", None, HORIZON, stride)
    return np.stack([s["action"].reshape(-1, 2) for s in steps])   # (N,8,2)


# --------------------------------------------------------------------------- #
def per_junction(frames_raw, pred_lat, actual_lat, segments):
    rows = []
    for k, s in enumerate(segments, 1):
        m = (frames_raw >= s["legible"]) & (frames_raw < s["turn_done"])
        if not m.any():
            rows.append(dict(junction=k, note="no frames"))
            continue
        pl, al = pred_lat[m], actual_lat[m]
        flat = np.abs(al) < FLAT_TOL
        app = np.abs(pl[flat]) if flat.any() else np.array([np.nan])
        pk_a = np.abs(al).max()
        pk_p = np.abs(pl).max()
        sgn = np.nan
        big = np.abs(al) >= 0.01
        if big.any():
            w = np.abs(al[big])
            sgn = float(((np.sign(pl[big]) == np.sign(al[big])) * w).sum() / w.sum())
        rows.append(dict(
            junction=k, decision=s["decision"],
            frames=f'{s["legible"]}-{s["turn_done"]}',
            dir_acc=sgn,
            approach_mean_cm=float(np.nanmean(app)) * 100,
            approach_max_cm=float(np.nanmax(app)) * 100,
            peak_ratio=float(pk_p / pk_a) if pk_a > 0.02 else np.nan,
            lat_mae_cm=float(np.abs(pl - al).mean()) * 100))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--segments", required=True,
                    help="JSON from annotate_segments.py")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--image-topic", default="/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    args = ap.parse_args()

    segs = json.loads(Path(args.segments).read_text())
    segments = segs["segments"] if isinstance(segs, dict) else segs
    segments = sorted(segments, key=lambda s: s["legible"])
    print(f"{len(segments)} junctions: " +
          ", ".join(f'{s["legible"]}-{s["turn_done"]} {s["decision"]}'
                    for s in segments))

    print("[1/3] reading timestamps + odom (no image decode) ...")
    stamps, odom, itopic = read_stamps_and_odom(
        args.bag, args.image_topic, args.odom_topic, args.ros_distro)
    print(f"      {len(stamps)} frames, {len(odom)} odom samples, topic={itopic}")
    last = max(s["turn_done"] for s in segments)
    if last >= len(stamps):
        print(f"  !! annotation turn_done {last} >= frame count {len(stamps)} "
              f"-- indices may be offset; check the mp4 conversion")

    print("[2/3] ground-truth actions ...")
    A = build_actions(stamps, odom, args.stride)
    print(f"      {len(A)} steps @ stride {args.stride}")

    from eval_openloop import load_policy, predict
    print(f"[3/3] policy: {args.checkpoint}")
    policy = load_policy(args.checkpoint)

    raws, prompts, P = [], [], []
    t0 = time.time()
    for j, (raw, rgb) in enumerate(stream_frames(
            args.bag, itopic, args.ros_distro, args.stride)):
        if j >= len(A):
            break
        pr = prompt_at(raw, segments)
        P.append(np.asarray(predict(policy, rgb, pr, HORIZON),
                            dtype=np.float32).reshape(-1, 2))
        raws.append(raw)
        prompts.append(pr)
        if (j + 1) % 100 == 0:
            el = time.time() - t0
            print(f"      {j+1}/{len(A)} steps  ({el:.0f}s, "
                  f"{el/(j+1)*1000:.0f} ms/step)", flush=True)
    P = np.stack(P)
    A = A[:len(P)]
    raws = np.asarray(raws, float)
    prompts = np.asarray(prompts)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "series.npy", dict(
        frames=raws, prompts=prompts,
        pred_lat=P[:, -1, 1], actual_lat=A[:, -1, 1],
        pred_fwd=P[:, 0, 0], actual_fwd=A[:, 0, 0],
        pred_traj=P, actual_traj=A), allow_pickle=True)

    rows = per_junction(raws, P[:, -1, 1], A[:, -1, 1], segments)
    md = ["# Long-horizon eval", "",
          f"bag `{Path(args.bag).name}` · ckpt `{Path(args.checkpoint).name}` · "
          f"{len(P)} steps · {len(segments)} junctions", "",
          "| junction | decision | frames | dir acc | approach mean (cm) | "
          "approach max (cm) | peak ratio | lat MAE (cm) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "note" in r:
            md.append(f'| {r["junction"]} | — | — | — | — | — | — | — |')
            continue
        md.append(f'| {r["junction"]} | {r["decision"]} | {r["frames"]} | '
                  f'{r["dir_acc"]:.3f} | {r["approach_mean_cm"]:.2f} | '
                  f'{r["approach_max_cm"]:.2f} | {r["peak_ratio"]:.3f} | '
                  f'{r["lat_mae_cm"]:.2f} |')
    turn = np.char.startswith(prompts, "turn")
    rest = np.abs(P[~turn, -1, 1]).mean() * 100 if (~turn).any() else float("nan")
    md += ["",
           f"**Between junctions** (no turn prompt, {int((~turn).sum())} steps): "
           f"mean |lateral| = {rest:.2f} cm — false-turn tendency over the "
           f"straight corridors.",
           "",
           "Approach = predicted lateral while the turn prompt stands and the "
           "robot is still straight (patience). Rising values across junctions "
           "1→4 would mean patience degrades over a long episode."]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nseries -> {out/'series.npy'}\nreport -> {out/'report.md'}")


if __name__ == "__main__":
    main()
