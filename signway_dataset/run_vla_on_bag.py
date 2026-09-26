#!/usr/bin/env python3
"""
run_vla_on_bag.py -- run the policy on chosen frames of a bag and save what a
figure needs: the RGB frame the policy saw, and the trajectory it produced.

Streams the bag (one decoded frame in memory at a time), so a 30 GB recording is
fine.

    python signway_dataset/run_vla_on_bag.py --bag <b> --checkpoint "$CKPT" \
        --prompt turn_left --chain --chain-range 109:478:3 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def to_frame0(wp_i, pose_i, pose_0):
    """Waypoints from frame i expressed in frame 0's body coordinates.

    One prediction is 8 waypoints over 0.8 s -- about 0.7 m at 0.4 m/s, which
    projects to a stub at the bottom of the image. Chaining successive
    predictions into the first frame's coordinates is what produces a ribbon
    that sweeps up the corridor.

        q_0 = R(-yaw_0)(p_i - p_0) + R(yaw_i - yaw_0) q_i
    """
    import math
    x0, y0, th0 = pose_0
    xi, yi, thi = pose_i
    c0, s0 = math.cos(-th0), math.sin(-th0)
    dx, dy = xi - x0, yi - y0
    tx, ty = c0 * dx - s0 * dy, s0 * dx + c0 * dy
    d = thi - th0
    c, s_ = math.cos(d), math.sin(d)
    return [(tx + c * f - s_ * l, ty + s_ * f + c * l) for f, l in wp_i]


YUV = {"yuv422", "uyvy", "yuv422_yuy2", "yuyv"}
CH = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}
HORIZON = 8


def _accept(msg) -> bool:
    enc = (getattr(msg, "encoding", "") or "").lower()
    if enc not in YUV and enc not in CH:
        return False
    return len(msg.data) >= msg.width * (2 if enc in YUV else CH[enc])


def stream(bag, image_topic, distro, wanted):
    from bag_to_episode import _decode_image, _make_typestore, _open_reader

    ts = _make_typestore(distro)
    want = set(wanted)
    idx = 0
    with _open_reader(Path(bag), ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        topic = image_topic if image_topic in conns else next(
            (t for t in ("/c1/image_raw", "/image_raw") if t in conns), None)
        if topic is None:
            raise SystemExit(f"no image topic; bag has {sorted(conns)}")
        print(f"[bag] topic={topic}")
        for conn, _t, raw in reader.messages(connections=[conns[topic]]):
            msg = reader.deserialize(raw, conn.msgtype)
            if not _accept(msg):
                continue
            if idx in want:
                rgb = _decode_image(msg)
                if rgb is not None:
                    yield idx, rgb
                want.discard(idx)
                if not want:
                    return
            idx += 1


def read_stamps_odom(bag, image_topic, odom_topic, distro):
    """Frame timestamps + odom, without decoding any image."""
    from bag_to_episode import _make_typestore, _open_reader

    ts = _make_typestore(distro)
    stamps, odom = [], []
    with _open_reader(Path(bag), ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        itopic = image_topic if image_topic in conns else next(
            (t for t in ("/c1/image_raw", "/image_raw") if t in conns), None)
        for conn, t, raw in reader.messages(
                connections=[conns[itopic], conns[odom_topic]]):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == itopic:
                if _accept(msg):
                    stamps.append(t / 1e9)
            else:
                p = msg.pose.pose
                q = p.orientation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y ** 2 + q.z ** 2))
                odom.append((t / 1e9, float(p.position.x),
                             float(p.position.y), float(yaw)))
    return stamps, odom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="turn_left",
                    choices=["straight", "turn_left", "turn_right", "stop"])
    ap.add_argument("--frames", default="",
                    help="comma list of RAW frame indices, e.g. 1700,1850,2000")
    ap.add_argument("--image-topic", default="/image_raw")
    ap.add_argument("--ros-distro", default="humble")
    ap.add_argument("--chain", action="store_true",
                    help="chain predictions into the FIRST frame's coordinates")
    ap.add_argument("--chain-range", default=None,
                    help="START:END:STEP in raw frames, e.g. 109:478:3")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--also-straight", action="store_true",
                    help="also predict with prompt='straight' at each frame")
    args = ap.parse_args()

    if args.chain_range:
        a, b, st = (int(v) for v in args.chain_range.split(":"))
        frames = list(range(a, b + 1, st))
    else:
        frames = sorted(int(x) for x in args.frames.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import cv2
    from eval_openloop import load_policy, predict

    poses = None
    if args.chain:
        from bag_to_episode import interp_odom
        stamps, odom = read_stamps_odom(args.bag, args.image_topic,
                                        args.odom_topic, args.ros_distro)
        print(f"[odom] {len(odom)} samples for {len(stamps)} frames")
        poses = {}
        for f in frames:
            if f < len(stamps):
                poses[f] = interp_odom(odom, stamps[f])

    print(f"[policy] loading {args.checkpoint}")
    policy = load_policy(args.checkpoint)

    results = {}
    for idx, rgb in stream(args.bag, args.image_topic, args.ros_distro, frames):
        h, w = rgb.shape[:2]
        cv2.imwrite(str(out / f"frame_{idx:05d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        rec = {}
        prompts = [args.prompt] + (["straight"] if args.also_straight
                                   and args.prompt != "straight" else [])
        for p in prompts:
            wp = np.asarray(predict(policy, rgb, p, HORIZON),
                            dtype=np.float32).reshape(-1, 2)
            rec[p] = [[round(float(a), 5), round(float(b), 5)] for a, b in wp]
            print(f"  frame {idx}  prompt={p:11s} "
                  f"wp1=({wp[0,0]:+.3f},{wp[0,1]:+.3f}) "
                  f"wp8=({wp[-1,0]:+.3f},{wp[-1,1]:+.3f})")
        entry = dict(prompt=args.prompt, size=[w, h], wp=rec)
        if poses is not None and idx in poses:
            px, py, pyaw = poses[idx]
            entry["odom"] = dict(x=float(px), y=float(py), yaw=float(pyaw))
        results[str(idx)] = entry

    payload = dict(bag=str(args.bag), checkpoint=str(args.checkpoint),
                   frames=results)

    if args.chain and poses:
        base = min(int(k) for k in results if "odom" in results[k])
        p0 = poses[base]
        chained = []
        for k in sorted(results, key=int):
            e = results[k]
            if "odom" not in e:
                continue
            pi = (e["odom"]["x"], e["odom"]["y"], e["odom"]["yaw"])
            w0 = to_frame0(e["wp"][args.prompt][:1], pi, p0)
            chained.append([round(w0[0][0], 5), round(w0[0][1], 5)])
        last = max(results, key=lambda k: int(k))
        if "odom" in results[last]:
            pl = (results[last]["odom"]["x"], results[last]["odom"]["y"],
                  results[last]["odom"]["yaw"])
            chained += [[round(a, 5), round(b, 5)] for a, b in
                        to_frame0(results[last]["wp"][args.prompt], pl, p0)]
        payload["chained"] = dict(base_frame=base, prompt=args.prompt,
                                  wp=chained)
        ext = max(abs(c[0]) for c in chained) if chained else 0
        print(f"\n[chain] {len(chained)} points in frame {base}, "
              f"reaching {ext:.2f} m forward")
        print(f"[chain] overlay with:  --frame {base} --use-chained")

    (out / "traj.json").write_text(json.dumps(payload, indent=1))
    print(f"\n{len(results)} frames -> {out}")
    print("next: overlay_traj.py (runs anywhere with opencv, no GPU)")


if __name__ == "__main__":
    main()
