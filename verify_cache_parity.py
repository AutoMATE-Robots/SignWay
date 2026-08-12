#!/usr/bin/env python3
r"""
verify_cache_parity.py -- prove the cached pipeline produces EXACTLY what the
current pipeline produces. Run this before caching all 100 bags.

Compares, on one real bag:

  PATH A (current)  read_bag -> build_steps -> _resize(kept frames)
  PATH B (cached)   read_bag -> _resize(all frames) -> npz -> rebuild -> build_steps

Checks:
  1. images  bit-identical  (np.array_equal, not "close")
  2. actions bit-identical  (exact float equality, then max abs diff as a fallback report)
  3. prompts identical
  4. frame indices identical
  5. interp_odom round-trips exactly at its own nodes

Any FAIL means do not cache -- tell Claude what failed.

RUN (MSI, oft env, compute node):
    python verify_cache_parity.py --bag /users/1/munda057/SignWay/ros2_bags/rosbag2-keller-t25 \
                                  --flip 145 --decision turn_left --turn-done 407 --stride 3
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

for p in (os.path.expanduser("~/SignWay/tools"),
          os.path.expanduser("~/SignWay/signway_dataset")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bag_to_episode import build_steps, interp_odom, read_bag        # noqa: E402
from tfds_builder import (HORIZON, IMAGE_TOPIC, ODOM_TOPIC,          # noqa: E402
                          ROS_DISTRO, _resize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--flip", type=int, required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--turn-done", type=int, default=None)
    ap.add_argument("--stride", type=int, required=True)
    a = ap.parse_args()
    tdone = a.turn_done

    print(f"reading {Path(a.bag).name} ...")
    frames, odom = read_bag(Path(a.bag), IMAGE_TOPIC, ODOM_TOPIC, ROS_DISTRO)
    print(f"  {len(frames)} frames, {len(odom)} odom samples")

    # ---------------- PATH A : current pipeline ----------------
    steps_a = build_steps(frames, odom, a.flip, a.decision, tdone, HORIZON, a.stride)
    imgs_a = [_resize(s["image"]) for s in steps_a]        # resize AFTER selection
    acts_a = np.stack([s["action"] for s in steps_a])
    print(f"PATH A: {len(steps_a)} steps")

    # ---------------- PATH B : cached pipeline ----------------
    times = np.asarray([t for t, _ in frames], dtype=np.float64)
    imgs_all = np.stack([_resize(rgb) for _, rgb in frames]).astype(np.uint8)
    poses = np.asarray([interp_odom(odom, t) for t in times], dtype=np.float64)

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "c.npz"
        np.savez_compressed(cache, times=times, images=imgs_all, poses=poses)
        d = np.load(cache)
        t2, i2, p2 = d["times"], d["images"], d["poses"]

    frames_b = list(zip(t2, i2))
    odom_b = [(float(t), float(p[0]), float(p[1]), float(p[2]))
              for t, p in zip(t2, p2)]
    steps_b = build_steps(frames_b, odom_b, a.flip, a.decision, tdone, HORIZON, a.stride)
    imgs_b = [s["image"] for s in steps_b]                 # already resized
    acts_b = np.stack([s["action"] for s in steps_b])
    print(f"PATH B: {len(steps_b)} steps")

    # ---------------- compare ----------------
    fails = []

    if len(steps_a) != len(steps_b):
        fails.append(f"step count {len(steps_a)} vs {len(steps_b)}")
    else:
        n_img_bad = sum(0 if np.array_equal(x, y) else 1
                        for x, y in zip(imgs_a, imgs_b))
        print(f"images   : {len(imgs_a)-n_img_bad}/{len(imgs_a)} bit-identical")
        if n_img_bad:
            fails.append(f"{n_img_bad} images differ")

        exact = np.array_equal(acts_a, acts_b)
        dmax = float(np.abs(acts_a - acts_b).max())
        print(f"actions  : exact={exact}  max|diff|={dmax:.3e}")
        if not exact:
            # anything under float32 epsilon on ~0.5 m values is numerically identical
            if dmax > 0.0:
                fails.append(f"actions differ by {dmax:.3e}")
            else:
                print("           (unexpected -- should be exact with float64 poses)")

        pa = [s["prompt"] for s in steps_a]
        pb = [s["prompt"] for s in steps_b]
        print(f"prompts  : {'MATCH' if pa == pb else 'DIFFER'}")
        if pa != pb:
            fails.append("prompt schedules differ")

        fa = [s["frame_index"] for s in steps_a]
        fb = [s["frame_index"] for s in steps_b]
        print(f"frame idx: {'MATCH' if fa == fb else 'DIFFER'}")
        if fa != fb:
            fails.append("frame indices differ")

    # interp_odom round-trip at its own nodes
    probe = [int(x) for x in np.linspace(0, len(times) - 1, 25)]
    rt = max(float(np.abs(np.asarray(interp_odom(odom_b, float(times[i])))
                          - poses[i]).max()) for i in probe)
    print(f"interp round-trip: max|diff|={rt:.3e}")
    if rt > 1e-6:
        fails.append(f"interp_odom does not round-trip ({rt:.3e})")

    print()
    if fails:
        print("*** FAIL ***")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("*** PASS -- cached pipeline is identical to the current one ***")
    print("Reminder: the cached builder must NOT call _resize again "
          "(images are already 224x224).")


if __name__ == "__main__":
    main()