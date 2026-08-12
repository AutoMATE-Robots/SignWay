import sys, os
sys.argv.append("--signway")
import numpy as np
from pathlib import Path
from bag_to_episode import read_bag, build_steps
from eval_openloop import load_policy, predict, _attach_gt_trajectory

pol = load_policy(os.environ["CKPT"])
BAGS = [("rosbag2-keller-e1", 190, "turn_right", 522),
        ("rosbag2-keller-e2", 190, "turn_left", 443),
        ("rosbag2-keller-e3", 0, "straight", None)]
root = "/users/1/munda057/SignWay/ros2_bags"
for name, flip, dec, tdone in BAGS:
    frames, odom = read_bag(Path(f"{root}/{name}"), "/c1/image_raw", "/odom", "humble")
    steps = build_steps(frames, odom, flip, dec, tdone, 8, 2)
    _attach_gt_trajectory(steps, 8)
    print(f"\n===== {name} ({dec}) =====")
    print("frame | prompt      | PRED wp8 (fwd,left) | ACTUAL wp8 (fwd,left)")
    for i in range(0, len(steps), 12):
        s = steps[i]
        p = predict(pol, s["image"], s["prompt"], 8)
        g = s["trajectory"]
        print(f"{s['frame_index']:5d} | {s['prompt']:11s} | "
              f"({p[-1][0]:+.3f},{p[-1][1]:+.3f}) | ({g[-1][0]:+.3f},{g[-1][1]:+.3f})")
