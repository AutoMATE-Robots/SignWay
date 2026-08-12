import sys, os
sys.argv.append("--signway")
import numpy as np
from pathlib import Path
from bag_to_episode import read_bag, build_steps
from eval_openloop import load_policy, predict

pol = load_policy(os.environ["CKPT"])
bag = Path("/users/1/munda057/SignWay/ros2_bags/rosbag2-keller-t1")
frames, odom = read_bag(bag, "/c1/image_raw", "/odom", "humble")
steps = build_steps(frames, odom, 159, "turn_right", 380, 8, 2)
cases = [
    ("approach", steps[30]["image"]),
    ("pre-junction", steps[110]["image"]),
    ("mid-turn", steps[160]["image"]),
    ("NOISE", np.random.randint(0, 255, steps[30]["image"].shape, dtype=np.uint8)),
    ("BLACK", np.zeros_like(steps[30]["image"])),
]
for name, im in cases:
    p = predict(pol, im, "turn_right", 8)
    print(f"{name:12s} -> wp8 = ({p[-1][0]:+.4f}, {p[-1][1]:+.4f})")
