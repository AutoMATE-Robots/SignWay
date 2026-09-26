"""
tfds_builder.py -- package SignWay bags into an RLDS/TFDS dataset for OpenVLA-OFT.

FINAL SPEC:  input = image + prompt ;  output = trajectory (from odometry).
No input goal-pose. The prompt flips from "straight" to the decision at the legible frame
(supplied per bag), so "left"/"right" means "turn at the junction you're approaching", and the
model learns the timing from the image. See bag_to_episode.py for the full rationale.

CACHE-BACKED (Aug 2026): raw bags live on Cloudflare R2; MSI keeps only small .npz caches
(times, images already 224x224, poses). BAGS entries are BAG NAMES, resolved to
$SIGNWAY_CACHE_DIR/<name>.npz. Proven bit-identical to the old read_bag path by
verify_cache_parity.py -- images byte-equal, actions exact, interp round-trip 0.0.

    export SIGNWAY_CACHE_DIR=$SCRATCH/bag_cache
    python -c "from tfds_builder import SignwayDataset; \
               SignwayDataset(data_dir='$SCRATCH/rlds_v7').download_and_prepare()"

Each step:
    observation/image        (224,224,3) uint8  -- forward camera frame (from cache)
    action                   (16,) float32      -- 8 cumulative waypoints (fwd,left), robot frame
    language_instruction     string             -- straight/turn_left/turn_right/stop
    is_first/is_last/is_terminal, discount, reward(=0, unused)
"""
from __future__ import annotations

import csv as _csv
import os as _os
import sys
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

# tfds_builder.py lives in signway_dataset/; bag_to_episode.py lives in tools/. Make it findable.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_ROOT, _ROOT / "tools", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    _HAVE_TFDS = True
except Exception:
    _HAVE_TFDS = False

from bag_to_episode import build_steps

IMAGE_TOPIC = "/c1/image_raw"      # kept for reference; caches record which topic they used
ODOM_TOPIC = "/odom"
ROS_DISTRO = "humble"
HORIZON = 8
IMAGE_SIZE = (224, 224)

CACHE_DIR = Path(_os.environ.get(
    "SIGNWAY_CACHE_DIR",
    _os.path.join(_os.environ.get("SCRATCH", "/scratch.global"), "bag_cache")))

# Held-out anchor. NEVER move these into train -- they are the fixed ruler that makes
# "more data helped" a comparable claim across model versions (v4 -> v5 -> v6 -> v7).
VAL_BAGS = {"rosbag2-keller-e1", "rosbag2-keller-e2", "rosbag2-keller-e3"}

# ------------------------------------------------------------------------------------------
# (split, bag_name, flip_frame, decision, turn_done_frame, stride)
#   flip_frame / turn_done_frame : RAW frame indices at the bag's native fps
#   stride : 2 for the ~20.7fps bags (t1-t8, e1-e3), 3 for the ~31fps bags -> ~10 Hz dataset
#   straight bags use flip_frame 0 (the prompt is "straight" before AND after the flip, so
#   the value is irrelevant -- but it must not be None, or `i < flip_frame` raises TypeError)
# 93 train / 3 val.  Turn frames: left 3795, right 4281 (ratio 0.89).  Straights: 21 bags.
# ------------------------------------------------------------------------------------------
BAGS = [
    ("train", "rosbag2-keller-t1", 159, "turn_right", 380, 2),
    ("train", "rosbag2-keller-t2", 153, "turn_left", 375, 2),
    ("train", "rosbag2-keller-t3", 150, "turn_right", 699, 2),
    ("train", "rosbag2-keller-t4", 240, "straight", None, 2),
    ("train", "rosbag2-keller-t5", 230, "turn_left", 391, 2),
    ("train", "rosbag2-keller-t6", 280, "straight", None, 2),
    ("train", "rosbag2-keller-t7", 111, "turn_left", 480, 2),
    ("train", "rosbag2-keller-t8", 114, "turn_right", 435, 2),
    ("train", "rosbag2-keller-t9", 208, "turn_right", 569, 3),
    ("train", "rosbag2-keller-t10", 180, "straight", None, 3),
    ("train", "rosbag2-keller-t11", 2, "turn_right", 168, 3),
    ("train", "rosbag2-keller-t12", 33, "turn_left", 219, 3),
    ("train", "rosbag2-keller-t13", 0, "turn_left", 333, 3),
    ("train", "rosbag2-keller-t14", 32, "turn_right", 318, 3),
    ("train", "rosbag2-keller-t15", 162, "turn_left", 445, 3),
    ("train", "rosbag2-keller-t16", 160, "turn_right", 468, 3),
    ("train", "rosbag2-keller-t17", 40, "turn_right", 347, 3),
    ("train", "rosbag2-keller-t18", 10, "straight", None, 3),
    ("train", "rosbag2-keller-t19", 110, "turn_left", 442, 3),
    ("train", "rosbag2-keller-t20", 254, "straight", None, 3),
    ("train", "rosbag2-keller-t21", 186, "turn_left", 481, 3),
    ("train", "rosbag2-keller-t22", 295, "straight", None, 3),
    ("train", "rosbag2-keller-t23", 303, "turn_right", 510, 3),
    ("train", "rosbag2-keller-t24", 199, "straight", None, 3),
    ("train", "rosbag2-keller-t25", 145, "turn_left", 407, 3),
    ("train", "rosbag2-keller-t26", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t27", 4, "straight", None, 3),
    ("train", "rosbag2-keller-t28", 220, "turn_left", 421, 3),
    ("train", "rosbag2-keller-t29", 238, "turn_left", 477, 3),
    ("train", "rosbag2-keller-t30", 36, "turn_left", 615, 3),
    ("train", "rosbag2-keller-t31", 0, "turn_right", 427, 3),
    ("train", "rosbag2-keller-t32", 90, "turn_right", 502, 3),
    ("train", "rosbag2-keller-t33", 68, "turn_left", 488, 3),
    ("train", "rosbag2-keller-t34", 120, "turn_right", 476, 3),
    ("train", "rosbag2-keller-t35", 46, "turn_left", 413, 3),
    ("train", "rosbag2-keller-t36", 43, "turn_left", 537, 3),
    ("train", "rosbag2-keller-t37", 0, "turn_left", 253, 3),
    ("train", "rosbag2-keller-t38", 18, "turn_right", 422, 3),
    ("train", "rosbag2-keller-t39", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t40", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t41", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t42", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t43", 213, "turn_left", 496, 3),
    ("train", "rosbag2-keller-t44", 0, "turn_right", 327, 3),
    ("train", "rosbag2-keller-t45", 0, "turn_left", 387, 3),
    ("train", "rosbag2-keller-t46", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t47", 83, "turn_right", 505, 3),
    ("train", "rosbag2-keller-t48", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t49", 78, "turn_left", 524, 3),
    ("train", "rosbag2-keller-t50", 90, "turn_left", 463, 3),
    ("train", "rosbag2-keller-t51", 0, "turn_left", 293, 3),
    ("train", "rosbag2-keller-t52", 0, "turn_right", 362, 3),
    ("train", "rosbag2-keller-t53", 126, "turn_right", 686, 3),
    ("train", "rosbag2-keller-t54", 51, "turn_right", 522, 3),
    ("train", "rosbag2-keller-t55", 57, "turn_right", 485, 3),
    ("train", "rosbag2-keller-t56", 39, "turn_right", 353, 3),
    ("train", "rosbag2-keller-t57", 94, "turn_right", 460, 3),
    ("train", "rosbag2-keller-t58", 68, "turn_right", 453, 3),
    ("train", "rosbag2-keller-t59", 9, "turn_left", 390, 3),
    ("train", "rosbag2-keller-t60", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t61", 0, "turn_left", 371, 3),
    ("train", "rosbag2-keller-t62", 42, "turn_left", 395, 3),
    ("train", "rosbag2-keller-t63", 0, "turn_right", 245, 3),
    ("train", "rosbag2-keller-t64", 80, "turn_right", 482, 3),
    ("train", "rosbag2-keller-t65", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t66", 180, "turn_right", 547, 3),
    ("train", "rosbag2-keller-t67", 0, "turn_right", 377, 3),
    ("train", "rosbag2-keller-t68", 50, "turn_left", 409, 3),
    ("train", "rosbag2-keller-t69", 0, "turn_right", 398, 3),
    ("train", "rosbag2-keller-t70", 0, "turn_right", 381, 3),
    ("train", "rosbag2-keller-t71", 0, "turn_right", 363, 3),
    ("train", "rosbag2-keller-t72", 0, "turn_right", 175, 3),
    ("train", "rosbag2-keller-t73", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t74", 0, "turn_left", 430, 3),
    ("train", "rosbag2-keller-t75", 50, "turn_left", 319, 3),
    ("train", "rosbag2-keller-t76", 172, "turn_left", 481, 3),
    ("train", "rosbag2-keller-t77", 0, "turn_left", 259, 3),
    ("train", "rosbag2-keller-t78", 273, "turn_right", 516, 3),
    ("train", "rosbag2-keller-t79", 20, "turn_left", 331, 3),
    ("train", "rosbag2-keller-t80", 40, "turn_right", 282, 3),
    ("train", "rosbag2-keller-t81", 0, "turn_right", 246, 3),
    ("train", "rosbag2-keller-t82", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t83", 0, "turn_right", 240, 3),
    ("train", "rosbag2-keller-t84", 0, "turn_left", 234, 3),
    ("train", "rosbag2-keller-t85", 0, "turn_left", 194, 3),
    ("train", "rosbag2-keller-t86", 45, "turn_left", 288, 3),
    ("train", "rosbag2-keller-t87", 0, "turn_left", 221, 3),
    ("train", "rosbag2-keller-t88", 0, "turn_right", 164, 3),
    ("train", "rosbag2-keller-t89", 82, "turn_right", 294, 3),
    ("train", "rosbag2-keller-t90", 0, "turn_left", 296, 3),
    ("train", "rosbag2-keller-t91", 0, "turn_right", 283, 3),
    ("train", "rosbag2-keller-t92", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t93", 0, "straight", None, 3),
    
    # ---- Aug 20, Tate/, 0.4 m/s, long approaches (t94-t118) ----
    ("train", "rosbag2-keller-t94", 384, "turn_right", 716, 3),
    ("train", "rosbag2-keller-t95", 401, "turn_left", 701, 3),
    ("train", "rosbag2-keller-t96", 229, "turn_left", 551, 3),
    ("train", "rosbag2-keller-t97", 213, "straight", None, 3),
    ("train", "rosbag2-keller-t98", 379, "straight", None, 3),
    ("train", "rosbag2-keller-t99", 369, "turn_left", 674, 3),
    ("train", "rosbag2-keller-t100", 348, "straight", None, 3),
    ("train", "rosbag2-keller-t101", 309, "turn_right", 600, 3),
    ("train", "rosbag2-keller-t102", 232, "turn_left", 484, 3),
    ("train", "rosbag2-keller-t103", 256, "turn_right", 452, 3),
    ("train", "rosbag2-keller-t104", 307, "turn_right", 625, 3),
    ("train", "rosbag2-keller-t105", 405, "turn_right", 619, 3),
    ("train", "rosbag2-keller-t106", 318, "turn_left", 536, 3),
    ("train", "rosbag2-keller-t107", 300, "turn_left", 466, 3),
    ("train", "rosbag2-keller-t108", 238, "straight", None, 3),
    ("train", "rosbag2-keller-t109", 277, "straight", None, 3),
    ("train", "rosbag2-keller-t110", 204, "turn_right", 419, 3),
    ("train", "rosbag2-keller-t111", 460, "straight", None, 3),
    ("train", "rosbag2-keller-t112", 444, "turn_right", 691, 3),
    ("train", "rosbag2-keller-t113", 451, "turn_right", 743, 3),
    ("train", "rosbag2-keller-t114", 429, "straight", None, 3),
    ("train", "rosbag2-keller-t115", 718, "turn_right", 961, 3),
    ("train", "rosbag2-keller-t116", 710, "turn_left", 1115, 3),
    ("train", "rosbag2-keller-t117", 780, "turn_right", 1292, 3),
    ("train", "rosbag2-keller-t118", 221, "turn_left", 546, 3),

# ---- Aug 21, Tate/Keller, 0.4 m/s (t119-t168) ----
    ("train", "rosbag2-keller-t119", 300, "turn_left", 891, 3),
    ("train", "rosbag2-keller-t120", 0, "turn_right", 365, 3),
    ("train", "rosbag2-keller-t121", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t122", 0, "straight", None, 3),
    ("train", "rosbag2-keller-t123", 88, "turn_right", 588, 3),
    ("train", "rosbag2-keller-t124", 129, "turn_right", 398, 3),
    ("train", "rosbag2-keller-t125", 25, "straight", None, 3),
    ("train", "rosbag2-keller-t126", 76, "turn_right", 365, 3),
    ("train", "rosbag2-keller-t127", 123, "straight", None, 3),
    ("train", "rosbag2-keller-t128", 142, "turn_left", 337, 3),
    ("train", "rosbag2-keller-t129", 96, "straight", None, 3),
    ("train", "rosbag2-keller-t130", 268, "turn_left", 636, 3),
    ("train", "rosbag2-keller-t131", 19, "straight", None, 3),
    ("train", "rosbag2-keller-t132", 257, "turn_right", 650, 3),
    ("train", "rosbag2-keller-t133", 50, "straight", None, 3),
    ("train", "rosbag2-keller-t134", 163, "turn_right", 605, 3),
    ("train", "rosbag2-keller-t135", 184, "turn_right", 515, 3),
    ("train", "rosbag2-keller-t136", 22, "turn_right", 315, 3),
    ("train", "rosbag2-keller-t137", 42, "straight", None, 3),
    ("train", "rosbag2-keller-t138", 171, "turn_right", 670, 3),
    ("train", "rosbag2-keller-t139", 23, "turn_left", 206, 3),
    ("train", "rosbag2-keller-t140", 75, "straight", None, 3),
    ("train", "rosbag2-keller-t141", 123, "turn_right", 570, 3),
    ("train", "rosbag2-keller-t142", 15, "turn_right", 557, 3),
    ("train", "rosbag2-keller-t143", 35, "straight", None, 3),
    ("train", "rosbag2-keller-t144", 358, "turn_right", 695, 3),
    ("train", "rosbag2-keller-t145", 240, "turn_right", 573, 3),
    ("train", "rosbag2-keller-t146", 264, "turn_right", 582, 3),
    ("train", "rosbag2-keller-t147", 104, "turn_left", 644, 3),
    ("train", "rosbag2-keller-t148", 115, "turn_left", 458, 3),
    ("train", "rosbag2-keller-t149", 73, "turn_left", 425, 3),
    ("train", "rosbag2-keller-t150", 99, "turn_left", 411, 3),
    ("train", "rosbag2-keller-t151", 105, "turn_right", 423, 3),
    ("train", "rosbag2-keller-t152", 77, "straight", None, 3),
    ("train", "rosbag2-keller-t153", 180, "turn_right", 571, 3),
    ("train", "rosbag2-keller-t155", 35, "straight", None, 3),
    ("train", "rosbag2-keller-t156", 123, "turn_left", 610, 3),
    ("train", "rosbag2-keller-t157", 167, "turn_right", 503, 3),
    ("train", "rosbag2-keller-t158", 168, "turn_left", 473, 3),
    ("train", "rosbag2-keller-t159", 22, "straight", None, 3),
    ("train", "rosbag2-keller-t160", 69, "turn_right", 489, 3),
    ("train", "rosbag2-keller-t161", 56, "straight", None, 3),
    ("train", "rosbag2-keller-t162", 17, "turn_right", 423, 3),
    ("train", "rosbag2-keller-t163", 19, "straight", None, 3),
    ("train", "rosbag2-keller-t164", 147, "turn_right", 353, 3),
    ("train", "rosbag2-keller-t165", 128, "turn_left", 434, 3),
    ("train", "rosbag2-keller-t166", 23, "straight", None, 3),
    ("train", "rosbag2-keller-t167", 170, "turn_right", 531, 3),
    ("train", "rosbag2-keller-t168", 31, "straight", None, 3),
    
    ("val", "rosbag2-keller-e1", 190, "turn_right", 522, 2),
    ("val", "rosbag2-keller-e2", 190, "turn_left", 443, 2),
    ("val", "rosbag2-keller-e3", 191, "straight", None, 2),
]


def _load_cache(name):
    """Return (frames, odom) shaped exactly as build_steps expects.

    The cache stores poses already interpolated at each frame time, so the odom list we
    hand back has a sample at every frame timestamp; interp_odom then returns those values
    exactly (verified round-trip 0.0). Accepts a bare bag name, a path to a bag dir (only
    the basename is used), or a direct .npz path.
    """
    p = Path(name)
    if p.suffix != ".npz":
        p = CACHE_DIR / f"{Path(name).name}.npz"
    if not p.exists():
        raise FileNotFoundError(f"cache not found: {p}  (is SIGNWAY_CACHE_DIR set?)")
    d = np.load(p)
    times, images, poses = d["times"], d["images"], d["poses"]
    frames = list(zip(times, images))
    odom = [(float(t), float(q[0]), float(q[1]), float(q[2]))
            for t, q in zip(times, poses)]
    return frames, odom


def _bags_from_csv(path):
    """Optional: build BAGS from annotation_sheet.csv instead of the literal above.
    Rows with a blank decision are skipped, so a partially-annotated sheet still builds."""
    out, skipped = [], 0
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            bag = (r.get("bag") or "").strip()
            dec = (r.get("decision") or "").strip()
            if not bag or not dec:
                skipped += 1
                continue
            flip_s = (r.get("flip_frame") or "").strip()
            td_s = (r.get("turn_done_frame") or "").strip()
            flip = int(float(flip_s)) if flip_s else 0
            tdone = int(float(td_s)) if td_s else None
            stride = int(float((r.get("stride") or "1").strip()))
            split = "val" if bag in VAL_BAGS else "train"
            out.append((split, bag, flip, dec, tdone, stride))
    print(f"[builder] {len(out)} annotated bags from {path} ({skipped} unannotated, skipped)")
    return out


_ANN = _os.environ.get("SIGNWAY_ANNOTATIONS", "")
if _ANN and Path(_ANN).exists():
    BAGS = _bags_from_csv(_ANN)


# ---------------------------------------------------------------------------
# SIGNWAY_FLIP_ZERO: relabel TRAIN turn bags with flip_frame=0 so the frames
# before the original flip carry the turn prompt instead of "straight". Same
# images, same actions -- only the prompt changes. Those frames then teach
# "standing turn command, correct action is still straight", which is the
# patience signal the policy is short on. val is left alone.
# ---------------------------------------------------------------------------
if _os.environ.get("SIGNWAY_FLIP_ZERO", "").lower() in ("1", "true", "yes"):
    _before = sum(f for sp, nm, f, dec, td, st in BAGS
                  if sp == "train" and dec != "straight")
    BAGS = [(sp, nm, (0 if (sp == "train" and dec != "straight") else f), dec, td, st)
            for sp, nm, f, dec, td, st in BAGS]
    _n = sum(1 for sp, nm, f, dec, td, st in BAGS if sp == "train" and dec != "straight")
    print(f"[builder] SIGNWAY_FLIP_ZERO: {_n} train turn bags relabelled to flip_frame=0 "
          f"({_before} raw frames moved into the turn-prompt regime)")


def _episode_steps(bag_path: str, flip_frame: int, decision: str, turn_done, stride: int = 1):
    frames, odom = _load_cache(bag_path)
    steps = build_steps(frames, odom, flip_frame, decision, turn_done, HORIZON, stride)
    n = len(steps)
    for j, s in enumerate(steps):
        yield {
            # already 224x224 in the cache -- do NOT resize again
            "observation": {"image": np.asarray(s["image"], dtype=np.uint8)},
            "action": s["action"].astype(np.float32),
            "language_instruction": s["prompt"],
            "is_first": j == 0,
            "is_last": j == n - 1,
            "is_terminal": j == n - 1,
            "discount": np.float32(1.0),
            "reward": np.float32(0.0),
        }


if _HAVE_TFDS:

    class SignwayDataset(tfds.core.GeneratorBasedBuilder):
        """SignWay trajectory VLA dataset (RLDS)."""
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "image+prompt -> trajectory; prompt flips at legible frame."}

        def _info(self):
            return self.dataset_info_from_configs(features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8,
                            doc="forward camera RGB, 224x224"),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(16,), dtype=np.float32,
                        doc="ViNT-style cumulative waypoints: next 8 strided-frame positions "
                            "in the current robot frame, flattened [x1,y1,...,x8,y8] metres; "
                            "trained one-shot (NUM_ACTIONS_CHUNK=1, ACTION_DIM=16)"),
                    "language_instruction": tfds.features.Text(
                        doc="prompt: straight/turn_left/turn_right/stop"),
                    "is_first": np.bool_, "is_last": np.bool_, "is_terminal": np.bool_,
                    "discount": np.float32, "reward": np.float32,
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "bag_path": tfds.features.Text(),
                    "flip_frame": tfds.features.Scalar(dtype=np.int32),
                    "decision": tfds.features.Text(),
                }),
            }))

        def _split_generators(self, dl_manager):
            splits = {}
            for split, path, flip, dec, tdone, stride in BAGS:
                splits.setdefault(split, []).append((path, flip, dec, tdone, stride))
            return {name: self._generate_examples(specs) for name, specs in splits.items()}

        def _generate_examples(self, specs) -> Iterator[Tuple[str, dict]]:
            for path, flip, dec, tdone, stride in specs:
                steps = list(_episode_steps(path, flip, dec, tdone, stride))
                if not steps:
                    print(f"[builder] WARNING: {path} produced 0 steps -- too short? skipping")
                    continue
                yield Path(path).name, {
                    "steps": steps,
                    "episode_metadata": {"bag_path": path,
                                         "flip_frame": np.int32(flip), "decision": dec},
                }