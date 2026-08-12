#!/usr/bin/env python3
# one-off: one-shot patch switching tfds_builder.py from ros2 bags to .npz caches, applied 2026-08-09, kept for provenance
r"""
patch_builder_for_cache.py -- switch tfds_builder.py from reading ros2 bags to
reading the per-bag .npz caches, and let the BAGS list come from the annotation CSV.

THREE CHANGES
  1. _episode_steps reads a cache (.npz) instead of calling read_bag on a bag dir.
     build_steps is called EXACTLY as before, so annotations, stride, prompt schedule
     and waypoint maths are untouched -- proven bit-identical by verify_cache_parity.py.
  2. The _resize() call in the yield is dropped. Cached images are ALREADY 224x224
     (resized at extract time with the same tf.image.resize). Resizing again would be
     a 224->224 no-op at best and a needless float round-trip at worst.
  3. BAGS can be loaded from annotation_sheet.csv via $SIGNWAY_ANNOTATIONS, so the
     96-row list is generated, not typed. Falls back to the hardcoded BAGS if unset.

RUN (on MSI):
    python patch_builder_for_cache.py
    # then, when building:
    export SIGNWAY_CACHE_DIR=$SCRATCH/bag_cache
    export SIGNWAY_ANNOTATIONS=$SCRATCH/annotation_sheet.csv

Idempotent: running it twice is safe.
"""
import ast
import shutil
import sys
from pathlib import Path

P = Path.home() / "SignWay" / "signway_dataset" / "tfds_builder.py"
src = P.read_text()

if "SIGNWAY_CACHE_DIR" in src:
    print("already patched -- nothing to do")
    sys.exit(0)

shutil.copy(P, P.with_suffix(".py.prebag"))
print(f"backup written to {P.with_suffix('.py.prebag')}")

# ---- 1. constants + CSV-driven BAGS -------------------------------------------
anchor = 'IMAGE_SIZE = (224, 224)'
assert anchor in src, "IMAGE_SIZE anchor not found"
addition = '''IMAGE_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# Cache-backed input. Raw bags live on R2; MSI only keeps small .npz caches
# holding (times, images 224x224, poses). See extract_cache.py.
# ---------------------------------------------------------------------------
import csv as _csv
import os as _os

CACHE_DIR = Path(_os.environ.get(
    "SIGNWAY_CACHE_DIR",
    _os.path.join(_os.environ.get("SCRATCH", "/scratch.global"), "bag_cache")))

# Held-out anchor. Never move these into train -- they are the fixed ruler that
# makes "more data helped" a comparable claim across model versions.
VAL_BAGS = {"rosbag2-keller-e1", "rosbag2-keller-e2", "rosbag2-keller-e3"}


def _load_cache(name):
    """Return (frames, odom) shaped exactly as build_steps expects.

    The cache stores poses already interpolated at each frame time, so the odom
    list we hand back has a sample at every frame timestamp; interp_odom then
    returns those values exactly (verified round-trip 0.0).
    """
    p = Path(name)
    if p.suffix != ".npz":
        p = CACHE_DIR / f"{Path(name).name}.npz"
    if not p.exists():
        raise FileNotFoundError(f"cache not found: {p}")
    d = np.load(p)
    times, images, poses = d["times"], d["images"], d["poses"]
    frames = list(zip(times, images))
    odom = [(float(t), float(q[0]), float(q[1]), float(q[2]))
            for t, q in zip(times, poses)]
    return frames, odom


def _bags_from_csv(path):
    """Build the BAGS list from annotation_sheet.csv. Rows with no decision are
    skipped, so a partially-annotated sheet still builds."""
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
'''
src = src.replace(anchor, addition, 1)

# ---- 2. _episode_steps reads the cache ----------------------------------------
old_ep = ("def _episode_steps(bag_path: str, flip_frame: int, decision: str, "
          "turn_done, stride: int = 1):\n"
          "    frames, odom = read_bag(Path(bag_path), IMAGE_TOPIC, ODOM_TOPIC, ROS_DISTRO)")
new_ep = ("def _episode_steps(bag_path: str, flip_frame: int, decision: str, "
          "turn_done, stride: int = 1):\n"
          "    frames, odom = _load_cache(bag_path)")
assert old_ep in src, "_episode_steps anchor not found"
src = src.replace(old_ep, new_ep, 1)

# ---- 3. drop the redundant resize ---------------------------------------------
old_img = '"observation": {"image": _resize(s["image"])},'
new_img = ('"observation": {"image": np.asarray(s["image"], dtype=np.uint8)},'
           '  # already 224x224 from the cache')
assert old_img in src, "_resize call site not found"
src = src.replace(old_img, new_img, 1)

# ---- 4. BAGS from CSV when the env var is set ----------------------------------
# appended after the existing BAGS literal so the hardcoded list stays as fallback
marker = "IMAGE_TOPIC = "
assert marker in src, "IMAGE_TOPIC anchor not found"
override = '''_ANN = _os.environ.get("SIGNWAY_ANNOTATIONS", "")
if _ANN and Path(_ANN).exists():
    BAGS = _bags_from_csv(_ANN)

IMAGE_TOPIC = '''
src = src.replace(marker, override, 1)

ast.parse(src)
P.write_text(src)
print("patched OK:")
print("  - _episode_steps -> _load_cache(.npz)")
print("  - _resize dropped from the yield (cached images already 224x224)")
print("  - BAGS from $SIGNWAY_ANNOTATIONS when set")




